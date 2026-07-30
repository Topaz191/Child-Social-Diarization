"""SpeakAhead（发言准备度）：独立视觉先验，只进入 visual_probs，不进 visual_conf。"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from csd.constants import SPEAKERS
from csd.perception.head_pose import SlotVisualTimeline

logger = logging.getLogger(__name__)


def _load_prepare_mod():
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "scripts" / "prepare_readiness_xianyang.py"
    spec = importlib.util.spec_from_file_location("prepare_readiness_xianyang", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_train_mod():
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "scripts" / "train_readiness_lstm.py"
    spec = importlib.util.spec_from_file_location("train_readiness_lstm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def timeline_to_frame_df(timeline: SlotVisualTimeline, fps: float) -> pd.DataFrame:
    """把 SlotVisualTimeline 转成与 prepare 一致的帧表（含 speaker/t/头姿嘴动）。"""
    rows: List[dict] = []
    for slot_id, frames in (timeline.frames or {}).items():
        spk = timeline.slot_to_speaker.get(slot_id)
        if not spk:
            continue
        for frame_idx, pose in frames.items():
            rows.append(
                {
                    "speaker": str(spk).upper(),
                    "t": float(frame_idx) / float(fps),
                    "yaw": float(pose.yaw),
                    "pitch": float(pose.pitch),
                    "roll": float(pose.roll),
                    "mouth_opening": float(pose.mouth_opening),
                    "visibility": float(pose.visibility),
                    "side_face_weight": float(pose.side_face_weight),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["speaker", "t", "yaw", "pitch", "roll", "mouth_opening", "visibility", "side_face_weight"]
        )
    return pd.DataFrame(rows).sort_values(["speaker", "t"]).reset_index(drop=True)


def segment_mouth_visibility(
    timeline: SlotVisualTimeline,
    fps: float,
    start: float,
    end: float,
    speakers: Sequence[str] = SPEAKERS,
) -> Dict[str, float]:
    """段内各说话人平均嘴部可见性（供遮挡排除）。"""
    out = {s: 0.0 for s in speakers}
    counts = {s: 0 for s in speakers}
    f0 = int(max(0, start * fps))
    f1 = int(max(f0, end * fps))
    for slot_id, frames in (timeline.frames or {}).items():
        spk = str(timeline.slot_to_speaker.get(slot_id, "")).upper()
        if spk not in out:
            continue
        for fi, pose in frames.items():
            if f0 <= int(fi) <= f1:
                out[spk] += float(pose.visibility)
                counts[spk] += 1
    for s in speakers:
        if counts[s] > 0:
            out[s] /= counts[s]
    return out


class SpeakAheadScorer:
    """加载 readiness LSTM，对每个说话人在 t_pred 处输出准备度 ∈ [0,1]。"""

    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        import torch

        self.device = torch.device(device if device != "auto" else "cpu")
        train_mod = _load_train_mod()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        input_dim = int(ckpt.get("input_dim", len(ckpt.get("mean", []))))
        hidden = int(ckpt.get("hidden", 32))
        bidirectional = bool(ckpt.get("bidirectional", False))
        self.model = train_mod.ReadinessLSTM(
            input_dim=input_dim,
            hidden=hidden,
            bidirectional=bidirectional,
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.feature_cols = train_mod.resolve_feature_cols(input_dim, ckpt.get("feature_cols"))
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(ckpt["std"], dtype=np.float32).reshape(-1)
        self.seq_len = int(ckpt.get("seq_len", 16))
        self.window_sec = float(ckpt.get("window_sec", 0.75))
        self._ensure_feat = train_mod._ensure_feature_columns
        self._prep = _load_prepare_mod()
        self._feat_cache_key: Optional[int] = None
        self._feat_df: Optional[pd.DataFrame] = None
        logger.info(
            "SpeakAhead 已加载: %s dim=%d window=%.2fs",
            ckpt_path,
            input_dim,
            self.window_sec,
        )

    def _features_from_timeline(self, timeline: SlotVisualTimeline, fps: float) -> pd.DataFrame:
        key = id(timeline)
        if self._feat_cache_key == key and self._feat_df is not None:
            return self._feat_df
        raw = timeline_to_frame_df(timeline, fps)
        if len(raw) == 0:
            self._feat_cache_key, self._feat_df = key, raw
            return raw
        enriched = self._prep.enrich_frame_features(raw)
        enriched = self._ensure_feat(enriched, self.feature_cols)
        self._feat_cache_key, self._feat_df = key, enriched
        return enriched

    def score_at(
        self,
        timeline: SlotVisualTimeline,
        fps: float,
        t_pred: float,
        speakers: Sequence[str] = SPEAKERS,
    ) -> Dict[str, float]:
        """在 t_pred 处对每人过去 window_sec 打 readiness。"""
        from csd.eval.turn_event_protocol import score_window_readiness

        feat = self._features_from_timeline(timeline, fps)
        scores: Dict[str, float] = {s: 0.0 for s in speakers}
        if feat is None or len(feat) == 0:
            return scores
        for spk in speakers:
            s = score_window_readiness(
                self.model,
                feat,
                spk,
                t_pred,
                window_sec=self.window_sec,
                seq_len=self.seq_len,
                feature_cols=self.feature_cols,
                mean=self.mean,
                std=self.std,
                device=self.device,
            )
            if s is not None:
                scores[spk] = float(np.clip(s, 0.0, 1.0))
        return scores

    def score_segment(
        self,
        timeline: SlotVisualTimeline,
        fps: float,
        start: float,
        end: float,
        speakers: Sequence[str] = SPEAKERS,
        anchor: str = "onset",
    ) -> Dict[str, float]:
        """
        段级准备度（与训练对齐）。

        训练正样本窗口为 [onset - window_sec, onset]，故默认在段起点 onset 打分。
        长段应先按 hop 切成子段再调用，而不是在 mid 偷懒。
        """
        if anchor == "mid":
            t_pred = 0.5 * (float(start) + float(end))
        else:
            # onset：看开口前 window；略加 lead=0 保持与训练一致
            t_pred = float(start)
        return self.score_at(timeline, fps, t_pred, speakers=speakers)

    def score_onset(
        self,
        timeline: SlotVisualTimeline,
        fps: float,
        onset: float,
        speakers: Sequence[str] = SPEAKERS,
    ) -> Dict[str, float]:
        """显式按开口时刻打分（诊断 / GT onset 评估用）。"""
        return self.score_at(timeline, fps, float(onset), speakers=speakers)
