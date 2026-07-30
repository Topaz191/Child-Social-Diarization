"""
MuVAP 对齐的话轮事件协议（Silent / Active × Shift-Hold / NSP）。

协议对齐自 arXiv:2606.16731 §6（参数可调）:
  - 段前至少 solo_min_sec 单人连续说话
  - 与下一段间隙 gap ∈ (0, gap_max_sec]
  - Silent: t = end + offset_sec
  - Active: t = end - offset_sec（仍在当前段内）
  - Shift: 下一段说话人 ≠ 当前；Hold: 相同
  - NSP 标签: 下一段说话人（Hold 时即当前说话人）

SpeakAhead 决策（无额外 probe）:
  - 在 t 处对每个说话人算 readiness（过去 window_sec）
  - NSP: argmax readiness
  - Shift-Hold: pred_next != prev_speaker → Shift
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


STUDENT_SPEAKERS = ("S1", "S2", "S3", "S4")


@dataclass
class TurnEvent:
    mode: str  # silent | active
    t_pred: float
    prev_speaker: str
    next_speaker: str
    label_shift: int  # 1=SHIFT, 0=HOLD
    gap: float
    seg_end: float
    session_id: str = ""


def build_muvap_events(
    segments: Sequence[dict],
    *,
    solo_min_sec: float = 1.0,
    gap_max_sec: float = 3.0,
    offset_sec: float = 0.1,
    speakers: Sequence[str] = STUDENT_SPEAKERS,
    session_id: str = "",
) -> List[TurnEvent]:
    """从 GT 说话段构造 Silent/Active 事件。"""
    segs = sorted(
        [s for s in segments if str(s.get("speaker", "")).upper() in speakers],
        key=lambda x: float(x["start"]),
    )
    events: List[TurnEvent] = []
    for i, cur in enumerate(segs):
        if i + 1 >= len(segs):
            break
        nxt = segs[i + 1]
        prev_spk = str(cur["speaker"]).upper()
        next_spk = str(nxt["speaker"]).upper()
        start, end = float(cur["start"]), float(cur["end"])
        nxt_start = float(nxt["start"])
        dur = end - start
        gap = nxt_start - end
        if dur < solo_min_sec:
            continue
        if gap <= 0 or gap > gap_max_sec:
            continue
        # 要求当前段末尾 solo_min 内无人重叠抢话（用相邻段近似：段本身时长已够）
        label_shift = 0 if next_spk == prev_spk else 1

        # Silent: 开口结束后 +offset
        t_silent = end + offset_sec
        if t_silent < nxt_start:  # 仍在间隙内
            events.append(
                TurnEvent(
                    mode="silent",
                    t_pred=t_silent,
                    prev_speaker=prev_spk,
                    next_speaker=next_spk,
                    label_shift=label_shift,
                    gap=gap,
                    seg_end=end,
                    session_id=session_id,
                )
            )

        # Active: 结束前 -offset，且保证前方仍有足够 solo
        t_active = end - offset_sec
        if t_active - start >= solo_min_sec - 1e-6 and t_active > start:
            events.append(
                TurnEvent(
                    mode="active",
                    t_pred=t_active,
                    prev_speaker=prev_spk,
                    next_speaker=next_spk,
                    label_shift=label_shift,
                    gap=gap,
                    seg_end=end,
                    session_id=session_id,
                )
            )
    return events


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return float("nan")
    f1s = []
    for cls in (0, 1):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def score_window_readiness(
    model,
    feat_df,
    speaker: str,
    t_pred: float,
    *,
    window_sec: float,
    seq_len: int,
    feature_cols: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device,
) -> Optional[float]:
    """某人在 t_pred 处、过去 window_sec 的 readiness。"""
    import torch

    t0, t1 = t_pred - window_sec, t_pred
    if t0 < 0:
        return None
    spk_df = feat_df[feat_df["speaker"].astype(str).str.upper() == speaker].sort_values("t")
    sub = spk_df[(spk_df["t"] >= t0 - 1e-6) & (spk_df["t"] <= t1 + 1e-6)]
    if len(sub) < max(2, seq_len // 4):
        return None
    ts = sub["t"].to_numpy(dtype=np.float64)
    feats = sub[list(feature_cols)].to_numpy(dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    ts, feats = ts[uniq], feats[uniq]
    if len(ts) < 2:
        return None
    grid = np.linspace(t0, t1, seq_len)
    mat = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    for j in range(len(feature_cols)):
        mat[:, j] = np.interp(grid, ts, feats[:, j]).astype(np.float32)
    mat = (mat - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(mat[None]).to(device)
        return float(model.readiness_score(x).cpu().item())


def predict_event_with_readiness(
    model,
    feat_df,
    event: TurnEvent,
    *,
    speakers: Sequence[str],
    window_sec: float,
    seq_len: int,
    feature_cols: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device,
) -> Optional[dict]:
    scores: Dict[str, float] = {}
    for spk in speakers:
        s = score_window_readiness(
            model,
            feat_df,
            spk,
            event.t_pred,
            window_sec=window_sec,
            seq_len=seq_len,
            feature_cols=feature_cols,
            mean=mean,
            std=std,
            device=device,
        )
        if s is not None:
            scores[spk] = s
    if event.prev_speaker not in scores or len(scores) < 2:
        return None
    pred_next = max(scores, key=scores.get)
    pred_shift = 0 if pred_next == event.prev_speaker else 1
    return {
        "mode": event.mode,
        "t_pred": event.t_pred,
        "prev_speaker": event.prev_speaker,
        "next_speaker_gt": event.next_speaker,
        "label_shift": event.label_shift,
        "pred_shift": pred_shift,
        "pred_next": pred_next,
        "scores": scores,
        "session_id": event.session_id,
        "correct_shift": int(pred_shift == event.label_shift),
        "correct_nsp": int(pred_next == event.next_speaker),
    }


def _summarize_mode(rows: List[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "shift_hold_macro_f1": float("nan"),
            "shift_hold_acc": float("nan"),
            "nsp_acc": float("nan"),
            "n_shift": 0,
            "n_hold": 0,
        }
    y_true = [r["label_shift"] for r in rows]
    y_pred = [r["pred_shift"] for r in rows]
    nsp_acc = float(np.mean([r["correct_nsp"] for r in rows]))
    return {
        "n": len(rows),
        "shift_hold_macro_f1": macro_f1(y_true, y_pred),
        "shift_hold_acc": float(np.mean([r["correct_shift"] for r in rows])),
        "nsp_acc": nsp_acc,
        "n_shift": int(sum(y_true)),
        "n_hold": int(len(y_true) - sum(y_true)),
    }


def evaluate_readiness_on_events(
    model,
    session_dirs: Sequence,
    *,
    feature_cols: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    window_sec: float,
    device,
    speakers: Sequence[str] = ("S1", "S2", "S3"),
    solo_min_sec: float = 1.0,
    gap_max_sec: float = 3.0,
    offset_sec: float = 0.1,
    ensure_feat_fn=None,
) -> dict:
    """
    在多个场次目录上评 MuVAP 事件协议。
    每场需要: gt_segments.json + frame_features.csv
    """
    import json
    from pathlib import Path

    import pandas as pd

    all_rows: List[dict] = []
    per_session = []
    for d in session_dirs:
        d = Path(d)
        seg_p = d / "gt_segments.json"
        feat_p = d / "frame_features.csv"
        if not seg_p.exists() or not feat_p.exists():
            continue
        segments = json.loads(seg_p.read_text(encoding="utf-8"))
        events = build_muvap_events(
            segments,
            solo_min_sec=solo_min_sec,
            gap_max_sec=gap_max_sec,
            offset_sec=offset_sec,
            speakers=speakers,
            session_id=d.name,
        )
        feat = pd.read_csv(feat_p)
        if ensure_feat_fn is not None:
            feat = ensure_feat_fn(feat, feature_cols)
        rows = []
        for ev in events:
            r = predict_event_with_readiness(
                model,
                feat,
                ev,
                speakers=speakers,
                window_sec=window_sec,
                seq_len=seq_len,
                feature_cols=feature_cols,
                mean=mean,
                std=std,
                device=device,
            )
            if r is not None:
                rows.append(r)
        per_session.append({"session": d.name, "n_events": len(events), "n_scored": len(rows), **_summarize_mode(rows)})
        all_rows.extend(rows)

    silent = [r for r in all_rows if r["mode"] == "silent"]
    active = [r for r in all_rows if r["mode"] == "active"]
    report = {
        "protocol": {
            "name": "muvap_aligned_turn_events",
            "ref": "arXiv:2606.16731 §6",
            "solo_min_sec": solo_min_sec,
            "gap_max_sec": gap_max_sec,
            "offset_sec": offset_sec,
            "decision": "nsp=argmax readiness; shift=(pred_next!=prev)",
        },
        "overall": _summarize_mode(all_rows),
        "silent": _summarize_mode(silent),
        "active": _summarize_mode(active),
        "n_sessions_used": int(sum(1 for s in per_session if s["n_scored"] > 0)),
        "per_session": per_session,
        "events_exportable": [  # 精简，避免巨大 report
            {
                "mode": r["mode"],
                "session_id": r["session_id"],
                "t_pred": r["t_pred"],
                "prev": r["prev_speaker"],
                "next_gt": r["next_speaker_gt"],
                "pred_next": r["pred_next"],
                "label_shift": r["label_shift"],
                "pred_shift": r["pred_shift"],
            }
            for r in all_rows[:500]
        ],
    }
    return report


def events_to_dicts(events: Sequence[TurnEvent]) -> List[dict]:
    return [asdict(e) for e in events]
