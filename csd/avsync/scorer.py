"""运行时：对 VAD 段计算各说话人音–口同步分数。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from csd.avsync.matcher import SyncMatcher
from csd.avsync.mel import mel_for_time_range
from csd.avsync.mouth_crop import five_frame_indices, load_mouth_tensor_for_window
from csd.avsync.mtd_backend import MTDFeatureExtractor
from csd.constants import SPEAKERS
from csd.perception.face_tracker import FaceTrack

logger = logging.getLogger(__name__)


class AVSyncScorer:
    """
    冻结 MTD 提特征 +（可选）训练好的 SyncMatcher。
    若无 matcher，则用 MTD 预训练 logit 的 sigmoid 作为分数。
    """

    def __init__(
        self,
        extractor: Optional[MTDFeatureExtractor] = None,
        matcher_path: Optional[Path] = None,
        device: str = "auto",
        window_sec: float = 0.2,
    ):
        self.extractor = extractor or MTDFeatureExtractor(device=device, freeze=True)
        self.window_sec = window_sec
        self.matcher: Optional[SyncMatcher] = None
        self.device = self.extractor.device
        if matcher_path is not None and Path(matcher_path).is_file():
            ckpt = torch.load(str(matcher_path), map_location="cpu")
            self.matcher = SyncMatcher(
                feat_dim=int(ckpt.get("feat_dim", 200)),
                hidden=int(ckpt.get("hidden", 256)),
                mode=str(ckpt.get("mode", "mlp")),
            )
            self.matcher.load_state_dict(ckpt["model_state"])
            self.matcher.to(self.device)
            self.matcher.eval()
            logger.info("已加载 SyncMatcher: %s", matcher_path)

    def score_segment(
        self,
        video_path: str,
        wav: np.ndarray,
        sr: int,
        tracks: Dict[int, FaceTrack],
        spk_to_track_ids: Dict[str, List[int]],
        start_time: float,
        end_time: float,
        fps: float,
    ) -> Dict[str, float]:
        """返回各说话人同步分 [0,1]；无法估计则为 0。"""
        mid = 0.5 * (start_time + end_time)
        w0, w1 = mid - self.window_sec / 2, mid + self.window_sec / 2
        # 段较长时在段内多点取 max
        centers = [mid]
        if end_time - start_time > 0.6:
            centers = [
                start_time + 0.25 * (end_time - start_time),
                mid,
                start_time + 0.75 * (end_time - start_time),
            ]
        mel = mel_for_time_range(wav, sr, w0, w1, mel_width=16)
        scores = {s: 0.0 for s in SPEAKERS}

        for spk in SPEAKERS:
            tids = spk_to_track_ids.get(spk) or []
            if not tids:
                continue
            track = max(
                (tracks[t] for t in tids if t in tracks),
                key=lambda tr: len(tr.detections),
                default=None,
            )
            if track is None:
                continue
            best = 0.0
            for c in centers:
                fidx = five_frame_indices(int(round(c * fps)), fps)
                face = load_mouth_tensor_for_window(video_path, track, fidx)
                if face is None:
                    continue
                mel_c = mel_for_time_range(wav, sr, c - self.window_sec / 2, c + self.window_sec / 2, mel_width=16)
                v, a, logit = self.extractor.encode_numpy(face, mel_c, return_logit=True)
                if self.matcher is not None:
                    with torch.no_grad():
                        vt = torch.from_numpy(v[None]).to(self.device)
                        at = torch.from_numpy(a[None]).to(self.device)
                        p = float(self.matcher.score(vt, at).cpu().item())
                else:
                    p = float(1.0 / (1.0 + np.exp(-(logit or 0.0))))
                best = max(best, p)
            scores[spk] = best
        return scores
