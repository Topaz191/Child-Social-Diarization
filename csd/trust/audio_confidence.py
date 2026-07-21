"""声纹可信度估计：判断当前时间窗口内音频身份线索是否可靠。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from csd.constants import SPEAKERS
from csd.core.utils import cosine_similarity, l2_normalize
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment


@dataclass
class SegmentVoiceStats:
    index: int
    start_time: float
    end_time: float
    duration: float
    raw_sims: Dict[str, float]
    margin: float
    top1: str
    embedding: Optional[np.ndarray] = None


class AudioConfidenceEstimator:
    """模块1.1：margin 为主，一致性/段长为辅，聚类仅作弱辅助。"""

    def __init__(
        self,
        window_sec: float = 4.0,
        target_margin: float = 0.12,
        weights: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.window_sec = window_sec
        self.target_margin = target_margin
        self.weights = weights or (0.50, 0.30, 0.15, 0.05)

    def build_segment_stats(
        self,
        segments: Sequence[SpeechSegment],
        extractor: SpeakerEmbeddingExtractor,
    ) -> List[SegmentVoiceStats]:
        return [self._stats_for_segment(i, seg, extractor) for i, seg in enumerate(segments)]

    def _stats_for_segment(
        self,
        index: int,
        segment: SpeechSegment,
        extractor: SpeakerEmbeddingExtractor,
    ) -> SegmentVoiceStats:
        raw_sims = {s: 0.0 for s in SPEAKERS}
        margin = 0.0
        top1 = "S1"
        if segment.speaker_embedding is not None and extractor.enrolled_speakers:
            emb = l2_normalize(segment.speaker_embedding)
            for spk in SPEAKERS:
                ref = extractor.enrolled_speakers.get(spk.lower())
                if ref is not None:
                    raw_sims[spk] = max(0.0, cosine_similarity(emb, ref))
            ordered = sorted(raw_sims.values(), reverse=True)
            margin = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else float(ordered[0] if ordered else 0.0)
            top1 = max(raw_sims, key=raw_sims.get)
        return SegmentVoiceStats(
            index=index,
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            raw_sims=raw_sims,
            margin=margin,
            top1=top1,
            embedding=l2_normalize(segment.speaker_embedding) if segment.speaker_embedding is not None else None,
        )

    def _window_stats(self, all_stats: Sequence[SegmentVoiceStats], center_idx: int) -> List[SegmentVoiceStats]:
        center = all_stats[center_idx]
        t_mid = 0.5 * (center.start_time + center.end_time)
        half = self.window_sec / 2.0
        return [s for s in all_stats if s.end_time > t_mid - half and s.start_time < t_mid + half]

    @staticmethod
    def _consistency_score(window: Sequence[SegmentVoiceStats]) -> float:
        if not window:
            return 0.0
        labels = [s.top1 for s in window]
        counts: Dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        return max(counts.values()) / len(window)

    @staticmethod
    def _duration_score(window: Sequence[SegmentVoiceStats]) -> float:
        if not window:
            return 0.0
        return float(np.mean([min(s.duration / 1.0, 1.0) for s in window]))

    @staticmethod
    def _cluster_aux(window: Sequence[SegmentVoiceStats]) -> float:
        embs = [s.embedding for s in window if s.embedding is not None]
        if len(embs) < 3:
            return 0.5
        labels = [s.top1 for s in window if s.embedding is not None][: len(embs)]
        if len(set(labels)) < 2:
            return 0.5
        try:
            from sklearn.metrics import davies_bouldin_score
            dbi = float(davies_bouldin_score(np.stack(embs), labels))
            return float(np.clip(1.0 - dbi / 3.0, 0.0, 1.0))
        except Exception:
            return 0.5

    def confidence_for_segment(self, all_stats: Sequence[SegmentVoiceStats], center_idx: int) -> float:
        window = self._window_stats(all_stats, center_idx)
        if not window:
            return 0.0
        margin_score = float(np.clip(np.mean([s.margin for s in window]) / self.target_margin, 0.0, 1.0))
        w_m, w_c, w_d, w_k = self.weights
        conf = (
            w_m * margin_score
            + w_c * self._consistency_score(window)
            + w_d * self._duration_score(window)
            + w_k * self._cluster_aux(window)
        )
        return float(np.clip(conf, 0.0, 1.0))

    def compute_all(
        self,
        segments: Sequence[SpeechSegment],
        extractor: SpeakerEmbeddingExtractor,
    ) -> List[float]:
        stats = self.build_segment_stats(segments, extractor)
        return [self.confidence_for_segment(stats, i) for i in range(len(stats))]
