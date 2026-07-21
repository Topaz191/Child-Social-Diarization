"""说话人序列 CRF：维特比解码平滑段级标签。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from csd.constants import SPEAKERS


class SpeakerSequenceCRF:
    """线性链 CRF：仅 S1/S2/S3，局部发射分优先于全局惯性。"""

    def __init__(
        self,
        speakers: Sequence[str],
        stay_prob: float = 0.62,
        min_emit_prob: float = 1e-4,
        emission_scale: float = 4.0,
    ):
        self.speakers = list(speakers)
        self.states = list(speakers)
        self.n_states = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.min_emit_prob = min_emit_prob
        self.emission_scale = emission_scale
        self.trans = self._build_transition(stay_prob)

    def _build_transition(self, stay_prob: float) -> np.ndarray:
        n = self.n_states
        off_diag = (1.0 - stay_prob) / max(n - 1, 1)
        trans = np.full((n, n), off_diag, dtype=np.float64)
        np.fill_diagonal(trans, stay_prob)
        trans = np.clip(trans, 1e-6, 1.0)
        trans /= trans.sum(axis=1, keepdims=True)
        return np.log(trans)

    @staticmethod
    def _normalize_emit(scores: Dict[str, float], speakers: Sequence[str]) -> Dict[str, float]:
        keys = [k for k in speakers if k in scores]
        if not keys:
            return {s: 1.0 / len(speakers) for s in speakers}
        vals = np.array([max(0.0, scores.get(k, 0.0)) for k in keys], dtype=np.float64)
        if vals.sum() < 1e-8:
            return {k: 1.0 / len(keys) for k in keys}
        vals /= vals.sum()
        return {k: float(v) for k, v in zip(keys, vals)}

    def decode(self, segment_emissions: List[Dict[str, float]]) -> List[str]:
        if not segment_emissions:
            return []

        t_len = len(segment_emissions)
        n = self.n_states
        log_trans = self.trans

        emit_log = np.full((t_len, n), np.log(self.min_emit_prob), dtype=np.float64)
        for t, raw in enumerate(segment_emissions):
            norm = self._normalize_emit(raw, self.speakers)
            for spk, prob in norm.items():
                idx = self.state_index.get(spk)
                if idx is not None:
                    emit_log[t, idx] = self.emission_scale * np.log(max(prob, self.min_emit_prob))

        dp = np.full((t_len, n), -np.inf, dtype=np.float64)
        back = np.zeros((t_len, n), dtype=np.int32)
        dp[0] = emit_log[0]

        for t in range(1, t_len):
            for j in range(n):
                scores = dp[t - 1] + log_trans[:, j]
                back[t, j] = int(np.argmax(scores))
                dp[t, j] = scores[back[t, j]] + emit_log[t, j]

        path = np.zeros(t_len, dtype=np.int32)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(t_len - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]

        return [self.states[i] for i in path]
