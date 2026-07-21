"""模块4：双低可信度时的保守时序更新。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from csd.constants import SPEAKERS


class ConservativeUpdater:
    """
    当 audio_conf 与 visual_conf 均低于阈值时，不向当前观测大幅偏移，
    而是与历史说话人分布做指数平滑（保守更新）。
    """

    def __init__(
        self,
        dual_low_threshold: float = 0.35,
        history_momentum: float = 0.72,
        uniform_prior: bool = True,
        visual_led_min_vc: float = 0.52,
        visual_led_margin_max: float = 0.18,
        visual_led_ac_gap: float = 0.05,
    ):
        self.dual_low_threshold = dual_low_threshold
        self.history_momentum = history_momentum
        self.uniform_prior = uniform_prior
        self.visual_led_min_vc = visual_led_min_vc
        self.visual_led_margin_max = visual_led_margin_max
        self.visual_led_ac_gap = visual_led_ac_gap

    @classmethod
    def from_config(cls, config) -> "ConservativeUpdater":
        return cls(
            visual_led_min_vc=getattr(config, "router_visual_led_min_vc", 0.52),
            visual_led_margin_max=getattr(config, "router_visual_led_margin_max", 0.18),
        )

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        vals = np.array([max(0.0, scores.get(s, 0.0)) for s in SPEAKERS], dtype=np.float64)
        if vals.sum() < 1e-8:
            return {s: 1.0 / len(SPEAKERS) for s in SPEAKERS}
        vals /= vals.sum()
        return {s: float(v) for s, v in zip(SPEAKERS, vals)}

    def apply_sequence(
        self,
        fused_scores_list: List[Dict[str, float]],
        audio_confs: Sequence[float],
        visual_confs: Sequence[float],
        voice_margins: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        updated: List[Dict[str, float]] = []
        modes: List[str] = []
        history = {s: 1.0 / len(SPEAKERS) for s in SPEAKERS} if self.uniform_prior else {s: 0.0 for s in SPEAKERS}

        for i, raw in enumerate(fused_scores_list):
            ac = audio_confs[i] if i < len(audio_confs) else 0.0
            vc = visual_confs[i] if i < len(visual_confs) else 0.0
            margin = voice_margins[i] if voice_margins is not None and i < len(voice_margins) else 1.0
            current = self._normalize(raw)

            if ac < self.dual_low_threshold and vc < self.dual_low_threshold:
                m = self.history_momentum
                blended = {
                    s: m * history.get(s, 0.0) + (1.0 - m) * current.get(s, 0.0)
                    for s in SPEAKERS
                }
                out = self._normalize(blended)
                modes.append("conservative")
            else:
                out = current
                if (
                    vc >= self.visual_led_min_vc
                    and vc > ac + self.visual_led_ac_gap
                    and margin <= self.visual_led_margin_max
                ):
                    modes.append("visual_led")
                elif ac >= 0.65 and ac >= vc:
                    modes.append("audio_led")
                else:
                    modes.append("balanced")

            updated.append(out)
            history = out

        return updated, modes
