"""态势调度器：MRAF 式双模态可信度归一化权重 + 融合（含路由校准）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from csd.constants import SPEAKERS


class SituationRouter:
    """
    根据 audio_conf 与 visual_conf 计算模态权重（MRAF Eq.11 风格）：
      w_a = r_a / (r_a + r_v + eps)
      w_v = r_v / (r_a + r_v + eps)
    final = w_a * audio_probs + w_v * visual_probs

    路由校准：声纹 margin 高 / 嘴动弱 / 声视 top1 冲突时限制 visual 权重。
    """

    def __init__(
        self,
        visual_lip_weight: float = 0.55,
        visual_attn_weight: float = 0.45,
        attn_only_penalty: float = 0.45,
        eps: float = 1e-6,
        voice_margin_cap: float = 0.18,
        visual_cap_strong_audio: float = 0.28,
        visual_cap_weak_lip: float = 0.22,
        lip_min_for_visual: float = 0.12,
        disagree_margin: float = 0.15,
        visual_conf_floor: float = 0.25,
    ):
        self.visual_lip_weight = visual_lip_weight
        self.visual_attn_weight = visual_attn_weight
        self.attn_only_penalty = attn_only_penalty
        self.eps = eps
        self.voice_margin_cap = voice_margin_cap
        self.visual_cap_strong_audio = visual_cap_strong_audio
        self.visual_cap_weak_lip = visual_cap_weak_lip
        self.lip_min_for_visual = lip_min_for_visual
        self.disagree_margin = disagree_margin
        self.visual_conf_floor = visual_conf_floor

    @classmethod
    def from_config(cls, config) -> "SituationRouter":
        return cls(
            voice_margin_cap=getattr(config, "router_voice_margin_cap", 0.18),
            visual_cap_strong_audio=getattr(config, "router_visual_cap_strong_audio", 0.28),
            visual_cap_weak_lip=getattr(config, "router_visual_cap_weak_lip", 0.22),
            lip_min_for_visual=getattr(config, "router_lip_min_for_visual", 0.12),
            disagree_margin=getattr(config, "router_disagree_margin", 0.15),
        )

    @staticmethod
    def modality_weights(audio_conf: float, visual_conf: float, eps: float = 1e-6) -> Tuple[float, float]:
        ra = float(np.clip(audio_conf, 0.0, 1.0))
        rv = float(np.clip(visual_conf, 0.0, 1.0))
        w_a = ra / (ra + rv + eps)
        w_v = rv / (ra + rv + eps)
        return w_a, w_v

    @staticmethod
    def voice_margin(audio_probs: Dict[str, float]) -> float:
        vals = sorted((float(v) for v in audio_probs.values()), reverse=True)
        if len(vals) >= 2:
            return vals[0] - vals[1]
        return vals[0] if vals else 0.0

    @staticmethod
    def _normalize_probs(scores: Dict[str, float]) -> Dict[str, float]:
        vals = np.array([max(0.0, scores.get(s, 0.0)) for s in SPEAKERS], dtype=np.float64)
        if vals.sum() < 1e-8:
            return {s: 1.0 / len(SPEAKERS) for s in SPEAKERS}
        vals /= vals.sum()
        return {s: float(v) for s, v in zip(SPEAKERS, vals)}

    def build_visual_probs(
        self,
        lip_scores: Dict[str, float],
        attention_received: Dict[str, float],
        voice_probs: Dict[str, float],
        visual_conf: float,
    ) -> Dict[str, float]:
        """visual_conf 低时使用中性先验，避免噪声视觉信号主导。"""
        if visual_conf < self.visual_conf_floor:
            return {s: 1.0 / len(SPEAKERS) for s in SPEAKERS}

        raw = {
            spk: self.visual_lip_weight * lip_scores.get(spk, 0.0)
            + self.visual_attn_weight * attention_received.get(spk, 0.0)
            for spk in SPEAKERS
        }
        for spk in SPEAKERS:
            if lip_scores.get(spk, 0.0) < 0.10 and voice_probs.get(spk, 0.0) < 0.20:
                if attention_received.get(spk, 0.0) > 0.15:
                    raw[spk] *= self.attn_only_penalty
        probs = self._normalize_probs(raw)
        scale = float(np.clip(visual_conf, 0.0, 1.0))
        uniform = 1.0 / len(SPEAKERS)
        return {s: scale * probs[s] + (1.0 - scale) * uniform for s in SPEAKERS}

    def _calibrate_weights(
        self,
        w_a: float,
        w_v: float,
        audio_confidence: float,
        voice_margin: float,
        lip_scores: Dict[str, float],
        audio_probs: Dict[str, float],
        visual_probs: Dict[str, float],
    ) -> Tuple[float, float]:
        lip_max = max(lip_scores.values()) if lip_scores else 0.0

        if voice_margin >= self.voice_margin_cap and audio_confidence >= 0.55:
            w_v = min(w_v, self.visual_cap_strong_audio)
            w_a = 1.0 - w_v

        if lip_max < self.lip_min_for_visual:
            w_v = min(w_v, self.visual_cap_weak_lip)
            w_a = 1.0 - w_v

        audio_top = max(audio_probs, key=audio_probs.get)
        visual_top = max(visual_probs, key=visual_probs.get)
        if (
            audio_top != visual_top
            and voice_margin >= self.disagree_margin
            and audio_confidence >= 0.50
        ):
            w_v = min(w_v, self.visual_cap_weak_lip)
            w_a = 1.0 - w_v

        return w_a, w_v

    def fuse(
        self,
        audio_confidence: float,
        visual_confidence: float,
        audio_probs: Dict[str, float],
        lip_scores: Dict[str, float],
        attention_received: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, float, Dict[str, float]]:
        margin = self.voice_margin(audio_probs)
        w_a, w_v = self.modality_weights(audio_confidence, visual_confidence, self.eps)
        visual_p = self.build_visual_probs(lip_scores, attention_received, audio_probs, visual_confidence)
        w_a, w_v = self._calibrate_weights(
            w_a, w_v, audio_confidence, margin, lip_scores, audio_probs, visual_p
        )
        audio_n = self._normalize_probs(audio_probs)

        fused = {spk: w_a * audio_n.get(spk, 0.0) + w_v * visual_p.get(spk, 0.0) for spk in SPEAKERS}
        return fused, w_a, w_v, visual_p
