"""儿童嘴动幅度标定：把 MAR 活跃度映射为 [0,1] 说话似然。

训练产物 lip_amp_model.pt 含：
  - 手写窗口特征 → 小 MLP → P(speaking | frontal lip)
  - 正样本活跃度分位数（替换 visual_conf 里硬编码 0.015）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


FEATURE_NAMES = (
    "mean_mar",
    "std_mar",
    "max_mar",
    "p90_mar",
    "activity",
    "mean_side",
    "mean_abs_yaw",
)


def mouth_activity(mars: np.ndarray) -> float:
    mars = np.asarray(mars, dtype=np.float64)
    if mars.size == 0:
        return 0.0
    if mars.size == 1:
        return float(mars[0])
    return float(np.var(mars) + np.mean(np.abs(np.diff(mars))))


def window_feature_vector(
    mars: Sequence[float],
    side_weights: Sequence[float],
    yaws: Sequence[float],
) -> Optional[np.ndarray]:
    mars_a = np.asarray(mars, dtype=np.float64)
    if mars_a.size < 2:
        return None
    sides = np.asarray(side_weights, dtype=np.float64) if len(side_weights) else np.ones_like(mars_a)
    yaws_a = np.asarray(yaws, dtype=np.float64) if len(yaws) else np.zeros_like(mars_a)
    n = min(len(mars_a), len(sides), len(yaws_a))
    mars_a, sides, yaws_a = mars_a[:n], sides[:n], yaws_a[:n]
    feat = np.array(
        [
            float(np.mean(mars_a)),
            float(np.std(mars_a)),
            float(np.max(mars_a)),
            float(np.percentile(mars_a, 90)),
            mouth_activity(mars_a),
            float(np.mean(sides)),
            float(np.mean(np.abs(yaws_a))),
        ],
        dtype=np.float32,
    )
    return feat


class LipAmpMLP(nn.Module):
    """轻量 MLP：窗口特征 → logit(说话)。"""

    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


@dataclass
class LipAmplitudeCalibrator:
    """推理侧：活跃度分位归一化 + 可选 MLP 分数。"""

    activity_scale: float = 0.015
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    model: Optional[LipAmpMLP] = None
    pos_activity_p50: float = 0.0
    pos_activity_p90: float = 0.015
    min_side_face: float = 0.45

    def normalize_activity(self, activity: float) -> float:
        scale = max(float(self.activity_scale), 1e-6)
        return float(np.clip(activity / scale, 0.0, 1.0))

    def score_from_arrays(
        self,
        mars: Sequence[float],
        side_weights: Sequence[float],
        yaws: Optional[Sequence[float]] = None,
    ) -> float:
        yaws = yaws if yaws is not None else [0.0] * len(mars)
        feat = window_feature_vector(mars, side_weights, yaws)
        if feat is None:
            return 0.0
        activity = float(feat[FEATURE_NAMES.index("activity")])
        side = float(feat[FEATURE_NAMES.index("mean_side")])
        rule = self.normalize_activity(activity)
        if side < self.min_side_face:
            rule *= float(np.clip(side / max(self.min_side_face, 1e-6), 0.0, 1.0))

        if self.model is None or self.mean is None or self.std is None:
            return rule

        x = (feat - self.mean) / self.std
        with torch.no_grad():
            p = float(self.model.prob(torch.from_numpy(x[None])).cpu().item())
        # 规则尺度与学习分数融合，避免小样本 MLP 过冲
        return float(np.clip(0.5 * rule + 0.5 * p, 0.0, 1.0))

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "LipAmplitudeCalibrator":
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        model = LipAmpMLP(
            input_dim=int(ckpt.get("input_dim", len(FEATURE_NAMES))),
            hidden=int(ckpt.get("hidden", 32)),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return cls(
            activity_scale=float(ckpt.get("activity_scale", ckpt.get("pos_activity_p75", 0.015))),
            mean=np.asarray(ckpt["mean"], dtype=np.float32),
            std=np.asarray(ckpt["std"], dtype=np.float32),
            model=model,
            pos_activity_p50=float(ckpt.get("pos_activity_p50", 0.0)),
            pos_activity_p90=float(ckpt.get("pos_activity_p90", 0.015)),
            min_side_face=float(ckpt.get("min_side_face", 0.45)),
        )

    def to_json_stats(self) -> Dict[str, float]:
        return {
            "activity_scale": self.activity_scale,
            "pos_activity_p50": self.pos_activity_p50,
            "pos_activity_p90": self.pos_activity_p90,
            "min_side_face": self.min_side_face,
        }


def compute_activity_percentiles(activities: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(activities), dtype=np.float64)
    if arr.size == 0:
        return {"p50": 0.0, "p75": 0.015, "p90": 0.015}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }
