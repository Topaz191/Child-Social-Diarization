"""下游音–口匹配头：学习冻结 MTD 特征之间的同步关系。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SyncMatcher(nn.Module):
    """
    输入冻结视觉/音频 prenet 特征 [B,D]，输出同步 logit。

    结构：双塔投影 + 拼接 MLP（或点积）。
    """

    def __init__(
        self,
        feat_dim: int = 200,
        hidden: int = 256,
        dropout: float = 0.1,
        mode: str = "mlp",  # mlp | dot
    ):
        super().__init__()
        self.mode = mode
        self.vis_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.aud_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        if mode == "dot":
            self.head: Optional[nn.Module] = None
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, vis_feat: torch.Tensor, aud_feat: torch.Tensor) -> torch.Tensor:
        v = self.vis_proj(vis_feat)
        a = self.aud_proj(aud_feat)
        if self.mode == "dot":
            return (v * a).sum(dim=-1)
        assert self.head is not None
        return self.head(torch.cat([v, a], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score(self, vis_feat: torch.Tensor, aud_feat: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(vis_feat, aud_feat))
