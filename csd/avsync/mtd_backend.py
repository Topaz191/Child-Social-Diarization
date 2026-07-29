"""冻结 MTDVocaLiST：加载预训练 SyncTransformer，提取音/视 CNN 特征或 sync logit。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MTD_ROOT = ROOT / "third_party" / "MTDVocaLiST"
DEFAULT_WEIGHTS = ROOT / "models" / "mtdvocalist" / "pure_MTDVocaLiST.pth"


def _ensure_mtd_on_path(mtd_root: Path) -> None:
    root = str(mtd_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class MTDFeatureExtractor(nn.Module):
    """
    冻结 MTDVocaLiST。

    输入约定（与官方一致）:
      face: [B, 15, 48, 96]  — 5 帧 RGB 沿通道拼接，[0,1]
      mel:  [B, 1, 80, 80]   — mel（不足则 pad）

    输出:
      vis_feat / aud_feat: [B, 200]  — prenet 池化特征（供下游匹配头）
      logit: 预训练同步分类 logit（可选，零样本基线）
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        mtd_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        d_model: int = 200,
        freeze: bool = True,
    ):
        super().__init__()
        self.mtd_root = Path(mtd_root) if mtd_root else DEFAULT_MTD_ROOT
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.d_model = d_model
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not (self.mtd_root / "models" / "student_thin_200_all.py").exists():
            raise FileNotFoundError(
                f"未找到 MTDVocaLiST 源码: {self.mtd_root}\n"
                "请运行: python scripts/setup_mtdvocalist.py"
            )
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"未找到权重: {self.weights_path}\n"
                "请运行: python scripts/setup_mtdvocalist.py"
            )

        _ensure_mtd_on_path(self.mtd_root)
        from models.student_thin_200_all import SyncTransformer  # type: ignore

        self.model = SyncTransformer(d_model=d_model)
        cpk = torch.load(str(self.weights_path), map_location="cpu")
        state = cpk["state_dict"] if isinstance(cpk, dict) and "state_dict" in cpk else cpk
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("MTD load missing keys: %d", len(missing))
        if unexpected:
            logger.warning("MTD load unexpected keys: %d", len(unexpected))
        self.model.to(self.device)
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        logger.info("MTDVocaLiST 已加载 (freeze=%s, device=%s): %s", freeze, self.device, self.weights_path)

    @torch.no_grad()
    def encode_pair(
        self,
        face: torch.Tensor,
        mel: torch.Tensor,
        return_logit: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        face: [B,15,48,96] or [15,48,96]
        mel:  [B,1,80,W] — W 若非 80 则右侧 pad/裁到 80
        """
        if face.dim() == 3:
            face = face.unsqueeze(0)
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        mel = self._pad_mel(mel)
        face = face.to(self.device, dtype=torch.float32)
        mel = mel.to(self.device, dtype=torch.float32)

        B = face.shape[0]
        # 与官方 forward 一致地跑 prenet
        vid = self.model.vid_prenet(face.view(B, -1, 3, 48, 96).permute(0, 2, 3, 4, 1).contiguous())
        aud = self.model.aud_prenet(mel)
        # vid: [B,C,1,1,T] → pool; aud: [B,C,1,T]
        vis_feat = vid.flatten(2).mean(dim=-1)  # [B,C]
        aud_feat = aud.flatten(2).mean(dim=-1)
        # L2 归一化便于下游
        vis_feat = F.normalize(vis_feat, dim=-1)
        aud_feat = F.normalize(aud_feat, dim=-1)

        logit = None
        if return_logit:
            logit, _ = self.model(face, mel)
        return vis_feat, aud_feat, logit

    @staticmethod
    def _pad_mel(mel: torch.Tensor) -> torch.Tensor:
        """保证时间维=80: [B,1,80,T] → [B,1,80,80]。"""
        b, c, f, t = mel.shape
        if t == 80:
            return mel
        if t > 80:
            return mel[..., :80]
        pad = torch.zeros(b, c, f, 80 - t, dtype=mel.dtype, device=mel.device)
        return torch.cat([mel, pad], dim=-1)

    def encode_numpy(
        self,
        face_np: np.ndarray,
        mel_np: np.ndarray,
        return_logit: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        face = torch.from_numpy(np.asarray(face_np, dtype=np.float32))
        mel = torch.from_numpy(np.asarray(mel_np, dtype=np.float32))
        if mel.ndim == 2:
            mel = mel[None, ...]
        if mel.ndim == 3 and mel.shape[0] != 1:
            # [80,T] already handled; [1,80,T] ok
            pass
        vis, aud, logit = self.encode_pair(face, mel, return_logit=return_logit)
        v = vis.squeeze(0).cpu().numpy()
        a = aud.squeeze(0).cpu().numpy()
        lg = float(logit.squeeze().cpu().item()) if logit is not None else None
        return v, a, lg
