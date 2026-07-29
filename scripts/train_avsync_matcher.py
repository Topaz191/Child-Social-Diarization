#!/usr/bin/env python3
"""
训练下游 SyncMatcher：冻结 MTD 特征 → 学习口动与发音是否同步。

输入: output/avsync_xianyang/merged_all/avsync_features.npz
输出: avsync_matcher.pt + train_report.json

用法:
  python scripts/train_avsync_matcher.py --data-dir output/avsync_xianyang/merged_all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.avsync.matcher import SyncMatcher
from csd.core.utils import setup_logging

logger = logging.getLogger("train_avsync")


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        correct = 0.0
        for p in pos:
            correct += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
        auc = correct / (len(pos) * len(neg))
    return {
        "accuracy": acc,
        "auc": float(auc),
        "pos_mean": float(pos.mean()) if len(pos) else 0.0,
        "neg_mean": float(neg.mean()) if len(neg) else 0.0,
    }


def _split(
    V: np.ndarray, A: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(len(y) * val_ratio)) if len(y) >= 5 else max(1, len(y) // 5)
    n_val = min(n_val, len(y) - 1) if len(y) > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return V[train_idx], A[train_idx], y[train_idx], V[val_idx], A[val_idx], y[val_idx]


def train(
    data_dir: Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
    mode: str = "mlp",
) -> dict:
    npz = data_dir / "avsync_features.npz"
    if not npz.exists():
        raise FileNotFoundError(f"缺少 {npz}，请先运行 prepare_avsync_xianyang.py")
    data = np.load(npz)
    V, A, y = data["V"], data["A"], data["y"]
    if len(y) < 4:
        raise RuntimeError(f"样本过少: {len(y)}")

    # 基线：若 meta 含 mtd_logit，可顺带报告零样本 AUC
    mtd_auc = float("nan")
    meta_path = data_dir / "avsync_samples_meta.json"
    if meta_path.exists():
        samples = json.loads(meta_path.read_text(encoding="utf-8")).get("samples", [])
        if len(samples) == len(y) and all("mtd_logit" in s for s in samples):
            logits = np.array([float(s["mtd_logit"]) for s in samples], dtype=np.float64)
            probs = 1.0 / (1.0 + np.exp(-logits))
            mtd_auc = _metrics(y, probs)["auc"]
            logger.info("冻结 MTD 零样本 AUC=%.3f", mtd_auc)

    Vtr, Atr, ytr, Vva, Ava, yva = _split(V, A, y, val_ratio, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncMatcher(feat_dim=int(V.shape[-1]), hidden=hidden, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Vtr),
            torch.from_numpy(Atr),
            torch.from_numpy(ytr.astype(np.float32)),
        ),
        batch_size=min(batch_size, len(ytr)),
        shuffle=True,
    )

    best_auc, best_state = -1.0, None
    history: List[dict] = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for vb, ab, yb in loader:
            vb, ab, yb = vb.to(device), ab.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(vb, ab), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            tr_p = torch.sigmoid(model(torch.from_numpy(Vtr).to(device), torch.from_numpy(Atr).to(device))).cpu().numpy()
            va_p = (
                torch.sigmoid(model(torch.from_numpy(Vva).to(device), torch.from_numpy(Ava).to(device))).cpu().numpy()
                if len(yva)
                else np.array([])
            )
        tr_m = _metrics(ytr, tr_p)
        va_m = _metrics(yva, va_p) if len(yva) else {"accuracy": float("nan"), "auc": float("nan"), "pos_mean": 0, "neg_mean": 0}
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "train": tr_m, "val": va_m})
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            logger.info(
                "ep%02d loss=%.4f train_auc=%.3f val_auc=%.3f",
                ep,
                history[-1]["loss"],
                tr_m["auc"],
                va_m["auc"],
            )
        score = va_m["auc"] if va_m["auc"] == va_m["auc"] else tr_m["auc"]
        if score >= best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        all_p = torch.sigmoid(
            model(torch.from_numpy(V).to(device), torch.from_numpy(A).to(device))
        ).cpu().numpy()
    all_m = _metrics(y, all_p)

    ckpt = {
        "model_state": model.state_dict(),
        "feat_dim": int(V.shape[-1]),
        "hidden": hidden,
        "mode": mode,
        "metrics": {"all": all_m, "best_val_auc": best_auc, "mtd_zero_shot_auc": mtd_auc},
    }
    pt = data_dir / "avsync_matcher.pt"
    torch.save(ckpt, pt)
    report = {
        "n_samples": int(len(y)),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "all_metrics": all_m,
        "best_val_auc": best_auc,
        "mtd_zero_shot_auc": mtd_auc,
        "model_path": str(pt),
        "history_tail": history[-5:],
    }
    (data_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "完成: val_auc=%.3f all_auc=%.3f mtd_zero_shot=%.3f → %s",
        best_auc,
        all_m["auc"],
        mtd_auc,
        pt,
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "output" / "avsync_xianyang" / "merged_all")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="mlp", choices=["mlp", "dot"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    train(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
