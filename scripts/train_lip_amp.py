#!/usr/bin/env python3
"""
训练儿童嘴动幅度标定器（小 MLP + 活跃度分位数）。

输入: output/lip_amp_xianyang/merged_all/lip_amp_samples.npz
输出: lip_amp_model.pt + train_report.json

用法:
  python scripts/train_lip_amp.py --data-dir output/lip_amp_xianyang/merged_all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.trust.lip_amplitude import FEATURE_NAMES, LipAmpMLP, compute_activity_percentiles

logger = logging.getLogger("train_lip_amp")


def _normalize(X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _split_by_session(
    X: np.ndarray,
    y: np.ndarray,
    meta: List[dict],
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = [str(m.get("session_id") or m.get("speaker") or i) for i, m in enumerate(meta)]
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio)) if len(uniq) >= 2 else 0
    val_keys = set(uniq[:n_val])
    tr = [i for i, k in enumerate(keys) if k not in val_keys]
    va = [i for i, k in enumerate(keys) if k in val_keys]
    if not tr or not va:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_val = max(1, int(len(y) * val_ratio))
        return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]
    return X[tr], y[tr], X[va], y[va]


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


def train(
    data_dir: Path,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    dropout: float = 0.15,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> dict:
    npz = data_dir / "lip_amp_samples.npz"
    if not npz.exists():
        raise FileNotFoundError(f"缺少 {npz}，请先跑 prepare_lip_amp_xianyang.py")

    data = np.load(npz)
    X, y = data["X"], data["y"]
    if len(y) < 8:
        raise RuntimeError(f"样本过少 ({len(y)})")

    meta: List[dict] = []
    meta_path = data_dir / "lip_amp_samples_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("samples", [])
    if len(meta) != len(y):
        meta = [{"session_id": "unknown", "label": int(yi)} for yi in y]

    # 正样本活跃度分位 → 替换硬编码 0.015
    pos_act = [float(m.get("activity", 0.0)) for m in meta if int(m.get("label", 0)) == 1]
    if not pos_act:
        pos_act = X[y == 1, list(FEATURE_NAMES).index("activity")].tolist()
    pct = compute_activity_percentiles(pos_act)
    activity_scale = max(pct["p75"], 1e-4)

    Xn, mean, std = _normalize(X)
    Xtr, ytr, Xva, yva = _split_by_session(Xn, y, meta, val_ratio, seed)
    logger.info(
        "样本=%d train=%d val=%d | pos_act p50=%.5f p75=%.5f p90=%.5f scale=%.5f",
        len(y),
        len(ytr),
        len(yva),
        pct["p50"],
        pct["p75"],
        pct["p90"],
        activity_scale,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LipAmpMLP(input_dim=X.shape[-1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=min(batch_size, len(ytr)),
        shuffle=True,
    )

    best_auc, best_state, bad = -1.0, None, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            tr_p = torch.sigmoid(model(torch.from_numpy(Xtr).to(device))).cpu().numpy()
            va_p = torch.sigmoid(model(torch.from_numpy(Xva).to(device))).cpu().numpy()
        tr_m, va_m = _metrics(ytr, tr_p), _metrics(yva, va_p)
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
            bad = 0
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                logger.info("早停 ep%02d best_val_auc=%.3f", ep, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        all_p = torch.sigmoid(model(torch.from_numpy(Xn).to(device))).cpu().numpy()
    all_m = _metrics(y, all_p)

    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": int(X.shape[-1]),
        "hidden": hidden,
        "feature_cols": list(FEATURE_NAMES),
        "mean": mean,
        "std": std,
        "activity_scale": activity_scale,
        "pos_activity_p50": pct["p50"],
        "pos_activity_p75": pct["p75"],
        "pos_activity_p90": pct["p90"],
        "min_side_face": 0.45,
        "metrics": {"all": all_m, "best_val_auc": best_auc},
    }
    pt_path = data_dir / "lip_amp_model.pt"
    torch.save(ckpt, pt_path)
    # 纯规则尺度也单独落盘，方便 visual_conf 无 torch 时使用
    (data_dir / "lip_amp_scale.json").write_text(
        json.dumps(
            {
                "activity_scale": activity_scale,
                "pos_activity_p50": pct["p50"],
                "pos_activity_p75": pct["p75"],
                "pos_activity_p90": pct["p90"],
                "feature_cols": list(FEATURE_NAMES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "n_samples": int(len(y)),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "device": str(device),
        "best_val_auc": best_auc,
        "all_metrics": all_m,
        "activity_scale": activity_scale,
        "percentiles": pct,
        "model_path": str(pt_path),
        "history_tail": history[-5:],
    }
    (data_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("模型已保存: %s | val_auc=%.3f scale=%.5f", pt_path, best_auc, activity_scale)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "output" / "lip_amp_xianyang" / "merged_all")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=10)
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
        dropout=args.dropout,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
