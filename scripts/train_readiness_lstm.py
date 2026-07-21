#!/usr/bin/env python3
"""
阶段1：轻量 LSTM readiness 分类器（CPU 可训）。

输入: output/readiness/{trial}_{test}/readiness_samples.npz
输出: readiness_model.pt + 验证曲线/指标

用法:
  python scripts/train_readiness_lstm.py
  python scripts/train_readiness_lstm.py --data-dir output/readiness/0820_post --epochs 40
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

logger = logging.getLogger("train_readiness")


class ReadinessLSTM(nn.Module):
    """1 层 LSTM + 线性头，输出发言准备度概率。"""

    def __init__(self, input_dim: int = 5, hidden: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        logits = self.head(out[:, -1, :]).squeeze(-1)
        return logits

    @torch.no_grad()
    def readiness_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


def _normalize(X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
    if mean is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    Xn = (X - mean) / std
    return Xn.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _split(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(len(y) * val_ratio)) if len(y) >= 5 else max(1, len(y) // 5)
    n_val = min(n_val, len(y) - 1) if len(y) > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    # 简易 AUC（Mann-Whitney）
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        # P(score_pos > score_neg)
        correct = 0.0
        for p in pos:
            correct += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
        auc = correct / (len(pos) * len(neg))
    return {"accuracy": acc, "auc": float(auc), "pos_mean": float(pos.mean()) if len(pos) else 0.0, "neg_mean": float(neg.mean()) if len(neg) else 0.0}


def evaluate_pre_speech_lift(
    model: ReadinessLSTM,
    feat_csv: Path,
    meta_json: Path,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    window_sec: float,
    device: torch.device,
) -> Dict[str, float]:
    """在正样本窗口上检查 readiness 是否呈上升趋势（后半段均值 > 前半段）。"""
    import pandas as pd

    if not feat_csv.exists() or not meta_json.exists():
        return {"lift_rate": float("nan"), "n_checked": 0}

    feat = pd.read_csv(feat_csv)
    samples = json.loads(meta_json.read_text(encoding="utf-8")).get("samples", [])
    pos = [s for s in samples if s.get("label") == 1]
    feature_cols = ["yaw", "pitch", "roll", "mouth_opening", "visibility"]
    lifts = []
    model.eval()
    with torch.no_grad():
        for s in pos:
            spk_df = feat[feat["speaker"] == s["speaker"]].sort_values("t")
            t0, t1 = float(s["t0"]), float(s["t1"])
            sub = spk_df[(spk_df["t"] >= t0 - 1e-6) & (spk_df["t"] <= t1 + 1e-6)]
            if len(sub) < 4:
                continue
            ts = sub["t"].to_numpy(dtype=np.float64)
            feats = sub[feature_cols].to_numpy(dtype=np.float64)
            _, uniq = np.unique(ts, return_index=True)
            ts, feats = ts[uniq], feats[uniq]
            if len(ts) < 2:
                continue
            grid = np.linspace(t0, t1, seq_len)
            mat = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
            for j in range(len(feature_cols)):
                mat[:, j] = np.interp(grid, ts, feats[:, j])
            mat = (mat - mean) / std
            # 逐步前缀分数：看后 1/3 vs 前 1/3
            scores = []
            for k in range(max(4, seq_len // 3), seq_len + 1):
                x = torch.from_numpy(mat[:k][None]).to(device)
                # 不足长度时右侧 pad 最后一帧
                if k < seq_len:
                    pad = mat[k - 1 : k].repeat(seq_len - k, axis=0)
                    full = np.concatenate([mat[:k], pad], axis=0)
                    x = torch.from_numpy(full[None]).to(device)
                scores.append(float(model.readiness_score(x).cpu().item()))
            if len(scores) < 2:
                continue
            third = max(1, len(scores) // 3)
            lifts.append(float(np.mean(scores[-third:]) > np.mean(scores[:third])))

    rate = float(np.mean(lifts)) if lifts else float("nan")
    return {"lift_rate": rate, "n_checked": int(len(lifts)), "window_sec": window_sec}


def train(
    data_dir: Path,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    npz = data_dir / "readiness_samples.npz"
    if not npz.exists():
        raise FileNotFoundError(f"缺少样本文件: {npz}，请先运行 prepare_readiness_dataset.py")

    data = np.load(npz)
    X, y = data["X"], data["y"]
    if len(y) < 4:
        raise RuntimeError(f"样本过少({len(y)})，无法训练。请检查特征抽取与切分。")

    Xn, mean, std = _normalize(X)
    Xtr, ytr, Xva, yva = _split(Xn, y, val_ratio, seed)

    device = torch.device("cpu")
    model = ReadinessLSTM(input_dim=X.shape[-1], hidden=hidden, num_layers=1, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=min(batch_size, len(ytr)),
        shuffle=True,
    )

    history = []
    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            tr_prob = torch.sigmoid(model(torch.from_numpy(Xtr).to(device))).cpu().numpy()
            va_prob = torch.sigmoid(model(torch.from_numpy(Xva).to(device))).cpu().numpy() if len(yva) else np.array([])
        tr_m = _metrics(ytr, tr_prob)
        va_m = _metrics(yva, va_prob) if len(yva) else {"accuracy": float("nan"), "auc": float("nan"), "pos_mean": 0, "neg_mean": 0}
        row = {"epoch": ep, "loss": float(np.mean(losses)), "train": tr_m, "val": va_m}
        history.append(row)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            logger.info(
                "ep%02d loss=%.4f train_acc=%.3f train_auc=%.3f val_acc=%.3f val_auc=%.3f",
                ep,
                row["loss"],
                tr_m["accuracy"],
                tr_m["auc"],
                va_m["accuracy"],
                va_m["auc"],
            )
        score = va_m["auc"] if va_m["auc"] == va_m["auc"] else tr_m["auc"]
        if score >= best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 全量指标
    with torch.no_grad():
        all_prob = torch.sigmoid(model(torch.from_numpy(Xn).to(device))).cpu().numpy()
    all_m = _metrics(y, all_prob)

    summary_path = data_dir / "dataset_summary.json"
    extra = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    window_sec = 0.75
    meta_extra = data_dir / "readiness_samples_meta.json"
    if meta_extra.exists():
        window_sec = float(json.loads(meta_extra.read_text(encoding="utf-8")).get("extra", {}).get("window_sec", 0.75))

    lift = evaluate_pre_speech_lift(
        model,
        data_dir / "frame_features.csv",
        data_dir / "readiness_samples_meta.json",
        mean,
        std,
        seq_len=int(X.shape[1]),
        window_sec=window_sec,
        device=device,
    )

    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": int(X.shape[-1]),
        "seq_len": int(X.shape[1]),
        "hidden": hidden,
        "feature_cols": ["yaw", "pitch", "roll", "mouth_opening", "visibility"],
        "mean": mean,
        "std": std,
        "window_sec": window_sec,
        "metrics": {"all": all_m, "lift": lift, "best_val_auc": best_auc},
    }
    pt_path = data_dir / "readiness_model.pt"
    torch.save(ckpt, pt_path)

    report = {
        "n_samples": int(len(y)),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "all_metrics": all_m,
        "lift": lift,
        "best_val_auc": best_auc,
        "pass_lift_60": bool(lift.get("lift_rate", 0) >= 0.6) if lift.get("lift_rate") == lift.get("lift_rate") else False,
        "pass_auc_above_chance": bool(all_m["auc"] > 0.55) if all_m["auc"] == all_m["auc"] else False,
        "model_path": str(pt_path),
        "history_tail": history[-5:],
        "dataset_summary": extra,
    }
    (data_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("模型已保存: %s", pt_path)
    logger.info(
        "验收: auc=%.3f (pos_mean=%.3f neg_mean=%.3f) lift_rate=%.3f (%d checked) pass_lift60=%s pass_auc=%s",
        all_m["auc"],
        all_m["pos_mean"],
        all_m["neg_mean"],
        lift.get("lift_rate", float("nan")),
        lift.get("n_checked", 0),
        report["pass_lift_60"],
        report["pass_auc_above_chance"],
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "output" / "readiness" / "0820_post")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
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
    )


if __name__ == "__main__":
    main()
