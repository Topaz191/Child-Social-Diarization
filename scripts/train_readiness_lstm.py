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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging

logger = logging.getLogger("train_readiness")

# 与 prepare_readiness_xianyang.FEATURE_COLS 保持一致
FEATURE_COLS_18 = [
    "yaw",
    "pitch",
    "roll",
    "mouth_opening",
    "visibility",
    "side_face_weight",
    "d_yaw",
    "d_pitch",
    "d_roll",
    "d_mouth",
    "mouth_mean_short",
    "mouth_std_short",
    "mouth_max_short",
    "mouth_trend",
    "others_mouth_mean",
    "others_mouth_max",
    "mouth_rel",
    "others_still",
]
FEATURE_COLS_5 = ["yaw", "pitch", "roll", "mouth_opening", "visibility"]
FEATURE_COLS_6 = FEATURE_COLS_5 + ["side_face_weight"]


class ReadinessLSTM(nn.Module):
    """轻量时序 readiness：单向/双向 LSTM + 线性头。"""

    def __init__(
        self,
        input_dim: int = 5,
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden),
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


def _split_by_session(
    X: np.ndarray,
    y: np.ndarray,
    meta: List[dict],
    val_ratio: float,
    seed: int,
):
    """按 session_id / video 场次划分，避免同场泄漏。"""
    rng = np.random.default_rng(seed)
    keys = []
    for i, m in enumerate(meta):
        key = m.get("session_id") or m.get("video") or m.get("speaker") or f"row{i}"
        keys.append(str(key))
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio)) if len(uniq) >= 2 else 0
    val_keys = set(uniq[:n_val])
    train_idx = [i for i, k in enumerate(keys) if k not in val_keys]
    val_idx = [i for i, k in enumerate(keys) if k in val_keys]
    if not train_idx or not val_idx:
        return _split(X, y, val_ratio, seed)
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


def resolve_feature_cols(
    input_dim: int,
    declared: Optional[Sequence[str]] = None,
) -> List[str]:
    """按模型输入维数解析特征列。声明与维数不一致时以 input_dim 为准（常见于旧 summary/ckpt 仍写 5 维）。"""
    if declared is not None:
        cols = [str(c) for c in declared]
        if len(cols) == input_dim:
            return cols
        logger.info(
            "忽略过期 feature_cols（declared=%d ≠ input_dim=%d），改用 %d 维默认列名",
            len(cols),
            input_dim,
            input_dim,
        )
    if input_dim == len(FEATURE_COLS_18):
        return list(FEATURE_COLS_18)
    if input_dim == len(FEATURE_COLS_6):
        return list(FEATURE_COLS_6)
    if input_dim == len(FEATURE_COLS_5):
        return list(FEATURE_COLS_5)
    raise ValueError(f"无法解析 feature_cols: input_dim={input_dim}")


def sync_feature_cols_metadata(data_dir: Path, feature_cols: Sequence[str], input_dim: int) -> None:
    """把 dataset_summary / samples_meta 里的 feature_cols 改成与 X 一致，避免下次再误读旧 5 维。"""
    cols = list(feature_cols)
    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        try:
            extra = json.loads(summary_path.read_text(encoding="utf-8"))
            old = extra.get("feature_cols")
            if not old or len(old) != input_dim:
                extra["feature_cols"] = cols
                summary_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("已更新 %s feature_cols → %d 维", summary_path.name, input_dim)
        except Exception as exc:
            logger.warning("更新 dataset_summary 失败: %s", exc)
    meta_path = data_dir / "readiness_samples_meta.json"
    if meta_path.exists():
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
            extra = blob.setdefault("extra", {})
            old = extra.get("feature_cols")
            if not old or len(old) != input_dim:
                extra["feature_cols"] = cols
                meta_path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("已更新 %s extra.feature_cols → %d 维", meta_path.name, input_dim)
        except Exception as exc:
            logger.warning("更新 readiness_samples_meta 失败: %s", exc)


def _try_enrich_frame_features(feat):
    """旧 csv 只有 base 列时，按 prepare 脚本补齐动态/相对特征。"""
    try:
        import importlib.util

        prep = ROOT / "scripts" / "prepare_readiness_xianyang.py"
        if not prep.exists():
            return feat
        spec = importlib.util.spec_from_file_location("prepare_readiness_xianyang", prep)
        if spec is None or spec.loader is None:
            return feat
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enrich_frame_features(feat)
    except Exception as exc:
        logger.warning("enrich_frame_features 失败，缺失列将填 0: %s", exc)
        return feat


def _ensure_feature_columns(feat, feature_cols: Sequence[str]):
    import pandas as pd

    missing = [c for c in feature_cols if c not in feat.columns]
    if missing:
        feat = _try_enrich_frame_features(feat)
    still = [c for c in feature_cols if c not in feat.columns]
    if still:
        logger.warning("帧特征仍缺列，填 0: %s", still)
        for c in still:
            feat[c] = 0.0
    return feat


def _lift_scores_on_table(
    model: ReadinessLSTM,
    feat,
    pos_samples: Sequence[dict],
    feature_cols: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> List[float]:
    lifts: List[float] = []
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    if len(feature_cols) != len(mean) or len(mean) != len(std):
        logger.warning(
            "lift 跳过: cols=%d mean=%d std=%d 不一致",
            len(feature_cols),
            len(mean),
            len(std),
        )
        return lifts

    model.eval()
    with torch.no_grad():
        for s in pos_samples:
            spk_df = feat[feat["speaker"] == s["speaker"]].sort_values("t")
            t0, t1 = float(s["t0"]), float(s["t1"])
            sub = spk_df[(spk_df["t"] >= t0 - 1e-6) & (spk_df["t"] <= t1 + 1e-6)]
            if len(sub) < 4:
                continue
            ts = sub["t"].to_numpy(dtype=np.float64)
            feats = sub[list(feature_cols)].to_numpy(dtype=np.float64)
            _, uniq = np.unique(ts, return_index=True)
            ts, feats = ts[uniq], feats[uniq]
            if len(ts) < 2:
                continue
            grid = np.linspace(t0, t1, seq_len)
            mat = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
            for j in range(len(feature_cols)):
                mat[:, j] = np.interp(grid, ts, feats[:, j]).astype(np.float32)
            mat = (mat - mean) / std
            scores = []
            for k in range(max(4, seq_len // 3), seq_len + 1):
                if k < seq_len:
                    pad = np.repeat(mat[k - 1 : k], seq_len - k, axis=0)
                    full = np.concatenate([mat[:k], pad], axis=0)
                else:
                    full = mat
                x = torch.from_numpy(full[None]).to(device)
                scores.append(float(model.readiness_score(x).cpu().item()))
            if len(scores) < 2:
                continue
            third = max(1, len(scores) // 3)
            lifts.append(float(np.mean(scores[-third:]) > np.mean(scores[:third])))
    return lifts


def evaluate_pre_speech_lift(
    model: ReadinessLSTM,
    feat_csv: Path,
    meta_json: Path,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    window_sec: float,
    device: torch.device,
    feature_cols: Optional[Sequence[str]] = None,
    source_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, float]:
    """在正样本窗口上检查 readiness 是否呈上升趋势（后半段均值 > 前半段）。

    - feature_cols 必须与训练时 X 最后一维一致（支持 5/6/18 维）
    - merged_all 无 frame_features.csv 时，传入 source_dirs=merged_from 逐场评估
    """
    import pandas as pd

    cols = resolve_feature_cols(int(len(np.asarray(mean).reshape(-1))), feature_cols)
    jobs: List[Tuple[Path, Path]] = []
    if source_dirs:
        for d in source_dirs:
            d = Path(d)
            csv_p, meta_p = d / "frame_features.csv", d / "readiness_samples_meta.json"
            if csv_p.exists() and meta_p.exists():
                jobs.append((csv_p, meta_p))
    if not jobs and feat_csv.exists() and meta_json.exists():
        jobs.append((feat_csv, meta_json))
    if not jobs:
        return {
            "lift_rate": float("nan"),
            "n_checked": 0,
            "window_sec": window_sec,
            "feature_dim": len(cols),
            "reason": "no_frame_features",
        }

    lifts: List[float] = []
    for csv_p, meta_p in jobs:
        feat = pd.read_csv(csv_p)
        feat = _ensure_feature_columns(feat, cols)
        samples = json.loads(meta_p.read_text(encoding="utf-8")).get("samples", [])
        pos = [s for s in samples if s.get("label") == 1]
        lifts.extend(
            _lift_scores_on_table(model, feat, pos, cols, mean, std, seq_len, device)
        )

    rate = float(np.mean(lifts)) if lifts else float("nan")
    return {
        "lift_rate": rate,
        "n_checked": int(len(lifts)),
        "window_sec": window_sec,
        "feature_dim": len(cols),
        "n_sessions": int(len(jobs)),
    }


def train(
    data_dir: Path,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    bidirectional: bool = True,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int = 8,
    split_by_session: bool = True,
    eval_turn_events: bool = True,
) -> dict:
    npz = data_dir / "readiness_samples.npz"
    if not npz.exists():
        raise FileNotFoundError(f"缺少样本文件: {npz}，请先运行 prepare_readiness_dataset.py")

    data = np.load(npz)
    X, y = data["X"], data["y"]
    if len(y) < 4:
        raise RuntimeError(f"样本过少({len(y)})，无法训练。请检查特征抽取与切分。")

    meta: List[dict] = []
    meta_path = data_dir / "readiness_samples_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("samples", [])

    Xn, mean, std = _normalize(X)
    if split_by_session and len(meta) == len(y):
        Xtr, ytr, Xva, yva = _split_by_session(Xn, y, meta, val_ratio, seed)
        logger.info("按场次划分 train=%d val=%d", len(ytr), len(yva))
    else:
        Xtr, ytr, Xva, yva = _split(Xn, y, val_ratio, seed)

    device = torch.device("cpu")
    model = ReadinessLSTM(
        input_dim=X.shape[-1],
        hidden=hidden,
        num_layers=1,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=min(batch_size, len(ytr)),
        shuffle=True,
    )

    history = []
    best_auc, best_state = -1.0, None
    bad_epochs = 0
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
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                logger.info("早停于 ep%02d (best_val_auc=%.3f)", ep, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 全量指标
    with torch.no_grad():
        all_prob = torch.sigmoid(model(torch.from_numpy(Xn).to(device))).cpu().numpy()
    all_m = _metrics(y, all_prob)

    summary_path = data_dir / "dataset_summary.json"
    extra = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    window_sec = 0.75
    meta_blob: dict = {}
    meta_extra = data_dir / "readiness_samples_meta.json"
    if meta_extra.exists():
        meta_blob = json.loads(meta_extra.read_text(encoding="utf-8"))
        window_sec = float(meta_blob.get("extra", {}).get("window_sec", 0.75))
    declared_cols = extra.get("feature_cols") or meta_blob.get("extra", {}).get("feature_cols")
    feature_cols = resolve_feature_cols(int(X.shape[-1]), declared_cols)
    sync_feature_cols_metadata(data_dir, feature_cols, int(X.shape[-1]))
    source_dirs = meta_blob.get("extra", {}).get("merged_from") or extra.get("merged_from")
    if source_dirs:
        source_dirs = [Path(p) for p in source_dirs]

    lift = evaluate_pre_speech_lift(
        model,
        data_dir / "frame_features.csv",
        data_dir / "readiness_samples_meta.json",
        mean,
        std,
        seq_len=int(X.shape[1]),
        window_sec=window_sec,
        device=device,
        feature_cols=feature_cols,
        source_dirs=source_dirs,
    )

    turn_events = None
    if eval_turn_events:
        try:
            from csd.eval.turn_event_protocol import evaluate_readiness_on_events

            sess = source_dirs or ([data_dir] if (data_dir / "gt_segments.json").exists() else [])
            if sess:
                turn_events = evaluate_readiness_on_events(
                    model,
                    sess,
                    feature_cols=feature_cols,
                    mean=mean,
                    std=std,
                    seq_len=int(X.shape[1]),
                    window_sec=window_sec,
                    device=device,
                    ensure_feat_fn=_ensure_feature_columns,
                )
                turn_events.pop("events_exportable", None)
                logger.info(
                    "TurnEvents Silent n=%d sh_f1=%.3f nsp=%.3f | Active n=%d sh_f1=%.3f nsp=%.3f",
                    turn_events["silent"]["n"],
                    turn_events["silent"]["shift_hold_macro_f1"],
                    turn_events["silent"]["nsp_acc"],
                    turn_events["active"]["n"],
                    turn_events["active"]["shift_hold_macro_f1"],
                    turn_events["active"]["nsp_acc"],
                )
            else:
                logger.warning("跳过 turn-events：无 gt_segments.json 场次（等 prepare 写完后再跑 eval 脚本）")
        except Exception as exc:
            logger.warning("turn-events 评估失败: %s", exc)

    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": int(X.shape[-1]),
        "seq_len": int(X.shape[1]),
        "hidden": hidden,
        "bidirectional": bidirectional,
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "window_sec": window_sec,
        "metrics": {"all": all_m, "lift": lift, "best_val_auc": best_auc, "turn_events": turn_events},
    }
    pt_path = data_dir / "readiness_model.pt"
    torch.save(ckpt, pt_path)

    report = {
        "n_samples": int(len(y)),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "all_metrics": all_m,
        "lift": lift,
        "turn_events": turn_events,
        "best_val_auc": best_auc,
        "split_by_session": split_by_session,
        "bidirectional": bidirectional,
        "pass_lift_60": bool(lift.get("lift_rate", 0) >= 0.6) if lift.get("lift_rate") == lift.get("lift_rate") else False,
        "pass_auc_above_chance": bool(all_m["auc"] > 0.55) if all_m["auc"] == all_m["auc"] else False,
        "model_path": str(pt_path),
        "history_tail": history[-5:],
        "dataset_summary": extra,
    }
    (data_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if turn_events is not None:
        (data_dir / "turn_events_report.json").write_text(
            json.dumps(turn_events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    p.add_argument("--bidirectional", action="store_true", default=True)
    p.add_argument("--no-bidirectional", action="store_false", dest="bidirectional")
    p.add_argument("--split-by-session", action="store_true", default=True)
    p.add_argument("--no-split-by-session", action="store_false", dest="split_by_session")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--eval-turn-events", action="store_true", default=True, help="训练后跑 MuVAP 对齐事件评估")
    p.add_argument("--no-eval-turn-events", action="store_false", dest="eval_turn_events")
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
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        patience=args.patience,
        split_by_session=args.split_by_session,
        eval_turn_events=args.eval_turn_events,
    )


if __name__ == "__main__":
    main()
