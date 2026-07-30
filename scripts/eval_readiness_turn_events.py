#!/usr/bin/env python3
"""
SpeakAhead / readiness 的 MuVAP 对齐事件评估。

协议: Silent/Active × Shift-Hold (Macro-F1) + NSP (Acc)
详见 csd/eval/turn_event_protocol.py

用法（数据准备完成后）:
  python scripts/eval_readiness_turn_events.py \\
    --ckpt output/readiness_xianyang/merged_all/readiness_model.pt \\
    --session-root output/readiness_xianyang

  # 或显式场次目录
  python scripts/eval_readiness_turn_events.py \\
    --ckpt .../readiness_model.pt \\
    --sessions output/readiness_xianyang/0701_class1_g1_前测 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.eval.turn_event_protocol import evaluate_readiness_on_events

logger = logging.getLogger("eval_turn_events")


def _load_train_mod():
    import importlib.util

    path = ROOT / "scripts" / "train_readiness_lstm.py"
    spec = importlib.util.spec_from_file_location("train_readiness_lstm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_model(ckpt_path: Path, device: torch.device):
    mod = _load_train_mod()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    input_dim = int(ckpt.get("input_dim", len(ckpt.get("mean", []))))
    hidden = int(ckpt.get("hidden", 32))
    bidirectional = bool(ckpt.get("bidirectional", False))
    model = mod.ReadinessLSTM(
        input_dim=input_dim,
        hidden=hidden,
        bidirectional=bidirectional,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    feature_cols = mod.resolve_feature_cols(input_dim, ckpt.get("feature_cols"))
    mean = ckpt["mean"]
    std = ckpt["std"]
    seq_len = int(ckpt.get("seq_len", 16))
    window_sec = float(ckpt.get("window_sec", 0.75))
    return model, feature_cols, mean, std, seq_len, window_sec, mod._ensure_feature_columns


def _discover_sessions(session_root: Path, merged_meta: Path | None) -> List[Path]:
    dirs: List[Path] = []
    if merged_meta and merged_meta.exists():
        blob = json.loads(merged_meta.read_text(encoding="utf-8"))
        for p in blob.get("extra", {}).get("merged_from") or []:
            d = Path(p)
            if d.is_dir():
                dirs.append(d)
    if not dirs and session_root is not None and session_root.is_dir():
        for d in sorted(session_root.iterdir()):
            if d.is_dir() and (d / "gt_segments.json").exists() and (d / "frame_features.csv").exists():
                dirs.append(d)
    return dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuVAP 对齐的 readiness 话轮事件评估")
    p.add_argument("--ckpt", type=Path, required=True, help="readiness_model.pt")
    p.add_argument("--sessions", type=Path, nargs="*", default=None, help="场次目录列表")
    p.add_argument(
        "--session-root",
        type=Path,
        default=None,
        help="自动扫描含 gt_segments.json + frame_features.csv 的子目录",
    )
    p.add_argument(
        "--merged-meta",
        type=Path,
        default=None,
        help="merged_all/readiness_samples_meta.json，用其 merged_from",
    )
    p.add_argument("--out", type=Path, default=None, help="写出 JSON 报告路径")
    p.add_argument("--solo-min-sec", type=float, default=1.0)
    p.add_argument("--gap-max-sec", type=float, default=3.0)
    p.add_argument("--offset-sec", type=float, default=0.1)
    p.add_argument("--speakers", type=str, nargs="*", default=["S1", "S2", "S3"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    device = torch.device("cpu")
    model, feature_cols, mean, std, seq_len, window_sec, ensure_feat = _load_model(args.ckpt, device)

    if args.sessions:
        sessions = [Path(s) for s in args.sessions]
    else:
        merged_meta = args.merged_meta
        if merged_meta is None and args.ckpt.parent.name == "merged_all":
            cand = args.ckpt.parent / "readiness_samples_meta.json"
            if cand.exists():
                merged_meta = cand
        root = args.session_root
        if root is None:
            root = args.ckpt.parent.parent if args.ckpt.parent.name == "merged_all" else args.ckpt.parent
        sessions = _discover_sessions(root, merged_meta)

    if not sessions:
        raise FileNotFoundError(
            "未找到可用场次（需要 gt_segments.json + frame_features.csv）。"
            "请等 prepare 完成，或传入 --sessions / --session-root。"
        )

    logger.info("评估场次 %d 个 | ckpt=%s", len(sessions), args.ckpt)
    report = evaluate_readiness_on_events(
        model,
        sessions,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        window_sec=window_sec,
        device=device,
        speakers=tuple(args.speakers),
        solo_min_sec=args.solo_min_sec,
        gap_max_sec=args.gap_max_sec,
        offset_sec=args.offset_sec,
        ensure_feat_fn=ensure_feat,
    )
    out = args.out or (args.ckpt.parent / "turn_events_report.json")
    # 完整事件列表另存，主报告去掉超长 export
    events = report.pop("events_exportable", [])
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "turn_events_samples.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Silent: n=%d sh_f1=%.3f nsp_acc=%.3f | Active: n=%d sh_f1=%.3f nsp_acc=%.3f | → %s",
        report["silent"]["n"],
        report["silent"]["shift_hold_macro_f1"],
        report["silent"]["nsp_acc"],
        report["active"]["n"],
        report["active"]["shift_hold_macro_f1"],
        report["active"]["nsp_acc"],
        out,
    )


if __name__ == "__main__":
    main()
