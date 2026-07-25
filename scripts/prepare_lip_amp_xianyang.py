#!/usr/bin/env python3
"""
儿童嘴动幅度弱监督样本准备（咸阳）。

正样本：转录中该人明确说话的时间窗 + 偏正脸（side_face_weight / |yaw| 门控）
负样本：远离说话段的静音窗（同样要求偏正脸，避免侧脸噪声当负例）

优先复用 readiness 流水线已抽的 frame_features.csv；否则现场抽取。

用法:
  python scripts/prepare_lip_amp_xianyang.py --from-manifest output/xianyang/manifest.json
  python scripts/prepare_lip_amp_xianyang.py --reuse-readiness-root output/readiness_xianyang --skip-extract
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.data.xianyang import (
    load_xianyang_segments,
    parse_xianyang_video_name,
    resolve_video_position_map,
)
from csd.trust.lip_amplitude import FEATURE_NAMES, window_feature_vector

logger = logging.getLogger("prepare_lip_amp")

# 复用 readiness 脚本中的抽帧实现（同仓 scripts/，避免重复 YOLO/Mesh 逻辑）
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "prepare_readiness_xianyang",
    ROOT / "scripts" / "prepare_readiness_xianyang.py",
)
_prep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prep)
export_frame_features = _prep.export_frame_features
STUDENT_SPEAKERS = _prep.STUDENT_SPEAKERS


def _is_frontal_row(row: pd.Series, min_side: float, max_abs_yaw: float) -> bool:
    side = float(row.get("side_face_weight", 0.0))
    yaw = abs(float(row.get("yaw", 99.0)))
    return side >= min_side or yaw <= max_abs_yaw


def _window_rows(
    spk_df: pd.DataFrame,
    t0: float,
    t1: float,
    min_side: float,
    max_abs_yaw: float,
    min_frames: int,
) -> Optional[pd.DataFrame]:
    sub = spk_df[(spk_df["t"] >= t0 - 1e-6) & (spk_df["t"] <= t1 + 1e-6)].copy()
    if len(sub) < min_frames:
        return None
    frontal = sub[sub.apply(lambda r: _is_frontal_row(r, min_side, max_abs_yaw), axis=1)]
    # 窗口内偏正脸帧占比不够则丢弃
    if len(frontal) < max(min_frames, int(0.6 * len(sub))):
        return None
    return frontal


def _feat_from_rows(rows: pd.DataFrame) -> Optional[np.ndarray]:
    return window_feature_vector(
        rows["mouth_opening"].tolist(),
        rows["side_face_weight"].tolist(),
        rows["yaw"].tolist(),
    )


def _overlaps_any(t0: float, t1: float, intervals: Sequence[Tuple[float, float]], margin: float) -> bool:
    for a, b in intervals:
        if t1 > a - margin and t0 < b + margin:
            return True
    return False


def cut_lip_amp_samples(
    feat_df: pd.DataFrame,
    segments: List[dict],
    *,
    window_sec: float = 0.6,
    hop_sec: float = 0.3,
    min_side: float = 0.55,
    max_abs_yaw: float = 30.0,
    min_frames: int = 3,
    neg_margin: float = 2.0,
    seed: int = 42,
    max_neg_per_gap: int = 4,
    session_id: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    positives: List[Tuple[np.ndarray, dict]] = []
    negatives: List[Tuple[np.ndarray, dict]] = []

    by_spk = {spk: g.sort_values("t").reset_index(drop=True) for spk, g in feat_df.groupby("speaker")}
    all_intervals = [(float(s["start"]), float(s["end"])) for s in segments]
    duration = max((float(s["end"]) for s in segments), default=0.0)
    if len(feat_df):
        duration = max(duration, float(feat_df["t"].max()))

    # --- 正：说话段内滑动窗（偏正脸）---
    for i, seg in enumerate(segments):
        spk = str(seg["speaker"]).upper()
        if spk not in by_spk:
            continue
        s0, s1 = float(seg["start"]), float(seg["end"])
        if s1 - s0 < window_sec * 0.8:
            continue
        t = s0
        while t + window_sec <= s1 + 1e-6:
            t0, t1 = t, t + window_sec
            rows = _window_rows(by_spk[spk], t0, t1, min_side, max_abs_yaw, min_frames)
            if rows is not None:
                feat = _feat_from_rows(rows)
                if feat is not None:
                    positives.append(
                        (
                            feat,
                            {
                                "label": 1,
                                "speaker": spk,
                                "t0": round(t0, 3),
                                "t1": round(t1, 3),
                                "seg_idx": i,
                                "kind": "speaking_frontal",
                                "session_id": session_id,
                                "activity": float(feat[FEATURE_NAMES.index("activity")]),
                                "mean_side": float(feat[FEATURE_NAMES.index("mean_side")]),
                            },
                        )
                    )
            t += hop_sec

    # --- 负：段间静音（偏正脸）---
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for seg in sorted_segs:
        if float(seg["start"]) - cursor >= window_sec + 2 * neg_margin:
            gaps.append((cursor + neg_margin, float(seg["start"]) - neg_margin))
        cursor = max(cursor, float(seg["end"]))
    if duration - cursor >= window_sec + 2 * neg_margin:
        gaps.append((cursor + neg_margin, duration - neg_margin))

    speakers = [s for s in by_spk if s in STUDENT_SPEAKERS]
    for g0, g1 in gaps:
        span = g1 - g0
        if span < window_sec:
            continue
        n_take = min(max_neg_per_gap, max(1, int(span // max(window_sec, 1e-6))))
        for _ in range(n_take):
            if not speakers:
                break
            spk = str(rng.choice(speakers))
            own = [(float(s["start"]), float(s["end"])) for s in segments if s["speaker"] == spk]
            for _try in range(10):
                t0 = float(rng.uniform(g0, max(g0, g1 - window_sec)))
                t1 = t0 + window_sec
                if _overlaps_any(t0, t1, all_intervals, margin=neg_margin):
                    continue
                if _overlaps_any(t0, t1, own, margin=neg_margin):
                    continue
                rows = _window_rows(by_spk[spk], t0, t1, min_side, max_abs_yaw, min_frames)
                if rows is None:
                    continue
                feat = _feat_from_rows(rows)
                if feat is None:
                    continue
                negatives.append(
                    (
                        feat,
                        {
                            "label": 0,
                            "speaker": spk,
                            "t0": round(t0, 3),
                            "t1": round(t1, 3),
                            "seg_idx": -1,
                            "kind": "silence_frontal",
                            "session_id": session_id,
                            "activity": float(feat[FEATURE_NAMES.index("activity")]),
                            "mean_side": float(feat[FEATURE_NAMES.index("mean_side")]),
                        },
                    )
                )
                break

    n = min(len(positives), len(negatives))
    if n == 0:
        logger.warning("lip-amp 样本为空: pos=%d neg=%d session=%s", len(positives), len(negatives), session_id)
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:n] + negatives[:n]
    rng.shuffle(chosen)
    X = np.stack([c[0] for c in chosen], axis=0).astype(np.float32)
    y = np.array([c[1]["label"] for c in chosen], dtype=np.int64)
    meta = [c[1] for c in chosen]
    logger.info(
        "lip-amp 切分: pos候选=%d neg候选=%d 平衡=%d session=%s",
        len(positives),
        len(negatives),
        len(meta),
        session_id,
    )
    return X, y, meta


def save_dataset(out_dir: Path, X: np.ndarray, y: np.ndarray, meta: List[dict], extra: Optional[dict] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "lip_amp_samples.npz"
    np.savez_compressed(npz, X=X, y=y)
    (out_dir / "lip_amp_samples_meta.json").write_text(
        json.dumps({"samples": meta, "extra": extra or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "n_samples": int(len(y)),
        "n_pos": int((y == 1).sum()) if len(y) else 0,
        "n_neg": int((y == 0).sum()) if len(y) else 0,
        "X_shape": list(X.shape),
        "feature_cols": list(FEATURE_NAMES),
        "npz": str(npz),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写: %s", summary)
    return npz


def resolve_session_dir_name(parsed: dict) -> str:
    return f"{parsed['date']}_{parsed['sheet']}_{parsed['test']}_{parsed['camera_tag']}"


def process_one(
    video: Path,
    excel: Path,
    out_root: Path,
    *,
    readiness_root: Optional[Path],
    frame_skip: int,
    window_sec: float,
    hop_sec: float,
    min_side: float,
    max_abs_yaw: float,
    neg_margin: float,
    seed: int,
    skip_extract: bool,
    position_map_paths: Sequence[Path],
    require_position_map: bool = True,
    require_video_start_abs: bool = True,
) -> Optional[Path]:
    parsed = parse_xianyang_video_name(video.name)
    if parsed is None:
        logger.error("无法解析文件名: %s", video.name)
        return None

    pos = resolve_video_position_map(video.name, position_map_paths, require_confirmed=True)
    if require_position_map and pos is None:
        logger.error("缺少 confirmed 位置标注，跳过: %s", video.name)
        return None

    video_start_abs = None if pos is None else pos.get("video_start_abs")
    if require_video_start_abs and video_start_abs is None:
        logger.error("缺少 video_start_abs，跳过: %s", video.name)
        return None

    left_to_right = pos["left_to_right"] if pos else ["S1", "S2", "S3"]
    session = resolve_session_dir_name(parsed)
    out_dir = out_root / session
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = load_xianyang_segments(
        excel,
        parsed["sheet"],
        parsed["phase"],
        speakers=STUDENT_SPEAKERS,
        min_duration=0.4,
        video_start_abs=video_start_abs,
        align_to_video=True,
    )
    in_frame = set(left_to_right)
    segments = [s for s in segments if s["speaker"] in in_frame]
    (out_dir / "gt_segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    # 特征来源：本目录 / readiness 目录 / 重新抽取
    feat_csv = out_dir / "frame_features.csv"
    reused = False
    if readiness_root is not None:
        cand = readiness_root / session / "frame_features.csv"
        if cand.exists():
            feat_csv = cand
            reused = True
    if skip_extract and not feat_csv.exists():
        logger.error("skip-extract 但找不到 frame_features: %s", feat_csv)
        return None

    if feat_csv.exists() and (skip_extract or reused or (out_dir / "frame_features.csv").exists()):
        if reused and feat_csv != out_dir / "frame_features.csv":
            # 复制一份到 lip_amp 输出，便于独立打包
            df = pd.read_csv(feat_csv)
            df.to_csv(out_dir / "frame_features.csv", index=False)
            meta_src = feat_csv.with_suffix(".meta.json")
            if meta_src.exists():
                (out_dir / "frame_features.meta.json").write_text(meta_src.read_text(encoding="utf-8"), encoding="utf-8")
            feat_df = df
        else:
            feat_df = pd.read_csv(feat_csv if feat_csv.exists() else out_dir / "frame_features.csv")
        logger.info("复用帧特征: %s (%d 行)", feat_csv, len(feat_df))
    else:
        if not video.exists():
            raise FileNotFoundError(video)
        feat_df, _, _ = export_frame_features(
            video,
            parsed["cameras"],
            segments,
            out_dir / "frame_features.csv",
            frame_skip=frame_skip,
            left_to_right=left_to_right,
        )

    needed = {"mouth_opening", "side_face_weight", "yaw", "speaker", "t"}
    missing = needed - set(feat_df.columns)
    if missing:
        raise RuntimeError(f"frame_features 缺列 {missing}: {feat_csv}")

    X, y, meta = cut_lip_amp_samples(
        feat_df,
        segments,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_side=min_side,
        max_abs_yaw=max_abs_yaw,
        neg_margin=neg_margin,
        seed=seed,
        session_id=session,
    )
    return save_dataset(
        out_dir,
        X,
        y,
        meta,
        extra={
            **parsed,
            "left_to_right": left_to_right,
            "video": str(video),
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "min_side": min_side,
            "max_abs_yaw": max_abs_yaw,
            "n_gt_segments": len(segments),
            "feature_cols": list(FEATURE_NAMES),
        },
    )


def merge_npzs(session_dirs: Sequence[Path], out_dir: Path) -> Path:
    Xs, ys, metas = [], [], []
    for d in session_dirs:
        npz = d / "lip_amp_samples.npz"
        meta_p = d / "lip_amp_samples_meta.json"
        if not npz.exists():
            continue
        data = np.load(npz)
        if len(data["y"]) == 0:
            continue
        Xs.append(data["X"])
        ys.append(data["y"])
        if meta_p.exists():
            samples = json.loads(meta_p.read_text(encoding="utf-8")).get("samples", [])
            for s in samples:
                s.setdefault("session_id", d.name)
            metas.extend(samples)
    if not Xs:
        raise RuntimeError("没有可合并的 lip-amp 样本")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return save_dataset(out_dir, X, y, metas, extra={"merged_from": [str(d) for d in session_dirs]})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="咸阳儿童嘴动幅度样本准备")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--from-manifest", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "lip_amp_xianyang")
    p.add_argument("--reuse-readiness-root", type=Path, default=ROOT / "output" / "readiness_xianyang")
    p.add_argument("--position-maps-dir", type=Path, default=ROOT / "ref" / "position_maps")
    p.add_argument("--merge-out", type=Path, default=None)
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--window-sec", type=float, default=0.6)
    p.add_argument("--hop-sec", type=float, default=0.3)
    p.add_argument("--min-side", type=float, default=0.55, help="偏正脸：side_face_weight 下限")
    p.add_argument("--max-abs-yaw", type=float, default=30.0, help="偏正脸：|yaw| 上限（度）")
    p.add_argument("--neg-margin", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--require-position-map", action="store_true", default=True)
    p.add_argument("--no-require-position-map", action="store_false", dest="require_position_map")
    p.add_argument("--require-video-start-abs", action="store_true", default=True)
    p.add_argument("--no-require-video-start-abs", action="store_false", dest="require_video_start_abs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    map_paths = sorted(args.position_maps_dir.glob("*.json")) if args.position_maps_dir.exists() else []
    videos: List[Path] = []

    if args.video is not None:
        videos = [args.video]
    elif args.from_manifest is not None:
        man = json.loads(args.from_manifest.read_text(encoding="utf-8"))
        for item in man.get("sessions") or man.get("videos") or []:
            if not item.get("matched", True):
                continue
            p = Path(item.get("path") or "")
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                videos.append(p)
    else:
        raise SystemExit("请指定 --video 或 --from-manifest")

    done = []
    for i, video in enumerate(videos):
        if args.limit and i >= args.limit:
            break
        parsed = parse_xianyang_video_name(video.name)
        if parsed is None:
            continue
        session = resolve_session_dir_name(parsed)
        out_dir = args.out_root / session
        if args.skip_existing and (out_dir / "lip_amp_samples.npz").exists():
            logger.info("跳过已有: %s", session)
            done.append(out_dir)
            continue
        try:
            path = process_one(
                video,
                args.excel,
                args.out_root,
                readiness_root=args.reuse_readiness_root if args.reuse_readiness_root.exists() else None,
                frame_skip=args.frame_skip,
                window_sec=args.window_sec,
                hop_sec=args.hop_sec,
                min_side=args.min_side,
                max_abs_yaw=args.max_abs_yaw,
                neg_margin=args.neg_margin,
                seed=args.seed + i,
                skip_extract=args.skip_extract,
                position_map_paths=map_paths,
                require_position_map=args.require_position_map,
                require_video_start_abs=args.require_video_start_abs,
            )
            if path is not None:
                done.append(path.parent)
        except Exception as exc:  # noqa: BLE001
            logger.exception("处理失败 %s: %s", video.name, exc)

    merge_out = args.merge_out or (args.out_root / "merged_all")
    if done:
        merge_npzs(done, merge_out)
        logger.info("合并完成: %s", merge_out)
    else:
        logger.warning("无成功场次，未合并")


if __name__ == "__main__":
    main()
