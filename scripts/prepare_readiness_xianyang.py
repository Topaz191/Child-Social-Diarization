#!/usr/bin/env python3
"""
从咸阳视频 + 转录标注 Excel 导出帧特征并切分 readiness 正负样本。

不依赖 position_gt_verify.json：用文件名机位（如 S1S2）按画面从左到右对齐。

用法（单场）:
  python scripts/prepare_readiness_xianyang.py \\
    --video video/xianyang/0701/class1/0701-前测-五年级1班-第1组-S2S3-小组汇报.mp4

用法（按 manifest 批量，跳过已有样本）:
  python scripts/index_xianyang_dataset.py
  python scripts/prepare_readiness_xianyang.py --from-manifest output/xianyang/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.config import ASDConfig
from csd.core.utils import setup_logging
from csd.data.xianyang import (
    load_xianyang_segments,
    parse_xianyang_video_name,
    resolve_video_position_map,
    scan_xianyang_videos,
)
from csd.perception.face_tracker import FaceTracker
from csd.perception.head_pose import HeadPoseAnalyzer
from csd.social.position_speaker_mapper import PositionSpeakerMapper

logger = logging.getLogger("prepare_readiness_xy")

FEATURE_COLS = ["yaw", "pitch", "roll", "mouth_opening", "visibility"]
STUDENT_SPEAKERS = ("S1", "S2", "S3", "S4")


def assign_slots_by_camera_ltr(
    slots,
    cameras: Sequence[str],
) -> Dict[int, str]:
    """画面从左到右的槽位 → 文件名机位顺序中的说话人。"""
    ordered = sorted(slots, key=lambda s: s.mean_x)
    cams = [c.upper() for c in cameras]
    mapping: Dict[int, str] = {}
    for i, slot in enumerate(ordered):
        if i < len(cams):
            mapping[slot.cluster_id] = cams[i]
        else:
            mapping[slot.cluster_id] = f"SLOT{slot.cluster_id}"
    return mapping


def export_frame_features(
    video: Path,
    cameras: Sequence[str],
    segments: List[dict],
    out_csv: Path,
    frame_skip: int = 3,
    pre_pad_sec: float = 1.2,
    left_to_right: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, float, Dict[int, str]]:
    config = ASDConfig(
        frame_skip=frame_skip,
        speech_pose_frame_skip=max(2, frame_skip // 2),
        speech_pose_pad_sec=0.15,
        face_backend=os.environ.get("CSD_FACE_BACKEND", "auto"),
        yolov8_face_model=os.environ.get("YOLOV8_FACE_WEIGHTS", ""),
        model_cache_dir=Path(os.environ.get("CSD_MODEL_CACHE", str(ROOT / "models"))),
    )
    tracker = FaceTracker(config)
    logger.info("人脸跟踪: %s", video)
    tracker.process_video(str(video))
    tracks, fps = tracker.tracks, tracker.fps

    order = [str(s).upper() for s in (left_to_right or cameras)]
    if len(order) < 3:
        for s in ("S1", "S2", "S3"):
            if s not in order:
                order.append(s)
        order = order[:3]
    n_slots = int(getattr(config, "primary_group_n_slots", 3) or 3)
    n_slots = max(n_slots, len(order))
    mapper = PositionSpeakerMapper(config)
    size_ratio = float(getattr(config, "track_face_size_min_ratio_to_max", 0.25) or 0.25)
    slots = mapper.extract_position_slots(
        tracks, n_slots=n_slots, face_size_min_ratio=size_ratio
    )
    if not slots:
        raise RuntimeError(f"未检测到稳定人脸槽位: {video}")

    slot_to_tracks = mapper._cluster_to_tracks(tracks, n_slots=n_slots)
    slot_to_speaker = assign_slots_by_camera_ltr(slots, order)
    slot_positions = {s.cluster_id: (s.mean_x, s.mean_y) for s in slots}
    logger.info("槽位→说话人(左→中→右): %s | order=%s", slot_to_speaker, order)

    speech_intervals = [(max(0.0, s["start"] - pre_pad_sec), s["end"]) for s in segments]
    analyzer = HeadPoseAnalyzer(config)
    timeline = analyzer.build_slot_timeline(
        str(video),
        tracks,
        slot_to_tracks,
        slot_to_speaker,
        slot_positions,
        fps,
        speech_intervals=speech_intervals,
    )

    rows = []
    for slot_id, frames in timeline.frames.items():
        spk = slot_to_speaker.get(slot_id, f"SLOT{slot_id}")
        for frame_idx, pose in sorted(frames.items()):
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "t": float(frame_idx) / float(fps),
                    "slot_id": int(slot_id),
                    "speaker": spk,
                    "yaw": float(pose.yaw),
                    "pitch": float(pose.pitch),
                    "roll": float(pose.roll),
                    "mouth_opening": float(pose.mouth_opening),
                    "visibility": float(pose.visibility),
                    "side_face_weight": float(pose.side_face_weight),
                }
            )

    df = pd.DataFrame(rows).sort_values(["speaker", "t"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    meta = {
        "video": str(video),
        "fps": fps,
        "frame_skip": frame_skip,
        "n_rows": len(df),
        "cameras": list(cameras),
        "slot_to_speaker": {str(k): v for k, v in slot_to_speaker.items()},
        "speakers": sorted(df["speaker"].unique().tolist()) if len(df) else [],
    }
    out_csv.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("帧特征已写: %s (%d 行, fps=%.2f)", out_csv, len(df), fps)
    return df, float(fps), slot_to_speaker


def _window_matrix(
    spk_df: pd.DataFrame,
    t0: float,
    t1: float,
    seq_len: int,
    feature_cols: Sequence[str] = FEATURE_COLS,
) -> Optional[np.ndarray]:
    sub = spk_df[(spk_df["t"] >= t0 - 1e-6) & (spk_df["t"] <= t1 + 1e-6)]
    if len(sub) < max(2, seq_len // 4):
        return None
    ts = sub["t"].to_numpy(dtype=np.float64)
    feats = sub[list(feature_cols)].to_numpy(dtype=np.float64)
    _, uniq = np.unique(ts, return_index=True)
    ts, feats = ts[uniq], feats[uniq]
    if len(ts) < 2:
        return None
    grid = np.linspace(t0, t1, seq_len)
    out = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
    for j in range(len(feature_cols)):
        out[:, j] = np.interp(grid, ts, feats[:, j]).astype(np.float32)
    return out


def _speaking_mask(segments: List[dict], speaker: str) -> List[Tuple[float, float]]:
    return [(s["start"], s["end"]) for s in segments if s["speaker"] == speaker]


def _overlaps_any(t0: float, t1: float, intervals: Sequence[Tuple[float, float]], margin: float) -> bool:
    for a, b in intervals:
        if t1 > a - margin and t0 < b + margin:
            return True
    return False


def cut_samples(
    feat_df: pd.DataFrame,
    segments: List[dict],
    window_sec: float = 0.75,
    seq_len: int = 16,
    neg_margin: float = 2.0,
    seed: int = 42,
    max_neg_per_gap: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    positives: List[Tuple[np.ndarray, dict]] = []
    negatives: List[Tuple[np.ndarray, dict]] = []

    by_spk = {spk: g.sort_values("t").reset_index(drop=True) for spk, g in feat_df.groupby("speaker")}
    all_intervals = [(s["start"], s["end"]) for s in segments]
    duration = max((s["end"] for s in segments), default=0.0)
    if len(feat_df):
        duration = max(duration, float(feat_df["t"].max()))

    for i, seg in enumerate(segments):
        spk = seg["speaker"]
        if spk not in by_spk:
            continue
        onset = float(seg["start"])
        t0, t1 = onset - window_sec, onset
        if t0 < 0:
            continue
        mat = _window_matrix(by_spk[spk], t0, t1, seq_len)
        if mat is None:
            continue
        positives.append(
            (
                mat,
                {
                    "label": 1,
                    "speaker": spk,
                    "t0": round(t0, 3),
                    "t1": round(t1, 3),
                    "seg_idx": i,
                    "onset": round(onset, 3),
                    "kind": "pre_speech",
                },
            )
        )

    sorted_segs = sorted(segments, key=lambda x: x["start"])
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for seg in sorted_segs:
        if seg["start"] - cursor >= window_sec + 2 * neg_margin:
            gaps.append((cursor + neg_margin, seg["start"] - neg_margin))
        cursor = max(cursor, seg["end"])
    if duration - cursor >= window_sec + 2 * neg_margin:
        gaps.append((cursor + neg_margin, duration - neg_margin))

    speakers_with_feat = [s for s in by_spk if s in STUDENT_SPEAKERS]
    for g0, g1 in gaps:
        span = g1 - g0
        if span < window_sec:
            continue
        n_take = min(max_neg_per_gap, max(1, int(span // max(window_sec, 1e-6))))
        for _ in range(n_take):
            if not speakers_with_feat:
                break
            spk = str(rng.choice(speakers_with_feat))
            own = _speaking_mask(segments, spk)
            for _try in range(8):
                t0 = float(rng.uniform(g0, max(g0, g1 - window_sec)))
                t1 = t0 + window_sec
                if _overlaps_any(t0, t1, all_intervals, margin=neg_margin):
                    continue
                if _overlaps_any(t0, t1, own, margin=neg_margin):
                    continue
                mat = _window_matrix(by_spk[spk], t0, t1, seq_len)
                if mat is None:
                    continue
                negatives.append(
                    (
                        mat,
                        {
                            "label": 0,
                            "speaker": spk,
                            "t0": round(t0, 3),
                            "t1": round(t1, 3),
                            "seg_idx": -1,
                            "onset": None,
                            "kind": "silence",
                        },
                    )
                )
                break

    n = min(len(positives), len(negatives))
    if n == 0:
        logger.warning("正/负样本为空: pos=%d neg=%d", len(positives), len(negatives))
        return (
            np.zeros((0, seq_len, len(FEATURE_COLS)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:n] + negatives[:n]
    rng.shuffle(chosen)
    X = np.stack([c[0] for c in chosen], axis=0)
    y = np.array([c[1]["label"] for c in chosen], dtype=np.int64)
    meta = [c[1] for c in chosen]
    logger.info(
        "样本切分完成: pos候选=%d neg候选=%d 平衡后=%d (各%d)",
        len(positives),
        len(negatives),
        len(meta),
        n,
    )
    return X, y, meta


def save_dataset(out_dir: Path, X: np.ndarray, y: np.ndarray, meta: List[dict], extra: Optional[dict] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "readiness_samples.npz"
    np.savez_compressed(npz_path, X=X, y=y)
    (out_dir / "readiness_samples_meta.json").write_text(
        json.dumps({"samples": meta, "extra": extra or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "n_samples": int(len(y)),
        "n_pos": int((y == 1).sum()) if len(y) else 0,
        "n_neg": int((y == 0).sum()) if len(y) else 0,
        "X_shape": list(X.shape),
        "feature_cols": FEATURE_COLS,
        "npz": str(npz_path),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("数据集已写: %s", summary)
    return npz_path


def process_one(
    video: Path,
    excel: Path,
    out_root: Path,
    frame_skip: int,
    window_sec: float,
    seq_len: int,
    neg_margin: float,
    seed: int,
    skip_extract: bool,
    student_only: bool,
    position_map_paths: Optional[Sequence[Path]] = None,
    require_position_map: bool = False,
) -> Optional[Path]:
    parsed = parse_xianyang_video_name(video.name)
    if parsed is None:
        logger.error("无法解析文件名: %s", video.name)
        return None

    pos = None
    if position_map_paths:
        pos = resolve_video_position_map(video.name, position_map_paths, require_confirmed=True)
    if require_position_map and pos is None:
        logger.error("缺少已确认的位置标注，跳过: %s", video.name)
        return None
    left_to_right = pos["left_to_right"] if pos else ["S1", "S2", "S3"]
    if pos:
        logger.info("使用人工位置标注: %s", left_to_right)
    else:
        logger.warning("未找到 confirmed 位置标注，回退默认左中右 S1,S2,S3（请尽快人工确认）")

    out_dir = out_root / f"{parsed['date']}_{parsed['sheet']}_{parsed['test']}_{parsed['camera_tag']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = out_dir / "frame_features.csv"
    speakers = STUDENT_SPEAKERS if student_only else None
    segments = load_xianyang_segments(
        excel,
        parsed["sheet"],
        parsed["phase"],
        speakers=speakers,
        min_duration=0.4,
        video_start_abs=None if pos is None else pos.get("video_start_abs"),
        align_to_video=True,
    )
    # 只保留画面中出现的说话人段，避免把画外说话人对齐到错误人脸
    in_frame = set(left_to_right)
    segments = [s for s in segments if s["speaker"] in in_frame]
    time_base = segments[0].get("time_base") if segments else None
    logger.info(
        "处理 %s | sheet=%s phase=%s order=%s | video_start_abs=%s | time_base=%s | 画内学生段=%d",
        video.name,
        parsed["sheet"],
        parsed["phase"],
        left_to_right,
        None if pos is None else pos.get("video_start_abs_str"),
        time_base,
        len(segments),
    )
    (out_dir / "gt_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "session_meta.json").write_text(
        json.dumps(
            {
                "video": str(video),
                **parsed,
                "left_to_right": left_to_right,
                "position_map_source": None if pos is None else pos.get("source"),
                "video_start_abs": None if pos is None else pos.get("video_start_abs"),
                "video_start_abs_str": None if pos is None else pos.get("video_start_abs_str"),
                "time_base": time_base,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if skip_extract and feat_csv.exists():
        feat_df = pd.read_csv(feat_csv)
        fps = 25.0
        meta_path = feat_csv.with_suffix(".meta.json")
        if meta_path.exists():
            fps = float(json.loads(meta_path.read_text(encoding="utf-8")).get("fps", 25.0))
        logger.info("复用特征: %s (%d 行)", feat_csv, len(feat_df))
    else:
        if not video.exists():
            raise FileNotFoundError(video)
        feat_df, fps, _ = export_frame_features(
            video,
            parsed["cameras"],
            segments,
            feat_csv,
            frame_skip=frame_skip,
            left_to_right=left_to_right,
        )

    X, y, meta = cut_samples(
        feat_df,
        segments,
        window_sec=window_sec,
        seq_len=seq_len,
        neg_margin=neg_margin,
        seed=seed,
    )
    return save_dataset(
        out_dir,
        X,
        y,
        meta,
        extra={
            **parsed,
            "left_to_right": left_to_right,
            "position_map_source": None if pos is None else pos.get("source"),
            "window_sec": window_sec,
            "seq_len": seq_len,
            "neg_margin": neg_margin,
            "fps": fps,
            "n_gt_segments": len(segments),
            "n_feature_rows": int(len(feat_df)),
            "video": str(video),
        },
    )


def merge_npzs(session_dirs: Sequence[Path], out_dir: Path) -> Path:
    """把多场次 npz 合并成一个训练文件。"""
    Xs, ys, metas = [], [], []
    for d in session_dirs:
        npz = d / "readiness_samples.npz"
        meta_p = d / "readiness_samples_meta.json"
        if not npz.exists():
            continue
        data = np.load(npz)
        if len(data["y"]) == 0:
            continue
        Xs.append(data["X"])
        ys.append(data["y"])
        if meta_p.exists():
            metas.extend(json.loads(meta_p.read_text(encoding="utf-8")).get("samples", []))
    if not Xs:
        raise RuntimeError("没有可合并的样本")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return save_dataset(out_dir, X, y, metas, extra={"merged_from": [str(d) for d in session_dirs]})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="咸阳 readiness 样本准备")
    p.add_argument("--video", type=Path, default=None, help="单个视频路径")
    p.add_argument("--from-manifest", type=Path, default=None, help="manifest.json，批量处理 matched 场次")
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "readiness_xianyang")
    p.add_argument("--merge-out", type=Path, default=None, help="合并所有场次到该目录（训练用）")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--window-sec", type=float, default=0.75)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--neg-margin", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="已有 readiness_samples.npz 则跳过")
    p.add_argument("--include-teacher", action="store_true", help="正样本也包含 T（默认只要学生）")
    p.add_argument("--limit", type=int, default=None, help="最多处理 N 个视频（试跑）")
    p.add_argument(
        "--position-maps",
        type=Path,
        nargs="*",
        default=None,
        help="人工位置标注 JSON；默认自动读取 ref/position_maps/*.json",
    )
    p.add_argument(
        "--require-position-map",
        action="store_true",
        help="没有 confirmed 位置标注则跳过该视频",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    student_only = not args.include_teacher
    map_dir = ROOT / "ref" / "position_maps"
    position_maps = args.position_maps
    if position_maps is None and map_dir.exists():
        position_maps = sorted(map_dir.glob("*.json"))

    videos: List[Path] = []
    if args.video is not None:
        videos = [args.video]
    elif args.from_manifest is not None:
        man = json.loads(args.from_manifest.read_text(encoding="utf-8"))
        videos = [Path(s["path"]) for s in man.get("sessions", []) if s.get("matched")]
    else:
        videos = [m.path for m in scan_xianyang_videos(args.video_root) if m.parse_ok]

    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        raise SystemExit("没有待处理视频。请先放入 video/xianyang/{date}/classN/ 并检查命名。")

    done_dirs: List[Path] = []
    for video in videos:
        parsed = parse_xianyang_video_name(video.name)
        if parsed is None:
            logger.warning("跳过无法解析: %s", video)
            continue
        out_dir = args.out_root / f"{parsed['date']}_{parsed['sheet']}_{parsed['test']}_{parsed['camera_tag']}"
        if args.skip_existing and (out_dir / "readiness_samples.npz").exists():
            logger.info("已存在，跳过: %s", out_dir.name)
            done_dirs.append(out_dir)
            continue
        result = process_one(
            video=video,
            excel=args.excel,
            out_root=args.out_root,
            frame_skip=args.frame_skip,
            window_sec=args.window_sec,
            seq_len=args.seq_len,
            neg_margin=args.neg_margin,
            seed=args.seed,
            skip_extract=args.skip_extract,
            student_only=student_only,
            position_map_paths=position_maps,
            require_position_map=args.require_position_map,
        )
        if result is not None:
            done_dirs.append(out_dir)

    merge_out = args.merge_out or (args.out_root / "merged_all")
    if done_dirs:
        try:
            merge_npzs(done_dirs, merge_out)
            logger.info("合并训练集: %s", merge_out)
        except RuntimeError as exc:
            logger.warning("%s", exc)


if __name__ == "__main__":
    main()
