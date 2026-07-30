#!/usr/bin/env python3
"""
抽取咸阳视频的共享视觉特征缓存（轨迹 / 唇轮廓 / 头姿 / 视线注意力）。

输出: output/visual_cache/<session_key>/
  meta.json, tracks.npz, mesh.npz, attention.npz, lips_timeseries.parquet|csv

用法:
  python scripts/extract_visual_cache_xianyang.py \\
    --from-manifest output/xianyang/manifest.json \\
    --require-position-map --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

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
from csd.perception.visual_cache import (
    build_visual_cache_for_video,
    cache_dir_for,
    has_tracks_cache,
    session_key_from_video,
)

logger = logging.getLogger("extract_visual_cache")

STUDENTS = ("S1", "S2", "S3", "S4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="抽取共享视觉特征缓存")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--from-manifest", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "visual_cache")
    p.add_argument("--position-maps", type=Path, nargs="*", default=None)
    p.add_argument("--require-position-map", action="store_true")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--dense-skip", type=int, default=1)
    p.add_argument("--speech-pad-sec", type=float, default=1.0)
    p.add_argument("--full-mesh", action="store_true", help="额外保存全脸 mesh 坐标（体积大）")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-sec", type=float, default=None)
    return p.parse_args()


def process_one(
    video: Path,
    excel: Path,
    out_root: Path,
    *,
    position_map_paths: Sequence[Path],
    require_position_map: bool,
    frame_skip: int,
    dense_skip: int,
    speech_pad_sec: float,
    full_mesh: bool,
    max_sec: Optional[float],
) -> Optional[Path]:
    meta = parse_xianyang_video_name(video.name)
    if not meta:
        logger.warning("无法解析文件名: %s", video.name)
        return None

    pos = resolve_video_position_map(video, position_map_paths or [])
    if require_position_map and (pos is None or not pos.get("confirmed")):
        logger.warning("缺少 confirmed position_map，跳过: %s", video.name)
        return None
    if require_position_map and pos is not None and pos.get("video_start_abs") is None:
        logger.warning("缺少 video_start_abs，跳过: %s", video.name)
        return None

    left_to_right = list(pos.get("left_to_right") or meta.get("cameras") or []) if pos else list(meta.get("cameras") or [])
    segments = []
    if excel.exists():
        try:
            segments = load_xianyang_segments(
                excel,
                meta["sheet"],
                meta["phase"],
                speakers=STUDENTS,
                video_start_abs=None if pos is None else pos.get("video_start_abs"),
                align_to_video=True,
            )
            segments = [s for s in segments if s.get("speaker") in STUDENTS]
        except Exception as exc:
            logger.warning("加载 GT 段失败（仍抽取稀疏特征）: %s", exc)

    speech_intervals = [(float(s["start"]), float(s["end"])) for s in segments] if segments else None
    session_dir = cache_dir_for(video, out_root)
    cfg = ASDConfig(frame_skip=max(1, frame_skip))
    return build_visual_cache_for_video(
        video,
        session_dir,
        config=cfg,
        left_to_right=left_to_right,
        speech_intervals=speech_intervals,
        frame_skip=frame_skip,
        dense_skip=dense_skip,
        speech_pad_sec=speech_pad_sec,
        full_mesh=full_mesh,
        max_sec=max_sec,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
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

    done = 0
    for video in videos:
        if not video.is_absolute():
            video = ROOT / video
        key = session_key_from_video(video)
        session_dir = args.out_root / key
        if args.skip_existing and (session_dir / "mesh.npz").exists() and has_tracks_cache(session_dir):
            logger.info("跳过已有: %s", key)
            done += 1
            continue
        try:
            d = process_one(
                video,
                args.excel,
                args.out_root,
                position_map_paths=position_maps or [],
                require_position_map=args.require_position_map,
                frame_skip=args.frame_skip,
                dense_skip=args.dense_skip,
                speech_pad_sec=args.speech_pad_sec,
                full_mesh=args.full_mesh,
                max_sec=args.max_sec,
            )
        except Exception as exc:
            logger.exception("失败 %s: %s", video.name, exc)
            continue
        if d is not None:
            done += 1
            logger.info("OK %s → %s", video.name, d)
    logger.info("完成 %d / %d", done, len(videos))


if __name__ == "__main__":
    main()
