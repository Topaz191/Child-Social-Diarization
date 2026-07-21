#!/usr/bin/env python3
"""
扫描 video/xianyang，与咸阳转录标注 Excel 自动对应，写出 manifest。

目录约定:
  video/xianyang/{MMDD}/class{N}/*.mp4

文件名约定:
  {日期}-{前|中|后测}-{年级}年级{班}班-第{组}组-{机位}-{活动}.mp4
  例: 0701-前测-五年级1班-第2组-S1S2-小组讨论.mp4

标注约定:
  sheet 名 = {年级}-{班}-{组}，如 5-1-2；表内按 前测/中测/后测 分段。

用法:
  python scripts/index_xianyang_dataset.py
  python scripts/index_xianyang_dataset.py --date 0701 --class-id 1
  python scripts/index_xianyang_dataset.py --dump-segments --sheet 5-1-1 --phase 前测
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.data.xianyang import (
    build_xianyang_manifest,
    iter_matched_sessions,
    load_xianyang_segments,
)

logger = logging.getLogger("index_xianyang")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="索引咸阳视频并与标注 Excel 对齐")
    p.add_argument(
        "--video-root",
        type=Path,
        default=ROOT / "video" / "xianyang",
        help="视频根目录（其下为 {date}/classN/）",
    )
    p.add_argument(
        "--excel",
        type=Path,
        default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx",
        help="转录标注 Excel",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "xianyang" / "manifest.json",
        help="manifest 输出路径",
    )
    p.add_argument("--date", type=str, default=None, help="只打印某日期，如 0701")
    p.add_argument("--class-id", type=int, default=None, help="只打印某班级，如 1")
    p.add_argument("--phase", type=str, default=None, choices=["前测", "中测", "后测"])
    p.add_argument("--no-count-segments", action="store_true", help="不统计标注段数（更快）")
    p.add_argument("--dump-segments", action="store_true", help="导出某 sheet/phase 的说话段 JSON")
    p.add_argument("--sheet", type=str, default=None, help="配合 --dump-segments")
    p.add_argument(
        "--segments-out",
        type=Path,
        default=None,
        help="说话段 JSON 输出路径；默认 output/xianyang/segments_{sheet}_{phase}.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.dump_segments:
        if not args.sheet or not args.phase:
            raise SystemExit("--dump-segments 需要同时指定 --sheet 与 --phase")
        segs = load_xianyang_segments(args.excel, args.sheet, args.phase)
        out = args.segments_out or (
            ROOT / "output" / "xianyang" / f"segments_{args.sheet}_{args.phase}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "excel": str(args.excel),
            "sheet": args.sheet,
            "phase": args.phase,
            "n_segments": len(segs),
            "n_student": sum(1 for s in segs if str(s["speaker"]).startswith("S")),
            "segments": segs,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("已写说话段: %s (%d 段)", out, len(segs))
        return

    if not args.video_root.exists():
        raise FileNotFoundError(f"视频根目录不存在: {args.video_root}")
    if not args.excel.exists():
        raise FileNotFoundError(f"标注 Excel 不存在: {args.excel}")

    manifest = build_xianyang_manifest(
        args.video_root,
        args.excel,
        count_segments=not args.no_count_segments,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "扫描完成: videos=%d matched=%d parse_failed=%d -> %s",
        manifest["n_videos"],
        manifest["n_matched"],
        manifest["n_parse_failed"],
        args.out,
    )

    shown = list(
        iter_matched_sessions(
            manifest,
            date=args.date,
            class_id=args.class_id,
            phase=args.phase,
            require_matched=False,
        )
    )
    # 若指定了过滤条件，也把未匹配的同条件视频打出来
    if args.date or args.class_id is not None or args.phase:
        pool = manifest["sessions"]
        shown = [
            s
            for s in pool
            if (args.date is None or s.get("date") == args.date)
            and (args.class_id is None or s.get("class_id") == args.class_id)
            and (args.phase is None or s.get("phase") == args.phase)
        ]

    for s in shown:
        flag = "OK" if s.get("matched") else "MISS"
        logger.info(
            "[%s] %s | sheet=%s %s | cams=%s | segs=%s/%s | %s",
            flag,
            s.get("date"),
            s.get("sheet"),
            s.get("phase"),
            s.get("camera_tag"),
            s.get("n_student_segments"),
            s.get("n_segments"),
            Path(s["path"]).name,
        )

    missing = manifest.get("sheets_without_video") or []
    if missing:
        # 只预览前若干条，避免刷屏
        preview = missing[:12]
        logger.info("有标注但尚未放入视频的 sheet/phase: %d 条（预览 %d）", len(missing), len(preview))
        for m in preview:
            logger.info("  - %s %s (n_segments=%s)", m["sheet"], m["phase"], m["n_segments"])


if __name__ == "__main__":
    main()
