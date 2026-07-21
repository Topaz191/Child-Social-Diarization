#!/usr/bin/env python3
"""
为 xianyang 视频生成「画面位置 ↔ 说话人」人工标注模板。

你只需打开视频，确认画面从左到右分别是谁，填写/修改 left_to_right，
并把 confirmed 改为 true。

用法:
  python scripts/make_position_map_templates.py --date 0701 --class-id 1
  python scripts/make_position_map_templates.py --date 0701 --class-id 1 --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.data.xianyang import parse_xianyang_video_name, scan_xianyang_videos

import logging

logger = logging.getLogger("make_pos_map")


def _parse_roster(df: pd.DataFrame) -> Dict[str, str]:
    """从 sheet 表头解析 S1/S2/S3 → 姓名。"""
    roster: Dict[str, str] = {}
    for j in range(min(6, df.shape[1])):
        cell = df.iloc[2, j] if df.shape[0] > 2 else None
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            continue
        text = str(cell).strip()
        m = re.match(r"(S\d)\s*(.*)$", text, re.IGNORECASE)
        if not m:
            continue
        spk = m.group(1).upper()
        name = m.group(2).strip()
        roster.setdefault(spk, name or spk)
    return dict(sorted(roster.items()))


def _first_note(df: pd.DataFrame) -> Optional[str]:
    for i in range(min(20, len(df))):
        if df.shape[1] <= 5:
            break
        v = df.iloc[i, 5]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s and s.lower() != "note":
            return s
    return None


def build_entry(video_path: Path, excel: Path) -> Optional[Dict[str, Any]]:
    parsed = parse_xianyang_video_name(video_path.name)
    if parsed is None:
        return None
    sheet = parsed["sheet"]
    try:
        df = pd.read_excel(excel, sheet_name=sheet, header=None)
    except ValueError:
        df = None

    roster = _parse_roster(df) if df is not None else {}
    note = _first_note(df) if df is not None else None
    cams = list(parsed["cameras"])
    # 画面稳定 3 人入镜；文件名机位不代表人数
    roster_keys = sorted(roster.keys()) if roster else ["S1", "S2", "S3"]
    if len(roster_keys) < 3:
        for s in ("S1", "S2", "S3"):
            if s not in roster_keys:
                roster_keys.append(s)
        roster_keys = roster_keys[:3]

    return {
        "video_name": video_path.name,
        "rel_path": str(video_path),  # 生成时写入；人工可忽略
        "date": parsed["date"],
        "class_id": parsed["class_id"],
        "grade": parsed["grade"],
        "group": parsed["group"],
        "phase": parsed["phase"],
        "test": parsed["test"],
        "sheet": sheet,
        "camera_tag": parsed["camera_tag"],
        "activity": parsed["activity"],
        "roster": roster,
        "in_frame_expected": roster_keys[:3],
        "annotation_note": note,
        # ---- 请人工填写/确认 ----
        "left_to_right": roster_keys[:3],  # 占位：左/中/右，务必按实际座位修改
        "confirmed": False,
        "notes": "",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--class-id", type=int, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="默认 ref/position_maps/{date}_class{N}.json",
    )
    p.add_argument("--force", action="store_true", help="覆盖已有文件（会丢掉已填写内容）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    out = args.out or (ROOT / "ref" / "position_maps" / f"{args.date}_class{args.class_id}.json")

    if out.exists() and not args.force:
        raise SystemExit(f"已存在 {out}。若要保留已填内容请直接编辑；强制重建请加 --force")

    videos = [
        m
        for m in scan_xianyang_videos(args.video_root)
        if m.parse_ok and m.date == args.date and m.class_id == args.class_id
    ]
    if not videos:
        raise SystemExit(f"未找到视频: {args.video_root}/{args.date}/class{args.class_id}/")

    entries = []
    for m in videos:
        e = build_entry(m.path, args.excel)
        if e:
            # 相对路径更可读
            try:
                e["rel_path"] = str(m.path.relative_to(args.video_root))
            except ValueError:
                e["rel_path"] = str(m.path)
            entries.append(e)

    doc = {
        "version": 1,
        "date": args.date,
        "class_id": args.class_id,
        "excel": str(args.excel.name),
        "how_to_fill": [
            "每个视频画面稳定入镜 3 名本组学生（左/中/右）；远处其他组人脸会被尺寸过滤掉。",
            "打开对应 mp4，从观众视角按从左到右填写 left_to_right，必须是 3 个：S1/S2/S3 的一个排列。",
            "文件名里的 S1S2/S2S3 只表示录音/剪辑机位，不代表画面只有两人。",
            "核对无误后把 confirmed 改为 true。",
            "T（老师）一般不写进 left_to_right。",
        ],
        "position_labels": {
            "3人": ["左", "中", "右"],
        },
        "expected_in_frame": 3,
        "videos": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写模板: %s (%d 个视频)", out, len(entries))
    for e in entries:
        logger.info(
            "  %s | sheet=%s | 预填 left_to_right=%s | roster=%s",
            e["video_name"],
            e["sheet"],
            e["left_to_right"],
            e["roster"],
        )


if __name__ == "__main__":
    main()
