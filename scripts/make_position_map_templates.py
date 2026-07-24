#!/usr/bin/env python3
"""
为 xianyang 生成「画面位置 ↔ 说话人」人工标注模板。

默认按 Excel sheet（{年级}-{班}-{组}）展开各组，不依赖本地是否已有 mp4。
你只需打开对应视频，确认画面从左到右分别是谁，填写 left_to_right，并把 confirmed 改为 true。

用法:
  python scripts/make_position_map_templates.py --date 0701 --class-id 3 --phase 前测
  python scripts/make_position_map_templates.py --date 0706 --class-id 3 --phase 后测 --force
  # 旧行为：只按本地已有视频生成
  python scripts/make_position_map_templates.py --date 0701 --class-id 3 --from-videos --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.utils import setup_logging
from csd.data.xianyang import (
    PHASE_TO_TEST,
    list_excel_sheets,
    parse_video_start_abs_from_note,
    parse_xianyang_video_name,
    scan_xianyang_videos,
)

logger = logging.getLogger("make_pos_map")

_SHEET_RE = re.compile(r"^(?P<grade>\d+)-(?P<class_id>\d+)-(?P<group>\d+)$")
_CAMERA_HINT_RE = re.compile(r"S\d(?:S\d)+", re.IGNORECASE)


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


def _roster_keys(roster: Dict[str, str]) -> List[str]:
    keys = sorted(roster.keys()) if roster else ["S1", "S2", "S3"]
    if len(keys) < 3:
        for s in ("S1", "S2", "S3"):
            if s not in keys:
                keys.append(s)
        keys = keys[:3]
    return keys[:3]


def _camera_from_note(note: Optional[str], default: str = "S2S3") -> str:
    if not note:
        return default
    m = _CAMERA_HINT_RE.search(note.replace(" ", ""))
    return m.group(0).upper() if m else default


def build_entry_from_fields(
    *,
    date: str,
    class_id: int,
    grade: int,
    group: int,
    phase: str,
    camera_tag: str,
    activity: str,
    roster: Dict[str, str],
    note: Optional[str],
    video_name: Optional[str] = None,
    rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    keys = _roster_keys(roster)
    sheet = f"{grade}-{class_id}-{group}"
    if not video_name:
        video_name = f"{date}-{phase}-五年级{class_id}班-第{group}组-{camera_tag}-{activity}.mp4"
    if rel_path is None:
        rel_path = str(Path(date) / f"class{class_id}" / video_name)
    v_abs, v_str = parse_video_start_abs_from_note(note, camera_tag=camera_tag)
    return {
        "video_name": video_name,
        "rel_path": rel_path,
        "date": date,
        "class_id": class_id,
        "grade": grade,
        "group": group,
        "phase": phase,
        "test": PHASE_TO_TEST[phase],
        "sheet": sheet,
        "camera_tag": camera_tag,
        "activity": activity,
        "roster": roster,
        "in_frame_expected": keys,
        "annotation_note": note,
        "video_start_abs": v_abs,
        "video_start_abs_str": v_str,
        "left_to_right": keys,  # 占位：左/中/右，务必按实际座位修改
        "confirmed": False,
        "notes": "",
    }


def build_entry_from_video(video_path: Path, excel: Path) -> Optional[Dict[str, Any]]:
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
    return build_entry_from_fields(
        date=parsed["date"],
        class_id=parsed["class_id"],
        grade=parsed["grade"],
        group=parsed["group"],
        phase=parsed["phase"],
        camera_tag=parsed["camera_tag"],
        activity=parsed["activity"],
        roster=roster,
        note=note,
        video_name=video_path.name,
        rel_path=str(video_path),
    )


def list_class_sheets(excel: Path, class_id: int, grade: Optional[int] = None) -> List[Tuple[str, int, int]]:
    """返回 [(sheet_name, grade, group), ...]，按组号排序。"""
    out: List[Tuple[str, int, int]] = []
    for name in list_excel_sheets(excel):
        m = _SHEET_RE.match(str(name).strip())
        if not m:
            continue
        g = int(m.group("grade"))
        c = int(m.group("class_id"))
        group = int(m.group("group"))
        if c != class_id:
            continue
        if grade is not None and g != grade:
            continue
        out.append((name, g, group))
    out.sort(key=lambda x: (x[1], x[2]))
    return out


def build_entries_from_excel(
    excel: Path,
    *,
    date: str,
    class_id: int,
    phase: str,
    grade: Optional[int] = None,
    activity: str = "小组讨论",
) -> List[Dict[str, Any]]:
    sheets = list_class_sheets(excel, class_id, grade=grade)
    if not sheets:
        raise SystemExit(f"Excel 中未找到 class_id={class_id} 的 sheet（期望形如 5-{class_id}-N）")

    entries: List[Dict[str, Any]] = []
    for sheet_name, g, group in sheets:
        df = pd.read_excel(excel, sheet_name=sheet_name, header=None)
        roster = _parse_roster(df)
        note = _first_note(df)
        camera = _camera_from_note(note)
        entries.append(
            build_entry_from_fields(
                date=date,
                class_id=class_id,
                grade=g,
                group=group,
                phase=phase,
                camera_tag=camera,
                activity=activity,
                roster=roster,
                note=note,
            )
        )
    return entries


def build_entries_from_videos(
    video_root: Path,
    excel: Path,
    *,
    date: str,
    class_id: int,
) -> List[Dict[str, Any]]:
    videos = [
        m
        for m in scan_xianyang_videos(video_root)
        if m.parse_ok and m.date == date and m.class_id == class_id
    ]
    if not videos:
        raise SystemExit(f"未找到视频: {video_root}/{date}/class{class_id}/")

    entries: List[Dict[str, Any]] = []
    for m in videos:
        e = build_entry_from_video(m.path, excel)
        if not e:
            continue
        try:
            e["rel_path"] = str(m.path.relative_to(video_root))
        except ValueError:
            e["rel_path"] = str(m.path)
        entries.append(e)
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--date", type=str, required=True, help="写入模板的日期，如 0701（用于拼 video_name）")
    p.add_argument("--class-id", type=int, required=True)
    p.add_argument(
        "--phase",
        type=str,
        default="前测",
        choices=["前测", "中测", "后测"],
        help="该日期对应的测试阶段（默认前测；0704 中测、0706 后测等请显式指定）",
    )
    p.add_argument("--grade", type=int, default=None, help="只取该年级的 sheet；默认该班全部年级匹配项")
    p.add_argument("--activity", type=str, default="讨论", help="占位活动名，写入猜测的 video_name")
    p.add_argument(
        "--from-videos",
        action="store_true",
        help="改为只扫描本地 mp4（旧行为）；默认按 Excel sheet 造全部组",
    )
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

    if args.from_videos:
        entries = build_entries_from_videos(
            args.video_root, args.excel, date=args.date, class_id=args.class_id
        )
    else:
        entries = build_entries_from_excel(
            args.excel,
            date=args.date,
            class_id=args.class_id,
            phase=args.phase,
            grade=args.grade,
            activity=args.activity,
        )

    doc = {
        "version": 1,
        "date": args.date,
        "class_id": args.class_id,
        "excel": str(args.excel.name),
        "how_to_fill": [
            "每个视频画面稳定入镜 3 名本组学生（左/中/右）；远处其他组人脸会被尺寸过滤掉。",
            "打开对应 mp4，从观众视角按从左到右填写 left_to_right，必须是 3 个：S1/S2/S3 的一个排列。",
            "文件名里的 S1S2/S2S3 只表示录音/剪辑机位，不代表画面只有两人。",
            "模板默认按 Excel 各组生成；video_name 为占位猜测，若与真实文件名不一致请改正。",
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
    logger.info("已写模板: %s (%d 个视频/组)", out, len(entries))
    for e in entries:
        logger.info(
            "  group=%s sheet=%s | %s | roster=%s",
            e["group"],
            e["sheet"],
            e["video_name"],
            e["roster"],
        )


if __name__ == "__main__":
    main()
