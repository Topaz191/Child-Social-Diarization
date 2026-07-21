#!/usr/bin/env python3
"""校验 position_map JSON：结构、说话人是否合法、是否已确认。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def validate(path: Path) -> int:
    doc = json.loads(path.read_text(encoding="utf-8"))
    videos = doc.get("videos") or []
    errors = 0
    confirmed = 0
    for i, v in enumerate(videos):
        name = v.get("video_name", f"#{i}")
        ltr = v.get("left_to_right")
        if not isinstance(ltr, list) or not ltr:
            print(f"[ERR] {name}: left_to_right 为空")
            errors += 1
            continue
        for spk in ltr:
            if not re.fullmatch(r"S\d+", str(spk).upper()):
                print(f"[ERR] {name}: 非法说话人 {spk!r}（应为 S1/S2/…）")
                errors += 1
        expected = [s.upper() for s in (v.get("in_frame_expected") or [])]
        got = [str(s).upper() for s in ltr]
        if len(got) != 3:
            print(f"[WARN] {name}: left_to_right 长度={len(got)}，预期本组 3 人（左/中/右）")
        if expected and sorted(got) != sorted(expected):
            print(
                f"[WARN] {name}: left_to_right={got} 与 in_frame_expected {expected} 集合不一致"
            )
        if len(got) != len(set(got)):
            print(f"[ERR] {name}: left_to_right 有重复")
            errors += 1
        if v.get("confirmed") is True:
            confirmed += 1
            print(f"[OK]  {name}: {' → '.join(got)}")
        else:
            print(f"[TODO] {name}: {' → '.join(got)}  (confirmed=false)")
    print(f"\n合计 {len(videos)} 个视频，已确认 {confirmed}，错误 {errors}")
    return 1 if errors else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=ROOT / "ref" / "position_maps" / "0701_class1.json",
    )
    args = p.parse_args()
    raise SystemExit(validate(args.json_path))


if __name__ == "__main__":
    main()
