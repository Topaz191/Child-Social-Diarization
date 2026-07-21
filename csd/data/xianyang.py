"""
咸阳（xianyang）小学生干预数据：视频目录 ↔ 转录标注 Excel 自动对应。

约定目录：
  video/xianyang/{MMDD}/class{N}/*.mp4

约定文件名（示例）：
  0701-前测-五年级1班-第1组-S2S3-小组汇报.mp4
  0701-前测-5年级1班-第2组-S1S2-小组讨论.mp4

约定标注：
  ref/202507-xianyang-小学生转录标注.xlsx
  sheet 名 = {年级}-{班}-{组}，如 5-1-2
  表内用「前测/中测/后测」分段，列 Start/End/Speaker/Content
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PHASES = ("前测", "中测", "后测")
PHASE_TO_TEST = {"前测": "pre", "中测": "mid", "后测": "post"}
CN_DIGIT = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

# 0701-前测-五年级1班-第2组-S1S2-小组讨论.mp4
_VIDEO_RE = re.compile(
    r"^(?P<date>\d{3,4})-"
    r"(?P<phase>前测|中测|后测)-"
    r"(?P<grade_raw>\d+|[一二三四五六七八九十]+)年级"
    r"(?P<class_id>\d+)班-"
    r"第(?P<group>\d+)组-"
    r"(?P<cameras>S\d(?:S\d)*)-"
    r"(?P<activity>.+)$",
    re.IGNORECASE,
)
_CLASS_DIR_RE = re.compile(r"^class(?P<n>\d+)$", re.IGNORECASE)
_CAMERA_RE = re.compile(r"S\d", re.IGNORECASE)


def _cn_or_int(token: str) -> Optional[int]:
    token = str(token).strip()
    if token.isdigit():
        return int(token)
    if token in CN_DIGIT:
        return CN_DIGIT[token]
    # 十一 / 十二 …
    if token.startswith("十"):
        rest = token[1:]
        return 10 if not rest else 10 + CN_DIGIT.get(rest, 0)
    return None


def parse_time_to_seconds(value: Any) -> Optional[float]:
    """把 Excel 时间（time / timedelta / HH:MM:SS / 秒）转为秒。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, datetime):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, (int, float)):
        # Excel 有时把一天内时间存成 0~1 的分数
        v = float(value)
        if 0 <= v < 1.5:
            return v * 86400.0
        return v
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    if s.isdigit() or re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.replace("：", ":").split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except ValueError:
        return None
    return None


def sheet_id_for(grade: int, class_id: int, group: int) -> str:
    return f"{grade}-{class_id}-{group}"


def parse_xianyang_video_name(name: str) -> Optional[Dict[str, Any]]:
    """解析视频文件名（可带或不带扩展名）。"""
    stem = Path(name).stem.strip()
    # 兼容偶发空格：五年级 1班
    stem = re.sub(r"\s+", "", stem)
    m = _VIDEO_RE.match(stem)
    if not m:
        return None
    grade = _cn_or_int(m.group("grade_raw"))
    if grade is None:
        return None
    cameras = [c.upper() for c in _CAMERA_RE.findall(m.group("cameras"))]
    return {
        "date": m.group("date"),
        "phase": m.group("phase"),
        "grade": grade,
        "class_id": int(m.group("class_id")),
        "group": int(m.group("group")),
        "cameras": cameras,
        "camera_tag": m.group("cameras").upper(),
        "activity": m.group("activity"),
        "sheet": sheet_id_for(grade, int(m.group("class_id")), int(m.group("group"))),
        "test": PHASE_TO_TEST[m.group("phase")],
    }


@dataclass
class XianyangVideoMeta:
    path: Path
    date: str
    class_id: int
    grade: int
    group: int
    phase: str
    sheet: str
    test: str
    cameras: List[str]
    camera_tag: str
    activity: str
    parse_ok: bool = True
    folder_class_mismatch: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.date}/{self.sheet}/{self.phase}/{self.camera_tag}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["key"] = self.key
        return d


def scan_xianyang_videos(video_root: Path) -> List[XianyangVideoMeta]:
    """
    扫描 video/xianyang 下所有 mp4。
    只认 {date}/class{N}/ 结构；其它位置的视频记入 notes 但仍尽量解析文件名。
    """
    video_root = Path(video_root)
    out: List[XianyangVideoMeta] = []
    if not video_root.exists():
        return out

    for path in sorted(video_root.rglob("*.mp4")):
        rel_parts = path.relative_to(video_root).parts
        folder_date = rel_parts[0] if len(rel_parts) >= 2 else ""
        folder_class: Optional[int] = None
        if len(rel_parts) >= 3:
            cm = _CLASS_DIR_RE.match(rel_parts[1])
            if cm:
                folder_class = int(cm.group("n"))

        parsed = parse_xianyang_video_name(path.name)
        notes: List[str] = []
        if parsed is None:
            out.append(
                XianyangVideoMeta(
                    path=path,
                    date=folder_date or "unknown",
                    class_id=folder_class or -1,
                    grade=-1,
                    group=-1,
                    phase="unknown",
                    sheet="",
                    test="",
                    cameras=[],
                    camera_tag="",
                    activity="",
                    parse_ok=False,
                    notes=["filename_not_matched"],
                )
            )
            continue

        mismatch = folder_class is not None and folder_class != parsed["class_id"]
        if mismatch:
            notes.append(f"folder_class={folder_class} != name_class={parsed['class_id']}")
        if folder_date and folder_date != parsed["date"]:
            notes.append(f"folder_date={folder_date} != name_date={parsed['date']}")
        if folder_class is None:
            notes.append("not_under_classN_folder")

        out.append(
            XianyangVideoMeta(
                path=path,
                date=parsed["date"],
                class_id=parsed["class_id"],
                grade=parsed["grade"],
                group=parsed["group"],
                phase=parsed["phase"],
                sheet=parsed["sheet"],
                test=parsed["test"],
                cameras=list(parsed["cameras"]),
                camera_tag=parsed["camera_tag"],
                activity=parsed["activity"],
                parse_ok=True,
                folder_class_mismatch=mismatch,
                notes=notes,
            )
        )
    return out


def _normalize_header_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip().lower().replace(" ", "")


def _is_phase_header(row: Sequence[Any]) -> Optional[str]:
    if not row:
        return None
    phase = str(row[0]).strip() if pd.notna(row[0]) else ""
    if phase not in PHASES:
        return None
    cells = [_normalize_header_cell(x) for x in row[:6]]
    if any(c in {"start", "开始时间", "开始"} for c in cells[1:]):
        return phase
    if any(c == "speaker" for c in cells):
        return phase
    return None


def _find_col(header_row: Sequence[Any], aliases: Sequence[str]) -> Optional[int]:
    wanted = {a.lower().replace(" ", "") for a in aliases}
    for i, cell in enumerate(header_row):
        key = _normalize_header_cell(cell)
        if key in wanted:
            return i
    return None


def _parse_sheet_phases(df: pd.DataFrame, sheet: str) -> Dict[str, List[Dict[str, Any]]]:
    """把已读入的 sheet DataFrame 拆成前/中/后测说话段。"""
    out: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PHASES}
    current_phase: Optional[str] = None
    cols: Dict[str, Optional[int]] = {}

    for i in range(len(df)):
        row = df.iloc[i].tolist()
        phase_hdr = _is_phase_header(row)
        if phase_hdr is not None:
            current_phase = phase_hdr
            cols = {
                "start": _find_col(row, ["Start", "开始时间", "开始"]) or 1,
                "end": _find_col(row, ["End", "结束时间", "结束"]) or 2,
                "speaker": _find_col(row, ["Speaker", "说话人"]) or 3,
                "content": _find_col(row, ["Content", "内容", "转写"]) or 4,
                "note": _find_col(row, ["Note", "备注"]) or 5,
                "apt": _find_col(row, ["APT", "apt"]),
                "reg": _find_col(row, ["REG", "reg"]),
            }
            continue

        if current_phase is None or not cols:
            continue

        spk_i = cols["speaker"]
        if spk_i is None or spk_i >= len(row) or pd.isna(row[spk_i]):
            continue
        spk = str(row[spk_i]).strip().upper()
        if spk == "SPEAKER" or not re.fullmatch(r"S\d+|T\d*", spk):
            continue

        start = parse_time_to_seconds(row[cols["start"]] if cols["start"] < len(row) else None)
        end = parse_time_to_seconds(row[cols["end"]] if cols["end"] < len(row) else None)
        if start is None or end is None or end <= start:
            continue

        def _cell(key: str) -> Any:
            idx = cols.get(key)
            if idx is None or idx >= len(row):
                return None
            v = row[idx]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return v

        content = _cell("content")
        note = _cell("note")
        out[current_phase].append(
            {
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "speaker": spk,
                "content": None if content is None else str(content),
                "note": None if note is None else str(note),
                "apt": _cell("apt"),
                "reg": _cell("reg"),
                "phase": current_phase,
                "sheet": sheet,
                "test": PHASE_TO_TEST[current_phase],
            }
        )
    return out


def load_xianyang_segments(
    excel_path: Path,
    sheet: str,
    phase: str,
    speakers: Optional[Sequence[str]] = None,
    min_duration: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    从指定 sheet / 阶段加载说话段。
    返回字段: start, end, speaker, duration, content, note, apt, reg, phase, sheet
    """
    if phase not in PHASES:
        raise ValueError(f"phase 必须是 {PHASES} 之一，收到: {phase}")

    df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    segs = _parse_sheet_phases(df, sheet).get(phase, [])
    allow = {s.upper() for s in speakers} if speakers else None
    out = []
    for s in segs:
        if allow is not None and s["speaker"] not in allow:
            continue
        if s["duration"] < min_duration:
            continue
        out.append(s)
    return out


def list_excel_sheets(excel_path: Path) -> List[str]:
    return list(pd.ExcelFile(excel_path).sheet_names)


def load_xianyang_sheet_all_phases(
    excel_path: Path,
    sheet: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """一次读入某 sheet，按前/中/后测拆成三段列表。"""
    df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    return _parse_sheet_phases(df, sheet)


def build_xianyang_manifest(
    video_root: Path,
    excel_path: Path,
    count_segments: bool = True,
    student_only_count: bool = True,
) -> Dict[str, Any]:
    """扫描视频并与 Excel sheet 对齐，生成可落盘的 manifest。"""
    videos = scan_xianyang_videos(video_root)
    sheets = set(list_excel_sheets(excel_path)) if Path(excel_path).exists() else set()

    sessions: List[Dict[str, Any]] = []
    unmatched_videos: List[Dict[str, Any]] = []
    sheet_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def _segs_for(sheet: str, phase: str) -> List[Dict[str, Any]]:
        if sheet not in sheet_cache:
            sheet_cache[sheet] = load_xianyang_sheet_all_phases(excel_path, sheet)
        return sheet_cache[sheet].get(phase, [])

    for meta in videos:
        if not meta.parse_ok:
            unmatched_videos.append(meta.to_dict())
            continue

        item = meta.to_dict()
        item["sheet_exists"] = meta.sheet in sheets
        item["matched"] = bool(meta.sheet in sheets and not meta.folder_class_mismatch)

        if count_segments and item["sheet_exists"]:
            try:
                segs = _segs_for(meta.sheet, meta.phase)
            except Exception as exc:  # noqa: BLE001
                item["segment_error"] = str(exc)
                segs = []
            item["n_segments"] = len(segs)
            if student_only_count:
                item["n_student_segments"] = sum(1 for s in segs if s["speaker"].startswith("S"))
            else:
                item["n_student_segments"] = item["n_segments"]
        else:
            item["n_segments"] = None
            item["n_student_segments"] = None

        sessions.append(item)

    covered = {(s["sheet"], s["phase"]) for s in sessions if s.get("sheet_exists")}
    sheets_without_video = []
    if count_segments:
        for sh in sorted(sheets):
            try:
                by_phase = sheet_cache.get(sh) or load_xianyang_sheet_all_phases(excel_path, sh)
                sheet_cache[sh] = by_phase
            except Exception:  # noqa: BLE001
                continue
            for ph in PHASES:
                if (sh, ph) in covered:
                    continue
                n = len(by_phase.get(ph, []))
                if n > 0:
                    sheets_without_video.append({"sheet": sh, "phase": ph, "n_segments": n})

    return {
        "video_root": str(Path(video_root)),
        "excel": str(Path(excel_path)),
        "n_videos": len(videos),
        "n_matched": sum(1 for s in sessions if s.get("matched")),
        "n_parse_failed": len(unmatched_videos),
        "sessions": sessions,
        "unmatched_videos": unmatched_videos,
        "sheets_without_video": sheets_without_video,
        "excel_sheets": sorted(sheets),
    }


def iter_matched_sessions(
    manifest: Dict[str, Any],
    *,
    date: Optional[str] = None,
    class_id: Optional[int] = None,
    phase: Optional[str] = None,
    require_matched: bool = True,
) -> Iterable[Dict[str, Any]]:
    for s in manifest.get("sessions", []):
        if require_matched and not s.get("matched"):
            continue
        if date is not None and s.get("date") != date:
            continue
        if class_id is not None and s.get("class_id") != class_id:
            continue
        if phase is not None and s.get("phase") != phase:
            continue
        yield s


def load_position_map_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    读取人工位置标注 JSON，按 video_name 索引。
    JSON 由 scripts/make_position_map_templates.py 生成。
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for item in doc.get("videos") or []:
        name = item.get("video_name")
        if name:
            out[str(name)] = item
    return out


def left_to_right_to_ref_x(left_to_right: Sequence[str]) -> Dict[str, float]:
    """把从左到右的说话人列表转为归一化 x 参考位置（供槽位对齐）。"""
    n = len(left_to_right)
    if n <= 0:
        return {}
    if n == 1:
        return {str(left_to_right[0]).upper(): 0.5}
    return {
        str(spk).upper(): float(i) / float(n - 1) * 0.7 + 0.15
        for i, spk in enumerate(left_to_right)
    }


def resolve_video_position_map(
    video_name: str,
    map_paths: Sequence[Path],
    *,
    require_confirmed: bool = True,
) -> Optional[Dict[str, Any]]:
    """在若干 position_map JSON 中查找该视频的人工标注。"""
    for path in map_paths:
        p = Path(path)
        if not p.exists():
            continue
        by_name = load_position_map_file(p)
        item = by_name.get(video_name)
        if not item:
            continue
        if require_confirmed and not item.get("confirmed"):
            return None
        ltr = item.get("left_to_right") or []
        if not ltr:
            return None
        return {
            "left_to_right": [str(s).upper() for s in ltr],
            "speaker_ref_x": left_to_right_to_ref_x(ltr),
            "confirmed": bool(item.get("confirmed")),
            "source": str(p),
            "raw": item,
        }
    return None
