"""外置配套音频：与视频同时间轴（t=0 对齐）。

约定：咸阳「同一日期 + 同一班 + 同一组」只有一路配套音频
（不区分机位 S1S2/S2S3；多机位视频共用这一路）。

目录推荐：
  audio/xianyang/{MMDD}/class{N}/g{G}.wav

同门 merged 目录也可直接放在：
  audio/merged audio/202507-小学-咸阳/{MMDD}-{前测|中测|后测}/*.wav
文件名需能解析出 date/class/group（与现有 audio 命名一致即可）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

AUDIO_EXTS: Sequence[str] = (".wav", ".WAV", ".flac", ".FLAC", ".m4a", ".M4A", ".mp3", ".MP3", ".aac", ".AAC")

# 0701-前测-五年级1班-第1组-讨论-c.wav  /  0701-前测-5年级1班-第1组.wav
_AUDIO_NAME_RE = re.compile(
    r"^(?P<date>\d{3,4})-"
    r"(?:(?P<phase>前测|中测|后测)-)?"
    r"(?:(?P<grade_raw>\d+|[一二三四五六七八九十]+)年级)?"
    r"(?P<class_id>\d+)班-"
    r"第(?P<group>\d+)组",
    re.IGNORECASE,
)
# 扁平短名：0701_class1_g1 / g1（仅在 class 目录内）
_FLAT_KEY_RE = re.compile(
    r"^(?:(?P<date>\d{3,4})[_-])?(?:class(?P<class_id>\d+)[_-])?g(?P<group>\d+)$",
    re.IGNORECASE,
)
_CLASS_DIR_RE = re.compile(r"^class(?P<n>\d+)$", re.IGNORECASE)
_DATE_DIR_RE = re.compile(r"^(?P<date>\d{3,4})")


def default_xianyang_audio_root(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "audio" / "xianyang"


def project_audio_root(project_root: Optional[Path] = None) -> Path:
    """仓库根下的 audio/（可含 xianyang、merged audio 等子树）。"""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "audio"


def companion_search_roots(audio_root: Optional[Path] = None) -> List[Path]:
    """
    配套音频搜索根（按优先级）：
      audio/xianyang/
      audio/merged audio/   （同门 merged 目录，含空格）
      audio/merged_audio/ / mergedaudio/
      audio/                （整树兜底）
    """
    if audio_root is not None:
        ar = Path(audio_root)
        if ar.name.lower() == "xianyang" and ar.parent.name.lower() == "audio":
            base = ar.parent
            roots = [
                ar,
                base / "merged audio",
                base / "merged_audio",
                base / "mergedaudio",
                base,
            ]
        elif ar.name.lower() == "audio":
            roots = [
                ar / "xianyang",
                ar / "merged audio",
                ar / "merged_audio",
                ar / "mergedaudio",
                ar,
            ]
        else:
            roots = [ar]
    else:
        base = project_audio_root()
        roots = [
            base / "xianyang",
            base / "merged audio",
            base / "merged_audio",
            base / "mergedaudio",
            base,
        ]
    out: List[Path] = []
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve()) if r.exists() else str(r)
        except OSError:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        if r.exists():
            out.append(r)
    return out


def video_rel_under_xianyang(video_path: Path, video_root: Optional[Path] = None) -> Optional[Path]:
    video_path = Path(video_path).resolve()
    if video_root is not None:
        try:
            return video_path.relative_to(Path(video_root).resolve())
        except ValueError:
            pass
    parts = video_path.parts
    for i, p in enumerate(parts):
        if p.lower() == "xianyang" and i + 1 < len(parts):
            return Path(*parts[i + 1 :])
    return None


def parse_session_key_from_name(name: str) -> Optional[Dict[str, Any]]:
    """从视频或音频文件名解析 date / class_id / group（及可选 phase）。"""
    stem = Path(name).stem
    m = _AUDIO_NAME_RE.match(stem)
    if m:
        return {
            "date": m.group("date"),
            "class_id": int(m.group("class_id")),
            "group": int(m.group("group")),
            "phase": m.group("phase"),
        }
    try:
        from csd.data.xianyang import parse_xianyang_video_name

        parsed = parse_xianyang_video_name(name if name.lower().endswith(".mp4") else f"{stem}.mp4")
        if parsed:
            return {
                "date": parsed["date"],
                "class_id": int(parsed["class_id"]),
                "group": int(parsed["group"]),
                "phase": parsed.get("phase"),
            }
    except Exception:
        pass
    m2 = _FLAT_KEY_RE.match(stem)
    if m2 and m2.group("group"):
        out = {
            "date": m2.group("date"),
            "class_id": int(m2.group("class_id")) if m2.group("class_id") else None,
            "group": int(m2.group("group")),
            "phase": None,
        }
        return out
    return None


def session_key(date: str, class_id: int, group: int) -> Tuple[str, int, int]:
    return (str(date), int(class_id), int(group))


def canonical_audio_path(
    date: str,
    class_id: int,
    group: int,
    *,
    audio_root: Optional[Path] = None,
    ext: str = ".wav",
) -> Path:
    """推荐落盘路径：audio/xianyang/{date}/class{N}/g{G}.wav"""
    root = Path(audio_root) if audio_root is not None else default_xianyang_audio_root()
    return root / str(date) / f"class{int(class_id)}" / f"g{int(group)}{ext}"


def _phase_from_path(path: Path) -> Optional[str]:
    text = str(path).replace("\\", "/")
    for ph in ("前测", "中测", "后测"):
        if ph in text:
            return ph
    return None


def _pick_best_audio(
    cands: List[Path],
    *,
    prefer_phase: Optional[str] = None,
) -> Optional[Path]:
    if not cands:
        return None
    scored = []
    for p in cands:
        name_len = len(p.name)
        phase_hit = 0
        if prefer_phase and _phase_from_path(p) == prefer_phase:
            phase_hit = -10
        scored.append((phase_hit, name_len, str(p)))
    scored.sort()
    return Path(scored[0][2])


def _key_from_audio_file(path: Path, audio_root: Path) -> Optional[Tuple[str, int, int]]:
    """结合文件名与目录推断 (date, class_id, group)。"""
    info = parse_session_key_from_name(path.name)
    date = info.get("date") if info else None
    class_id = info.get("class_id") if info else None
    group = info.get("group") if info else None

    try:
        rel = path.resolve().relative_to(Path(audio_root).resolve())
        parts = rel.parts
        # .../0701/class1/xxx.wav
        if len(parts) >= 3:
            dm = _DATE_DIR_RE.match(parts[0])
            if dm:
                date = date or dm.group("date")
            cm = _CLASS_DIR_RE.match(parts[1])
            if cm:
                class_id = class_id if class_id is not None else int(cm.group("n"))
        # .../202507-小学-咸阳/0701-前测/xxx.wav → date from folder or filename
        for part in parts:
            dm = _DATE_DIR_RE.match(part)
            if dm and date is None:
                date = dm.group("date")
                break
    except ValueError:
        pass

    if date is None or class_id is None or group is None:
        return None
    return session_key(date, class_id, group)


def index_companion_audios(audio_root: Optional[Path] = None) -> Dict[Tuple[str, int, int], Path]:
    """扫描一个音频根目录，按 (date, class, group) 建索引。"""
    root = Path(audio_root) if audio_root is not None else default_xianyang_audio_root()
    index: Dict[Tuple[str, int, int], Path] = {}
    if not root.exists():
        return index
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in AUDIO_EXTS:
            continue
        key = _key_from_audio_file(p, root)
        if key is None:
            continue
        prev = index.get(key)
        if prev is None or len(p.name) < len(prev.name):
            index[key] = p
    return index


def resolve_companion_audio(
    video_path: Path,
    *,
    audio_root: Optional[Path] = None,
    video_root: Optional[Path] = None,
    explicit: Optional[Path] = None,
    extra_dirs: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """
    解析外置配套音频。时间轴：音频 t=0 = 视频第 0 帧。

    优先级：
      1. explicit（CLI --audio）
      2. audio/xianyang 规范路径 g{N}.wav
      3. audio/xianyang、audio/merged audio/... 等搜索根中按 date+class+group 匹配
      4. 视频旁同名音频
      5. extra_dirs
    """
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        logger.warning("指定音频不存在: %s", p)

    video_path = Path(video_path)
    primary = Path(audio_root) if audio_root is not None else default_xianyang_audio_root()

    info = parse_session_key_from_name(video_path.name)
    if info is None:
        rel = video_rel_under_xianyang(video_path, video_root=video_root)
        if rel is not None and len(rel.parts) >= 2:
            cm = _CLASS_DIR_RE.match(rel.parts[1])
            _ = cm  # 文件名解析失败时无法可靠匹配
            pass

    if info is not None and info.get("date") is not None and info.get("class_id") is not None:
        key = session_key(info["date"], info["class_id"], info["group"])
        prefer_phase = info.get("phase")

        # 1) 规范路径（仅 xianyang 根）
        xy_root = primary if primary.name.lower() == "xianyang" else default_xianyang_audio_root()
        for ext in AUDIO_EXTS:
            cand = canonical_audio_path(
                info["date"], info["class_id"], info["group"], audio_root=xy_root, ext=ext
            )
            if cand.exists():
                logger.info("使用配套音频(规范路径): %s", cand)
                return cand.resolve()

        # 2) 多根目录搜索
        cands: List[Path] = []
        roots = companion_search_roots(primary)
        if extra_dirs:
            for d in extra_dirs:
                dp = Path(d)
                if dp.exists() and dp not in roots:
                    roots.append(dp)
        for root in roots:
            for p in root.rglob("*"):
                if not p.is_file() or p.suffix not in AUDIO_EXTS:
                    continue
                k = _key_from_audio_file(p, root)
                if k == key:
                    cands.append(p)
        best = _pick_best_audio(cands, prefer_phase=prefer_phase)
        if best is not None:
            logger.info("使用配套音频(date/class/group=%s phase=%s): %s", key, prefer_phase, best)
            return best.resolve()

    # 回退：视频旁同名
    for ext in AUDIO_EXTS:
        side = video_path.with_suffix(ext)
        if side.exists():
            logger.info("使用视频旁音频: %s", side)
            return side.resolve()
    return None
