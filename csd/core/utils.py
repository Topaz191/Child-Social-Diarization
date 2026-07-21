"""通用工具函数。"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个 bbox [x1,y1,x2,y2] 的 IoU。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def bbox_center_norm(box: np.ndarray, frame_w: int, frame_h: int) -> Tuple[float, float]:
    cx = (box[0] + box[2]) / 2.0 / frame_w
    cy = (box[1] + box[3]) / 2.0 / frame_h
    return float(cx), float(cy)


def time_to_frame(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def frame_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / fps


def extract_audio_ffmpeg(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Optional[Path]:
    """使用 ffmpeg 从视频提取单声道 16kHz WAV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            logger.error("ffmpeg 失败: %s", result.stderr[-500:])
            return None
        return output_path
    except FileNotFoundError:
        logger.error("未找到 ffmpeg，请安装并加入 PATH")
        return None


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("已保存 JSON: %s", path)


def load_wav_segment(
    wav_path: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
):
    """加载音频片段，返回 (waveform_tensor, sample_rate)。"""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        sr = sample_rate

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[1], end_sample)
    if end_sample <= start_sample:
        return None, sr
    return waveform[:, start_sample:end_sample], sr


def merge_short_segments(
    segments: List[Tuple[float, float]],
    min_duration: float,
    gap_merge: float = 0.2,
) -> List[Tuple[float, float]]:
    """合并过短的相邻语音段。"""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: x[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] < gap_merge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_duration]


def merge_intervals(
    segments: List[Tuple[float, float]],
    gap_merge: float = 0.2,
) -> List[Tuple[float, float]]:
    """合并重叠或相邻的时间区间。"""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: x[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_merge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def union_intervals(
    *segment_lists: List[Tuple[float, float]],
    gap_merge: float = 0.2,
) -> List[Tuple[float, float]]:
    """多个区间列表取并集后合并。"""
    combined: List[Tuple[float, float]] = []
    for segs in segment_lists:
        combined.extend(segs)
    return merge_intervals(combined, gap_merge=gap_merge)
