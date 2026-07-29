"""从人脸 bbox 裁剪嘴部 ROI（Wav2Lip / MTD：96×96 脸的下半 48×96）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from csd.perception.face_tracker import FaceTrack


def face_to_mouth_rgb(frame_bgr: np.ndarray, bbox: np.ndarray, out_w: int = 96, out_h: int = 96) -> np.ndarray:
    """返回嘴部 RGB uint8，形状 (48, 96, 3)。"""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_h // 2, out_w, 3), dtype=np.uint8)
    face = frame_bgr[y1:y2, x1:x2]
    face = cv2.resize(face, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mouth = face[out_h // 2 :, :, :]  # 48x96
    return cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)


def stack_mouth_window(mouths_rgb: Sequence[np.ndarray]) -> np.ndarray:
    """5 帧嘴部 RGB → MTD 输入 [15, 48, 96] float32 in [0,1]。"""
    if len(mouths_rgb) != 5:
        raise ValueError(f"需要 5 帧嘴部图，得到 {len(mouths_rgb)}")
    # Wav2Lip/SyncNet: concat along channel → 15x48x96
    arr = np.concatenate([m.transpose(2, 0, 1) for m in mouths_rgb], axis=0)  # 15,48,96
    return (arr.astype(np.float32) / 255.0)


def interpolate_bbox(track: FaceTrack, frame_idx: int, max_gap: int = 18) -> Optional[np.ndarray]:
    """轨迹内 bbox 插值；找不到则 None。"""
    if frame_idx in track.detections:
        return track.detections[frame_idx].bbox.copy()
    keys = sorted(track.detections.keys())
    if not keys:
        return None
    if frame_idx < keys[0] or frame_idx > keys[-1]:
        return None
    left = None
    right = None
    for k in keys:
        if k < frame_idx:
            left = k
        elif k > frame_idx:
            right = k
            break
    if left is None or right is None:
        return None
    if right - left > max_gap:
        return None
    a = track.detections[left].bbox.astype(np.float32)
    b = track.detections[right].bbox.astype(np.float32)
    t = (frame_idx - left) / max(right - left, 1)
    return (a * (1.0 - t) + b * t).astype(np.float32)


def load_mouth_tensor_for_window(
    video_path: str,
    track: FaceTrack,
    frame_indices: Sequence[int],
    max_gap: int = 18,
) -> Optional[np.ndarray]:
    """读取指定帧，裁嘴并堆成 [15,48,96]。缺帧过多则返回 None。"""
    if len(frame_indices) != 5:
        raise ValueError("frame_indices 必须长度为 5")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    mouths: List[np.ndarray] = []
    try:
        for fi in frame_indices:
            bbox = interpolate_bbox(track, int(fi), max_gap=max_gap)
            if bbox is None:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            mouths.append(face_to_mouth_rgb(frame, bbox))
    finally:
        cap.release()
    return stack_mouth_window(mouths)


def five_frame_indices(center_f: int, fps: float, target_fps: float = 25.0) -> List[int]:
    """以 center 为中心取等效 25fps 的 5 帧（映射回原视频帧号）。"""
    # 在 25fps 时间轴上取 center-2 ... center+2
    step = fps / target_fps
    return [max(0, int(round(center_f + (k - 2) * step))) for k in range(5)]
