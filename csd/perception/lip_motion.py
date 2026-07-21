"""模块4：嘴部运动检测（基于 MediaPipe Face Mesh 的 MAR 指标）。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack
from csd.core.utils import time_to_frame

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh 嘴部关键点索引
# 参考: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
MOUTH_OUTER = [61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146]
# 简化 MAR：用上唇/下唇/嘴角
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


class LipMotionAnalyzer:
    """计算嘴部纵横比 (MAR) 及其时序活跃度。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self._face_mesh = None

    def _load_model(self) -> None:
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("请安装 mediapipe: pip install mediapipe") from e

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=getattr(self.config, "head_mesh_static_roi", True),
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=getattr(self.config, "head_mesh_det_conf", 0.35),
            min_tracking_confidence=getattr(self.config, "head_mesh_track_conf", 0.35),
        )
        logger.info("MediaPipe Face Mesh 已加载")

    @staticmethod
    def _compute_mar(landmarks, frame_w: int, frame_h: int) -> float:
        """根据关键点计算嘴部纵横比 MAR。"""
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * frame_w, lm.y * frame_h])

        try:
            top = pt(MOUTH_TOP)
            bottom = pt(MOUTH_BOTTOM)
            left = pt(MOUTH_LEFT)
            right = pt(MOUTH_RIGHT)
        except (IndexError, AttributeError):
            return 0.0

        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / horizontal)

    def compute_mar_in_roi(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[float]:
        """在 bbox ROI 内用 Face Mesh 计算 MAR。"""
        self._load_model()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        pad = int(0.2 * max(x2 - x1, y2 - y1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        rh, rw = roi.shape[:2]
        return self._compute_mar(lm, rw, rh)

    @staticmethod
    def _activity_from_mar_values(mar_values: List[float]) -> float:
        if len(mar_values) < 2:
            return 0.0
        mar_arr = np.array(mar_values)
        return float(np.var(mar_arr) + np.mean(np.abs(np.diff(mar_arr))))

    def compute_all_activities(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        identity_to_tracks: Dict[int, List[int]],
        start_time: float,
        end_time: float,
        fps: float,
    ) -> Dict[int, float]:
        """为每个 identity 计算嘴部运动活跃度（单次遍历视频）。"""
        self._load_model()
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)

        # track_id -> frame_idx -> bbox
        track_frames: Dict[int, Dict[int, np.ndarray]] = {}
        for tid, track in tracks.items():
            frames = {
                f: track.detections[f].bbox
                for f in track.detections
                if start_frame <= f <= end_frame
            }
            if frames:
                track_frames[tid] = frames

        if not track_frames:
            return {iid: 0.0 for iid in identity_to_tracks}

        mar_by_track: Dict[int, List[float]] = {tid: [] for tid in track_frames}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {iid: 0.0 for iid in identity_to_tracks}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            for tid, frame_map in track_frames.items():
                if frame_idx not in frame_map:
                    continue
                mar = self.compute_mar_in_roi(frame, frame_map[frame_idx])
                if mar is not None:
                    mar_by_track[tid].append(mar)
        cap.release()

        activities: Dict[int, float] = {}
        for identity_id, track_ids in identity_to_tracks.items():
            max_activity = 0.0
            for tid in track_ids:
                max_activity = max(max_activity, self._activity_from_mar_values(mar_by_track.get(tid, [])))
            activities[identity_id] = max_activity
        return activities
