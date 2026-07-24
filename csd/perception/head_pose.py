"""头部姿态与嘴部特征提取（MediaPipe Face Mesh + PnP）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack
from csd.core.utils import time_to_frame

logger = logging.getLogger(__name__)

# PnP 用 6 点 3D 人脸模型（毫米，与 OpenCV 常用配置一致）
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)
_LM_IDX = (1, 152, 33, 263, 61, 291)
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291


@dataclass
class HeadPoseFrame:
    """单帧单人脸头部特征。"""

    yaw: float
    pitch: float
    roll: float
    mouth_opening: float
    visibility: float
    side_face_weight: float


@dataclass
class SlotVisualTimeline:
    """每个画面槽位随时间变化的视觉特征。"""

    slot_to_speaker: Dict[int, str]
    speaker_to_slot: Dict[str, int]
    slot_positions: Dict[int, Tuple[float, float]]
    # slot_id -> frame_idx -> HeadPoseFrame
    frames: Dict[int, Dict[int, HeadPoseFrame]]


class HeadPoseAnalyzer:
    """基于 Face Mesh 估计头部欧拉角与嘴部开合度。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self._face_mesh = None
        self._side_yaw_deg = getattr(config, "side_face_yaw_deg", 35.0)

    def _load_model(self) -> None:
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("请安装 mediapipe: pip install 'mediapipe==0.10.14'") from e

        # 新版 mediapipe（尤其部分 Linux/Py3.12 wheel）已移除 mp.solutions
        if not hasattr(mp, "solutions"):
            ver = getattr(mp, "__version__", "unknown")
            raise ImportError(
                f"当前 mediapipe=={ver} 没有 mp.solutions（Face Mesh 旧 API）。\n"
                f"本项目需要经典 Face Mesh。请在实例执行:\n"
                f"  pip uninstall -y mediapipe\n"
                f"  pip install 'mediapipe==0.10.14'\n"
                f"若仍失败，试: pip install 'mediapipe==0.10.13'"
            )

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=self.config.head_mesh_static_roi,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.head_mesh_det_conf,
            min_tracking_confidence=self.config.head_mesh_track_conf,
        )
        logger.info(
            "HeadPose Face Mesh 已加载 (static_roi=%s, det_conf=%.2f, mediapipe=%s)",
            self.config.head_mesh_static_roi,
            self.config.head_mesh_det_conf,
            getattr(mp, "__version__", "?"),
        )

    @staticmethod
    def _mar(landmarks, w: int, h: int) -> float:
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        top, bottom = pt(MOUTH_TOP), pt(MOUTH_BOTTOM)
        left, right = pt(MOUTH_LEFT), pt(MOUTH_RIGHT)
        horiz = np.linalg.norm(left - right)
        if horiz < 1e-6:
            return 0.0
        return float(np.linalg.norm(top - bottom) / horiz)

    @staticmethod
    def _mean_visibility(landmarks, indices: Tuple[int, ...]) -> float:
        """
        Face Mesh landmark.visibility 在 MediaPipe 中常为 0（未填充）。
        仅当存在正值时使用；否则视为 ROI 内检测成功，返回 1.0。
        """
        vals = []
        for idx in indices:
            lm = landmarks[idx]
            if hasattr(lm, "visibility"):
                v = float(lm.visibility)
                if v > 1e-6:
                    vals.append(v)
        if not vals:
            return 1.0
        return float(np.mean(vals))

    @staticmethod
    def interpolate_bbox_at_frame(
        track: FaceTrack,
        frame_idx: int,
        max_gap: int = 18,
    ) -> Optional[np.ndarray]:
        """在轨迹检测帧之间线性插值 bbox。"""
        if frame_idx in track.detections:
            return track.detections[frame_idx].bbox.astype(np.float64).copy()

        keys = sorted(track.detections.keys())
        if not keys:
            return None

        if frame_idx < keys[0]:
            gap = keys[0] - frame_idx
            return track.detections[keys[0]].bbox.astype(np.float64).copy() if gap <= max_gap else None
        if frame_idx > keys[-1]:
            gap = frame_idx - keys[-1]
            return track.detections[keys[-1]].bbox.astype(np.float64).copy() if gap <= max_gap else None

        lo = max(k for k in keys if k <= frame_idx)
        hi = min(k for k in keys if k >= frame_idx)
        if lo == hi:
            return track.detections[lo].bbox.astype(np.float64).copy()

        t = (frame_idx - lo) / max(hi - lo, 1)
        b0 = track.detections[lo].bbox.astype(np.float64)
        b1 = track.detections[hi].bbox.astype(np.float64)
        return b0 * (1.0 - t) + b1 * t

    @staticmethod
    def _speech_target_frames(
        speech_intervals: Sequence[Tuple[float, float]],
        fps: float,
        frame_skip: int,
        pad_sec: float,
    ) -> set:
        targets: set = set()
        for start, end in speech_intervals:
            s = max(0, int((start - pad_sec) * fps))
            e = int((end + pad_sec) * fps)
            step = max(1, frame_skip)
            for f in range(s, e + 1, step):
                targets.add(f)
        return targets

    @staticmethod
    def _rotation_to_euler(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[float, float, float]:
        rot_mat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rot_mat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch = float(euler[0])
        yaw = float(euler[1])
        roll = float(euler[2])
        return yaw, pitch, roll

    def _side_face_weight(self, yaw: float) -> float:
        return float(np.clip(1.0 - abs(yaw) / self._side_yaw_deg, 0.1, 1.0))

    def _pose_quality_score(self, pose: HeadPoseFrame) -> float:
        return float(pose.visibility * pose.side_face_weight)

    def analyze_roi(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[HeadPoseFrame]:
        """在 bbox ROI 内估计头部姿态。"""
        self._load_model()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        if min(rh, rw) < 24:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        image_points = np.array(
            [(lm[i].x * rw, lm[i].y * rh) for i in _LM_IDX],
            dtype=np.float64,
        )
        focal = rw
        center = (rw / 2.0, rh / 2.0)
        camera_matrix = np.array(
            [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        yaw, pitch, roll = self._rotation_to_euler(rvec, tvec)
        mar = self._mar(lm, rw, rh)
        vis = self._mean_visibility(lm, _LM_IDX)
        return HeadPoseFrame(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mouth_opening=mar,
            visibility=vis,
            side_face_weight=self._side_face_weight(yaw),
        )

    def _process_frame_slots(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracks: Dict[int, FaceTrack],
        track_to_slot: Dict[int, int],
        slot_frames: Dict[int, Dict[int, HeadPoseFrame]],
        use_interp: bool,
    ) -> None:
        slot_best: Dict[int, Tuple[float, HeadPoseFrame]] = {}
        max_gap = self.config.bbox_interp_max_gap

        for tid, track in tracks.items():
            if tid not in track_to_slot:
                continue
            if use_interp:
                bbox = self.interpolate_bbox_at_frame(track, frame_idx, max_gap=max_gap)
            else:
                if frame_idx not in track.detections:
                    continue
                bbox = track.detections[frame_idx].bbox

            if bbox is None:
                continue

            slot_id = track_to_slot[tid]
            pose = self.analyze_roi(frame, bbox)
            if pose is None:
                continue
            score = self._pose_quality_score(pose)
            prev = slot_best.get(slot_id)
            if prev is None or score > prev[0]:
                slot_best[slot_id] = (score, pose)

        for slot_id, (_, pose) in slot_best.items():
            prev_pose = slot_frames[slot_id].get(frame_idx)
            if prev_pose is None or self._pose_quality_score(pose) >= self._pose_quality_score(prev_pose):
                slot_frames[slot_id][frame_idx] = pose

    def build_slot_timeline(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        slot_to_tracks: Dict[int, List[int]],
        slot_to_speaker: Dict[int, str],
        slot_positions: Dict[int, Tuple[float, float]],
        fps: float,
        speech_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> SlotVisualTimeline:
        """遍历视频收集头部/嘴部时序；VAD 段内加密采样并插值 bbox。"""
        speaker_to_slot = {v: k for k, v in slot_to_speaker.items()}
        track_to_slot: Dict[int, int] = {}
        for slot_id, tids in slot_to_tracks.items():
            for tid in tids:
                track_to_slot[tid] = slot_id

        slot_frames: Dict[int, Dict[int, HeadPoseFrame]] = {sid: {} for sid in slot_to_speaker}

        speech_targets: set = set()
        if speech_intervals:
            speech_targets = self._speech_target_frames(
                speech_intervals,
                fps,
                self.config.speech_pose_frame_skip,
                self.config.speech_pose_pad_sec,
            )
            logger.info(
                "VAD 段内加密 pose 目标帧: %d (skip=%d, pad=%.2fs)",
                len(speech_targets),
                self.config.speech_pose_frame_skip,
                self.config.speech_pose_pad_sec,
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        frame_idx = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sparse_hits = dense_hits = 0
        logger.info("提取头部/嘴部特征: %s", video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in speech_targets:
                before = sum(len(v) for v in slot_frames.values())
                self._process_frame_slots(
                    frame, frame_idx, tracks, track_to_slot, slot_frames, use_interp=True
                )
                after = sum(len(v) for v in slot_frames.values())
                dense_hits += after - before
            elif frame_idx % self.config.frame_skip == 0:
                before = sum(len(v) for v in slot_frames.values())
                self._process_frame_slots(
                    frame, frame_idx, tracks, track_to_slot, slot_frames, use_interp=False
                )
                after = sum(len(v) for v in slot_frames.values())
                sparse_hits += after - before

            frame_idx += 1
            if frame_idx % 500 == 0 and total:
                logger.info("  头部特征进度: %d / %d (%.1f%%)", frame_idx, total, 100.0 * frame_idx / total)

        cap.release()
        logger.info(
            "头部特征完成: %s (sparse+%d dense+%d)",
            ", ".join(f"slot{sid}={len(slot_frames[sid])}帧" for sid in sorted(slot_frames)),
            sparse_hits,
            dense_hits,
        )
        return SlotVisualTimeline(
            slot_to_speaker=slot_to_speaker,
            speaker_to_slot=speaker_to_slot,
            slot_positions=slot_positions,
            frames=slot_frames,
        )

    @staticmethod
    def aggregate_segment_mouth(
        timeline: SlotVisualTimeline,
        speaker: str,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> Tuple[float, float]:
        """返回 (嘴动活跃度, 平均侧脸降权)。"""
        slot_id = timeline.speaker_to_slot.get(speaker)
        if slot_id is None:
            return 0.0, 0.0
        start_f = time_to_frame(start_time, fps)
        end_f = time_to_frame(end_time, fps)
        mars, weights = [], []
        for f, pose in timeline.frames.get(slot_id, {}).items():
            if start_f <= f <= end_f:
                mars.append(pose.mouth_opening)
                weights.append(pose.side_face_weight)
        if not mars:
            return 0.0, 0.0
        arr = np.array(mars)
        if len(arr) < 2:
            activity = float(arr[0])
        else:
            activity = float(np.var(arr) + np.mean(np.abs(np.diff(arr))))
        return activity, float(np.mean(weights))

    @staticmethod
    def segment_yaw_series(
        timeline: SlotVisualTimeline,
        speaker: str,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> List[float]:
        slot_id = timeline.speaker_to_slot.get(speaker)
        if slot_id is None:
            return []
        start_f = time_to_frame(start_time, fps)
        end_f = time_to_frame(end_time, fps)
        yaws = []
        for f, pose in sorted(timeline.frames.get(slot_id, {}).items()):
            if start_f <= f <= end_f:
                yaws.append(pose.yaw)
        return yaws
