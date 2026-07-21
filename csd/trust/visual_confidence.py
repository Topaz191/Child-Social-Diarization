"""视觉可信度估计：头部稳定、遮挡、侧脸、嘴动检测质量。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from csd.constants import SPEAKERS
from csd.perception.head_pose import HeadPoseFrame, SlotVisualTimeline
from csd.perception.vad_speaker import SpeechSegment


class VisualConfidenceEstimator:
    """
    模块1.2：估计窗口内视觉行为特征可靠度 visual_conf ∈ [0,1]。

    子指标：姿态稳定、低遮挡、侧脸惩罚、嘴动信号强度。
    """

    def __init__(
        self,
        window_sec: float = 4.0,
        target_yaw_std_deg: float = 12.0,
        weights: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
    ):
        self.window_sec = window_sec
        self.target_yaw_std_deg = target_yaw_std_deg
        self.weights = weights

    def _segment_poses(
        self,
        timeline: SlotVisualTimeline,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> List[HeadPoseFrame]:
        from csd.core.utils import time_to_frame

        start_f = time_to_frame(start_time, fps)
        end_f = time_to_frame(end_time, fps)
        poses: List[HeadPoseFrame] = []
        for slot_id in timeline.slot_to_speaker:
            for f, pose in timeline.frames.get(slot_id, {}).items():
                if start_f <= f <= end_f:
                    poses.append(pose)
        return poses

    def _pose_stability(self, poses: Sequence[HeadPoseFrame]) -> float:
        if len(poses) < 2:
            return 0.3 if poses else 0.0
        yaws = np.array([p.yaw for p in poses], dtype=np.float64)
        std = float(np.std(yaws))
        return float(np.clip(1.0 - std / self.target_yaw_std_deg, 0.0, 1.0))

    @staticmethod
    def _occlusion_score(poses: Sequence[HeadPoseFrame]) -> float:
        if not poses:
            return 0.0
        vis = np.array([p.visibility for p in poses], dtype=np.float64)
        low_ratio = float(np.mean(vis < 0.5))
        return float(np.clip(1.0 - low_ratio, 0.0, 1.0))

    @staticmethod
    def _frontal_score(poses: Sequence[HeadPoseFrame]) -> float:
        if not poses:
            return 0.0
        side_w = np.array([p.side_face_weight for p in poses], dtype=np.float64)
        return float(np.clip(np.mean(side_w), 0.0, 1.0))

    @staticmethod
    def _lip_signal_score(
        timeline: SlotVisualTimeline,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> float:
        from csd.perception.head_pose import HeadPoseAnalyzer

        activities = []
        for spk in SPEAKERS:
            act, _ = HeadPoseAnalyzer.aggregate_segment_mouth(
                timeline, spk, start_time, end_time, fps
            )
            activities.append(act)
        if not activities:
            return 0.0
        peak = max(activities)
        return float(np.clip(peak / 0.015, 0.0, 1.0))

    def confidence_for_segment(
        self,
        timeline: SlotVisualTimeline,
        segment: SpeechSegment,
        fps: float,
    ) -> float:
        t_mid = 0.5 * (segment.start_time + segment.end_time)
        half = self.window_sec / 2.0
        w_start, w_end = t_mid - half, t_mid + half

        poses = self._segment_poses(timeline, w_start, w_end, fps)
        stability = self._pose_stability(poses)
        occlusion = self._occlusion_score(poses)
        frontal = self._frontal_score(poses)
        lip_sig = self._lip_signal_score(timeline, w_start, w_end, fps)

        w_s, w_o, w_f, w_l = self.weights
        conf = w_s * stability + w_o * occlusion + w_f * frontal + w_l * lip_sig
        return float(np.clip(conf, 0.0, 1.0))

    def compute_all(
        self,
        timeline: SlotVisualTimeline,
        segments: Sequence[SpeechSegment],
        fps: float,
    ) -> List[float]:
        return [self.confidence_for_segment(timeline, seg, fps) for seg in segments]
