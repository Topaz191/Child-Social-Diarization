"""视觉可信度估计：头部稳定、遮挡、侧脸、嘴动检测质量。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from csd.constants import SPEAKERS
from csd.perception.head_pose import HeadPoseFrame, SlotVisualTimeline
from csd.perception.vad_speaker import SpeechSegment


class VisualConfidenceEstimator:
    """
    模块1.2：估计窗口内视觉行为特征可靠度 visual_conf ∈ [0,1]。

    子指标：姿态稳定、低遮挡、侧脸惩罚、嘴动信号强度。
    嘴动强度优先使用儿童嘴动标定器（lip_amp_model.pt / lip_amp_scale.json）。
    """

    def __init__(
        self,
        window_sec: float = 4.0,
        target_yaw_std_deg: float = 12.0,
        weights: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
        lip_amp_checkpoint: Optional[Path] = None,
        lip_activity_scale: float = 0.015,
    ):
        self.window_sec = window_sec
        self.target_yaw_std_deg = target_yaw_std_deg
        self.weights = weights
        self.lip_activity_scale = float(lip_activity_scale)
        self._lip_calibrator = None
        if lip_amp_checkpoint is not None and Path(lip_amp_checkpoint).exists():
            from csd.trust.lip_amplitude import LipAmplitudeCalibrator

            self._lip_calibrator = LipAmplitudeCalibrator.from_checkpoint(Path(lip_amp_checkpoint))
            self.lip_activity_scale = float(self._lip_calibrator.activity_scale)
        else:
            for cand in (
                Path("output/lip_amp_xianyang/merged_all/lip_amp_model.pt"),
                Path("output/lip_amp_xianyang/merged_all/lip_amp_scale.json"),
            ):
                if not cand.exists():
                    continue
                if cand.suffix == ".pt":
                    from csd.trust.lip_amplitude import LipAmplitudeCalibrator

                    self._lip_calibrator = LipAmplitudeCalibrator.from_checkpoint(cand)
                    self.lip_activity_scale = float(self._lip_calibrator.activity_scale)
                    break
                if cand.suffix == ".json":
                    import json

                    stats = json.loads(cand.read_text(encoding="utf-8"))
                    self.lip_activity_scale = float(stats.get("activity_scale", self.lip_activity_scale))
                    break

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

    def _lip_signal_score(
        self,
        timeline: SlotVisualTimeline,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> float:
        from csd.core.utils import time_to_frame
        from csd.perception.head_pose import HeadPoseAnalyzer

        if self._lip_calibrator is not None:
            scores = []
            start_f = time_to_frame(start_time, fps)
            end_f = time_to_frame(end_time, fps)
            for spk in SPEAKERS:
                slot_id = timeline.speaker_to_slot.get(spk)
                if slot_id is None:
                    continue
                mars, sides, yaws = [], [], []
                for f, pose in timeline.frames.get(slot_id, {}).items():
                    if start_f <= f <= end_f:
                        mars.append(pose.mouth_opening)
                        sides.append(pose.side_face_weight)
                        yaws.append(pose.yaw)
                if len(mars) >= 2:
                    scores.append(self._lip_calibrator.score_from_arrays(mars, sides, yaws))
            return float(max(scores)) if scores else 0.0

        activities = []
        for spk in SPEAKERS:
            act, _ = HeadPoseAnalyzer.aggregate_segment_mouth(
                timeline, spk, start_time, end_time, fps
            )
            activities.append(act)
        if not activities:
            return 0.0
        peak = max(activities)
        return float(np.clip(peak / max(self.lip_activity_scale, 1e-6), 0.0, 1.0))

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
