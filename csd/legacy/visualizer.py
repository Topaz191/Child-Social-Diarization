"""模块6：可视化输出与 JSON 结果导出。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from csd.perception.face_identity import FaceIdentityManager
from csd.perception.face_tracker import FaceTrack
from csd.legacy.fusion import SpeakerDecision
from csd.core.utils import frame_to_time, save_json

logger = logging.getLogger(__name__)

# BGR 颜色表
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 0),
    (255, 128, 0),
]


class ResultVisualizer:
    """生成标注视频与 JSON 时间戳。"""

    def __init__(self, identity_mgr: FaceIdentityManager):
        self.identity_mgr = identity_mgr

    def decisions_to_json(self, decisions: List[SpeakerDecision]) -> List[dict]:
        return [
            {
                "start_time": round(d.start_time, 3),
                "end_time": round(d.end_time, 3),
                "speaker_id": d.speaker_id,
                "speaker_label": d.speaker_label,
                "confidence": d.confidence,
                "voice_score": d.voice_score,
                "lip_score": d.lip_score,
                "conflict": d.conflict,
            }
            for d in decisions
        ]

    def _active_speaker_at(
        self,
        frame_idx: int,
        fps: float,
        decisions: List[SpeakerDecision],
    ) -> Optional[SpeakerDecision]:
        t = frame_to_time(frame_idx, fps)
        for dec in decisions:
            if dec.start_time <= t <= dec.end_time:
                return dec
        return None

    def render_video(
        self,
        video_path: str,
        output_path: Path,
        tracks: Dict[int, FaceTrack],
        frame_tracks: Dict[int, List[FaceTrack]],
        decisions: List[SpeakerDecision],
        fps: float,
    ) -> None:
        """输出带说话人高亮框的标注视频。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

        frame_idx = 0
        logger.info("生成标注视频: %s", output_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            active_dec = self._active_speaker_at(frame_idx, fps, decisions)
            active_identity = active_dec.speaker_id if active_dec else -1

            # 绘制当前帧所有轨迹
            active_tracks = frame_tracks.get(frame_idx, [])
            for track in active_tracks:
                if frame_idx not in track.detections:
                    continue
                det = track.detections[frame_idx]
                identity_id = self.identity_mgr.get_identity_for_track(track.track_id)
                if identity_id is None:
                    continue

                x1, y1, x2, y2 = det.bbox.astype(int)
                color = COLORS[identity_id % len(COLORS)]
                is_speaking = identity_id == active_identity and active_dec is not None

                thickness = 4 if is_speaking else 2
                if is_speaking:
                    # 说话人高亮：加粗 + 半透明填充
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                label = self.identity_mgr.identity_label(identity_id)
                if is_speaking:
                    label = f"*{label}* ({active_dec.confidence:.2f})"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # 顶部状态栏
            if active_dec:
                status = f"Speaking: {active_dec.speaker_label} [{active_dec.start_time:.1f}-{active_dec.end_time:.1f}s]"
                cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.info("标注视频已保存: %s", output_path)

    def save_results(
        self,
        decisions: List[SpeakerDecision],
        json_path: Path,
        video_path: Optional[str] = None,
        output_video_path: Optional[Path] = None,
        tracks: Optional[Dict[int, FaceTrack]] = None,
        frame_tracks: Optional[Dict[int, List[FaceTrack]]] = None,
        fps: float = 25.0,
    ) -> None:
        save_json(self.decisions_to_json(decisions), json_path)
        if video_path and output_video_path and tracks and frame_tracks:
            self.render_video(
                video_path, output_video_path, tracks, frame_tracks, decisions, fps
            )
