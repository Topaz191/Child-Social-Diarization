"""完整多模态主动说话人检测流水线。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from csd.core.config import ASDConfig
from csd.perception.face_identity import FaceIdentityManager
from csd.perception.face_tracker import FaceTracker
from csd.legacy.fusion import MultimodalFusion, SpeakerDecision
from csd.perception.lip_motion import LipMotionAnalyzer
from csd.core.utils import extract_audio_ffmpeg, save_json, setup_logging
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, VADProcessor
from csd.legacy.visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


class ActiveSpeakerPipeline:
    """编排模块1-6的完整处理流程。"""

    def __init__(self, config: Optional[ASDConfig] = None):
        self.config = config or ASDConfig()
        self.config.ensure_dirs()

        self.face_tracker = FaceTracker(self.config)
        self.identity_mgr = FaceIdentityManager(self.config)
        self.vad = VADProcessor(self.config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(self.config)
        self.lip_analyzer = LipMotionAnalyzer(self.config)
        self.fusion: Optional[MultimodalFusion] = None
        self.visualizer: Optional[ResultVisualizer] = None

    def run(self, video_path: str) -> List[SpeakerDecision]:
        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        stem = Path(video_path).stem
        out_dir = self.config.output_dir
        audio_path = out_dir / f"{stem}_audio.wav"
        json_path = out_dir / f"{stem}_speakers.json"
        video_out_path = out_dir / f"{stem}_annotated.mp4"

        # --- 提取音频 ---
        logger.info("=" * 50)
        logger.info("步骤 0: 从视频提取音频")
        extracted = extract_audio_ffmpeg(
            Path(video_path), audio_path, self.config.sample_rate
        )
        if extracted is None:
            raise RuntimeError("音频提取失败，请确认已安装 ffmpeg")
        audio_path = extracted

        # --- 模块1: 人脸检测与跟踪 ---
        logger.info("=" * 50)
        logger.info("步骤 1: 人脸检测与位置跟踪")
        frame_tracks = self.face_tracker.process_video(video_path)
        tracks = self.face_tracker.tracks
        fps = self.face_tracker.fps

        # --- 模块2: 人脸识别与身份关联 ---
        logger.info("=" * 50)
        logger.info("步骤 2: 人脸识别与身份关联")
        self.identity_mgr.build_identities(tracks)

        # --- 模块3: VAD + 声纹 ---
        logger.info("=" * 50)
        logger.info("步骤 3: VAD 语音活动检测")
        speech_segments = self.vad.detect(Path(audio_path))

        if self.config.speaker_enroll_dir:
            logger.info("加载预注册声纹库: %s", self.config.speaker_enroll_dir)
            self.speaker_extractor.enroll_from_directory(
                Path(self.config.speaker_enroll_dir)
            )

        logger.info("步骤 3b: 声纹特征提取")
        segments = self.speaker_extractor.process_segments(
            Path(audio_path), speech_segments
        )

        # --- 模块4-5: 嘴动 + 多模态融合 ---
        logger.info("=" * 50)
        logger.info("步骤 4-5: 嘴部运动检测与多模态联合判定")
        self.fusion = MultimodalFusion(
            self.config,
            self.identity_mgr,
            self.speaker_extractor,
            self.lip_analyzer,
        )
        self.fusion.link_enrolled_speakers()
        decisions = self.fusion.process_all(video_path, tracks, segments, fps)

        # --- 模块6: 输出 ---
        logger.info("=" * 50)
        logger.info("步骤 6: 生成 JSON 与标注视频")
        self.visualizer = ResultVisualizer(self.identity_mgr)
        save_json(self.visualizer.decisions_to_json(decisions), json_path)
        if self.config.generate_video:
            self.visualizer.render_video(
                video_path, video_out_path, tracks, frame_tracks, decisions, fps
            )
        else:
            logger.info("跳过标注视频生成 (--no-video)")

        logger.info("=" * 50)
        logger.info("处理完成!")
        logger.info("  JSON: %s", json_path)
        logger.info("  视频: %s", video_out_path)
        return decisions
