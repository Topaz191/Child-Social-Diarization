"""基于已知位置-说话人映射 + 声纹标签的说话人分离。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack, FaceTracker
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment, VADProcessor

logger = logging.getLogger(__name__)

# 由 verify_position_with_gt.py + Excel 标准标签验证的固定映射
DEFAULT_POSITION_TO_SPEAKER = {
    0: "S1",  # 右中 (0.67, 0.42)
    2: "S2",  # 左中 (0.18, 0.41)
    1: "S3",  # 中中 (0.45, 0.44)
}


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str
    confidence: float
    voice_speaker: Optional[str] = None
    voice_score: float = 0.0
    visual_speaker: Optional[str] = None
    visual_score: float = 0.0
    conflict: bool = False


class KnownPositionDiarizer:
    """
    已知画面位置 ↔ 说话人对应关系时的分离策略：
    - 声纹：speakers/post 预注册 s1/s2/s3
    - 视觉：各位置簇嘴动活跃度 → 映射到 S1/S2/S3
    - 融合：加权得分，冲突时降权
    """

    def __init__(
        self,
        config: ASDConfig,
        position_to_speaker: Optional[Dict[int, str]] = None,
        voice_weight: float = 0.55,
        visual_weight: float = 0.45,
    ):
        self.config = config
        self.position_to_speaker = position_to_speaker or DEFAULT_POSITION_TO_SPEAKER.copy()
        self.voice_weight = voice_weight
        self.visual_weight = visual_weight
        self.mapper = PositionSpeakerMapper(config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(config)
        self.tracks: Dict[int, FaceTrack] = {}
        self.fps: float = 25.0

    @classmethod
    def load_mapping_from_verify_json(cls, path: Path) -> Dict[int, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = {}
        for spk, info in data.get("gt_speaker_to_position", {}).items():
            mapping[int(info["cluster_id"])] = spk.upper()
        if len(mapping) < 3:
            raise ValueError(f"验证 JSON 中映射不完整: {path}")
        return mapping

    def prepare(
        self,
        video_path: str,
        enroll_dir: Path,
        frame_skip: Optional[int] = None,
    ) -> None:
        if frame_skip is not None:
            self.config.frame_skip = frame_skip
        tracker = FaceTracker(self.config)
        tracker.process_video(video_path)
        self.tracks = tracker.tracks
        self.fps = tracker.fps
        self.speaker_extractor.enroll_from_directory(enroll_dir)
        logger.info(
            "已加载 %d 条人脸轨迹, 声纹库: %s",
            len(self.tracks),
            list(self.speaker_extractor.enrolled_speakers.keys()),
        )

    def _visual_scores(self, video_path: str, seg: SpeechSegment) -> Dict[str, float]:
        lip = self.mapper.compute_lip_by_cluster(
            video_path,
            self.tracks,
            seg.start_time,
            seg.end_time,
            self.fps,
            n_slots=3,
        )
        scores: Dict[str, float] = {}
        for cid, act in lip.items():
            spk = self.position_to_speaker.get(cid)
            if spk:
                scores[spk] = max(scores.get(spk, 0.0), act)
        return scores

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        if max(vals) - min(vals) < 1e-8:
            return {k: 1.0 / len(scores) for k in scores}
        return {k: (v - min(vals)) / (max(vals) - min(vals)) for k, v in scores.items()}

    def decide_segment(self, video_path: str, seg: SpeechSegment) -> DiarizationSegment:
        voice_name, voice_score = None, 0.0
        if seg.speaker_embedding is not None:
            voice_name, voice_score = self.speaker_extractor.match_speaker_best(
                seg.speaker_embedding
            )
            if voice_name:
                voice_name = voice_name.upper()

        visual_raw = self._visual_scores(video_path, seg)
        visual_norm = self._normalize(visual_raw)

        voice_norm: Dict[str, float] = {}
        if voice_name:
            voice_norm[voice_name] = max(0.0, voice_score)

        speakers = set(visual_norm) | set(voice_norm)
        combined: Dict[str, float] = {}
        for spk in speakers:
            combined[spk] = (
                self.voice_weight * voice_norm.get(spk, 0.0)
                + self.visual_weight * visual_norm.get(spk, 0.0)
            )

        best = max(combined, key=combined.get) if combined else "UNKNOWN"
        conf = combined.get(best, 0.0)

        best_voice = max(voice_norm, key=voice_norm.get) if voice_norm else None
        best_visual = max(visual_norm, key=visual_norm.get) if visual_norm else None
        conflict = (
            best_voice is not None
            and best_visual is not None
            and best_voice != best_visual
            and voice_norm.get(best_voice, 0) > 0.25
            and visual_norm.get(best_visual, 0) > 0.25
        )
        if conflict:
            conf *= 0.85

        visual_spk = best_visual
        visual_sc = visual_norm.get(best_visual, 0.0) if best_visual else 0.0

        return DiarizationSegment(
            start=seg.start_time,
            end=seg.end_time,
            speaker=best,
            confidence=round(conf, 4),
            voice_speaker=voice_name,
            voice_score=round(voice_score, 4),
            visual_speaker=visual_spk,
            visual_score=round(visual_sc, 4),
            conflict=conflict,
        )

    def diarize_segments(
        self,
        video_path: str,
        segments: List[SpeechSegment],
    ) -> List[DiarizationSegment]:
        results = []
        for i, seg in enumerate(segments):
            dec = self.decide_segment(video_path, seg)
            results.append(dec)
            if (i + 1) % 20 == 0:
                logger.info("  已处理 %d/%d 段", i + 1, len(segments))
        return results

    def voice_only(self, segments: List[SpeechSegment]) -> List[DiarizationSegment]:
        out = []
        for seg in segments:
            name, score = None, 0.0
            if seg.speaker_embedding is not None:
                name, score = self.speaker_extractor.match_speaker_best(seg.speaker_embedding)
            out.append(
                DiarizationSegment(
                    start=seg.start_time,
                    end=seg.end_time,
                    speaker=(name or "UNKNOWN").upper(),
                    confidence=round(score, 4),
                    voice_speaker=(name or "").upper() or None,
                    voice_score=round(score, 4),
                )
            )
        return out

    def visual_only(self, video_path: str, segments: List[SpeechSegment]) -> List[DiarizationSegment]:
        out = []
        for seg in segments:
            visual_raw = self._visual_scores(video_path, seg)
            if not visual_raw:
                spk, sc = "UNKNOWN", 0.0
            else:
                spk = max(visual_raw, key=visual_raw.get)
                sc = visual_raw[spk]
            out.append(
                DiarizationSegment(
                    start=seg.start_time,
                    end=seg.end_time,
                    speaker=spk,
                    confidence=round(sc, 4),
                    visual_speaker=spk,
                    visual_score=round(sc, 4),
                )
            )
        return out
