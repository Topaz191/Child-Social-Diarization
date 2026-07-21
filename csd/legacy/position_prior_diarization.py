"""基于已知位置先验 + 声纹 + 嘴动的说话人分离。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack, FaceTracker
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.core.utils import cosine_similarity, l2_normalize
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment, VADProcessor

logger = logging.getLogger(__name__)

# 由 verify_position_with_gt.py + Excel 标准标签确认的固定映射
DEFAULT_POSITION_TO_SPEAKER = {
    0: "S1",  # 右中 (0.67, 0.42)
    1: "S3",  # 中中 (0.45, 0.44)
    2: "S2",  # 左中 (0.18, 0.41)
}


@dataclass
class DiarizationSegment:
    start_time: float
    end_time: float
    speaker: str
    confidence: float
    voice_speaker: Optional[str] = None
    voice_score: float = 0.0
    visual_speaker: Optional[str] = None
    lip_score: float = 0.0
    conflict: bool = False
    source: str = "fusion"


class PositionPriorDiarizer:
    """
    说话人分离：VAD 切段 + post 声纹匹配 + 嘴动位置（固定位置→人映射）融合。
    """

    def __init__(
        self,
        config: ASDConfig,
        position_to_speaker: Optional[Dict[int, str]] = None,
        voice_weight: float = 0.55,
        visual_weight: float = 0.45,
        voice_min_score: float = 0.0,
    ):
        self.config = config
        self.position_to_speaker = position_to_speaker or DEFAULT_POSITION_TO_SPEAKER.copy()
        self.voice_weight = voice_weight
        self.visual_weight = visual_weight
        self.voice_min_score = voice_min_score
        self.mapper = PositionSpeakerMapper(config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(config)
        self.vad = VADProcessor(config)

    @classmethod
    def load_position_map_from_json(cls, path: Path) -> Dict[int, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = {}
        for spk, info in data.get("gt_speaker_to_position", {}).items():
            mapping[int(info["cluster_id"])] = spk.upper()
        return mapping

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = np.array(list(scores.values()))
        if vals.max() - vals.min() < 1e-8:
            return {k: 1.0 / len(scores) for k in scores}
        normed = (vals - vals.min()) / (vals.max() - vals.min())
        return dict(zip(scores.keys(), normed.tolist()))

    def _voice_label(self, segment: SpeechSegment) -> Tuple[Optional[str], float]:
        if segment.speaker_embedding is None:
            return None, 0.0
        name, score = self.speaker_extractor.match_speaker_best(segment.speaker_embedding)
        if name is None:
            return None, 0.0
        return name.upper(), float(score)

    def _visual_label(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segment: SpeechSegment,
        fps: float,
    ) -> Tuple[Optional[str], float, Dict[int, float]]:
        lip_by_cluster = self.mapper.compute_lip_by_cluster(
            video_path, tracks, segment.start_time, segment.end_time, fps, n_slots=3
        )
        if not lip_by_cluster:
            return None, 0.0, {}

        speaker_lip: Dict[str, float] = {}
        for cid, lip_val in lip_by_cluster.items():
            spk = self.position_to_speaker.get(cid)
            if spk:
                speaker_lip[spk] = max(speaker_lip.get(spk, 0.0), lip_val)

        if not speaker_lip:
            return None, 0.0, lip_by_cluster

        best_spk = max(speaker_lip, key=speaker_lip.get)
        normed = self._normalize_scores(speaker_lip)
        return best_spk, float(normed.get(best_spk, 0.0)), lip_by_cluster

    def fuse_segment(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segment: SpeechSegment,
        fps: float,
    ) -> DiarizationSegment:
        voice_spk, voice_score = self._voice_label(segment)
        visual_spk, lip_score, _ = self._visual_label(video_path, tracks, segment, fps)

        scores: Dict[str, float] = {}
        for spk in ("S1", "S2", "S3"):
            v = voice_score if voice_spk == spk else 0.0
            l = lip_score if visual_spk == spk else 0.0
            # 若声纹对该人有分，用原始 cosine；否则仅嘴动
            if voice_spk == spk and segment.speaker_embedding is not None:
                ref = self.speaker_extractor.enrolled_speakers.get(spk.lower())
                if ref is not None:
                    v = max(0.0, cosine_similarity(segment.speaker_embedding, ref))
            if visual_spk == spk:
                l = lip_score
            scores[spk] = self.voice_weight * v + self.visual_weight * l

        conflict = (
            voice_spk is not None
            and visual_spk is not None
            and voice_spk != visual_spk
            and voice_score > 0.25
            and lip_score > 0.2
        )
        if conflict:
            for spk in scores:
                scores[spk] *= 0.85

        if max(scores.values()) < 1e-6:
            best = voice_spk or visual_spk or "UNKNOWN"
            conf = max(voice_score, lip_score)
            source = "voice" if voice_spk else "visual"
        else:
            best = max(scores, key=scores.get)
            conf = scores[best]
            source = "fusion"

        return DiarizationSegment(
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker=best,
            confidence=round(conf, 4),
            voice_speaker=voice_spk,
            voice_score=round(voice_score, 4),
            visual_speaker=visual_spk,
            lip_score=round(lip_score, 4),
            conflict=conflict,
            source=source,
        )

    def diarize(
        self,
        video_path: str,
        audio_path: Path,
        enroll_dir: Path,
        tracks: Optional[Dict[int, FaceTrack]] = None,
        fps: Optional[float] = None,
        speech_times: Optional[List[Tuple[float, float]]] = None,
    ) -> List[DiarizationSegment]:
        video_path = str(Path(video_path).resolve())
        self.speaker_extractor.enroll_from_directory(enroll_dir)

        if speech_times is None:
            speech_times = self.vad.detect(audio_path)
        segments = self.speaker_extractor.process_segments(audio_path, speech_times)

        if tracks is None or fps is None:
            tracker = FaceTracker(self.config)
            tracker.process_video(video_path)
            tracks = tracker.tracks
            fps = tracker.fps

        results = []
        for i, seg in enumerate(segments):
            dec = self.fuse_segment(video_path, tracks, seg, fps)
            results.append(dec)
            if (i + 1) % 20 == 0:
                logger.info("  已处理 %d/%d 语音段", i + 1, len(segments))
        return results

    @staticmethod
    def to_segment_dicts(items: List[DiarizationSegment]) -> List[dict]:
        return [
            {
                "start": d.start_time,
                "end": d.end_time,
                "speaker": d.speaker,
                "confidence": d.confidence,
                "voice_speaker": d.voice_speaker,
                "voice_score": d.voice_score,
                "visual_speaker": d.visual_speaker,
                "lip_score": d.lip_score,
                "conflict": d.conflict,
                "source": d.source,
            }
            for d in items
        ]
