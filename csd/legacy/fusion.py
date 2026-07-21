"""模块5：多模态联合判定说话人。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_identity import FaceIdentityManager
from csd.perception.face_tracker import FaceTrack
from csd.perception.lip_motion import LipMotionAnalyzer
from csd.core.utils import cosine_similarity, l2_normalize
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class SpeakerDecision:
    """单个语音段的说话人判定结果。"""

    start_time: float
    end_time: float
    speaker_id: int  # 人脸 identity_id，-1 表示未知
    speaker_label: str
    confidence: float
    voice_score: float = 0.0
    lip_score: float = 0.0
    position_score: float = 0.0
    conflict: bool = False
    enrolled_name: Optional[str] = None


class MultimodalFusion:
    """
    联合声纹 + 嘴部运动 + 位置稳定性判定说话人。
    动态维护 identity -> voice embedding 映射（无预注册时在线注册）。
    """

    def __init__(
        self,
        config: ASDConfig,
        identity_mgr: FaceIdentityManager,
        speaker_extractor: SpeakerEmbeddingExtractor,
        lip_analyzer: LipMotionAnalyzer,
    ):
        self.config = config
        self.identity_mgr = identity_mgr
        self.speaker_extractor = speaker_extractor
        self.lip_analyzer = lip_analyzer
        # identity_id -> 声纹模板（在线学习）
        self.identity_voice_templates: Dict[int, np.ndarray] = {}
        # enrolled name -> identity_id 映射
        self.enrolled_name_to_identity: Dict[str, int] = {}

    def _identity_to_tracks(self, tracks: Dict[int, FaceTrack]) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {}
        for tid, iid in self.identity_mgr.track_to_identity.items():
            mapping.setdefault(iid, []).append(tid)
        return mapping

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        vals = np.array(list(scores.values()))
        if vals.max() - vals.min() < 1e-8:
            return {k: 1.0 / len(scores) for k in scores}
        normed = (vals - vals.min()) / (vals.max() - vals.min())
        return dict(zip(scores.keys(), normed.tolist()))

    def _voice_scores(
        self,
        segment: SpeechSegment,
    ) -> Tuple[Dict[int, float], Optional[str], float]:
        """计算各 identity 的声纹匹配分，及预注册说话人名。"""
        emb = segment.speaker_embedding
        if emb is None:
            return {}, None, 0.0

        emb = l2_normalize(emb)

        # 优先匹配预注册声纹库
        enrolled_name, enrolled_score = self.speaker_extractor.match_speaker(emb)
        if enrolled_name:
            iid = self.enrolled_name_to_identity.get(enrolled_name)
            if iid is not None:
                return {iid: enrolled_score}, enrolled_name, enrolled_score

        scores: Dict[int, float] = {}
        for identity_id, template in self.identity_voice_templates.items():
            scores[identity_id] = max(0.0, cosine_similarity(emb, template))

        return scores, enrolled_name, enrolled_score

    def _lip_scores(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segment: SpeechSegment,
        fps: float,
    ) -> Dict[int, float]:
        identity_tracks = self._identity_to_tracks(tracks)
        activities = self.lip_analyzer.compute_all_activities(
            video_path,
            tracks,
            identity_tracks,
            segment.start_time,
            segment.end_time,
            fps,
        )
        return self._normalize_scores(activities)

    def _position_scores(self, tracks: Dict[int, FaceTrack]) -> Dict[int, float]:
        """位置稳定性：检测帧数越多、位置方差越小，分数越高。"""
        scores: Dict[int, float] = {}
        identity_tracks = self._identity_to_tracks(tracks)
        for iid, tids in identity_tracks.items():
            frame_counts = sum(len(tracks[t].detections) for t in tids if t in tracks)
            scores[iid] = float(frame_counts)
        return self._normalize_scores(scores)

    def decide_segment(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segment: SpeechSegment,
        fps: float,
    ) -> SpeakerDecision:
        voice_raw, enrolled_name, enrolled_score = self._voice_scores(segment)
        lip_raw = self._lip_scores(video_path, tracks, segment, fps)
        pos_raw = self._position_scores(tracks)

        all_identities = set(voice_raw) | set(lip_raw) | set(pos_raw)
        if not all_identities:
            return SpeakerDecision(
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker_id=-1,
                speaker_label="Unknown",
                confidence=0.0,
            )

        combined: Dict[int, float] = {}
        for iid in all_identities:
            v = voice_raw.get(iid, 0.0)
            l = lip_raw.get(iid, 0.0)
            p = pos_raw.get(iid, 0.0)
            combined[iid] = (
                self.config.voice_weight * v
                + self.config.lip_weight * l
                + self.config.position_weight * p
            )

        # 若声纹与嘴动指向不同人，对领先者降权并标记冲突
        best_voice = max(voice_raw, key=voice_raw.get) if voice_raw else None
        best_lip = max(lip_raw, key=lip_raw.get) if lip_raw else None
        conflict = (
            best_voice is not None
            and best_lip is not None
            and best_voice != best_lip
            and voice_raw.get(best_voice, 0) > 0.3
            and lip_raw.get(best_lip, 0) > 0.3
        )
        if conflict:
            for iid in combined:
                combined[iid] *= 1.0 - self.config.conflict_penalty

        best_id = max(combined, key=combined.get)
        confidence = float(combined[best_id])

        # 嘴动阈值过滤：若最佳者嘴动过低且无强声纹匹配，降低置信度
        lip_activity = lip_raw.get(best_id, 0.0)
        if lip_activity < 0.2 and enrolled_score < self.config.speaker_match_thresh:
            confidence *= 0.5

        label = self.identity_mgr.identity_label(best_id)
        if enrolled_name and best_id == self.enrolled_name_to_identity.get(enrolled_name):
            label = enrolled_name

        # 在线更新声纹模板
        if segment.speaker_embedding is not None and confidence > 0.4:
            self._update_voice_template(best_id, segment.speaker_embedding)

        return SpeakerDecision(
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker_id=best_id,
            speaker_label=label,
            confidence=round(confidence, 4),
            voice_score=round(voice_raw.get(best_id, 0.0), 4),
            lip_score=round(lip_raw.get(best_id, 0.0), 4),
            position_score=round(pos_raw.get(best_id, 0.0), 4),
            conflict=conflict,
            enrolled_name=enrolled_name,
        )

    def _update_voice_template(self, identity_id: int, embedding: np.ndarray) -> None:
        emb = l2_normalize(embedding)
        if identity_id in self.identity_voice_templates:
            old = self.identity_voice_templates[identity_id]
            self.identity_voice_templates[identity_id] = l2_normalize(0.8 * old + 0.2 * emb)
        else:
            self.identity_voice_templates[identity_id] = emb

    def link_enrolled_speakers(self) -> None:
        """将预注册声纹与最近的人脸 identity 关联（基于首次高置信匹配）。"""
        for name, emb in self.speaker_extractor.enrolled_speakers.items():
            # 暂存，待首次匹配时动态关联
            self.enrolled_name_to_identity.setdefault(name, -1)

    def process_all(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segments: List[SpeechSegment],
        fps: float,
    ) -> List[SpeakerDecision]:
        decisions = []
        for seg in segments:
            dec = self.decide_segment(video_path, tracks, seg, fps)
            decisions.append(dec)
            logger.info(
                "  [%.2f-%.2f]s -> %s (conf=%.3f, voice=%.3f, lip=%.3f%s)",
                dec.start_time,
                dec.end_time,
                dec.speaker_label,
                dec.confidence,
                dec.voice_score,
                dec.lip_score,
                ", CONFLICT" if dec.conflict else "",
            )
        return decisions
