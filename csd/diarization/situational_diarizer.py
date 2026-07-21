"""
CSD 态势感知说话人分离器。

模块1：audio_conf + visual_conf
模块2/3：SituationRouter 双模态权重融合
模块4：ConservativeUpdater 双低保守更新 + CRF 时序平滑
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from csd.constants import SPEAKERS
from csd.core.config import ASDConfig
from csd.core.utils import cosine_similarity, l2_normalize
from csd.legacy.position_prior_diarization import DEFAULT_POSITION_TO_SPEAKER
from csd.perception.face_tracker import FaceTrack, FaceTracker
from csd.perception.head_pose import HeadPoseAnalyzer, SlotVisualTimeline
from csd.perception.lip_motion import LipMotionAnalyzer
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment, VADProcessor
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.social.social_attention import SocialAttentionComputer
from csd.trust.audio_confidence import AudioConfidenceEstimator
from csd.trust.conservative_update import ConservativeUpdater
from csd.trust.situation_router import SituationRouter
from csd.trust.speaker_crf import SpeakerSequenceCRF
from csd.trust.visual_confidence import VisualConfidenceEstimator

logger = logging.getLogger(__name__)


@dataclass
class SituationalSegment:
    start_time: float
    end_time: float
    speaker: str
    confidence: float
    voice_probs: Dict[str, float] = field(default_factory=dict)
    lip_scores: Dict[str, float] = field(default_factory=dict)
    attention_received: Dict[str, float] = field(default_factory=dict)
    fused_scores: Dict[str, float] = field(default_factory=dict)
    visual_probs: Dict[str, float] = field(default_factory=dict)
    audio_confidence: float = 0.0
    visual_confidence: float = 0.0
    route_weight_audio: float = 0.0
    route_weight_visual: float = 0.0
    gate_mode: str = ""
    crf_speaker: str = ""
    voice_best: str = ""
    lip_best: str = ""


# 兼容旧名
IntegrativeSegment = SituationalSegment


class SituationalDiarizer:
    """动态可信度融合说话人分离（CSD 主推理器）。"""

    def __init__(
        self,
        config: ASDConfig,
        position_to_speaker: Optional[Dict[int, str]] = None,
        speaker_ref_x: Optional[Dict[str, float]] = None,
        voice_temperature: float = 8.0,
        attn_temperature: float = 15.0,
        crf_stay_prob: float = 0.62,
        use_conservative: bool = True,
    ):
        self.config = config
        self.position_to_speaker = position_to_speaker or DEFAULT_POSITION_TO_SPEAKER.copy()
        self.speaker_ref_x = speaker_ref_x or {"S1": 0.67, "S2": 0.18, "S3": 0.45}
        self.voice_temperature = voice_temperature
        self.use_conservative = use_conservative
        self.mapper = PositionSpeakerMapper(config)
        self.vad = VADProcessor(config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(config)
        self.head_analyzer = HeadPoseAnalyzer(config)
        self.lip_analyzer = LipMotionAnalyzer(config)
        self.attention = SocialAttentionComputer(temperature=attn_temperature)
        self.audio_conf_est = AudioConfidenceEstimator()
        self.visual_conf_est = VisualConfidenceEstimator()
        self.router = SituationRouter.from_config(config)
        self.conservative = ConservativeUpdater.from_config(config)
        self.crf = SpeakerSequenceCRF(SPEAKERS, stay_prob=crf_stay_prob)
        self._tracks: Optional[Dict[int, FaceTrack]] = None
        self._slot_to_tracks: Optional[Dict[int, List[int]]] = None
        self._slot_to_speaker: Optional[Dict[int, str]] = None

    IntegrativeDiarizer = None  # noqa — 兼容占位

    @classmethod
    def load_position_map_from_json(cls, path: Path) -> Dict[int, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(info["cluster_id"]): spk.upper() for spk, info in data.get("gt_speaker_to_position", {}).items()}

    @classmethod
    def load_speaker_ref_x_from_json(cls, path: Path) -> Dict[str, float]:
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster_x = {int(p["cluster_id"]): float(p["mean_x"]) for p in data.get("positions", [])}
        ref_x = {}
        for spk, info in data.get("gt_speaker_to_position", {}).items():
            cid = int(info["cluster_id"])
            if cid in cluster_x:
                ref_x[spk.upper()] = cluster_x[cid]
        return ref_x or {"S1": 0.67, "S2": 0.18, "S3": 0.45}

    def _softmax_dict(self, scores: Dict[str, float], temperature: float) -> Dict[str, float]:
        keys = list(scores.keys())
        vals = np.array([max(0.0, scores.get(k, 0.0)) for k in keys], dtype=np.float64)
        if vals.max() < 1e-8:
            return {k: 1.0 / len(keys) for k in keys}
        logits = vals * temperature - vals.max() * temperature
        probs = np.exp(logits)
        probs /= probs.sum()
        return {k: float(v) for k, v in zip(keys, probs)}

    def _voice_probs(self, segment: SpeechSegment) -> Dict[str, float]:
        if segment.speaker_embedding is None:
            return {s: 1.0 / len(SPEAKERS) for s in SPEAKERS}
        emb = l2_normalize(segment.speaker_embedding)
        raw = {}
        for spk in SPEAKERS:
            ref = self.speaker_extractor.enrolled_speakers.get(spk.lower())
            raw[spk] = max(0.0, cosine_similarity(emb, ref)) if ref is not None else 0.0
        return self._softmax_dict(raw, self.voice_temperature)

    def _lip_scores(self, video_path: str, segment: SpeechSegment, timeline: SlotVisualTimeline, fps: float) -> Dict[str, float]:
        raw, side_weights = {}, {}
        for spk in SPEAKERS:
            act, sw = self.head_analyzer.aggregate_segment_mouth(timeline, spk, segment.start_time, segment.end_time, fps)
            raw[spk] = act
            side_weights[spk] = sw
        if self._tracks and self._slot_to_tracks and self._slot_to_speaker and max(raw.values()) < 1e-5:
            spk_to_slot = {v: k for k, v in self._slot_to_speaker.items()}
            for spk in SPEAKERS:
                tids = self._slot_to_tracks.get(spk_to_slot.get(spk), [])
                if tids:
                    act = self.lip_analyzer.compute_all_activities(
                        video_path, self._tracks, {0: tids}, segment.start_time, segment.end_time, fps
                    )
                    raw[spk] = max(raw[spk], act.get(0, 0.0))
        normed = self._normalize_nonneg(raw)
        return {spk: normed.get(spk, 0.0) * side_weights.get(spk, 1.0) for spk in SPEAKERS}

    @staticmethod
    def _normalize_nonneg(scores: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(scores.values()), dtype=np.float64)
        if len(vals) == 0 or vals.max() - vals.min() < 1e-8:
            return {k: 1.0 / max(len(scores), 1) for k in scores} if scores else {}
        normed = (vals - vals.min()) / (vals.max() - vals.min())
        return dict(zip(scores.keys(), normed.tolist()))

    def _align_slots_to_speakers(self, slots) -> Dict[int, str]:
        used: set = set()
        slot_to_speaker = {}
        for slot in sorted(slots, key=lambda s: s.mean_x):
            best_spk, best_dist = None, 1e9
            for spk, rx in self.speaker_ref_x.items():
                if spk in used:
                    continue
                dist = abs(slot.mean_x - rx)
                if dist < best_dist:
                    best_dist, best_spk = dist, spk
            if best_spk:
                used.add(best_spk)
                slot_to_speaker[slot.cluster_id] = best_spk
        return slot_to_speaker

    def _build_timeline(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        fps: float,
        speech_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> SlotVisualTimeline:
        slots = self.mapper.extract_position_slots(tracks, n_slots=3)
        slot_positions = {s.cluster_id: (s.mean_x, s.mean_y) for s in slots}
        slot_to_tracks = self.mapper._cluster_to_tracks(tracks, n_slots=3)
        slot_to_speaker = self._align_slots_to_speakers(slots)
        self._slot_to_tracks, self._slot_to_speaker = slot_to_tracks, slot_to_speaker
        logger.info("槽位→说话人: %s", slot_to_speaker)
        return self.head_analyzer.build_slot_timeline(
            video_path,
            tracks,
            slot_to_tracks,
            slot_to_speaker,
            slot_positions,
            fps,
            speech_intervals=speech_intervals,
        )

    def diarize(
        self,
        video_path: str,
        audio_path: Path,
        enroll_dir: Path,
        tracks: Optional[Dict[int, FaceTrack]] = None,
        fps: Optional[float] = None,
        speech_times: Optional[List[Tuple[float, float]]] = None,
        use_crf: bool = True,
    ) -> List[SituationalSegment]:
        video_path = str(Path(video_path).resolve())
        self.speaker_extractor.enroll_from_directory(enroll_dir)
        if speech_times is None:
            speech_times = self.vad.detect(audio_path)
        segments = self.speaker_extractor.process_segments(audio_path, speech_times)

        if tracks is None or fps is None:
            tracker = FaceTracker(self.config)
            tracker.process_video(video_path)
            tracks, fps = tracker.tracks, tracker.fps
        self._tracks = tracks
        timeline = self._build_timeline(video_path, tracks, fps, speech_intervals=speech_times)

        audio_confs = self.audio_conf_est.compute_all(segments, self.speaker_extractor)
        visual_confs = self.visual_conf_est.compute_all(timeline, segments, fps)
        logger.info(
            "可信度 audio mean=%.3f visual mean=%.3f",
            float(np.mean(audio_confs)) if audio_confs else 0,
            float(np.mean(visual_confs)) if visual_confs else 0,
        )

        raw_fused: List[Dict[str, float]] = []
        voice_margins: List[float] = []
        meta: List[dict] = []
        for i, seg in enumerate(segments):
            ac = audio_confs[i] if i < len(audio_confs) else 0.0
            vc = visual_confs[i] if i < len(visual_confs) else 0.0
            voice_p = self._voice_probs(seg)
            lip_s = self._lip_scores(video_path, seg, timeline, fps)
            _, attn = self.attention.segment_attention(timeline, list(SPEAKERS), seg.start_time, seg.end_time, fps)
            margin = self.router.voice_margin(voice_p)
            voice_margins.append(margin)
            fused, w_a, w_v, visual_p = self.router.fuse(ac, vc, voice_p, lip_s, attn)
            raw_fused.append({k: round(v, 4) for k, v in fused.items()})
            meta.append(
                dict(
                    seg=seg, voice_p=voice_p, lip_s=lip_s, attn=attn, visual_p=visual_p,
                    ac=ac, vc=vc, w_a=w_a, w_v=w_v, margin=margin,
                )
            )

        if self.use_conservative:
            fused_list, modes = self.conservative.apply_sequence(
                raw_fused, audio_confs, visual_confs, voice_margins=voice_margins
            )
        else:
            fused_list, modes = raw_fused, ["balanced"] * len(raw_fused)

        results: List[SituationalSegment] = []
        for m, fused, mode in zip(meta, fused_list, modes):
            seg = m["seg"]
            best = max(fused, key=fused.get)
            results.append(
                SituationalSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=best,
                    confidence=round(fused[best], 4),
                    voice_probs={k: round(v, 4) for k, v in m["voice_p"].items()},
                    lip_scores={k: round(v, 4) for k, v in m["lip_s"].items()},
                    attention_received={k: round(v, 4) for k, v in m["attn"].items()},
                    fused_scores={k: round(v, 4) for k, v in fused.items()},
                    visual_probs={k: round(v, 4) for k, v in m["visual_p"].items()},
                    audio_confidence=round(m["ac"], 4),
                    visual_confidence=round(m["vc"], 4),
                    route_weight_audio=round(m["w_a"], 4),
                    route_weight_visual=round(m["w_v"], 4),
                    gate_mode=mode,
                    voice_best=max(m["voice_p"], key=m["voice_p"].get),
                    lip_best=max(m["lip_s"], key=m["lip_s"].get),
                )
            )

        if use_crf and results:
            labels = self.crf.decode([r.fused_scores for r in results])
            for r, spk in zip(results, labels):
                r.crf_speaker = spk
                r.speaker = spk
                r.confidence = round(r.fused_scores.get(spk, r.confidence), 4)

        return results

    @staticmethod
    def to_dicts(items: List[SituationalSegment]) -> List[dict]:
        return [
            {
                "start": d.start_time,
                "end": d.end_time,
                "speaker": d.speaker,
                "confidence": d.confidence,
                "gate_mode": d.gate_mode,
                "voice_best": d.voice_best,
                "lip_best": d.lip_best,
                "voice_probs": d.voice_probs,
                "lip_scores": d.lip_scores,
                "attention_received": d.attention_received,
                "fused_scores": d.fused_scores,
                "visual_probs": d.visual_probs,
                "audio_confidence": d.audio_confidence,
                "visual_confidence": d.visual_confidence,
                "route_weight_audio": d.route_weight_audio,
                "route_weight_visual": d.route_weight_visual,
                "crf_speaker": d.crf_speaker,
            }
            for d in items
        ]


# 兼容旧类名
IntegrativeDiarizer = SituationalDiarizer
