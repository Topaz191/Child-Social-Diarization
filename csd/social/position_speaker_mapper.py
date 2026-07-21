"""位置簇 ↔ 声纹簇 关联分析（基于嘴动 + 声纹标签）。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack
from csd.perception.lip_motion import LipMotionAnalyzer
from csd.core.utils import cosine_similarity, l2_normalize
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class PositionSlot:
    """画面中的一个固定座位/区域。"""

    cluster_id: int
    mean_x: float
    mean_y: float
    track_count: int
    detection_frames: int


@dataclass
class VoiceProfile:
    """声纹聚类得到的一个说话人。"""

    voice_id: int
    label: str
    segment_count: int
    total_duration: float
    centroid: Optional[np.ndarray] = None


@dataclass
class PositionSpeakerMapping:
    """位置与说话人的对应关系。"""

    position: PositionSlot
    voice: VoiceProfile
    association_score: float
    lip_evidence: float
    voice_evidence: float


@dataclass
class MappingResult:
    """完整映射分析结果。"""

    positions: List[PositionSlot]
    voices: List[VoiceProfile]
    mappings: List[PositionSpeakerMapping]
    cooccurrence: Dict[int, Dict[int, float]] = field(default_factory=dict)
    segment_details: List[dict] = field(default_factory=list)


class PositionSpeakerMapper:
    """
    利用「语音段声纹标签 + 同时段各位置嘴动活跃度」推断位置对应关系。

    思路：
    1. 人脸轨迹按 DBSCAN 位置聚类 → 若干固定座位
    2. 语音段 embedding 聚类 → 若干声纹说话人
    3. 每个语音段内，统计各位置簇嘴动活跃度
    4. 累加 cooccurrence[voice][position]，匈牙利算法做最优匹配
    """

    def __init__(self, config: ASDConfig):
        self.config = config
        self.lip_analyzer = LipMotionAnalyzer(config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(config)

    @staticmethod
    def extract_position_slots(
        tracks: Dict[int, FaceTrack],
        n_slots: int = 3,
        face_size_min_ratio: float = 0.25,
    ) -> List[PositionSlot]:
        """从轨迹中心点提取固定座位（KMeans，避免 DBSCAN 噪声碎片化）。"""
        from sklearn.cluster import KMeans

        valid = [t for t in tracks.values() if len(t.detections) >= 3]
        if not valid:
            return []

        # 再按脸尺寸过滤一次，避免远处其他组进入槽位
        if face_size_min_ratio > 0:
            max_area = max(t.mean_bbox_area for t in valid)
            if max_area > 1e-6:
                before = len(valid)
                valid = [t for t in valid if t.mean_bbox_area >= face_size_min_ratio * max_area]
                if len(valid) < before:
                    logger.info(
                        "槽位提取脸尺寸过滤: %d -> %d (ratio>=%.2f)",
                        before,
                        len(valid),
                        face_size_min_ratio,
                    )
        if not valid:
            return []

        coords = np.array([t.mean_position for t in valid])
        k = min(n_slots, len(coords))
        if k <= 0:
            return []

        if k == 1:
            labels = np.zeros(len(valid), dtype=int)
        else:
            labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(coords)

        slots: List[PositionSlot] = []
        for cid in range(k):
            members = [t for t, lab in zip(valid, labels) if lab == cid]
            xs = [t.mean_position[0] for t in members]
            ys = [t.mean_position[1] for t in members]
            frames = sum(len(t.detections) for t in members)
            slots.append(
                PositionSlot(
                    cluster_id=cid,
                    mean_x=float(np.mean(xs)),
                    mean_y=float(np.mean(ys)),
                    track_count=len(members),
                    detection_frames=frames,
                )
            )
        slots.sort(key=lambda s: s.mean_x)
        return slots

    def cluster_voices(
        self,
        segments: List[SpeechSegment],
        n_speakers: int = 2,
    ) -> Tuple[List[VoiceProfile], np.ndarray]:
        """
        对语音段 embedding 聚类，返回 VoiceProfile 列表及每段所属 voice_id。
        """
        valid = [(i, s) for i, s in enumerate(segments) if s.speaker_embedding is not None]
        if not valid:
            return [], np.full(len(segments), -1, dtype=int)

        indices, segs = zip(*valid)
        embs = np.stack([l2_normalize(s.speaker_embedding) for s in segs])
        n_speakers = min(n_speakers, len(embs))

        if n_speakers <= 1:
            labels = np.zeros(len(embs), dtype=int)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embs)

        centroids: Dict[int, np.ndarray] = {}
        for vid in range(n_speakers):
            mask = labels == vid
            if mask.any():
                centroids[vid] = l2_normalize(embs[mask].mean(axis=0))

        segment_labels = np.full(len(segments), -1, dtype=int)
        for idx, lab in zip(indices, labels):
            segment_labels[idx] = int(lab)

        profiles: List[VoiceProfile] = []
        for vid in range(n_speakers):
            mask = segment_labels == vid
            dur = sum(segments[i].duration for i in range(len(segments)) if mask[i])
            profiles.append(
                VoiceProfile(
                    voice_id=vid,
                    label=f"Voice_{vid}",
                    segment_count=int(mask.sum()),
                    total_duration=float(dur),
                    centroid=centroids.get(vid),
                )
            )
        profiles.sort(key=lambda v: -v.total_duration)
        return profiles, segment_labels

    def label_segments_with_enrollment(
        self,
        segments: List[SpeechSegment],
        speaker_extractor: SpeakerEmbeddingExtractor,
    ) -> Tuple[List[VoiceProfile], np.ndarray]:
        """用预注册声纹库 (s1/s2/s3) 标注每个语音段，替代无监督聚类。"""
        if not speaker_extractor.enrolled_speakers:
            return [], np.full(len(segments), -1, dtype=int)

        name_to_id = {name: i for i, name in enumerate(sorted(speaker_extractor.enrolled_speakers.keys()))}
        segment_labels = np.full(len(segments), -1, dtype=int)
        stats: Dict[str, list] = defaultdict(list)

        for i, seg in enumerate(segments):
            if seg.speaker_embedding is None:
                continue
            name, score = speaker_extractor.match_speaker_best(seg.speaker_embedding)
            if name is None:
                continue
            segment_labels[i] = name_to_id[name]
            stats[name].append((seg.duration, score))

        profiles: List[VoiceProfile] = []
        for name in sorted(speaker_extractor.enrolled_speakers.keys()):
            vid = name_to_id[name]
            mask = segment_labels == vid
            dur = sum(segments[i].duration for i in range(len(segments)) if mask[i])
            profiles.append(
                VoiceProfile(
                    voice_id=vid,
                    label=name.upper(),
                    segment_count=int(mask.sum()),
                    total_duration=float(dur),
                    centroid=speaker_extractor.enrolled_speakers[name],
                )
            )
        profiles.sort(key=lambda v: -v.total_duration)
        logger.info(
            "预注册声纹标注: %s",
            ", ".join(f"{p.label}={p.segment_count}段/{p.total_duration:.1f}s" for p in profiles),
        )
        return profiles, segment_labels

    def _cluster_to_tracks(self, tracks: Dict[int, FaceTrack], n_slots: int = 3) -> Dict[int, List[int]]:
        """将轨迹按 KMeans 位置槽位分组。"""
        from sklearn.cluster import KMeans

        ratio = float(getattr(self.config, "track_face_size_min_ratio_to_max", 0.25) or 0.0)
        valid = [(tid, t) for tid, t in tracks.items() if len(t.detections) >= 3]
        if ratio > 0 and valid:
            max_area = max(t.mean_bbox_area for _, t in valid)
            if max_area > 1e-6:
                valid = [(tid, t) for tid, t in valid if t.mean_bbox_area >= ratio * max_area]
        if not valid:
            return {}

        tids, track_list = zip(*valid)
        coords = np.array([t.mean_position for t in track_list])
        k = min(n_slots, len(coords))
        if k == 1:
            labels = np.zeros(len(valid), dtype=int)
        else:
            labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(coords)

        mapping: Dict[int, List[int]] = defaultdict(list)
        for tid, lab in zip(tids, labels):
            mapping[int(lab)].append(int(tid))
        return dict(mapping)

    def compute_lip_by_cluster(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        start_time: float,
        end_time: float,
        fps: float,
        n_slots: int = 3,
    ) -> Dict[int, float]:
        """某语音段内各位置簇的嘴动活跃度。"""
        cluster_tracks = self._cluster_to_tracks(tracks, n_slots=n_slots)
        if not cluster_tracks:
            return {}

        activities = self.lip_analyzer.compute_all_activities(
            video_path, tracks, cluster_tracks, start_time, end_time, fps
        )
        return activities

    def build_cooccurrence(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segments: List[SpeechSegment],
        segment_voice_labels: np.ndarray,
        fps: float,
        n_positions: int = 3,
    ) -> Tuple[Dict[int, Dict[int, float]], List[dict]]:
        """
        累加 cooccurrence[voice_id][cluster_id] = 加权嘴动证据。
        权重 = 段时长 × 嘴动活跃度。
        """
        cooc: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        details: List[dict] = []

        cluster_ids = list(range(n_positions))
        if not cluster_ids:
            return {}, details

        for i, seg in enumerate(segments):
            vid = int(segment_voice_labels[i])
            if vid < 0:
                continue

            lip_by_cluster = self.compute_lip_by_cluster(
                video_path, tracks, seg.start_time, seg.end_time, fps, n_slots=n_positions
            )
            if not lip_by_cluster:
                continue

            best_cluster = max(lip_by_cluster, key=lip_by_cluster.get)
            best_lip = lip_by_cluster[best_cluster]
            weight = seg.duration

            for cid, lip_act in lip_by_cluster.items():
                cooc[vid][cid] += lip_act * weight

            details.append(
                {
                    "start_time": round(seg.start_time, 2),
                    "end_time": round(seg.end_time, 2),
                    "voice_id": vid,
                    "lip_by_cluster": {int(k): round(v, 5) for k, v in lip_by_cluster.items()},
                    "best_lip_cluster": int(best_cluster),
                    "best_lip_score": round(best_lip, 5),
                }
            )

        return dict(cooc), details

    @staticmethod
    def assign_positions_to_voices(
        positions: List[PositionSlot],
        voices: List[VoiceProfile],
        cooc: Dict[int, Dict[int, float]],
    ) -> List[PositionSpeakerMapping]:
        """匈牙利算法：声纹簇 → 位置簇 最优匹配。"""
        if not positions or not voices:
            return []

        pos_ids = [p.cluster_id for p in positions]
        voice_ids = [v.voice_id for v in voices]
        cost = np.zeros((len(voice_ids), len(pos_ids)), dtype=np.float64)

        max_val = 0.0
        for vi, vid in enumerate(voice_ids):
            for pi, cid in enumerate(pos_ids):
                val = cooc.get(vid, {}).get(cid, 0.0)
                cost[vi, pi] = -val
                max_val = max(max_val, val)

        if max_val < 1e-8:
            logger.warning("嘴动-声纹共现矩阵几乎为空，匹配不可靠")

        row_ind, col_ind = linear_sum_assignment(cost)
        mappings: List[PositionSpeakerMapping] = []
        voice_map = {v.voice_id: v for v in voices}
        pos_map = {p.cluster_id: p for p in positions}

        for vi, pi in zip(row_ind, col_ind):
            vid = voice_ids[vi]
            cid = pos_ids[pi]
            score = -cost[vi, pi]
            lip_ev = score
            voice_prof = voice_map[vid]
            mappings.append(
                PositionSpeakerMapping(
                    position=pos_map[cid],
                    voice=voice_prof,
                    association_score=float(score),
                    lip_evidence=float(lip_ev),
                    voice_evidence=voice_prof.total_duration,
                )
            )
        mappings.sort(key=lambda m: m.position.mean_x)
        return mappings

    def analyze(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        segments: List[SpeechSegment],
        fps: float,
        n_speakers: int = 3,
        n_positions: int = 3,
        speaker_names: Optional[List[str]] = None,
        enroll_dir: Optional[Path] = None,
    ) -> MappingResult:
        """完整分析流程。"""
        positions = self.extract_position_slots(tracks, n_slots=n_positions)

        if enroll_dir:
            self.speaker_extractor.enroll_from_directory(Path(enroll_dir))
            voices, seg_labels = self.label_segments_with_enrollment(segments, self.speaker_extractor)
        else:
            voices, seg_labels = self.cluster_voices(segments, n_speakers=n_speakers)

        if speaker_names and not enroll_dir:
            for i, name in enumerate(speaker_names[: len(voices)]):
                voices[i].label = name

        cooc, details = self.build_cooccurrence(
            video_path, tracks, segments, seg_labels, fps, n_positions=n_positions
        )
        mappings = self.assign_positions_to_voices(positions, voices, cooc)

        return MappingResult(
            positions=positions,
            voices=voices,
            mappings=mappings,
            cooccurrence={
                int(vid): {int(cid): float(val) for cid, val in clusters.items()}
                for vid, clusters in cooc.items()
            },
            segment_details=details,
        )

    @staticmethod
    def result_to_dict(result: MappingResult) -> dict:
        """序列化为 JSON 友好格式。"""
        return {
            "positions": [
                {
                    "cluster_id": p.cluster_id,
                    "mean_x": round(p.mean_x, 4),
                    "mean_y": round(p.mean_y, 4),
                    "screen_position": PositionSpeakerMapper._describe_position(p.mean_x, p.mean_y),
                    "track_count": p.track_count,
                    "detection_frames": p.detection_frames,
                }
                for p in result.positions
            ],
            "voices": [
                {
                    "voice_id": v.voice_id,
                    "label": v.label,
                    "segment_count": v.segment_count,
                    "total_duration_sec": round(v.total_duration, 2),
                }
                for v in result.voices
            ],
            "position_to_speaker": [
                {
                    "cluster_id": m.position.cluster_id,
                    "mean_x": round(m.position.mean_x, 4),
                    "mean_y": round(m.position.mean_y, 4),
                    "screen_position": PositionSpeakerMapper._describe_position(
                        m.position.mean_x, m.position.mean_y
                    ),
                    "speaker": m.voice.label,
                    "voice_id": m.voice.voice_id,
                    "association_score": round(m.association_score, 4),
                    "voice_duration_sec": round(m.voice.total_duration, 2),
                }
                for m in result.mappings
            ],
            "cooccurrence_matrix": result.cooccurrence,
            "segment_lip_evidence": result.segment_details,
        }

    @staticmethod
    def _describe_position(x: float, y: float) -> str:
        h = "左" if x < 0.35 else ("右" if x > 0.65 else "中")
        v = "上" if y < 0.35 else ("下" if y > 0.65 else "中")
        return f"{h}{v}"
