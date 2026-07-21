"""模块2：人脸识别特征提取与身份关联。"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack
from csd.core.utils import cosine_similarity, l2_normalize

logger = logging.getLogger(__name__)


class FaceIdentityManager:
    """
    将跨帧 track 关联为稳定身份 ID。
    结合位置聚类 + 人脸 embedding 余弦相似度。
    """

    def __init__(self, config: ASDConfig):
        self.config = config
        self.identity_embeddings: Dict[int, np.ndarray] = {}  # identity_id -> mean emb
        self.track_to_identity: Dict[int, int] = {}  # track_id -> identity_id
        self._next_identity_id = 0

    def build_identities(self, tracks: Dict[int, FaceTrack]) -> None:
        """根据轨迹 embedding 与位置簇合并身份。"""
        track_list = list(tracks.values())
        if not track_list:
            return

        has_embedding = any(
            d.embedding is not None
            for t in track_list
            for d in t.detections.values()
        )
        if not has_embedding:
            self.build_identities_by_position_only(tracks)
            return

        # 先按 cluster_id 分组
        clusters: Dict[int, List[FaceTrack]] = defaultdict(list)
        for t in track_list:
            cid = t.cluster_id if t.cluster_id is not None else t.track_id
            clusters[cid].append(t)

        for cluster_tracks in clusters.values():
            self._merge_tracks_in_cluster(cluster_tracks)

        n_id = len(set(self.track_to_identity.values()))
        logger.info("身份关联完成: %d 个身份 (%d 条轨迹)", n_id, len(self.track_to_identity))

    def build_identities_by_position_only(self, tracks: Dict[int, FaceTrack]) -> None:
        """无 face embedding 时：同一位置簇内所有轨迹合并为一个身份。"""
        cluster_to_identity: Dict[int, int] = {}
        for track in tracks.values():
            cid = track.cluster_id if track.cluster_id is not None and track.cluster_id >= 0 else None
            if cid is None:
                continue
            if cid not in cluster_to_identity:
                cluster_to_identity[cid] = len(cluster_to_identity)
            self.track_to_identity[track.track_id] = cluster_to_identity[cid]

        for track in tracks.values():
            if track.track_id not in self.track_to_identity:
                self._assign_new_identity(track)

        n_id = len(set(self.track_to_identity.values()))
        logger.info("位置聚类身份: %d 个身份 (%d 条轨迹)", n_id, len(self.track_to_identity))

    def _merge_tracks_in_cluster(self, tracks: List[FaceTrack]) -> None:
        """同一位置簇内，用 embedding 进一步区分不同人。"""
        assigned: List[FaceTrack] = []

        for track in sorted(tracks, key=lambda t: len(t.detections), reverse=True):
            emb = track.mean_embedding
            if emb is None:
                self._assign_new_identity(track)
                assigned.append(track)
                continue

            emb = l2_normalize(emb)
            best_id = None
            best_sim = -1.0

            for identity_id, ref_emb in self.identity_embeddings.items():
                sim = cosine_similarity(emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = identity_id

            if best_id is not None and best_sim >= self.config.face_match_thresh:
                self.track_to_identity[track.track_id] = best_id
                # 更新身份模板（滑动平均）
                old = self.identity_embeddings[best_id]
                self.identity_embeddings[best_id] = l2_normalize(0.7 * old + 0.3 * emb)
            else:
                self._assign_new_identity(track, emb)
            assigned.append(track)

    def _assign_new_identity(self, track: FaceTrack, emb: Optional[np.ndarray] = None) -> int:
        identity_id = self._next_identity_id
        self._next_identity_id += 1
        self.track_to_identity[track.track_id] = identity_id
        if emb is not None:
            self.identity_embeddings[identity_id] = l2_normalize(emb)
        elif track.mean_embedding is not None:
            self.identity_embeddings[identity_id] = l2_normalize(track.mean_embedding)
        return identity_id

    def get_identity_for_track(self, track_id: int) -> Optional[int]:
        return self.track_to_identity.get(track_id)

    def get_identity_position(self, tracks: Dict[int, FaceTrack], identity_id: int) -> Tuple[float, float]:
        """获取某身份的平均归一化位置。"""
        xs, ys = [], []
        for tid, iid in self.track_to_identity.items():
            if iid == identity_id and tid in tracks:
                cx, cy = tracks[tid].mean_position
                xs.append(cx)
                ys.append(cy)
        if not xs:
            return 0.5, 0.5
        return float(np.mean(xs)), float(np.mean(ys))

    def identity_label(self, identity_id: int) -> str:
        return f"Person_{identity_id}"
