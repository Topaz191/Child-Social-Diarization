"""语音活动检测：Silero + embedding 能量/稳定性 + 嘴动 OR 融合。"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from csd.core.config import ASDConfig
from csd.perception.face_tracker import FaceTrack
from csd.perception.lip_motion import LipMotionAnalyzer
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.core.utils import cosine_similarity, merge_intervals, merge_short_segments, union_intervals
from csd.perception.vad_speaker import SpeakerEmbeddingExtractor, VADProcessor

logger = logging.getLogger(__name__)

SpeechSegmentTimes = List[Tuple[float, float]]


class SpeechActivityDetector:
    """多策略「有人说话 / 无人说话」检测。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self.silero = VADProcessor(config)
        self.speaker_extractor = SpeakerEmbeddingExtractor(config)
        self.lip_analyzer = LipMotionAnalyzer(config)
        self.mapper = PositionSpeakerMapper(config)

    def detect_silero(self, wav_path: Path) -> SpeechSegmentTimes:
        return self.silero.detect(wav_path)

    def detect_embedding(
        self,
        wav_path: Path,
        threshold: Optional[float] = None,
        window_size: Optional[float] = None,
        hop_size: Optional[float] = None,
    ) -> SpeechSegmentTimes:
        """滑动窗口 embedding + 能量/稳定性判 speech。"""
        import torch
        import torchaudio

        threshold = threshold if threshold is not None else self.config.embed_vad_threshold
        window_size = window_size if window_size is not None else self.config.embed_vad_window
        hop_size = hop_size if hop_size is not None else self.config.embed_vad_hop
        min_dur = self.config.min_speech_duration

        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.config.sample_rate)(waveform)
            sr = self.config.sample_rate

        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        if waveform.shape[1] < window_samples:
            return []

        embeddings = []
        timestamps = []
        for start in range(0, waveform.shape[1] - window_samples + 1, hop_samples):
            segment = waveform[:, start : start + window_samples]
            emb = self.speaker_extractor.extract_from_waveform(segment)
            embeddings.append(emb)
            timestamps.append(start / sr)

        if not embeddings:
            return []

        emb_arr = np.asarray(embeddings, dtype=np.float32)
        is_speech = self._speech_mask_from_embeddings(emb_arr, threshold)

        if self.config.embed_vad_enroll_gate and self.speaker_extractor.enrolled_speakers:
            refs = list(self.speaker_extractor.enrolled_speakers.values())
            for i, emb in enumerate(emb_arr):
                if not is_speech[i]:
                    continue
                best = max(cosine_similarity(emb, ref) for ref in refs)
                if best < self.config.embed_vad_enroll_thresh:
                    is_speech[i] = False
            logger.info(
                "Embedding-VAD 声纹门控: thresh=%.2f, 保留 %d/%d 窗",
                self.config.embed_vad_enroll_thresh,
                int(is_speech.sum()),
                len(is_speech),
            )

        segments = self._mask_to_segments(timestamps, is_speech, window_size, hop_size)
        segments = merge_short_segments(segments, min_dur)
        logger.info(
            "Embedding-VAD: threshold=%.2f, window=%.1fs → %d 段",
            threshold,
            window_size,
            len(segments),
        )
        return segments

    @staticmethod
    def _speech_mask_from_embeddings(
        embeddings: np.ndarray,
        threshold: float,
        stability_window: int = 3,
    ) -> np.ndarray:
        energies = np.linalg.norm(embeddings, axis=1)
        max_e = float(energies.max()) if energies.max() > 0 else 1.0
        norm_e = energies / max_e

        stability = np.zeros(len(embeddings), dtype=np.float32)
        for i in range(1, len(embeddings)):
            v1 = embeddings[i - 1] / (np.linalg.norm(embeddings[i - 1]) + 1e-8)
            v2 = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
            stability[i] = float(np.dot(v1, v2))

        half = stability_window // 2
        smoothed = np.zeros(len(embeddings), dtype=np.float32)
        for i in range(len(embeddings)):
            lo = max(0, i - half)
            hi = min(len(embeddings), i + half + 1)
            smoothed[i] = float(np.mean(stability[lo:hi]))

        combined = 0.6 * norm_e + 0.4 * smoothed
        # 底噪场景下稳定性长期偏高，需同时满足最低能量
        energy_floor = max(0.15, float(np.percentile(norm_e, 40)))
        return (norm_e >= energy_floor) & (combined > threshold)

    @staticmethod
    def _mask_to_segments(
        timestamps: List[float],
        is_speech: np.ndarray,
        window_size: float,
        hop_size: float,
    ) -> SpeechSegmentTimes:
        segments: SpeechSegmentTimes = []
        i = 0
        n = len(timestamps)
        while i < n:
            if not is_speech[i]:
                i += 1
                continue
            start = timestamps[i]
            j = i + 1
            while j < n and is_speech[j]:
                j += 1
            end = timestamps[j - 1] + window_size
            segments.append((start, end))
            i = j
        return merge_intervals(segments, gap_merge=hop_size)

    def detect_lip(
        self,
        video_path: str,
        tracks: Dict[int, FaceTrack],
        fps: float,
        threshold: Optional[float] = None,
        hop_size: Optional[float] = None,
    ) -> SpeechSegmentTimes:
        """单次遍历视频，按位置簇统计嘴动活跃度 → speech 段。"""
        threshold = threshold if threshold is not None else self.config.lip_speech_thresh
        hop_size = hop_size if hop_size is not None else self.config.embed_vad_hop
        min_dur = self.config.min_speech_duration

        cluster_tracks = self.mapper._cluster_to_tracks(tracks, n_slots=3)
        if not cluster_tracks:
            logger.warning("无有效人脸轨迹，跳过嘴动 VAD")
            return []

        track_to_cluster: Dict[int, int] = {}
        for cid, tids in cluster_tracks.items():
            for tid in tids:
                track_to_cluster[tid] = cid

        cluster_mars: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, self.config.frame_skip)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                t = frame_idx / fps
                for tid, track in tracks.items():
                    if frame_idx not in track.detections:
                        continue
                    cid = track_to_cluster.get(tid)
                    if cid is None:
                        continue
                    bbox = track.detections[frame_idx].bbox
                    mar = self.lip_analyzer.compute_mar_in_roi(frame, bbox)
                    if mar is not None:
                        cluster_mars[cid].append((t, mar))
            frame_idx += 1
        cap.release()

        duration = frame_idx / fps if frame_idx > 0 else 0.0
        if duration <= 0:
            return []

        active_bins: List[Tuple[float, float]] = []
        t = 0.0
        while t < duration:
            bin_end = min(t + hop_size, duration)
            max_act = 0.0
            for samples in cluster_mars.values():
                mars = [m for ts, m in samples if t <= ts < bin_end]
                act = LipMotionAnalyzer._activity_from_mar_values(mars)
                max_act = max(max_act, act)
            if max_act >= threshold:
                active_bins.append((t, bin_end))
            t += hop_size

        segments = merge_intervals(active_bins, gap_merge=hop_size)
        segments = merge_short_segments(segments, min_dur)
        logger.info("嘴动-VAD: threshold=%.4f → %d 段", threshold, len(segments))
        return segments

    def detect_lip_fullframe(
        self,
        video_path: str,
        fps: float = 25.0,
        threshold: Optional[float] = None,
        hop_size: Optional[float] = None,
    ) -> SpeechSegmentTimes:
        """不依赖人脸跟踪：全帧 Face Mesh，取所有脸中最大嘴动活跃度。"""
        threshold = threshold if threshold is not None else self.config.lip_speech_thresh
        hop_size = hop_size if hop_size is not None else self.config.embed_vad_hop
        min_dur = self.config.min_speech_duration

        self.lip_analyzer._load_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, self.config.frame_skip)
        frame_idx = 0
        time_mars: List[Tuple[float, float]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                t = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.lip_analyzer._face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    h, w = frame.shape[:2]
                    best_mar = 0.0
                    for face_lm in results.multi_face_landmarks:
                        mar = LipMotionAnalyzer._compute_mar(face_lm.landmark, w, h)
                        best_mar = max(best_mar, mar)
                    time_mars.append((t, best_mar))
            frame_idx += 1
        cap.release()

        duration = frame_idx / fps if frame_idx > 0 else 0.0
        if duration <= 0:
            return []

        active_bins: List[Tuple[float, float]] = []
        t = 0.0
        while t < duration:
            bin_end = min(t + hop_size, duration)
            mars = [m for ts, m in time_mars if t <= ts < bin_end]
            act = LipMotionAnalyzer._activity_from_mar_values(mars)
            if act >= threshold:
                active_bins.append((t, bin_end))
            t += hop_size

        segments = merge_intervals(active_bins, gap_merge=hop_size)
        segments = merge_short_segments(segments, min_dur)
        logger.info("全帧嘴动-VAD: threshold=%.4f → %d 段", threshold, len(segments))
        return segments

    def detect(
        self,
        wav_path: Path,
        mode: Optional[str] = None,
        video_path: Optional[str] = None,
        tracks: Optional[Dict[int, FaceTrack]] = None,
        fps: float = 25.0,
    ) -> Tuple[SpeechSegmentTimes, dict]:
        """
        按 vad_mode 返回语音段时间戳及各路明细。
        mode: silero | union | union_lip
        """
        mode = (mode or self.config.vad_mode).lower()
        details: dict = {"mode": mode}

        silero_segs: SpeechSegmentTimes = []
        embed_segs: SpeechSegmentTimes = []
        lip_segs: SpeechSegmentTimes = []

        if mode in ("silero", "union", "union_lip"):
            silero_segs = self.detect_silero(wav_path)
            details["silero_count"] = len(silero_segs)

        if mode in ("union", "union_lip"):
            embed_segs = self.detect_embedding(wav_path)
            details["embed_count"] = len(embed_segs)

        if mode == "union_lip":
            if video_path and tracks:
                try:
                    lip_segs = self.detect_lip(video_path, tracks, fps)
                    details["lip_source"] = "tracks"
                except Exception as exc:
                    logger.warning("轨迹嘴动 VAD 失败: %s", exc)
            if not lip_segs and video_path:
                try:
                    lip_segs = self.detect_lip_fullframe(video_path, fps)
                    details["lip_source"] = "fullframe"
                except Exception as exc:
                    logger.warning("全帧嘴动 VAD 失败，union_lip 退化为 union: %s", exc)
            details["lip_count"] = len(lip_segs)

        if mode == "silero":
            final = silero_segs
        elif mode == "union":
            final = union_intervals(silero_segs, embed_segs)
            final = merge_short_segments(final, self.config.min_speech_duration)
        elif mode == "union_lip":
            parts = [silero_segs, embed_segs]
            if lip_segs:
                parts.append(lip_segs)
            final = union_intervals(*parts)
            final = merge_short_segments(final, self.config.min_speech_duration)
        else:
            raise ValueError(f"未知 vad_mode: {mode}")

        details["final_count"] = len(final)
        details["silero_sec"] = sum(e - s for s, e in silero_segs)
        details["embed_sec"] = sum(e - s for s, e in embed_segs)
        details["lip_sec"] = sum(e - s for s, e in lip_segs)
        details["final_sec"] = sum(e - s for s, e in final)
        return final, details
