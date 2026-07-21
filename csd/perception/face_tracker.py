"""模块1：视频帧人脸检测、位置跟踪与位置聚类。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from csd.core.config import ASDConfig
from csd.core.backends import resolve_face_backend
from csd.core.model_assets import ensure_yolov8_face_weights
from csd.core.utils import bbox_center_norm, bbox_iou

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """单帧单个人脸检测结果。"""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    norm_pos: Tuple[float, float]  # 归一化中心坐标
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None  # 5 点或 106 点


@dataclass
class FaceTrack:
    """跨帧人脸轨迹。"""

    track_id: int
    detections: Dict[int, FaceDetection] = field(default_factory=dict)  # frame_idx -> det
    lost_count: int = 0
    cluster_id: Optional[int] = None  # DBSCAN 位置簇 ID

    @property
    def mean_position(self) -> Tuple[float, float]:
        if not self.detections:
            return 0.5, 0.5
        xs = [d.norm_pos[0] for d in self.detections.values()]
        ys = [d.norm_pos[1] for d in self.detections.values()]
        return float(np.mean(xs)), float(np.mean(ys))

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        embs = [d.embedding for d in self.detections.values() if d.embedding is not None]
        if not embs:
            return None
        return np.mean(embs, axis=0)

    def last_bbox(self) -> Optional[np.ndarray]:
        if not self.detections:
            return None
        last_frame = max(self.detections.keys())
        return self.detections[last_frame].bbox.copy()

    @property
    def mean_bbox_area(self) -> float:
        if not self.detections:
            return 0.0
        areas = []
        for d in self.detections.values():
            x1, y1, x2, y2 = d.bbox
            areas.append(float(max(0.0, x2 - x1) * max(0.0, y2 - y1)))
        return float(np.mean(areas)) if areas else 0.0

    @staticmethod
    def bbox_area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceTracker:
    """人脸检测 + IoU/位置匈牙利跟踪 + DBSCAN 位置聚类。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self._backend = resolve_face_backend(config.face_backend)
        self._app = None
        self._mp_detector = None
        self._yolo_model = None
        self.tracks: Dict[int, FaceTrack] = {}
        self._next_track_id = 0
        self.fps: float = 25.0
        self.frame_size: Tuple[int, int] = (0, 0)

    def _load_model(self) -> None:
        if self._backend == "yolov8face":
            self._load_yolov8face()
        elif self._backend == "retinaface":
            self._load_retinaface()
        elif self._backend == "insightface":
            self._load_insightface()
        else:
            self._load_mediapipe()

    def _load_yolov8face(self) -> None:
        if self._yolo_model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("请安装 ultralytics: pip install ultralytics") from e

        model_path = self.config.yolov8_face_model
        if not model_path:
            try:
                model_path = str(ensure_yolov8_face_weights(self.config.model_cache_dir))
            except RuntimeError as exc:
                logger.warning("YOLOv8-face 权重不可用 (%s)，回退 MediaPipe", exc)
                self._backend = "mediapipe"
                self._load_mediapipe()
                return
        logger.info("加载 YOLOv8-face: %s", model_path)
        self._yolo_model = YOLO(model_path)

    def _load_retinaface(self) -> None:
        """InsightFace SCRFD/RetinaFace 检测专用（无 embedding）。"""
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.warning("未安装 insightface，RetinaFace 回退 MediaPipe")
            self._backend = "mediapipe"
            self._load_mediapipe()
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device_id = 0 if self.config.resolve_device() == "cuda" else -1
        if device_id < 0:
            providers = ["CPUExecutionProvider"]

        logger.info("加载 RetinaFace/SCRFD 检测: %s", self.config.insightface_model)
        self._app = FaceAnalysis(
            name=self.config.insightface_model,
            root=str(self.config.model_cache_dir),
            providers=providers,
            allowed_modules=["detection"],
        )
        self._app.prepare(
            ctx_id=device_id,
            det_size=self.config.face_det_size,
            det_thresh=self.config.face_det_thresh,
        )

    def _load_insightface(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "请安装 insightface: pip install insightface onnxruntime"
            ) from e

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device_id = 0 if self.config.resolve_device() == "cuda" else -1
        if device_id < 0:
            providers = ["CPUExecutionProvider"]

        logger.info("加载 InsightFace 模型: %s", self.config.insightface_model)
        self._app = FaceAnalysis(
            name=self.config.insightface_model,
            root=str(self.config.model_cache_dir),
            providers=providers,
        )
        self._app.prepare(
            ctx_id=device_id,
            det_size=self.config.face_det_size,
            det_thresh=self.config.face_det_thresh,
        )

    def _load_mediapipe(self) -> None:
        if self._mp_detector is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("请安装 mediapipe: pip install mediapipe") from e

        logger.info("加载 MediaPipe FaceDetection")
        self._mp_face_detection = mp.solutions.face_detection
        self._mp_detector = self._mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.config.face_det_thresh,
        )

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> List[FaceDetection]:
        """检测单帧人脸并返回检测结果。"""
        self._load_model()
        if self._backend == "yolov8face" and self._yolo_model is not None:
            dets = self._detect_yolov8face(frame)
        elif self._backend in ("retinaface", "insightface") and self._app is not None:
            dets = self._detect_insightface(frame)
        elif self._backend == "mediapipe" or self._mp_detector is not None:
            dets = self._detect_mediapipe(frame)
        else:
            dets = []
        return self.filter_by_max_face_scale(dets)

    def filter_by_max_face_scale(
        self,
        detections: List[FaceDetection],
        min_ratio: Optional[float] = None,
    ) -> List[FaceDetection]:
        """只保留面积达到当帧最大脸 min_ratio 倍以上的检测（滤远处小人脸）。"""
        if len(detections) <= 1:
            return detections
        ratio = (
            self.config.face_size_min_ratio_to_max
            if min_ratio is None
            else float(min_ratio)
        )
        if ratio <= 0:
            return detections
        areas = [FaceTrack.bbox_area(d.bbox) for d in detections]
        max_area = max(areas)
        if max_area <= 1e-6:
            return detections
        kept = [d for d, a in zip(detections, areas) if a >= ratio * max_area]
        if len(kept) < len(detections):
            logger.debug(
                "脸尺寸过滤: %d -> %d (max_area=%.0f, ratio>=%.2f)",
                len(detections),
                len(kept),
                max_area,
                ratio,
            )
        return kept

    def filter_tracks_by_face_scale(
        self,
        tracks: Optional[Dict[int, FaceTrack]] = None,
        min_ratio: Optional[float] = None,
    ) -> Dict[int, FaceTrack]:
        """轨迹级过滤：丢掉平均脸远小于主组最大脸的轨迹。"""
        src = tracks if tracks is not None else self.tracks
        ratio = (
            self.config.track_face_size_min_ratio_to_max
            if min_ratio is None
            else float(min_ratio)
        )
        if ratio <= 0 or not src:
            return dict(src)
        areas = {tid: t.mean_bbox_area for tid, t in src.items() if len(t.detections) >= 1}
        if not areas:
            return dict(src)
        max_area = max(areas.values())
        if max_area <= 1e-6:
            return dict(src)
        kept = {
            tid: t
            for tid, t in src.items()
            if areas.get(tid, 0.0) >= ratio * max_area
        }
        dropped = len(src) - len(kept)
        if dropped > 0:
            logger.info(
                "轨迹脸尺寸过滤: %d -> %d (丢掉远处/过小脸 %d 条, ratio>=%.2f)",
                len(src),
                len(kept),
                dropped,
                ratio,
            )
        return kept

    def _detect_yolov8face(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        results = self._yolo_model.predict(
            frame,
            verbose=False,
            conf=self.config.face_det_thresh,
            iou=0.45,
            imgsz=max(self.config.face_det_size),
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: List[FaceDetection] = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = np.array(xyxy, dtype=np.float32)
            score = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.5
            norm_pos = bbox_center_norm(bbox, w, h)
            detections.append(
                FaceDetection(bbox=bbox, score=score, norm_pos=norm_pos, embedding=None)
            )
        return detections

    def _detect_insightface(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        faces = self._app.get(frame)
        detections: List[FaceDetection] = []
        for face in faces:
            bbox = face.bbox.astype(np.float32)
            norm_pos = bbox_center_norm(bbox, w, h)
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    score=float(face.det_score),
                    norm_pos=norm_pos,
                    embedding=face.normed_embedding.copy() if face.normed_embedding is not None else None,
                    landmarks=face.landmark_2d_106.copy() if hasattr(face, "landmark_2d_106") else None,
                )
            )
        return detections

    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        detections: List[FaceDetection] = []
        if not results.detections:
            return detections

        for det_mp in results.detections:
            bb = det_mp.location_data.relative_bounding_box
            x1 = max(0.0, bb.xmin * w)
            y1 = max(0.0, bb.ymin * h)
            x2 = min(float(w), (bb.xmin + bb.width) * w)
            y2 = min(float(h), (bb.ymin + bb.height) * h)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            norm_pos = bbox_center_norm(bbox, w, h)
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    score=float(det_mp.score[0]) if det_mp.score else 0.5,
                    norm_pos=norm_pos,
                    embedding=None,
                )
            )
        return detections

    def _match_cost(
        self,
        det: FaceDetection,
        track: FaceTrack,
        frame_w: int,
        frame_h: int,
    ) -> float:
        last_bbox = track.last_bbox()
        if last_bbox is None:
            return 1e6
        iou = bbox_iou(det.bbox, last_bbox)
        if iou < self.config.track_iou_thresh:
            # 位置距离作为备选（镜头固定时位置稳定）
            tx, ty = track.mean_position
            dx = (det.norm_pos[0] - tx) * frame_w
            dy = (det.norm_pos[1] - ty) * frame_h
            dist = np.sqrt(dx * dx + dy * dy) / max(frame_w, frame_h)
            if dist > 0.15:
                return 1e6
            return dist
        return 1.0 - iou

    def update_tracks(
        self,
        detections: List[FaceDetection],
        frame_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> List[FaceTrack]:
        """用匈牙利算法将当前帧检测与已有轨迹关联。"""
        active_tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if t.lost_count <= self.config.track_max_lost_frames
        }

        if not detections:
            for track in active_tracks.values():
                track.lost_count += 1
            return list(active_tracks.values())

        if not active_tracks:
            for det in detections:
                self._create_track(det, frame_idx)
            return list(self.tracks.values())

        track_ids = list(active_tracks.keys())
        cost = np.zeros((len(detections), len(track_ids)), dtype=np.float64)
        for i, det in enumerate(detections):
            for j, tid in enumerate(track_ids):
                cost[i, j] = self._match_cost(det, active_tracks[tid], frame_w, frame_h)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dets = set()
        matched_tracks = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= 1e5:
                continue
            tid = track_ids[j]
            self.tracks[tid].detections[frame_idx] = detections[i]
            self.tracks[tid].lost_count = 0
            matched_dets.add(i)
            matched_tracks.add(tid)

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det, frame_idx)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].lost_count += 1

        return [t for t in self.tracks.values() if t.lost_count <= self.config.track_max_lost_frames]

    def _create_track(self, det: FaceDetection, frame_idx: int) -> FaceTrack:
        track = FaceTrack(track_id=self._next_track_id)
        track.detections[frame_idx] = det
        self.tracks[self._next_track_id] = track
        self._next_track_id += 1
        return track

    def cluster_by_position(self) -> None:
        """对轨迹均值位置做 DBSCAN，区分不同座位/区域。"""
        valid_tracks = [t for t in self.tracks.values() if len(t.detections) >= 3]
        if len(valid_tracks) < 2:
            for t in valid_tracks:
                t.cluster_id = 0
            return

        positions = np.array([t.mean_position for t in valid_tracks])
        clustering = DBSCAN(
            eps=self.config.position_cluster_eps,
            min_samples=self.config.position_cluster_min_samples,
        ).fit(positions)

        for track, label in zip(valid_tracks, clustering.labels_):
            track.cluster_id = int(label) if label >= 0 else track.track_id

        logger.info(
            "位置聚类完成: %d 条轨迹, %d 个簇",
            len(valid_tracks),
            len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0),
        )

    def process_video(self, video_path: str) -> Dict[int, List[FaceTrack]]:
        """处理整个视频，返回 frame_idx -> 活跃轨迹列表。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w, h)

        frame_results: Dict[int, List[FaceTrack]] = {}
        frame_idx = 0

        logger.info("开始人脸检测与跟踪 [%s]: %s (%.1f fps, %dx%d)", self._backend, video_path, self.fps, w, h)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.config.frame_skip == 0:
                detections = self.detect_frame(frame, frame_idx)
                active = self.update_tracks(detections, frame_idx, w, h)
                frame_results[frame_idx] = active

            frame_idx += 1
            if frame_idx % 500 == 0:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or frame_idx
                logger.info("人脸检测进度: %d / %d 帧 (%.1f%%)", frame_idx, total, 100.0 * frame_idx / total)

        cap.release()
        self.tracks = self.filter_tracks_by_face_scale(self.tracks)
        self.cluster_by_position()
        logger.info("人脸跟踪完成: 共 %d 条轨迹", len(self.tracks))
        return frame_results
