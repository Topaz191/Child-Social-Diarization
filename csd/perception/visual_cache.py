"""共享视觉特征缓存：人脸轨迹、唇轮廓、头姿与 yaw 视线注意力。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from csd.core.config import ASDConfig
from csd.data.xianyang import parse_xianyang_video_name
from csd.perception.face_mesh_indices import LIPS, MAR4, index_catalog
from csd.perception.face_tracker import FaceDetection, FaceTrack, FaceTracker
from csd.perception.head_pose import HeadPoseAnalyzer, HeadPoseFrame
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.social.social_attention import SocialAttentionComputer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ELSEWHERE = "__elsewhere__"
DEFAULT_SPEAKERS = ("S1", "S2", "S3")


def session_key_from_video(video: Path) -> str:
    """与 avsync 目录命名对齐：0706_class1_g4_后测。"""
    meta = parse_xianyang_video_name(video.name)
    if not meta:
        return video.stem
    return f"{meta['date']}_class{meta['class_id']}_g{meta['group']}_{meta['phase']}"


def cache_dir_for(video: Path, root: Path) -> Path:
    return Path(root) / session_key_from_video(video)


def tracks_cache_path(session_dir: Path) -> Path:
    return Path(session_dir) / "tracks.npz"


def has_tracks_cache(session_dir: Path) -> bool:
    return tracks_cache_path(session_dir).is_file()


def save_tracks(
    tracks: Dict[int, FaceTrack],
    session_dir: Path,
    *,
    fps: float,
    frame_w: int,
    frame_h: int,
    frame_skip: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """稀疏落盘轨迹，可重建 FaceTrack。"""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    tids: List[int] = []
    frames: List[int] = []
    bboxes: List[List[float]] = []
    scores: List[float] = []
    clusters: List[int] = []
    for tid, tr in sorted(tracks.items()):
        cid = int(tr.cluster_id) if tr.cluster_id is not None else -1
        for fi, det in sorted(tr.detections.items()):
            tids.append(int(tid))
            frames.append(int(fi))
            bboxes.append([float(x) for x in det.bbox.tolist()])
            scores.append(float(det.score))
            clusters.append(cid)
    np.savez_compressed(
        tracks_cache_path(session_dir),
        track_id=np.asarray(tids, dtype=np.int32),
        frame_idx=np.asarray(frames, dtype=np.int32),
        bbox=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4) if bboxes else np.zeros((0, 4), np.float32),
        score=np.asarray(scores, dtype=np.float32),
        cluster_id=np.asarray(clusters, dtype=np.int32),
        fps=np.asarray([float(fps)], dtype=np.float32),
        frame_wh=np.asarray([int(frame_w), int(frame_h)], dtype=np.int32),
        frame_skip=np.asarray([int(frame_skip)], dtype=np.int32),
    )
    meta_path = session_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.update(
        {
            "schema_version": SCHEMA_VERSION,
            "fps": float(fps),
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "frame_skip": int(frame_skip),
            "n_tracks": len(tracks),
            "n_detections": len(tids),
            "has_tracks": True,
        }
    )
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return tracks_cache_path(session_dir)


def load_tracks(session_dir: Path) -> Tuple[Dict[int, FaceTrack], Dict[str, Any]]:
    """从 tracks.npz 重建轨迹；返回 (tracks, info)。"""
    path = tracks_cache_path(session_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    tracks: Dict[int, FaceTrack] = {}
    n = int(data["track_id"].shape[0])
    for i in range(n):
        tid = int(data["track_id"][i])
        fi = int(data["frame_idx"][i])
        bbox = np.asarray(data["bbox"][i], dtype=np.float32)
        score = float(data["score"][i])
        cid_raw = int(data["cluster_id"][i])
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # 归一化中心：若有 frame_wh 用真实尺寸，否则粗略用像素比（下游槽位更依赖相对 x）
        wh = data["frame_wh"] if "frame_wh" in data.files else np.array([1, 1])
        fw = float(wh[0]) if float(wh[0]) > 1 else 1.0
        fh = float(wh[1]) if float(wh[1]) > 1 else 1.0
        det = FaceDetection(
            bbox=bbox,
            score=score,
            norm_pos=(cx / fw, cy / fh),
        )
        if tid not in tracks:
            tracks[tid] = FaceTrack(track_id=tid, cluster_id=None if cid_raw < 0 else cid_raw)
        tracks[tid].detections[fi] = det
        if cid_raw >= 0:
            tracks[tid].cluster_id = cid_raw
    info = {
        "fps": float(data["fps"][0]) if "fps" in data.files else 25.0,
        "frame_w": int(data["frame_wh"][0]) if "frame_wh" in data.files else 0,
        "frame_h": int(data["frame_wh"][1]) if "frame_wh" in data.files else 0,
        "frame_skip": int(data["frame_skip"][0]) if "frame_skip" in data.files else 3,
        "n_tracks": len(tracks),
        "n_detections": n,
    }
    return tracks, info


@dataclass
class VisualCacheBundle:
    """一次加载的缓存视图。"""

    session_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    tracks: Dict[int, FaceTrack] = field(default_factory=dict)
    fps: float = 25.0
    frame_w: int = 0
    frame_h: int = 0
    # mesh 行
    mesh_frame_idx: Optional[np.ndarray] = None
    mesh_slot_id: Optional[np.ndarray] = None
    mesh_speaker: Optional[List[str]] = None
    mesh_bbox: Optional[np.ndarray] = None
    mesh_pose: Optional[np.ndarray] = None  # [N,6] yaw pitch roll mar vis side
    lips_xy: Optional[np.ndarray] = None  # [N,L,2]
    mar4_xy: Optional[np.ndarray] = None  # [N,4,2]
    densified: Optional[np.ndarray] = None
    mesh_xy: Optional[np.ndarray] = None  # optional full
    # attention
    attn_frame_idx: Optional[np.ndarray] = None
    attn: Optional[np.ndarray] = None  # [T,S,S+1]
    attn_speakers: List[str] = field(default_factory=list)


def load_visual_cache(session_dir: Path) -> VisualCacheBundle:
    session_dir = Path(session_dir)
    meta: Dict[str, Any] = {}
    mp = session_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    tracks: Dict[int, FaceTrack] = {}
    fps = float(meta.get("fps", 25.0))
    fw = int(meta.get("frame_w", 0))
    fh = int(meta.get("frame_h", 0))
    if has_tracks_cache(session_dir):
        tracks, info = load_tracks(session_dir)
        fps = float(info.get("fps", fps))
        fw = int(info.get("frame_w", fw))
        fh = int(info.get("frame_h", fh))

    bundle = VisualCacheBundle(session_dir=session_dir, meta=meta, tracks=tracks, fps=fps, frame_w=fw, frame_h=fh)
    mesh_path = session_dir / "mesh.npz"
    if mesh_path.is_file():
        m = np.load(mesh_path, allow_pickle=True)
        bundle.mesh_frame_idx = m["frame_idx"]
        bundle.mesh_slot_id = m["slot_id"]
        bundle.mesh_speaker = [str(x) for x in m["speaker"].tolist()]
        bundle.mesh_bbox = m["bbox"]
        bundle.mesh_pose = m["pose"]
        bundle.lips_xy = m["lips_xy"]
        bundle.mar4_xy = m["mar4_xy"]
        bundle.densified = m["densified"] if "densified" in m.files else None
        if "mesh_xy" in m.files:
            bundle.mesh_xy = m["mesh_xy"]

    attn_path = session_dir / "attention.npz"
    if attn_path.is_file():
        a = np.load(attn_path, allow_pickle=True)
        bundle.attn_frame_idx = a["frame_idx"]
        bundle.attn = a["attn"]
        bundle.attn_speakers = [str(x) for x in a["speakers"].tolist()]
    return bundle


def mesh_rows_to_frame_feature_rows(bundle: VisualCacheBundle) -> List[Dict[str, Any]]:
    """把 mesh 缓存转成 readiness 用的基础行（无 DYN/REL 衍生列）。"""
    if bundle.mesh_frame_idx is None or bundle.mesh_pose is None:
        return []
    rows: List[Dict[str, Any]] = []
    fps = max(float(bundle.fps), 1e-6)
    for i in range(len(bundle.mesh_frame_idx)):
        fi = int(bundle.mesh_frame_idx[i])
        pose = bundle.mesh_pose[i]
        rows.append(
            {
                "frame_idx": fi,
                "t": float(fi) / fps,
                "slot_id": int(bundle.mesh_slot_id[i]) if bundle.mesh_slot_id is not None else -1,
                "speaker": bundle.mesh_speaker[i] if bundle.mesh_speaker else "UNK",
                "yaw": float(pose[0]),
                "pitch": float(pose[1]),
                "roll": float(pose[2]),
                "mouth_opening": float(pose[3]),
                "visibility": float(pose[4]),
                "side_face_weight": float(pose[5]),
            }
        )
    return rows


def _speech_dense_frames(
    intervals: Sequence[Tuple[float, float]],
    fps: float,
    dense_skip: int,
    pad_sec: float,
) -> set:
    out: set = set()
    step = max(1, int(dense_skip))
    for t0, t1 in intervals:
        s = max(0, int((t0 - pad_sec) * fps))
        e = int((t1 + pad_sec) * fps)
        for f in range(s, e + 1, step):
            out.add(f)
    return out


def _landmarks_xy(
    landmarks,
    indices: Sequence[int],
    rw: int,
    rh: int,
    ox: int,
    oy: int,
) -> np.ndarray:
    pts = np.zeros((len(indices), 2), dtype=np.float32)
    for i, idx in enumerate(indices):
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pts[i, 0] = float(lm.x) * rw + ox
        pts[i, 1] = float(lm.y) * rh + oy
    return pts


def _attn_matrix(
    soft: Dict[str, Dict[str, float]],
    speakers: Sequence[str],
) -> np.ndarray:
    """[S, S+1]，最后一列 elsewhere。"""
    s_list = list(speakers)
    n = len(s_list)
    mat = np.zeros((n, n + 1), dtype=np.float32)
    for i, src in enumerate(s_list):
        dst_map = soft.get(src) or {}
        for j, dst in enumerate(s_list):
            if dst == src:
                continue
            mat[i, j] = float(dst_map.get(dst, 0.0))
        mat[i, n] = float(dst_map.get(ELSEWHERE, 0.0))
    return mat


def _write_lips_timeseries(
    session_dir: Path,
    *,
    fps: float,
    frame_idx: np.ndarray,
    speakers: List[str],
    lips_xy: np.ndarray,
    mar: np.ndarray,
    densified: np.ndarray,
) -> Path:
    """长表：便于唇动研究。优先 parquet，失败回退 csv。"""
    rows = []
    for i in range(len(frame_idx)):
        fi = int(frame_idx[i])
        t = float(fi) / max(fps, 1e-6)
        spk = speakers[i]
        m = float(mar[i])
        dens = int(densified[i])
        for pid, (x, y) in enumerate(lips_xy[i]):
            rows.append(
                {
                    "frame_idx": fi,
                    "t": t,
                    "speaker": spk,
                    "point_id": pid,
                    "mesh_idx": int(LIPS[pid]) if pid < len(LIPS) else -1,
                    "x": float(x),
                    "y": float(y),
                    "mar": m,
                    "densified": dens,
                }
            )
    out_parquet = session_dir / "lips_timeseries.parquet"
    out_csv = session_dir / "lips_timeseries.csv"
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        try:
            df.to_parquet(out_parquet, index=False)
            return out_parquet
        except Exception:
            df.to_csv(out_csv, index=False)
            return out_csv
    except Exception:
        # 极简 csv 手写
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("frame_idx,t,speaker,point_id,mesh_idx,x,y,mar,densified\n")
            for r in rows:
                f.write(
                    f"{r['frame_idx']},{r['t']:.4f},{r['speaker']},{r['point_id']},"
                    f"{r['mesh_idx']},{r['x']:.2f},{r['y']:.2f},{r['mar']:.6f},{r['densified']}\n"
                )
        return out_csv


def build_visual_cache_for_video(
    video: Path,
    session_dir: Path,
    *,
    config: Optional[ASDConfig] = None,
    left_to_right: Optional[Sequence[str]] = None,
    speech_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    frame_skip: int = 3,
    dense_skip: int = 1,
    speech_pad_sec: float = 1.0,
    full_mesh: bool = False,
    attention_temperature: float = 15.0,
    tracks: Optional[Dict[int, FaceTrack]] = None,
    fps: Optional[float] = None,
    max_sec: Optional[float] = None,
) -> Path:
    """
    抽取并落盘一场视频的视觉缓存。
    若传入 tracks 则跳过人脸检测；否则现场跟踪。
    """
    video = Path(video)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ASDConfig(frame_skip=max(1, int(frame_skip)))
    cfg.frame_skip = max(1, int(frame_skip))

    if tracks is None:
        tracker = FaceTracker(cfg)
        logger.info("人脸跟踪 → cache: %s", video.name)
        tracker.process_video(str(video), max_sec=max_sec)
        tracks = tracker.tracks
        fps_v = float(tracker.fps)
        fw = int(getattr(tracker, "frame_width", 0) or 0)
        fh = int(getattr(tracker, "frame_height", 0) or 0)
    else:
        fps_v = float(fps or 25.0)
        fw, fh = 0, 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    if fw <= 0:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    if fh <= 0:
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps_v <= 1e-6:
        fps_v = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = int(max_sec * fps_v) if max_sec and max_sec > 0 else total

    order = [str(s).upper() for s in (left_to_right or DEFAULT_SPEAKERS)]
    while len(order) < 3:
        for s in DEFAULT_SPEAKERS:
            if s not in order:
                order.append(s)
                break
    order = order[:3]

    mapper = PositionSpeakerMapper(cfg)
    slots = mapper.extract_position_slots(tracks, n_slots=3)
    slot_to_tracks = mapper._cluster_to_tracks(tracks, n_slots=3)
    if not slots:
        ranked = sorted(
            [(tid, t.mean_position[0], t.mean_position[1]) for tid, t in tracks.items() if len(t.detections) >= 2],
            key=lambda x: x[1],
        )
        slot_positions = {i: (mx, my) for i, (_, mx, my) in enumerate(ranked[:3])}
        slot_to_tracks = {i: [tid] for i, (tid, _, _) in enumerate(ranked[:3])}
        slot_to_speaker = {i: order[i] for i in range(len(slot_positions))}
    else:
        sorted_slots = sorted(slots, key=lambda s: s.mean_x)
        slot_to_speaker = {s.cluster_id: order[i] for i, s in enumerate(sorted_slots[: len(order)])}
        slot_positions = {s.cluster_id: (s.mean_x, s.mean_y) for s in slots}

    track_to_slot = {tid: sid for sid, tids in slot_to_tracks.items() for tid in tids}
    speakers = [slot_to_speaker[s] for s in sorted(slot_to_speaker.keys())]

    save_tracks(
        tracks,
        session_dir,
        fps=fps_v,
        frame_w=fw,
        frame_h=fh,
        frame_skip=int(frame_skip),
        extra_meta={
            "video": str(video),
            "video_name": video.name,
            "left_to_right": order,
            "slot_to_speaker": {str(k): v for k, v in slot_to_speaker.items()},
            "slot_positions": {str(k): list(v) for k, v in slot_positions.items()},
            "slot_to_tracks": {str(k): list(v) for k, v in slot_to_tracks.items()},
            "dense_skip": int(dense_skip),
            "speech_pad_sec": float(speech_pad_sec),
            "full_mesh": bool(full_mesh),
            "lip_indices": list(LIPS),
            "mar4_indices": list(MAR4),
            "index_catalog": {k: list(v) for k, v in index_catalog().items()},
            "face_backend": getattr(cfg, "face_backend", "auto"),
        },
    )

    dense_frames = (
        _speech_dense_frames(speech_intervals, fps_v, dense_skip, speech_pad_sec)
        if speech_intervals
        else set()
    )

    analyzer = HeadPoseAnalyzer(cfg)
    analyzer._load_model()
    attn_comp = SocialAttentionComputer(temperature=attention_temperature)

    # 累积
    m_frame: List[int] = []
    m_slot: List[int] = []
    m_spk: List[str] = []
    m_bbox: List[List[float]] = []
    m_pose: List[List[float]] = []
    m_lips: List[np.ndarray] = []
    m_mar4: List[np.ndarray] = []
    m_dens: List[int] = []
    m_full: List[np.ndarray] = []

    attn_frame: List[int] = []
    attn_mats: List[np.ndarray] = []

    frame_idx = 0
    logger.info(
        "提取 mesh/唇/注意力: %s (skip=%d dense=%d targets=%d)",
        video.name,
        frame_skip,
        dense_skip,
        len(dense_frames),
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frame and frame_idx >= max_frame:
            break

        is_dense = frame_idx in dense_frames
        is_sparse = frame_idx % max(1, int(frame_skip)) == 0
        if not (is_dense or is_sparse):
            frame_idx += 1
            continue

        # 本帧各槽最佳 pose + lips
        slot_best: Dict[int, Tuple[float, HeadPoseFrame, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        yaws: Dict[str, float] = {}
        positions: Dict[str, Tuple[float, float]] = {}

        for tid, track in tracks.items():
            if tid not in track_to_slot:
                continue
            use_interp = is_dense and frame_idx not in track.detections
            if use_interp:
                bbox = HeadPoseAnalyzer.interpolate_bbox_at_frame(
                    track, frame_idx, max_gap=cfg.bbox_interp_max_gap
                )
            else:
                if frame_idx not in track.detections:
                    continue
                bbox = track.detections[frame_idx].bbox
            if bbox is None:
                continue

            sid = track_to_slot[tid]
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox.astype(int)
            pad = int(0.25 * max(x2 - x1, y2 - y1))
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0 or min(roi.shape[:2]) < 24:
                continue
            rh, rw = roi.shape[:2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = analyzer._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue
            lm = results.multi_face_landmarks[0].landmark
            pose = analyzer.analyze_roi(frame, bbox)
            if pose is None:
                continue
            lips = _landmarks_xy(lm, LIPS, rw, rh, x1p, y1p)
            mar4 = _landmarks_xy(lm, MAR4, rw, rh, x1p, y1p)
            full = None
            if full_mesh:
                full = np.zeros((len(lm), 2), dtype=np.float32)
                for i, p in enumerate(lm):
                    full[i, 0] = float(p.x) * rw + x1p
                    full[i, 1] = float(p.y) * rh + y1p
            score = float(pose.visibility * pose.side_face_weight)
            prev = slot_best.get(sid)
            if prev is None or score > prev[0]:
                slot_best[sid] = (score, pose, bbox.astype(np.float32), lips, mar4, full)

        for sid, (_, pose, bbox, lips, mar4, full) in slot_best.items():
            spk = slot_to_speaker.get(sid, f"SLOT{sid}")
            m_frame.append(frame_idx)
            m_slot.append(int(sid))
            m_spk.append(spk)
            m_bbox.append(bbox.tolist())
            m_pose.append(
                [
                    float(pose.yaw),
                    float(pose.pitch),
                    float(pose.roll),
                    float(pose.mouth_opening),
                    float(pose.visibility),
                    float(pose.side_face_weight),
                ]
            )
            m_lips.append(lips)
            m_mar4.append(mar4)
            m_dens.append(1 if is_dense else 0)
            if full_mesh:
                m_full.append(full if full is not None else np.zeros((468, 2), np.float32))
            yaws[spk] = float(pose.yaw)
            positions[spk] = slot_positions.get(sid, (0.5, 0.5))

        if len(yaws) >= 2:
            raw = attn_comp.attention_scores(list(speakers), positions, yaws)
            soft = attn_comp.softmax_attention(raw, attn_comp.temperature)
            attn_frame.append(frame_idx)
            attn_mats.append(_attn_matrix(soft, speakers))

        frame_idx += 1
        if frame_idx % 500 == 0 and total:
            logger.info("  visual_cache 进度: %d / %d", frame_idx, total)

    cap.release()

    n = len(m_frame)
    lips_arr = np.stack(m_lips, axis=0) if n else np.zeros((0, len(LIPS), 2), np.float32)
    mar4_arr = np.stack(m_mar4, axis=0) if n else np.zeros((0, 4, 2), np.float32)
    pose_arr = np.asarray(m_pose, dtype=np.float32).reshape(-1, 6) if n else np.zeros((0, 6), np.float32)
    dens_arr = np.asarray(m_dens, dtype=np.int8)
    save_kw: Dict[str, Any] = {
        "frame_idx": np.asarray(m_frame, dtype=np.int32),
        "slot_id": np.asarray(m_slot, dtype=np.int32),
        "speaker": np.asarray(m_spk, dtype=object),
        "bbox": np.asarray(m_bbox, dtype=np.float32).reshape(-1, 4) if n else np.zeros((0, 4), np.float32),
        "pose": pose_arr,
        "lips_xy": lips_arr.astype(np.float32),
        "mar4_xy": mar4_arr.astype(np.float32),
        "densified": dens_arr,
    }
    if full_mesh and m_full:
        save_kw["mesh_xy"] = np.stack(m_full, axis=0).astype(np.float32)
    np.savez_compressed(session_dir / "mesh.npz", **save_kw)

    if attn_mats:
        np.savez_compressed(
            session_dir / "attention.npz",
            frame_idx=np.asarray(attn_frame, dtype=np.int32),
            attn=np.stack(attn_mats, axis=0).astype(np.float32),
            speakers=np.asarray(list(speakers), dtype=object),
        )
    else:
        np.savez_compressed(
            session_dir / "attention.npz",
            frame_idx=np.zeros((0,), dtype=np.int32),
            attn=np.zeros((0, len(speakers), len(speakers) + 1), dtype=np.float32),
            speakers=np.asarray(list(speakers), dtype=object),
        )

    if n:
        _write_lips_timeseries(
            session_dir,
            fps=fps_v,
            frame_idx=np.asarray(m_frame, dtype=np.int32),
            speakers=m_spk,
            lips_xy=lips_arr,
            mar=pose_arr[:, 3],
            densified=dens_arr,
        )

    # 更新 meta
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(
        {
            "schema_version": SCHEMA_VERSION,
            "has_mesh": True,
            "has_attention": True,
            "n_mesh_rows": n,
            "n_attn_frames": len(attn_frame),
            "speakers": list(speakers),
            "full_mesh": bool(full_mesh),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("visual_cache 完成: %s (tracks=%d mesh_rows=%d)", session_dir, len(tracks), n)
    return session_dir


def ensure_tracks(
    video: Path,
    cache_root: Path,
    *,
    config: Optional[ASDConfig] = None,
    left_to_right: Optional[Sequence[str]] = None,
    speech_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    frame_skip: int = 3,
    write_full_cache: bool = False,
) -> Tuple[Dict[int, FaceTrack], float, Path]:
    """
    优先读 tracks 缓存；没有则跟踪。
    write_full_cache=True 时若无完整 cache 则跑完整抽取；否则仅保证 tracks.npz。
    """
    session_dir = cache_dir_for(video, cache_root)
    if has_tracks_cache(session_dir):
        tracks, info = load_tracks(session_dir)
        logger.info("复用 visual_cache tracks: %s (n=%d)", session_dir.name, len(tracks))
        return tracks, float(info["fps"]), session_dir

    cfg = config or ASDConfig(frame_skip=max(1, int(frame_skip)))
    if write_full_cache:
        build_visual_cache_for_video(
            video,
            session_dir,
            config=cfg,
            left_to_right=left_to_right,
            speech_intervals=speech_intervals,
            frame_skip=frame_skip,
        )
        tracks, info = load_tracks(session_dir)
        return tracks, float(info["fps"]), session_dir

    tracker = FaceTracker(cfg)
    tracker.process_video(str(video))
    tracks = tracker.tracks
    fps = float(tracker.fps)
    cap = cv2.VideoCapture(str(video))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    save_tracks(
        tracks,
        session_dir,
        fps=fps,
        frame_w=fw,
        frame_h=fh,
        frame_skip=int(frame_skip),
        extra_meta={"video": str(video), "video_name": video.name, "left_to_right": list(left_to_right or [])},
    )
    logger.info("已写入 tracks 缓存: %s", session_dir)
    return tracks, fps, session_dir
