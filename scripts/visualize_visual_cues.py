#!/usr/bin/env python3
"""
可视化课堂视频上的视觉线索：人脸框、地标、头部姿态(yaw)、嘴开合、社交注意力。

用法:
  conda activate pyannote0
  python scripts/visualize_visual_cues.py ^
    --video "video/xianyang/0701/class3/0701-前测-五年级3班-第1组-S2S3-小组讨论.mp4" ^
    --max-sec 60 --frame-skip 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.core.config import ASDConfig
from csd.core.utils import setup_logging
from csd.data.xianyang import left_to_right_to_ref_x, resolve_video_position_map
from csd.perception.face_mesh_indices import FACE_OVAL, LEFT_EYE, LIPS, NOSE, RIGHT_EYE
from csd.perception.face_tracker import FaceTracker
from csd.perception.head_pose import HeadPoseAnalyzer, HeadPoseFrame
from csd.social.position_speaker_mapper import PositionSpeakerMapper
from csd.social.social_attention import SocialAttentionComputer

logger = logging.getLogger("visualize_cues")

_OVAL = FACE_OVAL
_LEFT_EYE = LEFT_EYE
_RIGHT_EYE = RIGHT_EYE
_LIPS = LIPS
_NOSE = NOSE

SPEAKER_COLORS = {
    "S1": (80, 180, 255),
    "S2": (80, 220, 120),
    "S3": (220, 140, 60),
}
DEFAULT_COLOR = (180, 180, 180)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化人脸/姿态/地标/注意力线索")
    p.add_argument("--video", type=str, required=True, help="输入视频路径")
    p.add_argument("--position-map", type=str, default="", help="position_map JSON；默认自动找 ref/position_maps")
    p.add_argument("--out-dir", type=str, default="", help="输出目录，默认 output/visual_cues/<stem>")
    p.add_argument("--frame-skip", type=int, default=3, help="处理步长（越大越快）")
    p.add_argument("--max-sec", type=float, default=90.0, help="最多处理前 N 秒；0=全程")
    p.add_argument("--preview-every-sec", type=float, default=5.0, help="另存预览图间隔（秒）")
    p.add_argument("--write-video", action="store_true", default=True, help="写出叠加视频")
    p.add_argument("--no-write-video", action="store_false", dest="write_video")
    return p.parse_args()


def _resolve_ltr(video_path: Path, position_map_arg: str) -> Tuple[List[str], Dict[str, float], Optional[dict]]:
    map_paths: List[Path] = []
    if position_map_arg:
        map_paths.append(Path(position_map_arg))
    else:
        map_paths.extend(sorted((ROOT / "ref" / "position_maps").glob("*.json")))

    item = resolve_video_position_map(video_path.name, map_paths, require_confirmed=False)
    if item is None:
        logger.warning("未找到 position_map，将只用左中右槽位标签 slot0/1/2")
        return ["S1", "S2", "S3"], {"S1": 0.15, "S2": 0.5, "S3": 0.85}, None

    ltr = [str(s).upper() for s in (item.get("left_to_right") or ["S1", "S2", "S3"])]
    if len(ltr) != 3:
        ltr = (ltr + ["S1", "S2", "S3"])[:3]
    ref_x = left_to_right_to_ref_x(ltr)
    logger.info(
        "position_map: left_to_right=%s confirmed=%s roster=%s",
        ltr,
        item.get("confirmed"),
        item.get("roster"),
    )
    return ltr, ref_x, item


def _align_slots_to_speakers(
    slot_positions: Dict[int, Tuple[float, float]],
    speaker_ref_x: Dict[str, float],
) -> Dict[int, str]:
    """按 x 位置把槽位对齐到说话人。"""
    slots = sorted(slot_positions.keys(), key=lambda s: slot_positions[s][0])
    speakers = sorted(speaker_ref_x.keys(), key=lambda s: speaker_ref_x[s])
    out: Dict[int, str] = {}
    for i, sid in enumerate(slots):
        if i < len(speakers):
            out[sid] = speakers[i]
    return out


def _draw_polyline(frame, pts: List[Tuple[int, int]], color, closed: bool = False):
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [arr], closed, color, 1, cv2.LINE_AA)


def _extract_mesh_overlay(
    analyzer: HeadPoseAnalyzer,
    frame: np.ndarray,
    bbox: np.ndarray,
) -> Tuple[Optional[HeadPoseFrame], Optional[np.ndarray], Optional[List]]:
    """返回 (pose, bbox, landmarks_full_xy list of (x,y) for all 468 or None)."""
    analyzer._load_model()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 24:
        return None, None, None

    rh, rw = roi.shape[:2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = analyzer._face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, np.array([x1, y1, x2, y2], dtype=np.float64), None

    lm = results.multi_face_landmarks[0].landmark
    pose = analyzer.analyze_roi(frame, bbox)
    pts = [(int(p.x * rw) + x1, int(p.y * rh) + y1) for p in lm]
    return pose, np.array([x1, y1, x2, y2], dtype=np.float64), pts


def _draw_landmarks(frame, pts: List[Tuple[int, int]], color):
    for idx in _OVAL:
        if idx < len(pts):
            cv2.circle(frame, pts[idx], 1, color, -1, cv2.LINE_AA)
    for group, closed in ((_LEFT_EYE, True), (_RIGHT_EYE, True), (_LIPS, True), (_NOSE, False)):
        ring = [pts[i] for i in group if i < len(pts)]
        _draw_polyline(frame, ring, color, closed=closed)


def _draw_yaw_arrow(frame, cx: int, cy: int, yaw_deg: float, color, length: int = 60):
    # yaw>0 大致朝画面右；与社交注意力约定一致
    rad = np.radians(yaw_deg)
    dx = int(length * np.sin(rad))
    dy = int(-length * np.cos(rad))
    tip = (cx + dx, cy + dy)
    cv2.arrowedLine(frame, (cx, cy), tip, color, 2, tipLength=0.25)
    cv2.circle(frame, (cx, cy), 3, color, -1)


def _draw_attention(
    frame,
    centers: Dict[str, Tuple[int, int]],
    attn: Dict[str, Dict[str, float]],
    thresh: float = 0.35,
):
    for src, dst_map in attn.items():
        if src not in centers:
            continue
        for dst, score in dst_map.items():
            if dst == "__elsewhere__" or dst not in centers:
                continue
            if score < thresh:
                continue
            c0, c1 = centers[src], centers[dst]
            color = SPEAKER_COLORS.get(src, DEFAULT_COLOR)
            thickness = 1 + int(2 * score)
            mid = ((c0[0] + c1[0]) // 2, (c0[1] + c1[1]) // 2)
            cv2.arrowedLine(frame, c0, c1, color, thickness, tipLength=0.12)
            cv2.putText(
                frame,
                f"{score:.2f}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )


def main() -> None:
    args = _parse_args()
    setup_logging()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / video_path
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "output" / "visual_cues" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    ltr, speaker_ref_x, pos_item = _resolve_ltr(video_path, args.position_map)
    config = ASDConfig(frame_skip=max(1, args.frame_skip), output_dir=out_dir)

    logger.info("人脸跟踪: %s (frame_skip=%d, max_sec=%s)", video_path.name, config.frame_skip, args.max_sec or "all")
    tracker = FaceTracker(config)
    tracker.process_video(str(video_path), max_sec=args.max_sec if args.max_sec and args.max_sec > 0 else None)
    tracks, fps = tracker.tracks, tracker.fps
    logger.info("轨迹数=%d fps=%.2f", len(tracks), fps)

    mapper = PositionSpeakerMapper(config)
    slots = mapper.extract_position_slots(tracks, n_slots=3)
    slot_positions = {s.cluster_id: (s.mean_x, s.mean_y) for s in slots}
    slot_to_tracks = mapper._cluster_to_tracks(tracks, n_slots=3)
    if not slot_positions:
        # 回退：按轨迹平均 x 分成最多 3 槽
        ranked = sorted(
            [(tid, t.mean_position[0], t.mean_position[1]) for tid, t in tracks.items() if len(t.detections) >= 3],
            key=lambda x: x[1],
        )
        slot_positions = {}
        slot_to_tracks = {}
        for i, (tid, mx, my) in enumerate(ranked[:3]):
            slot_positions[i] = (mx, my)
            slot_to_tracks[i] = [tid]
    slot_to_speaker = _align_slots_to_speakers(slot_positions, speaker_ref_x)
    track_to_slot = {tid: sid for sid, tids in slot_to_tracks.items() for tid in tids}
    logger.info("slot_to_speaker=%s positions=%s", slot_to_speaker, slot_positions)

    analyzer = HeadPoseAnalyzer(config)
    attn_comp = SocialAttentionComputer(temperature=15.0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = int(args.max_sec * fps) if args.max_sec and args.max_sec > 0 else total

    writer = None
    out_video = out_dir / f"{video_path.stem}_cues.mp4"
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 输出按处理帧率，便于快速回看
        writer = cv2.VideoWriter(str(out_video), fourcc, max(fps / config.frame_skip, 1.0), (w, h))

    stats = {
        "frames_processed": 0,
        "faces_drawn": 0,
        "mesh_ok": 0,
        "mesh_fail": 0,
        "yaw_mean": [],
        "mar_mean": [],
    }
    feat_path = out_dir / "frame_poses.jsonl"
    if feat_path.exists():
        feat_path.unlink()
    last_preview_t = -1e9
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= max_frame:
            break

        if frame_idx % config.frame_skip != 0:
            frame_idx += 1
            continue

        t = frame_idx / max(fps, 1e-6)
        vis = frame.copy()
        centers: Dict[str, Tuple[int, int]] = {}
        yaws: Dict[str, float] = {}
        positions: Dict[str, Tuple[float, float]] = {}
        pose_rows = []

        for tid, track in tracks.items():
            if tid not in track_to_slot:
                continue
            bbox = HeadPoseAnalyzer.interpolate_bbox_at_frame(track, frame_idx, max_gap=config.bbox_interp_max_gap)
            if bbox is None:
                continue
            sid = track_to_slot[tid]
            spk = slot_to_speaker.get(sid, f"slot{sid}")
            color = SPEAKER_COLORS.get(spk, DEFAULT_COLOR)

            pose, draw_bbox, pts = _extract_mesh_overlay(analyzer, frame, bbox)
            stats["faces_drawn"] += 1
            if pose is None:
                stats["mesh_fail"] += 1
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{spk} mesh-fail", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                continue

            stats["mesh_ok"] += 1
            x1, y1, x2, y2 = (draw_bbox if draw_bbox is not None else bbox).astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers[spk] = (cx, cy)
            yaws[spk] = float(pose.yaw)
            positions[spk] = slot_positions.get(sid, (cx / max(w, 1), cy / max(h, 1)))
            stats["yaw_mean"].append(pose.yaw)
            stats["mar_mean"].append(pose.mouth_opening)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if pts:
                _draw_landmarks(vis, pts, color)
            _draw_yaw_arrow(vis, cx, cy, pose.yaw, color)

            label = f"{spk} y={pose.yaw:.0f} p={pose.pitch:.0f} mar={pose.mouth_opening:.2f}"
            cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # 嘴开合条
            bar_w = int(40 * min(pose.mouth_opening / 0.5, 1.0))
            cv2.rectangle(vis, (x1, y2 + 4), (x1 + 40, y2 + 12), (40, 40, 40), -1)
            cv2.rectangle(vis, (x1, y2 + 4), (x1 + bar_w, y2 + 12), color, -1)

            pose_rows.append(
                {
                    "t": round(t, 3),
                    "frame": frame_idx,
                    "speaker": spk,
                    "yaw": round(pose.yaw, 2),
                    "pitch": round(pose.pitch, 2),
                    "roll": round(pose.roll, 2),
                    "mouth_opening": round(pose.mouth_opening, 4),
                    "visibility": round(pose.visibility, 3),
                    "side_face_weight": round(pose.side_face_weight, 3),
                }
            )

        # 注意力箭头
        if len(centers) >= 2:
            raw = attn_comp.attention_scores(list(centers.keys()), positions, yaws)
            soft = attn_comp.softmax_attention(raw, attn_comp.temperature)
            _draw_attention(vis, centers, soft, thresh=0.28)

        # 顶栏说明
        legend = "box=face  mesh=landmarks  arrow=yaw(gaze proxy)  link=attention  bar=MAR"
        cv2.rectangle(vis, (0, 0), (w, 48), (0, 0, 0), -1)
        cv2.putText(vis, f"t={t:.1f}s  {legend}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(
            vis,
            f"L->R: {ltr}  mesh_ok={stats['mesh_ok']} fail={stats['mesh_fail']}",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 120),
            1,
        )

        if writer is not None:
            writer.write(vis)

        if t - last_preview_t >= args.preview_every_sec:
            prev_path = preview_dir / f"t{int(t):04d}.jpg"
            cv2.imwrite(str(prev_path), vis)
            last_preview_t = t

        # 追加帧特征
        if pose_rows:
            feat_path = out_dir / "frame_poses.jsonl"
            with feat_path.open("a", encoding="utf-8") as f:
                for row in pose_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats["frames_processed"] += 1
        if stats["frames_processed"] % 40 == 0:
            logger.info("processed %d frames (t=%.1fs)", stats["frames_processed"], t)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    summary = {
        "video": str(video_path),
        "out_video": str(out_video) if args.write_video else None,
        "preview_dir": str(preview_dir),
        "left_to_right": ltr,
        "slot_to_speaker": {str(k): v for k, v in slot_to_speaker.items()},
        "slot_positions": {str(k): list(v) for k, v in slot_positions.items()},
        "frames_processed": stats["frames_processed"],
        "faces_drawn": stats["faces_drawn"],
        "mesh_ok": stats["mesh_ok"],
        "mesh_fail": stats["mesh_fail"],
        "mesh_success_rate": round(stats["mesh_ok"] / max(stats["mesh_ok"] + stats["mesh_fail"], 1), 4),
        "yaw_mean": float(np.mean(stats["yaw_mean"])) if stats["yaw_mean"] else None,
        "mar_mean": float(np.mean(stats["mar_mean"])) if stats["mar_mean"] else None,
        "position_map_item": {
            "confirmed": None if pos_item is None else pos_item.get("confirmed"),
            "roster": None if pos_item is None else pos_item.get("roster"),
            "video_name": None if pos_item is None else pos_item.get("video_name"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完成: %s", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n输出目录: {out_dir}")
    if args.write_video:
        print(f"叠加视频: {out_video}")
    print(f"预览图:   {preview_dir}")
    print(f"摘要:     {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
