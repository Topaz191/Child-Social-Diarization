#!/usr/bin/env python3
"""
从咸阳视频构造音–口同步弱标签样本，并用冻结 MTDVocaLiST 预提特征。

正样本：GT 说话人嘴部窗口 + 该段音频
负样本：同段音频 + 其他学生嘴部窗口

输出:
  output/avsync_xianyang/<session>/avsync_features.npz
  output/avsync_xianyang/merged_all/avsync_features.npz

用法:
  python scripts/setup_mtdvocalist.py
  python scripts/prepare_avsync_xianyang.py --video path/to.mp4
  python scripts/prepare_avsync_xianyang.py --from-manifest output/xianyang/manifest.json --require-position-map
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csd.avsync.mel import mel_for_time_range
from csd.avsync.mouth_crop import five_frame_indices, load_mouth_tensor_for_window
from csd.avsync.mtd_backend import MTDFeatureExtractor
from csd.core.config import ASDConfig
from csd.core.utils import extract_audio_ffmpeg, setup_logging
from csd.data.companion_audio import project_audio_root, resolve_companion_audio
from csd.data.xianyang import (
    load_xianyang_segments,
    parse_xianyang_video_name,
    resolve_video_position_map,
    scan_xianyang_videos,
)
from csd.perception.face_tracker import FaceTracker
from csd.social.position_speaker_mapper import PositionSpeakerMapper

logger = logging.getLogger("prepare_avsync")

STUDENTS = ("S1", "S2", "S3", "S4")


def _assign_ltr(slots, cameras: Sequence[str]) -> Dict[int, str]:
    order = [str(c).upper() for c in cameras]
    while len(order) < 3:
        for s in ("S1", "S2", "S3"):
            if s not in order:
                order.append(s)
                break
    order = order[:3]
    sorted_slots = sorted(slots, key=lambda s: s.mean_x)
    out = {}
    for i, slot in enumerate(sorted_slots[: len(order)]):
        out[slot.cluster_id] = order[i]
    return out


def _load_wav_np(wav_path: Path, sr: int = 16000) -> np.ndarray:
    import torchaudio

    wf, s = torchaudio.load(str(wav_path))
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    if s != sr:
        wf = torchaudio.transforms.Resample(s, sr)(wf)
    return wf.squeeze(0).numpy().astype(np.float32)


def _ensure_session_wav(
    video: Path,
    out_wav: Path,
    *,
    audio_root: Optional[Path],
) -> Optional[Path]:
    """优先外置配套音频（含 merged audio），否则从视频抽轨。"""
    if out_wav.exists():
        return out_wav
    companion = resolve_companion_audio(video, audio_root=audio_root)
    if companion is not None:
        # 统一落到 16k wav（若已是 wav 可直接复制/重采样）
        try:
            import shutil

            import torchaudio

            wf, sr = torchaudio.load(str(companion))
            if wf.shape[0] > 1:
                wf = wf.mean(dim=0, keepdim=True)
            if int(sr) != 16000:
                wf = torchaudio.transforms.Resample(int(sr), 16000)(wf)
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_wav), wf, 16000)
            logger.info("配套音频 → %s (from %s)", out_wav, companion)
            return out_wav
        except Exception as exc:
            logger.warning("配套音频转换失败，回退抽视频轨: %s | %s", companion, exc)
    if extract_audio_ffmpeg(video, out_wav, sample_rate=16000) is None:
        return None
    logger.info("视频轨音频 → %s", out_wav)
    return out_wav


def process_one(
    video: Path,
    excel: Path,
    out_root: Path,
    extractor: MTDFeatureExtractor,
    *,
    position_map_paths: Optional[Sequence[Path]] = None,
    require_position_map: bool = True,
    window_sec: float = 0.2,
    hop_sec: float = 0.2,
    max_windows_per_seg: int = 8,
    seed: int = 42,
    audio_root: Optional[Path] = None,
) -> Optional[Path]:
    meta = parse_xianyang_video_name(video.name)
    if not meta:
        logger.warning("文件名无法解析: %s", video.name)
        return None

    pos = resolve_video_position_map(video, position_map_paths or [])
    if require_position_map and (pos is None or not pos.get("confirmed")):
        logger.warning("缺少 confirmed position_map，跳过: %s", video.name)
        return None
    if pos is not None and pos.get("video_start_abs") is None and require_position_map:
        logger.warning("缺少 video_start_abs，跳过: %s", video.name)
        return None

    left_to_right = list(pos.get("left_to_right") or meta.get("cameras") or []) if pos else list(meta.get("cameras") or [])
    segments = load_xianyang_segments(
        excel,
        meta["sheet"],
        meta["phase"],
        speakers=STUDENTS,
        video_start_abs=None if pos is None else pos.get("video_start_abs"),
        align_to_video=True,
    )
    segments = [s for s in segments if s.get("speaker") in STUDENTS]
    if not segments:
        logger.warning("无学生说话段: %s", video.name)
        return None

    out_dir = out_root / f"{meta['date']}_class{meta['class_id']}_g{meta['group']}_{meta['phase']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio_16k.wav"
    if _ensure_session_wav(video, wav_path, audio_root=audio_root) is None:
        return None
    wav = _load_wav_np(wav_path)

    cfg = ASDConfig()
    tracker = FaceTracker(cfg)
    tracker.process_video(str(video))
    tracks, fps = tracker.tracks, float(tracker.fps)
    mapper = PositionSpeakerMapper(cfg)
    slots = mapper.extract_position_slots(tracks, n_slots=3)
    slot_to_tracks = mapper._cluster_to_tracks(tracks, n_slots=3)
    slot_to_spk = _assign_ltr(slots, left_to_right)
    spk_to_tracks: Dict[str, List[int]] = {s: [] for s in STUDENTS}
    for sid, spk in slot_to_spk.items():
        spk_to_tracks.setdefault(spk, []).extend(slot_to_tracks.get(sid, []))

    rng = np.random.default_rng(seed)
    vis_list, aud_list, y_list, meta_list = [], [], [], []

    for seg_i, seg in enumerate(segments):
        spk = seg["speaker"]
        t0, t1 = float(seg["start"]), float(seg["end"])
        if t1 - t0 < window_sec:
            continue
        # 窗口中心
        centers = np.arange(t0 + window_sec / 2, t1 - window_sec / 2 + 1e-6, hop_sec)
        if len(centers) == 0:
            centers = np.array([(t0 + t1) / 2.0])
        if len(centers) > max_windows_per_seg:
            centers = rng.choice(centers, size=max_windows_per_seg, replace=False)
            centers.sort()

        track_ids = spk_to_tracks.get(spk) or []
        if not track_ids:
            continue
        # 选检测最多的轨迹
        track = max((tracks[t] for t in track_ids if t in tracks), key=lambda tr: len(tr.detections), default=None)
        if track is None:
            continue

        other_spks = [s for s in STUDENTS if s != spk and spk_to_tracks.get(s)]
        for c in centers:
            w0, w1 = c - window_sec / 2, c + window_sec / 2
            center_f = int(round(c * fps))
            fidx = five_frame_indices(center_f, fps)
            face = load_mouth_tensor_for_window(str(video), track, fidx)
            if face is None:
                continue
            mel = mel_for_time_range(wav, 16000, w0, w1, mel_width=16)
            # pad 到 80 在 extractor 内完成；这里存 [1,80,16]
            try:
                v_feat, a_feat, logit = extractor.encode_numpy(face, mel, return_logit=True)
            except Exception as exc:
                logger.warning("特征提取失败: %s", exc)
                continue
            vis_list.append(v_feat)
            aud_list.append(a_feat)
            y_list.append(1)
            meta_list.append(
                {
                    "label": 1,
                    "speaker": spk,
                    "t0": round(w0, 3),
                    "t1": round(w1, 3),
                    "seg_idx": seg_i,
                    "mtd_logit": logit,
                    "kind": "pos",
                }
            )

            # 负样本：换其他人嘴
            if other_spks:
                neg_spk = str(rng.choice(other_spks))
                neg_ids = spk_to_tracks.get(neg_spk) or []
                neg_track = max(
                    (tracks[t] for t in neg_ids if t in tracks),
                    key=lambda tr: len(tr.detections),
                    default=None,
                )
                if neg_track is None:
                    continue
                face_n = load_mouth_tensor_for_window(str(video), neg_track, fidx)
                if face_n is None:
                    continue
                try:
                    v_n, a_n, logit_n = extractor.encode_numpy(face_n, mel, return_logit=True)
                except Exception:
                    continue
                vis_list.append(v_n)
                aud_list.append(a_n)
                y_list.append(0)
                meta_list.append(
                    {
                        "label": 0,
                        "speaker": neg_spk,
                        "true_speaker": spk,
                        "t0": round(w0, 3),
                        "t1": round(w1, 3),
                        "seg_idx": seg_i,
                        "mtd_logit": logit_n,
                        "kind": "neg_swap_face",
                    }
                )

    if not y_list:
        logger.warning("无有效样本: %s", video.name)
        return None

    V = np.stack(vis_list, axis=0).astype(np.float32)
    A = np.stack(aud_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    npz = out_dir / "avsync_features.npz"
    np.savez_compressed(npz, V=V, A=A, y=y)
    (out_dir / "avsync_samples_meta.json").write_text(
        json.dumps(
            {
                "samples": meta_list,
                "extra": {
                    "video": str(video),
                    "left_to_right": left_to_right,
                    "feat_dim": int(V.shape[-1]),
                    "window_sec": window_sec,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "V_shape": list(V.shape),
        "A_shape": list(A.shape),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写 %s | %s", npz, summary)
    return out_dir


def merge_dirs(session_dirs: Sequence[Path], out_dir: Path) -> Path:
    Vs, As, ys, metas = [], [], [], []
    for d in session_dirs:
        npz = d / "avsync_features.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        if len(data["y"]) == 0:
            continue
        Vs.append(data["V"])
        As.append(data["A"])
        ys.append(data["y"])
        mp = d / "avsync_samples_meta.json"
        if mp.exists():
            metas.extend(json.loads(mp.read_text(encoding="utf-8")).get("samples", []))
    if not Vs:
        raise RuntimeError("没有可合并的 avsync 样本")
    out_dir.mkdir(parents=True, exist_ok=True)
    V = np.concatenate(Vs, axis=0)
    A = np.concatenate(As, axis=0)
    y = np.concatenate(ys, axis=0)
    np.savez_compressed(out_dir / "avsync_features.npz", V=V, A=A, y=y)
    (out_dir / "avsync_samples_meta.json").write_text(
        json.dumps({"samples": metas, "extra": {"merged_from": [str(d) for d in session_dirs]}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {"n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "V_shape": list(V.shape)}
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("合并完成: %s %s", out_dir, summary)
    return out_dir / "avsync_features.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备音–口同步训练特征")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--from-manifest", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=ROOT / "video" / "xianyang")
    p.add_argument("--excel", type=Path, default=ROOT / "ref" / "202507-xianyang-小学生转录标注.xlsx")
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "avsync_xianyang")
    p.add_argument("--position-maps", type=Path, nargs="*", default=None)
    p.add_argument("--require-position-map", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="配套音频根；默认搜 audio/（含 xianyang 与 merged audio）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    map_dir = ROOT / "ref" / "position_maps"
    position_maps = args.position_maps
    if position_maps is None and map_dir.exists():
        position_maps = sorted(map_dir.glob("*.json"))

    audio_root = args.audio_root if args.audio_root is not None else project_audio_root()

    videos: List[Path] = []
    if args.video is not None:
        videos = [args.video]
    elif args.from_manifest is not None:
        man = json.loads(args.from_manifest.read_text(encoding="utf-8"))
        videos = [Path(s["path"]) for s in man.get("sessions", []) if s.get("matched")]
    else:
        videos = [m.path for m in scan_xianyang_videos(args.video_root) if m.parse_ok]
    if args.limit is not None:
        videos = videos[: args.limit]

    extractor = MTDFeatureExtractor(device=args.device, freeze=True)
    done: List[Path] = []
    for video in videos:
        meta = parse_xianyang_video_name(video.name)
        if not meta:
            logger.warning("跳过无法解析: %s", video.name)
            continue
        cand = args.out_root / f"{meta['date']}_class{meta['class_id']}_g{meta['group']}_{meta['phase']}"
        if args.skip_existing and (cand / "avsync_features.npz").exists():
            logger.info("跳过已有: %s", cand.name)
            done.append(cand)
            continue
        try:
            d = process_one(
                video,
                args.excel,
                args.out_root,
                extractor,
                position_map_paths=position_maps,
                require_position_map=args.require_position_map,
                audio_root=audio_root,
            )
        except Exception as exc:
            logger.exception("失败 %s: %s", video.name, exc)
            continue
        if d is not None:
            done.append(d)

    if done:
        merge_dirs(done, args.out_root / "merged_all")
    else:
        logger.warning("无场次成功")


if __name__ == "__main__":
    main()
