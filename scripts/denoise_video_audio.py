#!/usr/bin/env python3
"""
对视频音轨做降噪，画面原样拷贝，输出旁路文件（默认不覆盖原片）。

默认后端: ffmpeg afftdn（无需额外 Python 包）
可选: --backend noisereduce（需 pip install noisereduce soundfile）

用法:
  python scripts/denoise_video_audio.py path/to/video.mp4
  python scripts/denoise_video_audio.py path/to/video.mp4 --strength 12
  python scripts/denoise_video_audio.py path/to/dir --glob "*.mp4"
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("denoise_video")


def _which_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("未找到 ffmpeg，请先安装并加入 PATH")
    return exe


def denoise_ffmpeg(src: Path, dst: Path, strength: float = 12.0, noise_floor: float = -25.0) -> None:
    """
    afftdn: nf 越小越激进；nr 为降噪强度（约 0.01~97）。
    strength 映射到 nr；noise_floor 映射到 nf(dB)。
    """
    ffmpeg = _which_ffmpeg()
    nr = max(0.01, min(97.0, float(strength)))
    nf = float(noise_floor)
    # 高通去低频轰鸣 + FFT 降噪；视频流 copy
    af = f"highpass=f=80,afftdn=nr={nr}:nf={nf}"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "copy",
        "-af",
        af,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(dst),
    ]
    logger.info("ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def denoise_noisereduce(src: Path, dst: Path, prop_decrease: float = 0.8) -> None:
    import numpy as np

    try:
        import noisereduce as nr
        import soundfile as sf
    except ImportError as e:
        raise ImportError("需要: pip install noisereduce soundfile") from e

    ffmpeg = _which_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="denoise_") as td:
        td_path = Path(td)
        wav_in = td_path / "in.wav"
        wav_out = td_path / "out.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_in)],
            check=True,
        )
        audio, sr = sf.read(str(wav_in))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        cleaned = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
        sf.write(str(wav_out), cleaned, sr)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-i",
                str(wav_out),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(dst),
            ],
            check=True,
        )


def default_out_path(src: Path, out_dir: Path | None) -> Path:
    stem = src.stem
    if stem.endswith("_denoised"):
        name = src.name
    else:
        name = f"{stem}_denoised{src.suffix}"
    base = out_dir if out_dir is not None else src.parent
    return base / name


def process_one(
    src: Path,
    out_dir: Path | None,
    backend: str,
    strength: float,
    noise_floor: float,
    prop_decrease: float,
    overwrite: bool,
) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst = default_out_path(src, out_dir)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"已存在: {dst}（加 --overwrite 可覆盖）")
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("去噪: %s -> %s [%s]", src, dst, backend)
    if backend == "ffmpeg":
        denoise_ffmpeg(src, dst, strength=strength, noise_floor=noise_floor)
    elif backend == "noisereduce":
        denoise_noisereduce(src, dst, prop_decrease=prop_decrease)
    else:
        raise ValueError(f"未知 backend: {backend}")
    logger.info("完成: %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="视频音轨去噪")
    p.add_argument("input", type=Path, help="视频文件或目录")
    p.add_argument("--glob", type=str, default="*.mp4", help="目录模式下的匹配模式")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录；默认与源文件同目录")
    p.add_argument("--backend", choices=["ffmpeg", "noisereduce"], default="ffmpeg")
    p.add_argument("--strength", type=float, default=12.0, help="ffmpeg afftdn nr（越大越强）")
    p.add_argument("--noise-floor", type=float, default=-25.0, help="ffmpeg afftdn nf(dB)")
    p.add_argument("--prop-decrease", type=float, default=0.8, help="noisereduce 降噪比例")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    src = args.input
    files: list[Path]
    if src.is_dir():
        files = sorted(src.glob(args.glob))
    else:
        files = [src]
    if not files:
        raise SystemExit(f"未找到视频: {src}")

    for f in files:
        process_one(
            f,
            args.out_dir,
            args.backend,
            args.strength,
            args.noise_floor,
            args.prop_decrease,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
