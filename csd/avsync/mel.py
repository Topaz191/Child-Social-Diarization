"""Wav2Lip / MTDVocaLiST 风格 mel 频谱（16kHz）。"""

from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 800
HOP = 200
WIN = 800
N_MELS = 80
FMIN = 55
FMAX = 7600
PREEMPH = 0.97
REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100
MAX_ABS = 4.0


def preemphasis(wav: np.ndarray, k: float = PREEMPH) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    if len(wav) < 2:
        return wav.astype(np.float32)
    return np.append(wav[0], wav[1:] - k * wav[:-1]).astype(np.float32)


def melspectrogram(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """返回 [n_mels, T]，已按 Wav2Lip 习惯做 dB + 对称归一化到约 [-1,1]*MAX_ABS。"""
    import librosa

    y = preemphasis(wav)
    S = librosa.feature.melspectrogram(
        y=y.astype(np.float32),
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0,
    )
    # amp -> dB
    min_level = np.exp(MIN_LEVEL_DB / 20.0 * np.log(10.0))
    S = np.clip(S, a_min=min_level, a_max=None)
    S = 20.0 * np.log10(S) - REF_LEVEL_DB
    # normalize to [-max_abs, max_abs]
    S = np.clip((2.0 * MAX_ABS) * ((S - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS, -MAX_ABS, MAX_ABS)
    return S.astype(np.float32)


def crop_mel_window(mel: np.ndarray, center_frame: int, width: int = 16) -> np.ndarray:
    """从整段 mel 取宽为 width 的窗，不足则 pad。返回 [1, n_mels, width]。"""
    # MTD 测试用 80x80；短窗同步常用 16（对应约 0.2s）。两者都支持。
    t = mel.shape[1]
    half = width // 2
    start = int(center_frame - half)
    end = start + width
    out = np.zeros((mel.shape[0], width), dtype=np.float32)
    src_s = max(0, start)
    src_e = min(t, end)
    dst_s = src_s - start
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s:
        out[:, dst_s:dst_e] = mel[:, src_s:src_e]
    return out[None, ...]  # [1, 80, W]


def mel_for_time_range(
    wav: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    mel_width: int = 16,
) -> np.ndarray:
    """截取 [t0,t1] 音频并生成固定宽度 mel，形状 [1, 80, mel_width]。"""
    i0 = max(0, int(t0 * sr))
    i1 = min(len(wav), int(t1 * sr))
    if i1 <= i0:
        seg = np.zeros(int(0.2 * sr), dtype=np.float32)
    else:
        seg = wav[i0:i1].astype(np.float32)
    # 保证至少覆盖 mel_width hops
    need = (mel_width + 2) * HOP
    if len(seg) < need:
        pad = np.zeros(need - len(seg), dtype=np.float32)
        seg = np.concatenate([seg, pad], axis=0)
    mel = melspectrogram(seg, sr=sr)
    center = mel.shape[1] // 2
    return crop_mel_window(mel, center, width=mel_width)
