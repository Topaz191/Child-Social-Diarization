"""模块3：音频 VAD 与声纹特征提取。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from csd.core.config import ASDConfig
from csd.core.backends import resolve_speaker_backend
from csd.core.utils import cosine_similarity, l2_normalize, load_wav_segment, merge_short_segments

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """VAD 检测出的语音片段。"""

    start_time: float
    end_time: float
    speaker_embedding: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VADProcessor:
    """基于 Silero-VAD 的语音活动检测。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self._model = None
        self._get_speech_timestamps = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            logger.info("加载 Silero-VAD 模型 (pip 包)...")
            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        except ImportError:
            logger.info("加载 Silero-VAD 模型 (torch.hub)...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            (get_speech_timestamps, _, _, _, _) = utils
            self._get_speech_timestamps = get_speech_timestamps
        device = self.config.resolve_device()
        if device == "cuda":
            self._model = self._model.cuda()

    def detect(self, wav_path: Path) -> List[Tuple[float, float]]:
        """返回语音段时间戳列表 [(start, end), ...]。"""
        import torchaudio

        self._load_model()
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.config.sample_rate)(waveform)
            sr = self.config.sample_rate

        device = self.config.resolve_device()
        wav = waveform.squeeze()
        if device == "cuda":
            wav = wav.cuda()

        speech_timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=sr,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
            return_seconds=True,
        )

        segments = [(ts["start"], ts["end"]) for ts in speech_timestamps]
        segments = merge_short_segments(segments, self.config.min_speech_duration)
        logger.info("VAD 检测到 %d 个语音段", len(segments))
        return segments


class SpeakerEmbeddingExtractor:
    """声纹 embedding 提取（auto: pyannote > speechbrain）。"""

    def __init__(self, config: ASDConfig):
        self.config = config
        self._backend = resolve_speaker_backend(config.speaker_backend)
        self._model = None
        self._inference = None
        self.enrolled_speakers: Dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self._backend == "pyannote":
            self._load_pyannote()
        else:
            self._load_speechbrain()

    def _load_pyannote(self) -> None:
        from pyannote.audio import Inference, Model

        logger.info("加载 pyannote 声纹模型: %s", self.config.pyannote_embedding_model)
        token = self.config.get_hf_token()
        self._model = Model.from_pretrained(self.config.pyannote_embedding_model, token=token)
        self._inference = Inference(self._model, window="whole")

    def _load_speechbrain(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("加载 SpeechBrain 声纹模型: %s", self.config.speechbrain_model)
        self._model = EncoderClassifier.from_hparams(
            source=self.config.speechbrain_model,
            savedir=self.config.speechbrain_savedir,
            run_opts={"device": self.config.resolve_device()},
        )

    def extract_from_waveform(self, waveform: torch.Tensor) -> np.ndarray:
        """从波形 tensor (1, samples) 提取 embedding。"""
        self._load_model()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self._backend == "pyannote":
            with torch.no_grad():
                emb = self._inference({"waveform": waveform, "sample_rate": self.config.sample_rate})
            if isinstance(emb, dict):
                emb = emb.get("embedding", next(iter(emb.values())))
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            return l2_normalize(np.asarray(emb).squeeze())

        with torch.no_grad():
            emb = self._model.encode_batch(waveform)
        return l2_normalize(emb.squeeze().cpu().numpy())

    def extract_segment(self, wav_path: Path, start: float, end: float) -> Optional[np.ndarray]:
        waveform, _ = load_wav_segment(wav_path, start, end, self.config.sample_rate)
        if waveform is None or waveform.shape[1] < int(0.3 * self.config.sample_rate):
            return None
        return self.extract_from_waveform(waveform)

    def enroll_from_directory(self, enroll_dir: Path) -> None:
        """
        注册声纹库，支持两种目录结构：
        1) enroll_dir/s1/*.wav  （子目录）
        2) enroll_dir/s1-1-post-pure.wav  （flat，文件名前缀为说话人）
        """
        if not enroll_dir or not enroll_dir.exists():
            return

        import torchaudio
        from collections import defaultdict

        by_speaker: Dict[str, list] = defaultdict(list)

        # 子目录结构
        for speaker_dir in sorted(enroll_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for wav_file in speaker_dir.glob("*.wav"):
                by_speaker[speaker_dir.name.lower()].append(wav_file)
            for wav_file in speaker_dir.glob("*.WAV"):
                by_speaker[speaker_dir.name.lower()].append(wav_file)

        # flat 结构：s1-1-post-pure.wav
        for wav_file in sorted(enroll_dir.glob("*.wav")):
            if not wav_file.is_file():
                continue
            spk = wav_file.name.split("-")[0].lower()
            if spk:
                by_speaker[spk].append(wav_file)
        for wav_file in sorted(enroll_dir.glob("*.WAV")):
            if not wav_file.is_file():
                continue
            spk = wav_file.name.split("-")[0].lower()
            if spk:
                by_speaker[spk].append(wav_file)

        for speaker_name, files in sorted(by_speaker.items()):
            # Windows 下 *.wav / *.WAV 可能重复，去重
            unique_files = sorted({str(f.resolve()) for f in files})
            embs = []
            for wav_file in unique_files:
                wf, sr = torchaudio.load(str(wav_file))
                if sr != self.config.sample_rate:
                    wf = torchaudio.transforms.Resample(sr, self.config.sample_rate)(wf)
                embs.append(self.extract_from_waveform(wf))
            if embs:
                self.enrolled_speakers[speaker_name] = l2_normalize(np.mean(embs, axis=0))
                logger.info("注册声纹: %s (%d 文件)", speaker_name, len(embs))

    def match_speaker_best(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """返回余弦相似度最高的说话人（不受阈值限制，用于标注）。"""
        if not self.enrolled_speakers or embedding is None:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, ref in self.enrolled_speakers.items():
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score

    def match_speaker(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """与已注册声纹库匹配，返回 (speaker_name, score)。"""
        if not self.enrolled_speakers or embedding is None:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, ref in self.enrolled_speakers.items():
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < self.config.speaker_match_thresh:
            return None, best_score
        return best_name, best_score

    def process_segments(
        self,
        wav_path: Path,
        segments: List[Tuple[float, float]],
    ) -> List[SpeechSegment]:
        """为每个语音段提取声纹 embedding。"""
        results = []
        for start, end in segments:
            emb = self.extract_segment(wav_path, start, end)
            results.append(SpeechSegment(start_time=start, end_time=end, speaker_embedding=emb))
        return results
