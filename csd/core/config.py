"""全局配置与默认超参数。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ASDConfig:
    """主动说话人检测流水线配置。"""

    # 路径
    output_dir: Path = Path("output")
    model_cache_dir: Path = Path("models")

    # 视频 / 音频
    sample_rate: int = 16000
    min_speech_duration: float = 0.5  # VAD 最短语音段（秒）
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 100
    # 多 VAD 融合: silero | union | union_lip
    vad_mode: str = "silero"
    embed_vad_threshold: float = 0.3
    embed_vad_window: float = 1.0
    embed_vad_hop: float = 0.5
    embed_vad_enroll_gate: bool = True  # 需与 post 声纹相似才计为 speech
    embed_vad_enroll_thresh: float = 0.28
    lip_speech_thresh: float = 0.002  # 嘴动活跃度阈值（MAR 方差+差分）

    # 人脸检测与跟踪
    face_det_thresh: float = 0.35
    face_det_size: tuple = (640, 640)
    track_iou_thresh: float = 0.3
    track_max_lost_frames: int = 30
    position_cluster_eps: float = 0.08  # DBSCAN eps（归一化坐标）
    position_cluster_min_samples: int = 5
    # 只保留与当帧最大人脸同量级的脸，滤掉远处其他组（面积比）
    face_size_min_ratio_to_max: float = 0.25
    # 轨迹级二次过滤：轨迹平均脸面积须 >= 最大轨迹的该比例
    track_face_size_min_ratio_to_max: float = 0.25
    primary_group_n_slots: int = 3  # 本组稳定入镜人数（左/中/右）

    # 头部/嘴部 Face Mesh（ROI 内）
    head_mesh_det_conf: float = 0.35
    head_mesh_track_conf: float = 0.35
    head_mesh_static_roi: bool = True  # ROI 裁剪后逐帧检测，远距离更稳
    speech_pose_frame_skip: int = 2  # VAD 段内加密采样步长（1=每帧）
    speech_pose_pad_sec: float = 0.15  # VAD 段前后扩展
    bbox_interp_max_gap: int = 18  # bbox 插值最大间隔帧数（约 0.6s@30fps）

    # 人脸识别
    face_match_thresh: float = 0.45  # 余弦相似度阈值
    insightface_model: str = "buffalo_l"
    face_backend: str = "auto"  # auto / yolov8face / retinaface / insightface / mediapipe
    yolov8_face_model: str = ""  # 空则自动下载到 model_cache_dir

    # 融合路由校准
    router_voice_margin_cap: float = 0.18  # 声纹 margin 高于此值时限制视觉权重
    router_visual_cap_strong_audio: float = 0.28
    router_visual_cap_weak_lip: float = 0.22
    router_lip_min_for_visual: float = 0.12
    router_disagree_margin: float = 0.15  # 声纹明确且与视觉 top1 不一致时降 visual
    router_visual_led_min_vc: float = 0.52
    router_visual_led_margin_max: float = 0.18

    # 声纹
    speaker_match_thresh: float = 0.55
    speaker_enroll_min_duration: float = 1.0
    speaker_backend: str = "auto"  # auto / pyannote / speechbrain
    pyannote_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speechbrain_savedir: str = "tmp/speechbrain_ecapa"

    # 嘴部运动
    mar_activity_thresh: float = 0.002  # MAR 方差阈值
    lip_weight: float = 0.4
    voice_weight: float = 0.4
    position_weight: float = 0.2
    conflict_penalty: float = 0.3  # 声纹与嘴动不一致时降权

    # 设备
    device: str = "auto"  # auto / cpu / cuda
    frame_skip: int = 1  # 每隔 N 帧处理一次（1=每帧）

    # 可选：预注册声纹库目录（speaker_name/*.wav）
    speaker_enroll_dir: Optional[Path] = None
    generate_video: bool = True
    hf_token: Optional[str] = None  # 默认读环境变量 HF_TOKEN

    def get_hf_token(self) -> Optional[str]:
        if self.hf_token:
            return self.hf_token
        for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            val = os.environ.get(env_key)
            if val:
                return val
        # 与 ms.py / eval.py 等脚本共用项目根目录 config.json
        for cfg_path in (Path("config.json"), Path(__file__).resolve().parent.parent / "config.json"):
            if cfg_path.is_file():
                try:
                    import json
                    with open(cfg_path, encoding="utf-8") as f:
                        data = json.load(f)
                    token = data.get("HF_TOKEN") or data.get("hf_token")
                    if token:
                        return str(token)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
        # 与 real_pca_v1(1).py 等脚本一样，支持直接写在 hf_auth.py
        try:
            from hf_auth import HF_TOKEN as code_token
            if code_token:
                return str(code_token)
        except ImportError:
            pass
        return None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
