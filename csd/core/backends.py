"""后端自动检测与选择。"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_face_backend(requested: str = "auto") -> str:
    """
    人脸检测后端优先级（auto）:
    yolov8face > retinaface > insightface > mediapipe
    """
    if requested != "auto":
        return requested
    if _has_module("ultralytics"):
        logger.info("使用 YOLOv8-face 人脸检测（ultralytics）")
        return "yolov8face"
    if _has_module("insightface"):
        logger.info("使用 RetinaFace/SCRFD 人脸检测（insightface detection）")
        return "retinaface"
    if _has_module("mediapipe"):
        logger.info("未检测到 ultralytics/insightface，使用 MediaPipe 人脸检测")
        return "mediapipe"
    raise ImportError(
        "无人脸检测后端可用。请安装其一: "
        "pip install ultralytics  或  pip install insightface onnxruntime  或  pip install mediapipe"
    )


def resolve_speaker_backend(requested: str = "auto") -> str:
    """
    声纹后端优先级（auto）:
    pyannote > speechbrain
    """
    if requested != "auto":
        return requested
    if _has_module("pyannote"):
        logger.info("使用 pyannote 声纹模型（与 pyannote0 环境兼容）")
        return "pyannote"
    if _has_module("speechbrain"):
        return "speechbrain"
    raise ImportError(
        "无声纹后端可用。请安装其一: pip install pyannote.audio  或  pip install speechbrain"
    )
