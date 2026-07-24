"""预训练模型下载与路径解析。"""

from __future__ import annotations

import logging
import os
import socket
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# lindevs/yolov8-face (MIT)；国内实例优先走镜像
YOLOV8N_FACE_URLS = (
    "https://ghproxy.net/https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
    "https://mirror.ghproxy.com/https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
    "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
)


def ensure_yolov8_face_weights(cache_dir: Path, filename: str = "yolov8n-face-lindevs.pt") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target

    # 允许环境变量指定已下载好的权重，彻底跳过联网
    env_path = os.environ.get("YOLOV8_FACE_WEIGHTS", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file() and p.stat().st_size > 1_000_000:
            logger.info("使用环境变量 YOLOV8_FACE_WEIGHTS: %s", p)
            return p

    timeout = float(os.environ.get("CSD_MODEL_DOWNLOAD_TIMEOUT", "30"))
    logger.info("下载 YOLOv8-face 权重（单次超时 %.0fs）…", timeout)
    tmp = target.with_suffix(".tmp")
    last_err: Exception | None = None
    socket.setdefaulttimeout(timeout)
    for url in YOLOV8N_FACE_URLS:
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            logger.info("  尝试: %s", url)
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError(f"下载文件过小: {tmp.stat().st_size} bytes")
            tmp.replace(target)
            logger.info("YOLOv8-face 权重已保存: %s", target)
            return target
        except Exception as exc:
            last_err = exc
            logger.warning("  下载失败: %s", exc)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"无法下载 YOLOv8-face 权重。请任选其一：\n"
        f"  1) 本机下载后 scp 到 {target}\n"
        f"  2) export YOLOV8_FACE_WEIGHTS=/path/to/yolov8n-face-lindevs.pt\n"
        f"  3) 改用 MediaPipe：在 prepare 时设 face_backend=mediapipe\n"
        f"原始错误: {last_err}"
    ) from last_err
