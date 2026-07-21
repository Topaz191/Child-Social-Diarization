"""预训练模型下载与路径解析。"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# lindevs/yolov8-face (MIT)
YOLOV8N_FACE_URL = (
    "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
)
YOLOV8N_FACE_FALLBACK_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


def ensure_yolov8_face_weights(cache_dir: Path, filename: str = "yolov8n-face-lindevs.pt") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target

    logger.info("下载 YOLOv8-face 权重: %s", YOLOV8N_FACE_URL)
    tmp = target.with_suffix(".tmp")
    last_err: Exception | None = None
    for url in (YOLOV8N_FACE_URL, YOLOV8N_FACE_FALLBACK_URL):
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            logger.info("  尝试: %s", url)
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            logger.info("YOLOv8-face 权重已保存: %s", target)
            return target
        except Exception as exc:
            last_err = exc
            logger.warning("  下载失败: %s", exc)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"无法下载 YOLOv8-face 权重，请手动放置到 {target} "
        f"或设置 config.yolov8_face_model"
    ) from last_err
