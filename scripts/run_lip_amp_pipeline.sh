#!/usr/bin/env bash
# 儿童嘴动幅度标定：索引 → 抽/复用特征 → 切偏正脸说话样本 → 训练
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT_ARGS=()
SKIP_EXTRACT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT_ARGS+=(--limit "$2"); shift 2 ;;
    --skip-extract)
      SKIP_EXTRACT=1; shift ;;
    *)
      echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "[1/4] 索引视频 ↔ 标注"
python scripts/index_xianyang_dataset.py

echo "[2/4] 准备 lip-amp 样本（confirmed position_map + video_start_abs + 偏正脸）"
PREP=(python scripts/prepare_lip_amp_xianyang.py
  --from-manifest output/xianyang/manifest.json
  --require-position-map
  --require-video-start-abs
  --reuse-readiness-root output/readiness_xianyang
  --skip-existing
  "${LIMIT_ARGS[@]}")
if [[ "$SKIP_EXTRACT" -eq 1 ]]; then
  PREP+=(--skip-extract)
fi
"${PREP[@]}"

echo "[3/4] 训练嘴动幅度标定器"
python scripts/train_lip_amp.py \
  --data-dir output/lip_amp_xianyang/merged_all \
  --epochs 60 \
  --hidden 32

echo "[4/4] 完成"
echo "  模型: output/lip_amp_xianyang/merged_all/lip_amp_model.pt"
echo "  尺度: output/lip_amp_xianyang/merged_all/lip_amp_scale.json"
echo "  报告: output/lip_amp_xianyang/merged_all/train_report.json"
