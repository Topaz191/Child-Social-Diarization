#!/usr/bin/env bash
# 集群侧一键流水线（视频已放到 video/xianyang/{date}/classN/ 之后）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] 索引视频 ↔ 标注"
python scripts/index_xianyang_dataset.py

echo "[2/4] 抽特征 + 切样本（可加 --limit 2 试跑）"
python scripts/prepare_readiness_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map \
  --skip-existing \
  "$@"

echo "[3/4] 训练 LSTM"
python scripts/train_readiness_lstm.py \
  --data-dir output/readiness_xianyang/merged_all \
  --epochs 40 \
  --hidden 64

echo "[4/4] 完成。模型: output/readiness_xianyang/merged_all/readiness_model.pt"
