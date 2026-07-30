# SpeakAhead：发言准备度（Readiness）— 思路、流水线与训练结果

面向同门快速了解：**要解决什么问题、怎么做、代码在哪、训出了什么结果**。

仓库：https://github.com/Topaz191/Child-Social-Diarization  

---

## 1. 要解决什么

在儿童小组讨论 diarization 里，「谁即将开口」往往早于明显嘴动。SpeakAhead 用**纯视觉短时序**预测每个人的 **发言准备度 ∈ [0,1]**，作为 CSD 路由里的**前瞻先验**（进 `visual_probs`，不进 `visual_conf`）。

| | 说明 |
|--|------|
| 正样本 | 某人**真实开口前**约 `window_sec`（默认 0.75s）的头姿/嘴动序列 |
| 负样本 | 远离任意说话段的聆听/静音窗口 |
| 输出 | 每人一个 readiness 分数；推理时在网格点 / 段起点 `t` 上对 `[t−window, t]` 打分 |

与「当前谁在说」的嘴动/声纹不同：它回答的是 **谁马上要说**。

---

## 2. 方法思路（简图）

```text
视频 + 位置映射(左中右→S1/S2/S3) + Excel 说话段 GT
        │
        ▼
逐帧：头姿 yaw/pitch/roll + 嘴开合 + 可见性/侧脸
        │
        ▼
特征增强 → 18 维（静态6 + 差分/短窗8 + 组内相对4）
        │
        ▼
按 GT onset 切正样本窗口；远离说话段切负样本
        │
        ▼
单向 LSTM + 线性头 → readiness_score
        │
        ▼
接入主推理：SituationRouter / dense_grid 的 visual 侧
```

### 18 维特征

| 组 | 列 |
|----|-----|
| 静态 6 | `yaw, pitch, roll, mouth_opening, visibility, side_face_weight` |
| 动态 8 | `d_yaw, d_pitch, d_roll, d_mouth, mouth_mean/std/max_short, mouth_trend` |
| 相对 4 | `others_mouth_mean/max, mouth_rel, others_still` |

设计动机：开口前常见抬头/转向/嘴从静到动；组内相对量帮助「别人在听、我在准备」。

### 模型

- 轻量 **LSTM**（`scripts/train_readiness_lstm.py` 中 `ReadinessLSTM`）
- 输入：`(seq_len=16, feat=18)` 左右
- 损失：二分类；报告 accuracy / AUC / pos·neg 均值分数

---

## 3. 代码入口

| 文件 | 作用 |
|------|------|
| [`scripts/prepare_readiness_xianyang.py`](../scripts/prepare_readiness_xianyang.py) | 抽帧特征 + 切正负样本 → `readiness_samples.npz` |
| [`scripts/train_readiness_lstm.py`](../scripts/train_readiness_lstm.py) | 训练 → `readiness_model.pt` + `train_report.json` |
| [`scripts/eval_readiness_turn_events.py`](../scripts/eval_readiness_turn_events.py) | 话轮事件协议评估（可选） |
| [`csd/trust/speak_ahead.py`](../csd/trust/speak_ahead.py) | 推理封装：`score_at` / `score_onset` |
| [`csd/eval/turn_event_protocol.py`](../csd/eval/turn_event_protocol.py) | MuVAP 风格事件评估 |
| [`ref/readiness训练数据需求说明.md`](../ref/readiness训练数据需求说明.md) | 对外数据需求说明 |

一键流水线（若脚本存在）：`bash scripts/run_cluster_pipeline.sh --require-position-map`

### 复现训练（服务器）

```bash
cd /root/autodl-tmp/Child-Social-Diarization
git pull
# 视频: video/xianyang/{MMDD}/class{N}/
# 位置: ref/position_maps/*.json confirmed=true

python scripts/index_xianyang_dataset.py
python scripts/prepare_readiness_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map --skip-existing
python scripts/train_readiness_lstm.py \
  --data-dir output/readiness_xianyang/merged_all \
  --epochs 40 --hidden 64
```

---

## 4. 当前公开的训练结果（咸阳 merged）

路径：[`artifacts/readiness/`](../artifacts/readiness/)

| 文件 | 说明 |
|------|------|
| `readiness_model.pt` | 合并样本上训好的权重（约 105 KB） |
| `train_report.json` | 指标与末尾 epoch 曲线 |
| `dataset_summary.json` | 样本数、特征列、张量形状 |

### 数据规模

| 项 | 值 |
|----|-----|
| 样本数 | 820（正 410 / 负 410） |
| 张量 | `(820, 16, 18)` |
| 划分 | train 656 / val 164 |

### 指标（摘自 `train_report.json`）

| 指标 | 值 |
|------|-----|
| 全集 accuracy | ≈ **0.954** |
| 全集 AUC | ≈ **0.983** |
| best val AUC | ≈ **0.907** |
| 正样本均分 / 负样本均分 | ≈ 0.95 / 0.08 |

说明：

- 验证集 AUC ~0.90，说明在「开口前窗 vs 远离说话窗」二分类上可分。
- **lift** 检查未通过（`n_checked=0`）：窗口内上升趋势诊断样本不足，不代表分类无效。
- 接入 diarization 后仍可能受 **槽位/跟踪、听者准备度抬高** 等影响；见本地试跑纪要（完整工程 `CDA/ref/试跑系统架构与问题纪要-0706g4.md`）。

加载示例：

```python
from csd.trust.speak_ahead import SpeakAheadScorer
scorer = SpeakAheadScorer("artifacts/readiness/readiness_model.pt")
# scores = scorer.score_at(timeline, fps, t_pred)
```

---

## 5. 与完整 CSD 的关系

- **本仓库**：SpeakAhead 数据准备、训练、评估、以及音–口 sync 等可在 GPU 上跑的部分。
- **本地完整工程 CDA**：`SituationalDiarizer`、密网格、社交仲裁、试跑评测等主推理链；训好的 `readiness_model.pt` 拷回后通过 `--use-readiness` / `ASDConfig.use_readiness` 接入。

---

## 6. 已知局限（给同门的注意点）

1. 标签依赖 Excel 说话段 + 位置映射；映射错 → 正样本脏。  
2. 准备度高 ≠ 当前说话人（听者也会抬升）。  
3. 部分会话特征几乎无某说话人正例时，该人 readiness 会长期接近 0。  
4. 大文件（`frame_features.csv`、原始视频、全量 `npz`）不进 Git；训练在 `output/` 本地/服务器生成。
