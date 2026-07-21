# CDA_readiness — 发言准备度（Speech Readiness）训练包

面向远程 GPU / 超算的精简工程：从咸阳课堂视频抽取头部/嘴部时序特征，训练轻量 LSTM，预测「即将开口」概率。

不要把完整 `CDA` 原仓库（含数 GB 视频与 legacy 产物）推到 GitHub；本仓库已用 `.gitignore` 排除视频与 `output/`。

## 目录结构

```text
CDA_readiness/
├── csd/                          # 视觉特征与数据索引核心库
├── scripts/
│   ├── index_xianyang_dataset.py       # 视频 ↔ Excel 自动对应
│   ├── prepare_readiness_xianyang.py   # 抽帧特征 + 切正负样本
│   ├── train_readiness_lstm.py         # 训练 readiness LSTM
│   └── run_cluster_pipeline.sh         # 一键流水线
├── ref/
│   ├── 202507-xianyang-小学生转录标注.xlsx
│   └── readiness训练数据需求说明.md
├── video/xianyang/               # 空目录：在集群上下载视频到这里
├── output/                       # 运行产物（勿上传）
├── requirements.txt
└── README.md
```

## 视频放置约定（与本地一致）

```text
video/xianyang/{MMDD}/class{N}/*.mp4
例: video/xianyang/0701/class1/0701-前测-五年级1班-第2组-S1S2-小组讨论.mp4
```

- sheet 自动对应：`五年级1班第2组` → `5-1-2`
- 阶段对应文件名中的 `前测/中测/后测`
- 机位 `S1S2` / `S2S3` 用于画面从左到右对齐说话人（无需 position_gt JSON）

## 上传什么 / 不上传什么

| 上传到集群 | 不要上传 |
| :--- | :--- |
| 本目录全部代码 + `ref/*.xlsx` | `CDA/video/**` 大视频（在集群侧下载） |
| `requirements.txt` | `CDA/audio`、`legacy`、`output`、`*.pkl`、PNG |
| （可选）已切好的 `*.npz` 小样本 | `models/` 权重缓存（集群可重下） |

建议打包：

```bash
# 在本机 hhproject 目录
tar --exclude='CDA_readiness/video/**/*.mp4' \
    --exclude='CDA_readiness/output/**' \
    --exclude='**/__pycache__' \
    -czf CDA_readiness.tar.gz CDA_readiness
```

## 集群使用步骤

```bash
# 1) 解压并建环境
tar -xzf CDA_readiness.tar.gz && cd CDA_readiness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 按集群 CUDA 版本安装匹配的 torch / onnxruntime-gpu

# 2) 把网盘视频下到约定目录（在集群上直下，勿经本地中转）
#    video/xianyang/0701/class1/...
#    video/xianyang/0704/class2/...

# 3) 索引 → 抽特征切样本 → 训练
bash scripts/run_cluster_pipeline.sh
# 或分步：
python scripts/index_xianyang_dataset.py
python scripts/prepare_readiness_xianyang.py --from-manifest output/xianyang/manifest.json --limit 2  # 先试跑
python scripts/prepare_readiness_xianyang.py --from-manifest output/xianyang/manifest.json --skip-existing
python scripts/train_readiness_lstm.py --data-dir output/readiness_xianyang/merged_all --epochs 40 --hidden 64
```

## 产物位置

- 索引：`output/xianyang/manifest.json`
- 单场样本：`output/readiness_xianyang/{date}_{sheet}_{pre|mid|post}_{cams}/`
- 合并训练数据：`output/readiness_xianyang/merged_all/readiness_samples.npz`
- 模型：`output/readiness_xianyang/merged_all/readiness_model.pt`

## 与完整 CDA 的关系

- 本包只服务 **readiness LSTM 规模化训练**。
- 动态可信度融合 / SituationRouter 集成仍在原 `CDA` 仓库；模型训好后再拷回集成。
