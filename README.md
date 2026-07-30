# Child Social Diarization — SpeakAhead（发言准备度）训练包

**仓库：** https://github.com/Topaz191/Child-Social-Diarization  

面向 GPU 租赁平台 / 超算的精简工程：从课堂视频抽取头部/嘴部时序特征，训练轻量 LSTM，预测「即将开口」概率（SpeakAhead）。属于 Child Social Diarization（CSD）中的视觉准备度子模块。

**同门快速入口（思路 + 结果 + 代码地图）：**  
→ [`docs/SpeakAhead发言准备度.md`](docs/SpeakAhead发言准备度.md)  
→ 训练好的权重与报告：[`artifacts/readiness/`](artifacts/readiness/)

视频与完整 `output/` 不进 Git；在算力平台侧单独下载/挂载数据。小体积产物（`artifacts/`）已提交便于传阅。

## 推荐工作流（本机改代码 → GitHub → GPU 平台跑）

大多数 GPU 租赁平台都支持 `git clone` / `git pull`，这比每次打包上传更省事：

```text
本机 Cursor 改代码  →  git push 到 GitHub  →  平台上 git pull  →  训练
```

**本机（改完后）：**

```bash
cd Child-Social-Diarization   # 或你的本地目录 CDA_readiness
git add -A
git commit -m "描述本次改动"
git push
```

**GPU 平台（首次）：**

```bash
git clone https://github.com/Topaz191/Child-Social-Diarization.git
cd Child-Social-Diarization
pip install -r requirements.txt
# 按平台 CUDA 安装匹配的 torch / onnxruntime-gpu
```

**GPU 平台（之后每次开新任务）：**

```bash
cd Child-Social-Diarization
git pull
# 视频仍放在 video/xianyang/{date}/classN/ （网盘直下或平台持久盘）
bash scripts/run_cluster_pipeline.sh --require-position-map
```

数据（大视频）不要塞进 Git；用平台网盘/对象存储/持久化磁盘挂载到 `video/`。

## 目录结构

```text
Child-Social-Diarization/
├── csd/                          # 视觉特征与数据索引核心库
├── scripts/
│   ├── index_xianyang_dataset.py       # 视频 ↔ Excel 自动对应
│   ├── prepare_readiness_xianyang.py   # 抽帧特征 + 切正负样本
│   ├── train_readiness_lstm.py         # 训练 readiness LSTM
│   ├── prepare_lip_amp_xianyang.py     # 偏正脸说话段 → 嘴动幅度样本
│   ├── train_lip_amp.py                # 训练嘴动幅度标定器
│   ├── run_cluster_pipeline.sh         # readiness 一键流水线
│   └── run_lip_amp_pipeline.sh         # 嘴动幅度一键流水线
├── ref/
│   ├── 202507-xianyang-小学生转录标注.xlsx
│   ├── position_maps/                  # 画面左中右 ↔ S1/S2/S3
│   └── readiness训练数据需求说明.md
├── video/xianyang/               # 空目录：在算力侧下载视频到这里
├── output/                       # 运行产物（勿提交）
├── requirements.txt
└── README.md
```

## 视频放置约定

```text
video/xianyang/{MMDD}/class{N}/*.mp4
例: video/xianyang/0701/class1/0701-前测-五年级1班-第2组-S1S2-小组讨论.mp4
```

- sheet 自动对应：`五年级1班第2组` → `5-1-2`
- 阶段对应文件名中的 `前测/中测/后测`
- 画面稳定 3 人入镜；在 `ref/position_maps/*.json` 填写左→中→右并设 `confirmed: true`
- 远处其他组人脸会按相对最大脸尺寸过滤

## 算力平台使用步骤

```bash
# 1) 代码
git clone https://github.com/Topaz191/Child-Social-Diarization.git
cd Child-Social-Diarization
pip install -r requirements.txt
# 按平台 CUDA 安装匹配的 torch / onnxruntime-gpu

# 2) 数据：视频放到约定目录（网盘直下或挂载盘）
#    video/xianyang/0701/class1/...

# 3) 确认位置标注后跑流水线
python scripts/validate_position_maps.py ref/position_maps/0701_class1.json
bash scripts/run_cluster_pipeline.sh --require-position-map

# 或分步试跑：
python scripts/index_xianyang_dataset.py
python scripts/prepare_readiness_xianyang.py --from-manifest output/xianyang/manifest.json --require-position-map --limit 2
python scripts/prepare_readiness_xianyang.py --from-manifest output/xianyang/manifest.json --require-position-map --skip-existing
python scripts/train_readiness_lstm.py --data-dir output/readiness_xianyang/merged_all --epochs 40 --hidden 64
```

## 儿童嘴动幅度标定（lip-amp）

用「转录说话段 + 偏正脸」弱监督，学习儿童 MAR 活跃度尺度（替换 `visual_conf` 里硬编码 `0.015`）。

```bash
# 一键（可加 --limit 2 试跑；已有 readiness 特征时可加 --skip-extract）
bash scripts/run_lip_amp_pipeline.sh

# 或分步
python scripts/index_xianyang_dataset.py
python scripts/prepare_lip_amp_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map \
  --reuse-readiness-root output/readiness_xianyang \
  --skip-existing
python scripts/train_lip_amp.py --data-dir output/lip_amp_xianyang/merged_all
```

产物：

- `output/lip_amp_xianyang/merged_all/lip_amp_model.pt`
- `output/lip_amp_xianyang/merged_all/lip_amp_scale.json`（`activity_scale` ≈ 正样本活跃度 p75）
- `VisualConfidenceEstimator` 会自动加载上述文件（若存在）

## 产物位置

- 索引：`output/xianyang/manifest.json`
- 单场样本：`output/readiness_xianyang/{date}_{sheet}_{pre|mid|post}_{cams}/`
- 合并训练数据：`output/readiness_xianyang/merged_all/readiness_samples.npz`
- 模型：`output/readiness_xianyang/merged_all/readiness_model.pt`
- 报告：`output/readiness_xianyang/merged_all/train_report.json`
- 嘴动幅度：`output/lip_amp_xianyang/merged_all/`

## 与本地完整工程的关系

- 本仓库服务 **SpeakAhead / readiness** 与 **儿童嘴动幅度标定** 规模化训练。
- 动态可信度融合 / SituationRouter 集成仍在本地完整工程；模型训好后拷回集成。
