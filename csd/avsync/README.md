# 音–口同步（MTDVocaLiST → SyncMatcher）

冻结 [MTDVocaLiST](https://github.com/xjchenGit/MTDVocaLiST) 的视觉/音频 prenet，提取 200 维特征，再训练轻量 `SyncMatcher` 学习「口动 ↔ 发音」是否同步。分数串联进 `SituationRouter` 的视觉侧（增强/替换 `lip_scores`）。

## 一次跑通

```bash
# 1) 源码 + 权重（约 50MB）
python scripts/setup_mtdvocalist.py

# 2) 索引视频（若尚无）
python scripts/index_xianyang_dataset.py

# 3) 弱标签样本 + 冻结特征（优先外置配套音频，含 audio/merged audio/）
python scripts/prepare_avsync_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map \
  --audio-root audio

# 4) 训下游匹配头
python scripts/train_avsync_matcher.py --data-dir output/avsync_xianyang/merged_all

# 5) diarize 时开启
# ASDConfig.use_avsync = True
# ASDConfig.avsync_matcher_path = "output/avsync_xianyang/merged_all/avsync_matcher.pt"
```

## 配套音频目录

训练优先使用外置音频（`t=0` 对齐视频），搜索顺序见 `csd/data/companion_audio.py`：

```text
audio/xianyang/{MMDD}/class{N}/g{G}.wav          # 规范路径
audio/merged audio/202507-小学-咸阳/{MMDD}-前测/*.wav   # 同门 merged 目录
audio/merged_audio/... / audio/mergedaudio/...
```

文件名需能解析 `date + 班 + 组`（与现有 audio 命名一致）。找不到外置音频时才从视频轨抽取。

## 标签

| 标签 | 构造 |
|------|------|
| 正 | GT 说话人嘴部 5 帧窗 + 对应段音频 mel |
| 负 | **同一音频** + **其他学生**嘴部窗 |

## 模块

| 文件 | 作用 |
|------|------|
| `mtd_backend.py` | 加载冻结 SyncTransformer，输出 vis/aud 特征与预训练 logit |
| `matcher.py` | 可训练匹配头 |
| `scorer.py` | 段级各说话人 sync 分 |
| `mouth_crop.py` / `mel.py` | 嘴 ROI 与 mel |

权重路径：`models/mtdvocalist/pure_MTDVocaLiST.pth`  
精简源码：`third_party/MTDVocaLiST/models/`
