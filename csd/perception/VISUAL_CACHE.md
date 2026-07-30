# 共享视觉特征缓存（轨迹 / 唇轮廓 / 视线）

跨 SpeakAhead / AVsync / 可视化复用的中间产物，避免重复人脸检测，并为唇动与视线研究提供可分析的关键点序列。

## 一次抽取

```bash
python scripts/extract_visual_cache_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map \
  --skip-existing
```

常用参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--frame-skip` | 3 | 全局稀疏采样（轨迹+头姿+唇+注意力） |
| `--dense-skip` | 1 | GT 说话段 ±`--speech-pad-sec` 内加密唇轮廓 |
| `--speech-pad-sec` | 1.0 | 加密窗口前后扩展（秒） |
| `--full-mesh` | off | 额外存全脸 mesh（体积大） |

## 目录结构

`output/visual_cache/<session_key>/`，例如 `0706_class1_g4_后测`：

| 文件 | 内容 |
|------|------|
| `meta.json` | schema、fps、座位图、唇索引表、槽位映射 |
| `tracks.npz` | 稀疏人脸轨迹（可重建 `FaceTrack`） |
| `mesh.npz` | 每处理帧×槽位：bbox、头姿 6 标量、`lips_xy`、`mar4_xy`、可选 `mesh_xy` |
| `attention.npz` | 与 mesh 帧对齐的 yaw-proxy 注意力 `[T,S,S+1]`（末列 `__elsewhere__`） |
| `lips_timeseries.parquet` 或 `.csv` | 长表：`t,speaker,point_id,x,y,mar,densified` |

## 唇关键点

- **唇环** `LIPS`（约 32 点）：与可视化描摹一致，见 `csd/perception/face_mesh_indices.py`
- **MAR4** `(13,14,61,291)`：上/下唇内中点 + 左右嘴角，与正式 MAR 分数一致

视线当前为 **yaw + 槽位位置** 的 cos→softmax（非瞳孔眼动）；原始 `yaw/pitch/roll` 与注意力矩阵一并落盘，便于后续换模型。

## prepare 复用

```bash
# AVsync：优先读 tracks.npz，没有则检测并写回 tracks
python scripts/prepare_avsync_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map --audio-root audio \
  --visual-cache-root output/visual_cache

# Readiness：有 mesh.npz 则直接导出帧特征；否则复用 tracks 再扫 pose
python scripts/prepare_readiness_xianyang.py \
  --from-manifest output/xianyang/manifest.json \
  --require-position-map \
  --visual-cache-root output/visual_cache
```

## API

- `csd.perception.visual_cache.ensure_tracks` — 读/写轨迹
- `csd.perception.visual_cache.build_visual_cache_for_video` — 完整抽取
- `csd.perception.visual_cache.load_visual_cache` — 加载 bundle
