# 画面位置 ↔ 说话人 人工标注

每个 `{date}_class{N}.json` 对应一天一个班的视频。

## 场景约定

- 画面稳定 **3 人**入镜（左 / 中 / 右）
- 文件名里的 `S1S2` / `S2S3` 只是机位标签，**不是**入镜人数；请使用多人机位视频，单人机位（如 `-S1-`）不要用于 readiness
- 活动名与服务器统一为 **`讨论`**（不是「小组讨论」）
- 远处其他组脸更小，检测时会按「相对最大脸面积」过滤

## 你要改的字段

对每个视频主要改：

1. `left_to_right`：必须 3 个，例如 `["S2", "S1", "S3"]`（左→中→右）
2. `confirmed`：核对无误后改为 `true`
3. `video_start_abs` / `video_start_abs_str`：视频第 0 帧的绝对时刻（从 `annotation_note` 自动抽取）。**若为 `null`，默认视为无效，不参与 readiness 训练。** Excel 的 Start/End 若是相对视频时间则直接用；若是一天内绝对时刻，脚本会减去该字段对齐到视频时间轴。

## 生成 / 校验

默认**按 Excel sheet（如 `5-3-1`…`5-3-4`）生成该班全部组**，不依赖本地是否已有 mp4。  
`video_name` 为占位猜测（日期 + `--phase` + 组号）；若与真实文件名不一致请改正。

```bash
# 0701 前测 · 3 班 → Excel 里 5-3-* 各组都会进模板
python scripts/make_position_map_templates.py --date 0701 --class-id 3 --phase 前测

# 0706 后测
python scripts/make_position_map_templates.py --date 0706 --class-id 3 --phase 后测 --force

# 旧行为：只扫本地已有视频
python scripts/make_position_map_templates.py --date 0701 --class-id 3 --from-videos --force

python scripts/validate_position_maps.py ref/position_maps/0701_class3.json
```
