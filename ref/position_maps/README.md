# 画面位置 ↔ 说话人 人工标注

每个 `{date}_class{N}.json` 对应一天一个班的视频。

## 场景约定

- 画面稳定 **3 人**入镜（左 / 中 / 右）
- 文件名里的 `S1S2` / `S2S3` 只是机位标签，**不是**入镜人数
- 远处其他组脸更小，检测时会按「相对最大脸面积」过滤

## 你要改的字段

对每个视频只改两处：

1. `left_to_right`：必须 3 个，例如 `["S2", "S1", "S3"]`（左→中→右）
2. `confirmed`：核对无误后改为 `true`

## 生成 / 校验

```bash
python scripts/make_position_map_templates.py --date 0701 --class-id 1
python scripts/validate_position_maps.py ref/position_maps/0701_class1.json
```
