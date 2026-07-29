# 咸阳外置配套音频（与视频时间轴对齐）

小组讨论视频轨录音往往不清晰，可为每个讨论场次另备一路干净音频。

## 对齐键（重要）

**同一日期 + 同一班 + 同一组 → 只有一路配套音频**（不区分机位 `S1S2` / `S2S3`）。  
多机位剪辑视频共用这一路；靠文件名（或目录）提取 `date / class_id / group` 与视频对齐。

时间轴：**音频 t=0 = 视频第 0 帧**（前端对齐，不做偏移）。

## 推荐目录

```text
audio/xianyang/{MMDD}/class{N}/g{G}.wav
```

示例（0701 · 1班 · 第1组）：

```text
video/xianyang/0701/class1/0701-前测-5年级1班-第1组-S2S3-小组讨论.mp4
audio/xianyang/0701/class1/g1.wav
```

### 同门 merged 目录（也可直接用）

整树放到仓库 `audio/` 下即可被自动搜索，例如：

```text
audio/merged audio/202507-小学-咸阳/0701-前测/<音频文件>
audio/merged audio/202507-小学-咸阳/0706-后测/<音频文件>
```

服务器示例：

```text
/root/autodl-tmp/Child-Social-Diarization/audio/merged audio/202507-小学-咸阳/0701-前测/...
```

音频文件名格式与现有一致（含日期/班/组），例如 `0701-前测-五年级1班-第4组-讨论-c.wav`。  
解析时按 `date+class+group` 对齐视频；若同 key 多文件，优先路径中阶段（前测/中测/后测）与视频一致者。

也可用描述性文件名（仍放在对应 `date/class` 下，或扁平放在 `audio/xianyang/`），只要能解析出三者，例如：

```text
0701-前测-五年级1班-第1组-讨论-c.wav
0701-前测-五年级1班-第1组.wav
```

解析代码：`csd/data/companion_audio.py`。

## 优先级

1. CLI `--audio`
2. 规范路径 `audio/xianyang/{date}/class{N}/g{G}.*`
3. 同 key 的其它命名文件
4. 视频旁同名音频
5. 都没有 → 从视频轨 ffmpeg 抽取（回退）

## 格式

优先 **16 kHz 单声道 WAV**；也可用 flac / m4a / mp3。
