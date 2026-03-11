# Ultralight-Digital-Human 推理步骤（WeNet）

训练完成后，用新音频生成数字人视频的完整流程。推理音频路径：`/data/luochuan/digital-human-android-project/data/preview.mp3`。

## 1. 将 MP3 转为 16kHz 单声道 WAV

WeNet 特征提取要求 16kHz；`wenet_infer.py` 按 `.wav` 后缀生成 `*_wenet.npy`，因此先用 WAV。

```bash
ffmpeg -y -i /data/luochuan/digital-human-android-project/data/preview.mp3 \
  -ar 16000 -ac 1 \
  /data/luochuan/digital-human-android-project/data/preview.wav
```

## 2. 提取 WeNet 音频特征

在 `Ultralight-Digital-Human/data_utils` 下执行（依赖当前目录的 `conf/` 和 `encoder.onnx`）：

```bash
cd /data/luochuan/digital-human-android-project/Ultralight-Digital-Human/data_utils

python wenet_infer.py /data/luochuan/digital-human-android-project/data/preview.wav
```

会生成：`/data/luochuan/digital-human-android-project/data/preview_wenet.npy`。

## 3. 运行视频推理

训练每 5 个 epoch 保存一次，200 epoch 时最后一档为 `195.pth`。使用该 checkpoint：

```bash
cd /data/luochuan/digital-human-android-project/Ultralight-Digital-Human

python inference.py \
  --asr wenet \
  --dataset /data/luochuan/digital-human-android-project/data/raw \
  --audio_feat /data/luochuan/digital-human-android-project/data/preview_wenet.npy \
  --save_path /data/luochuan/digital-human-android-project/data/result_preview.mp4 \
  --checkpoint ./checkpoint_wenet/195.pth
```

输出为无声视频：`data/result_preview.mp4`（20fps）。

## 4. 合入原音频得到最终视频

```bash
ffmpeg -y \
  -i /data/luochuan/digital-human-android-project/data/result_preview.mp4 \
  -i /data/luochuan/digital-human-android-project/data/preview.wav \
  -c:v libx264 -c:a aac \
  /data/luochuan/digital-human-android-project/data/result_preview_with_audio.mp4
```

最终带口型与声音的视频：`data/result_preview_with_audio.mp4`。

---

## 路径与 checkpoint 说明

| 项 | 路径/说明 |
|----|-----------|
| 训练数据目录 | `--dataset` 使用 `data/raw`（含 `full_body_img/`、`landmarks/`） |
| 最后一档 checkpoint | `checkpoint_wenet/195.pth`（对应 epoch 196；每 5 个 epoch 存一次：0, 5, …, 195） |
| 若用其他 epoch | 把 `195.pth` 换成 `0.pth`、`5.pth`、…、`190.pth` 等 |

## 换用其他推理音频

1. 将新音频转为 16kHz WAV（若已是 WAV 可跳过）：  
   `ffmpeg -y -i your_audio.mp3 -ar 16000 -ac 1 your_audio.wav`
2. 在 `data_utils` 下：  
   `python wenet_infer.py /path/to/your_audio.wav`
3. 用生成的 `your_audio_wenet.npy` 跑 `inference.py` 的 `--audio_feat`，再用 `ffmpeg` 合入 `your_audio.wav`。
