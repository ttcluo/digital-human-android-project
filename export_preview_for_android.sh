#!/bin/bash
# 将 preview.mp3 转为 Android 端可用的 wenet_feat_stream.bin
# 用法: ./export_preview_for_android.sh [preview.mp3 路径]
# 默认从 data/preview.mp3 读取，输出到 android/app/src/main/assets/

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
ULTRA="${ROOT}/Ultralight-Digital-Human"
DATA_UTILS="${ULTRA}/data_utils"
ASSETS="${ROOT}/android/app/src/main/assets"

AUDIO="${1:-${ROOT}/data/preview.mp3}"
if [[ ! -f "$AUDIO" ]]; then
  echo "错误: 未找到音频文件 $AUDIO"
  echo "用法: $0 [preview.mp3 路径]"
  exit 1
fi

AUDIO="$(cd "$(dirname "$AUDIO")" && pwd)/$(basename "$AUDIO")"
BASE="${AUDIO%.*}"
WAV="${BASE}.wav"
NPY="${BASE}_wenet.npy"
BIN="${ASSETS}/wenet_feat_stream.bin"

echo "[1/4] 转 16kHz 单声道 wav..."
ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 "$WAV"

echo "[2/4] 提取 WeNet 特征..."
(cd "$DATA_UTILS" && python wenet_infer.py "$WAV")

if [[ ! -f "$NPY" ]]; then
  echo "错误: wenet_infer 未生成 $NPY"
  exit 1
fi

echo "[3/4] 导出 Android bin 格式..."
python "${ULTRA}/export_wenet_feat_for_android.py" --audio_npy "$NPY" --out "$BIN"

echo "[4/4] 复制 preview.mp3 到 assets..."
mkdir -p "$ASSETS"
cp "$AUDIO" "${ASSETS}/preview.mp3"

echo "完成: wenet_feat_stream.bin 和 preview.mp3 已写入 ${ASSETS}/"
