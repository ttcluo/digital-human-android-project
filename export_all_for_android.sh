#!/bin/bash
# 一键导出 Android 完整推理所需资源
# 用法: ./export_all_for_android.sh <音频路径> <dataset路径> [dataset_max_秒]
# 示例: ./export_all_for_android.sh data/preview.mp3 data/raw 20

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <音频路径> <dataset路径> [dataset_max_秒]"
  echo "  音频: mp3 或 wav，用于生成 wenet_feat_stream.bin 和 preview.mp3"
  echo "  dataset: 需包含 full_body_img/ 和 landmarks/"
  echo "  dataset_max_秒: 可选，默认 20，只导出前 N 秒帧（显著减小体积）"
  echo ""
  echo "示例: $0 data/preview.mp3 data/raw 20"
  exit 1
fi

AUDIO="$1"
DATASET="$2"
DATASET_MAX="${3:-20}"

echo "=== 1. 导出 WeNet 特征和音频 ==="
"$ROOT/export_preview_for_android.sh" "$AUDIO"

echo ""
echo "=== 2. 导出 Dataset（前 ${DATASET_MAX} 秒）==="
"$ROOT/export_dataset_for_android.sh" "$DATASET" "$DATASET_MAX"

echo ""
echo "=== 完成 ==="
echo "请确保 unet_ondevice_128.onnx 已放入 android/app/src/main/assets/"
echo "然后编译运行 Android 应用，进入「完整推理」生成视频。"
