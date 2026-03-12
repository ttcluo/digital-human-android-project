#!/bin/bash
# 将 dataset（full_body_img + landmarks）导出到 Android assets
# 用法: ./export_dataset_for_android.sh <dataset_dir> [max_seconds]
# dataset_dir 需包含 full_body_img/ 和 landmarks/
# max_seconds: 可选，只导出前 N 秒的帧（20fps，默认 20 秒≈400 帧），大幅减小体积

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
ASSETS="${ROOT}/android/app/src/main/assets/dataset"
FPS=20

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <dataset_dir> [max_seconds]"
  echo "  dataset_dir 需包含 full_body_img/ 和 landmarks/"
  echo "  max_seconds: 可选，默认 20，只导出前 N 秒（推理足够用，可显著减小体积）"
  echo "  示例: $0 data/raw        # 导出全部"
  echo "  示例: $0 data/raw 20    # 只导出前 20 秒"
  exit 1
fi

SRC="$(cd "$1" && pwd)"
MAX_SEC="${2:-20}"

if [[ ! -d "$SRC/full_body_img" ]] || [[ ! -d "$SRC/landmarks" ]]; then
  echo "错误: $SRC 需包含 full_body_img/ 和 landmarks/"
  exit 1
fi

mkdir -p "${ASSETS}/full_body_img" "${ASSETS}/landmarks"

if [[ "$MAX_SEC" == "0" ]] || [[ "$MAX_SEC" == "all" ]]; then
  echo "导出全部帧..."
  find "$SRC/full_body_img" -maxdepth 1 -name "*.jpg" -exec cp {} "${ASSETS}/full_body_img/" \;
  find "$SRC/landmarks" -maxdepth 1 -name "*.lms" -exec cp {} "${ASSETS}/landmarks/" \;
else
  MAX_FRAMES=$((MAX_SEC * FPS))
  echo "只导出前 ${MAX_SEC} 秒（${MAX_FRAMES} 帧）..."
  for i in $(seq 0 $((MAX_FRAMES - 1))); do
    [[ -f "$SRC/full_body_img/$i.jpg" ]] && cp "$SRC/full_body_img/$i.jpg" "${ASSETS}/full_body_img/"
    [[ -f "$SRC/landmarks/$i.lms" ]] && cp "$SRC/landmarks/$i.lms" "${ASSETS}/landmarks/"
  done
fi
IMG_COUNT=$(find "${ASSETS}/full_body_img" -name "*.jpg" 2>/dev/null | wc -l)
LMS_COUNT=$(find "${ASSETS}/landmarks" -name "*.lms" 2>/dev/null | wc -l)
echo "Dataset 已导出到 ${ASSETS} (${IMG_COUNT} 张图, ${LMS_COUNT} 个 landmarks)"
