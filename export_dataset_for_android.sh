#!/bin/bash
# 将 dataset（full_body_img + landmarks）导出到 Android assets
# 用法: ./export_dataset_for_android.sh <dataset_dir>
# dataset_dir 需包含 full_body_img/ 和 landmarks/

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
ASSETS="${ROOT}/android/app/src/main/assets/dataset"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <dataset_dir>"
  echo "  dataset_dir 需包含 full_body_img/ 和 landmarks/"
  echo "  示例: $0 data/raw 或 $0 /path/to/Ultralight-Digital-Human/data/raw"
  exit 1
fi

SRC="$(cd "$1" && pwd)"
if [[ ! -d "$SRC/full_body_img" ]] || [[ ! -d "$SRC/landmarks" ]]; then
  echo "错误: $SRC 需包含 full_body_img/ 和 landmarks/"
  exit 1
fi

mkdir -p "${ASSETS}/full_body_img" "${ASSETS}/landmarks"
find "$SRC/full_body_img" -maxdepth 1 -name "*.jpg" -exec cp {} "${ASSETS}/full_body_img/" \;
find "$SRC/landmarks" -maxdepth 1 -name "*.lms" -exec cp {} "${ASSETS}/landmarks/" \;
IMG_COUNT=$(find "${ASSETS}/full_body_img" -name "*.jpg" 2>/dev/null | wc -l)
LMS_COUNT=$(find "${ASSETS}/landmarks" -name "*.lms" 2>/dev/null | wc -l)
echo "Dataset 已导出到 ${ASSETS} (${IMG_COUNT} 张图, ${LMS_COUNT} 个 landmarks)"
