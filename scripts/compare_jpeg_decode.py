#!/usr/bin/env python3
"""
对比 cv2.imread 与 PIL 解码同一 JPEG 的差异，排查是否 JPEG 解码导致 server/Android 不一致。
用法:
  python scripts/compare_jpeg_decode.py data/raw/full_body_img/0.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="对比 cv2 vs PIL JPEG 解码")
    parser.add_argument("jpg_path", help="JPEG 路径，如 data/raw/full_body_img/0.jpg")
    parser.add_argument("-o", "--out", help="输出目录，保存解码结果 PNG")
    args = parser.parse_args()

    path = Path(args.jpg_path)
    if not path.exists():
        print(f"错误: 文件不存在 {path}")
        sys.exit(1)

    # cv2
    try:
        import cv2
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            print("错误: cv2.imread 返回 None")
            sys.exit(1)
    except ImportError:
        print("错误: 需要 cv2")
        sys.exit(1)

    # PIL
    try:
        from PIL import Image
        pil_img = Image.open(path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_pil = np.array(pil_img)
        img_pil_bgr = img_pil[:, :, ::-1]  # RGB -> BGR 便于与 cv2 对比
    except ImportError:
        print("错误: 需要 PIL (pip install Pillow)")
        sys.exit(1)

    if img_cv.shape != img_pil_bgr.shape:
        print(f"形状不一致: cv2={img_cv.shape} pil={img_pil_bgr.shape}")
        sys.exit(1)

    diff = np.abs(img_cv.astype(np.float64) - img_pil_bgr.astype(np.float64))
    print(f"=== JPEG 解码对比: {path.name} ===")
    print(f"  shape: {img_cv.shape}")
    print(f"  max_diff: {diff.max():.0f}, mean_diff: {diff.mean():.2f}")
    if diff.max() > 0:
        idx = np.unravel_index(np.argmax(diff.sum(axis=2)), diff.shape[:2])
        print(f"  最大差异位置 (y={idx[0]},x={idx[1]}): cv2={img_cv[idx]} pil={img_pil_bgr[idx]}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "decode_cv2.png"), img_cv)
        cv2.imwrite(str(out_dir / "decode_pil.png"), img_pil_bgr)
        diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "decode_diff_x10.png"), diff_vis)
        print(f"  已保存: {out_dir}/decode_*.png")


if __name__ == "__main__":
    main()
