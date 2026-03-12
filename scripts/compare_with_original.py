#!/usr/bin/env python3
"""
对比服务器/Android 推理输出与原始参考（数据集嘴部区域），定位色彩问题。

用法:
  # 对比 raw 输出与原始 0.jpg 嘴部区域
  python scripts/compare_with_original.py --dataset android/app/src/main/assets/dataset \\
    --server data/server_onnx_frame0_raw.png --android data/android_raw_frame0.png

  # 仅对比服务器
  python scripts/compare_with_original.py --dataset android/app/src/main/assets/dataset \\
    --server data/server_onnx_frame0_raw.png

  # 导出 BGR/RGB 互换版本供肉眼检查
  python scripts/compare_with_original.py ... --save_variants data/color_check/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取: {path}")
    return img


def load_original_mouth(dataset_dir: Path, frame_idx: int = 0) -> np.ndarray:
    """从数据集加载原始嘴部 160x160（与 inference 一致的 crop）。"""
    img_dir = dataset_dir / "full_body_img"
    lms_dir = dataset_dir / "landmarks"
    img_path = img_dir / f"{frame_idx}.jpg"
    lms_path = lms_dir / f"{frame_idx}.lms"
    if not img_path.exists() or not lms_path.exists():
        raise FileNotFoundError(f"缺少 {img_path} 或 {lms_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取: {img_path}")

    pts = []
    with open(lms_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    lms = np.array(pts, dtype=np.float32)
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width

    crop = img[ymin:ymax, xmin:xmax]
    crop168 = cv2.resize(crop, (168, 168), interpolation=cv2.INTER_LINEAR)
    return crop168[4:164, 4:164].copy()


def ensure_160x160(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] != (160, 160):
        return cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
    return img


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        print("警告: pip install scikit-image 以启用 SSIM")
        return -1.0
    if img1.ndim == 3:
        return ssim_fn(img1, img2, channel_axis=2, data_range=max_val)
    return ssim_fn(img1, img2, data_range=max_val)


def compare_one(name: str, pred: np.ndarray, orig: np.ndarray, save_dir: Optional[Path]) -> None:
    """orig 为 BGR（cv2 读取数据集）。pred 可能被存成 BGR 或 RGB。"""
    pred = ensure_160x160(pred)
    orig = ensure_160x160(orig)
    # pred 直接当 BGR 与 orig(BGR) 比
    p_as_bgr = psnr(pred, orig)
    s_as_bgr = ssim(pred, orig)
    # pred 当 RGB 时，需与 orig_rgb 比；pred_rgb 即 pred 通道顺序当作 R,G,B，与 orig 的 B,G,R 对应则 pred 的 ch0 应对 orig 的 ch2
    pred_swap = pred[:, :, ::-1]
    p_pred_as_rgb = psnr(pred_swap, orig)
    s_pred_as_rgb = ssim(pred_swap, orig)

    print(f"  {name}:")
    print(f"    pred 当 BGR 与 orig 比: PSNR={p_as_bgr:.2f} dB  SSIM={s_as_bgr:.4f}")
    print(f"    pred 当 RGB(swap 后) 与 orig 比: PSNR={p_pred_as_rgb:.2f} dB  SSIM={s_pred_as_rgb:.4f}")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{name}_orig.png"), orig)
        cv2.imwrite(str(save_dir / f"{name}_pred_as_is.png"), pred)
        cv2.imwrite(str(save_dir / f"{name}_pred_swap_rgb.png"), pred_swap)


def main():
    parser = argparse.ArgumentParser(description="对比推理输出与原始参考，排查色彩问题")
    parser.add_argument("--dataset", required=True, help="dataset 目录，含 full_body_img/ 和 landmarks/")
    parser.add_argument("--server", help="服务器 raw 输出 PNG 路径")
    parser.add_argument("--android", help="Android raw 输出 PNG 路径")
    parser.add_argument("--frame", type=int, default=0, help="参考帧索引")
    parser.add_argument("--save_variants", help="保存 BGR/RGB 变体到目录，供肉眼检查")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not (dataset / "full_body_img").exists() or not (dataset / "landmarks").exists():
        print(f"错误: dataset 需包含 full_body_img/ 和 landmarks/")
        sys.exit(1)

    orig = load_original_mouth(dataset, args.frame)
    print(f"原始参考: 帧 {args.frame} 嘴部 160x160 (BGR 来自 cv2.imread)")

    save_dir = Path(args.save_variants) if args.save_variants else None

    if args.server:
        server_path = Path(args.server)
        if not server_path.exists():
            print(f"错误: 服务器图不存在: {server_path}")
            sys.exit(1)
        pred_s = load_image(str(server_path))
        compare_one("server", pred_s, orig, save_dir)

    if args.android:
        android_path = Path(args.android)
        if not android_path.exists():
            print(f"错误: Android 图不存在: {android_path}")
            sys.exit(1)
        pred_a = load_image(str(android_path))
        if pred_a.shape[2] == 4:
            pred_a = cv2.cvtColor(pred_a, cv2.COLOR_BGRA2BGR)
        compare_one("android", pred_a, orig, save_dir)

    if save_dir:
        print(f"\n已保存变体到: {save_dir}")

    if not args.server and not args.android:
        print("错误: 需指定 --server 或 --android")


if __name__ == "__main__":
    main()
