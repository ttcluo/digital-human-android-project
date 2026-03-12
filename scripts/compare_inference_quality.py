#!/usr/bin/env python3
"""
对比服务器与 Android 推理结果的画质：PSNR、SSIM。
用法:
  python scripts/compare_inference_quality.py --server data/server_frame0.png --android /path/to/android_frame0.png
  python scripts/compare_inference_quality.py --server data/server_frame0.png --android android_frame0.png --crop
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_image(path: str) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取: {path}")
        return img
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
        return img[:, :, ::-1]  # RGB -> BGR


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """峰值信噪比 (dB)，越高越好。"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """结构相似度 [0,1]，越高越好。"""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        try:
            from scipy.ndimage import uniform_filter
            # 简化版 SSIM
            C1, C2 = 6.5025, 58.5225
            I1 = img1.astype(np.float64)
            I2 = img2.astype(np.float64)
            mu1 = uniform_filter(I1, size=11)
            mu2 = uniform_filter(I2, size=11)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = uniform_filter(I1 ** 2, size=11) - mu1_sq
            sigma2_sq = uniform_filter(I2 ** 2, size=11) - mu2_sq
            sigma12 = uniform_filter(I1 * I2, size=11) - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return float(np.mean(ssim_map))
        except ImportError:
            print("警告: 需要 skimage 或 scipy，安装: pip install scikit-image")
            return -1.0

    if img1.ndim == 3:
        return ssim_fn(img1, img2, channel_axis=2, data_range=max_val)
    return ssim_fn(img1, img2, data_range=max_val)


def extract_mouth_region(img: np.ndarray, crop_rect: tuple) -> np.ndarray:
    """提取嘴部区域 (xmin, ymin, xmax, ymax)，返回 168x168 中心 160x160。"""
    xmin, ymin, xmax, ymax = crop_rect
    crop = img[ymin:ymax, xmin:xmax]
    import cv2
    crop168 = cv2.resize(crop, (168, 168), interpolation=cv2.INTER_AREA)
    return crop168[4:164, 4:164]  # 160x160


def load_landmarks(lms_path: str) -> np.ndarray:
    """解析 .lms 文件，返回 [[x,y], ...]"""
    pts = []
    with open(lms_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float32)


def get_crop_rect_from_landmarks(lms: np.ndarray) -> tuple:
    """与 inference.py 一致：xmin=lms[1][0], ymin=lms[52][1], xmax=lms[31][0], ymax=ymin+width"""
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width
    return (xmin, ymin, xmax, ymax)


def main():
    parser = argparse.ArgumentParser(description="对比服务器与 Android 推理画质 (PSNR/SSIM)")
    parser.add_argument("--server", required=True, help="服务器帧 PNG 路径")
    parser.add_argument("--android", required=True, help="Android 帧 PNG 路径")
    parser.add_argument("--crop", action="store_true", help="仅对比嘴部 160x160 区域")
    parser.add_argument("--landmarks", type=str, help="landmarks/*.lms 路径，与 --crop 配合自动获取裁剪区")
    parser.add_argument("--xmin", type=int, default=0, help="嘴部裁剪 (与 --crop 配合，无 --landmarks 时用)")
    parser.add_argument("--ymin", type=int, default=0, help="嘴部裁剪")
    parser.add_argument("--xmax", type=int, default=0, help="0 表示需指定")
    parser.add_argument("--ymax", type=int, default=0, help="0 表示需指定")
    args = parser.parse_args()

    server_path = Path(args.server)
    android_path = Path(args.android)
    if not server_path.exists():
        print(f"错误: 服务器图不存在: {server_path}")
        sys.exit(1)
    if not android_path.exists():
        print(f"错误: Android 图不存在: {android_path}")
        sys.exit(1)

    img_s = load_image(str(server_path))
    img_a = load_image(str(android_path))

    if img_s.shape != img_a.shape:
        print(f"警告: 尺寸不一致 服务器={img_s.shape} Android={img_a.shape}")
        h, w = min(img_s.shape[0], img_a.shape[0]), min(img_s.shape[1], img_a.shape[1])
        img_s = img_s[:h, :w]
        img_a = img_a[:h, :w]

    rect = None
    if args.crop:
        if args.landmarks:
            lms = load_landmarks(args.landmarks)
            rect = get_crop_rect_from_landmarks(lms)
            print(f"  从 landmarks 获取裁剪区: {rect}")
        elif args.xmax > 0 and args.ymax > 0:
            rect = (args.xmin, args.ymin, args.xmax, args.ymax)
        else:
            print("错误: --crop 需配合 --landmarks 或 --xmin/ymin/xmax/ymax")
            sys.exit(1)

    if rect:
        region_s = extract_mouth_region(img_s, rect)
        region_a = extract_mouth_region(img_a, rect)
        region_name = "嘴部 160x160"
    else:
        region_s = img_s
        region_a = img_a
        region_name = "全图"

    psnr_val = psnr(region_s, region_a)
    ssim_val = ssim(region_s, region_a)

    print(f"=== 推理画质对比 ({region_name}) ===")
    print(f"  服务器: {server_path}")
    print(f"  Android: {android_path}")
    print(f"  PSNR: {psnr_val:.2f} dB  (越高越好，>30 通常可接受)")
    print(f"  SSIM: {ssim_val:.4f}  (越高越好，>0.95 通常可接受)")

    if not args.crop and region_s.shape[0] > 200:
        print("\n提示: 仅对比嘴部区域:")
        print("  python scripts/compare_inference_quality.py --server X --android Y --crop --landmarks data/raw/landmarks/0.lms")


if __name__ == "__main__":
    main()
