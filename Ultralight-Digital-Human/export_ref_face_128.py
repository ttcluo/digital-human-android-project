import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def crop_face_patch(img: np.ndarray, lms_path: str) -> np.ndarray:
    """复用 inference.py / datasetsss.py 的裁剪逻辑，得到 160x160 口部区域 patch。"""
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)

    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width

    crop_img = img[ymin:ymax, xmin:xmax]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    patch = crop_img[4:164, 4:164].copy()  # 160x160
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(
        "从训练视频帧中导出一张 128x128 的参考人脸 patch（与训练/推理同分布）"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据目录，需包含 full_body_img/ 与 landmarks/，例如 /data/.../data/raw",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="使用哪一帧作为参考图，默认 0",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./ref_face_128.png",
        help="输出 PNG 路径，默认 ./ref_face_128.png",
    )
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset_dir, "full_body_img")
    lms_dir = os.path.join(args.dataset_dir, "landmarks")

    img_path = os.path.join(img_dir, f"{args.frame_idx}.jpg")
    lms_path = os.path.join(lms_dir, f"{args.frame_idx}.lms")

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"未找到图像文件: {img_path}")
    if not os.path.isfile(lms_path):
        raise FileNotFoundError(f"未找到 landmarks 文件: {lms_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"cv2 读取图像失败: {img_path}")

    patch160 = crop_face_patch(img, lms_path)
    patch128 = cv2.resize(patch160, (128, 128), cv2.INTER_AREA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), patch128)
    print(f"[INFO] 已导出参考人脸 patch: {out_path} (shape={patch128.shape})")


if __name__ == "__main__":
    main()

