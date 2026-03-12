#!/usr/bin/env python3
"""
服务器端用 ONNX 推理并导出第 0 帧，与 Android 端对比以隔离 PyTorch vs ONNX 差异。

排查 Android 画质差异流程:
  1. 服务器导出 raw 输出:
     python scripts/inference_onnx_frame0.py --dataset data/raw --audio_bin android/app/src/main/assets/wenet_feat_stream.bin --onnx android/app/src/main/assets/unet_wenet_160.onnx -o data/server_onnx_frame0.png --raw

  2. Android 运行 FullInference（选 Ultralight 160），用 adb pull 拉取 android_raw_frame0.png:
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_raw_frame0.png data/

  3. 对比 raw 输出（排除贴回差异）:
     python scripts/compare_inference_quality.py --server data/server_onnx_frame0_raw.png --android data/android_raw_frame0.png

  4. 若 raw 仍差异大，用 --dump_inputs 导出输入，排查预处理
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def get_audio_features(features: np.ndarray, index: int) -> np.ndarray:
    """与 inference.py / export_wenet_feat_for_android.py 一致。"""
    left = index - 4
    right = index + 4
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = features[left:right].copy()
    if pad_left > 0:
        auds = np.concatenate([np.zeros_like(auds[:pad_left]), auds], axis=0)
    if pad_right > 0:
        auds = np.concatenate([auds, np.zeros_like(auds[:pad_right])], axis=0)
    return auds.reshape(128, 16, 32).astype(np.float32)


def parse_landmarks(lms_path: str) -> np.ndarray:
    pts = []
    with open(lms_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="服务器端 ONNX 推理，导出第 0 帧")
    parser.add_argument("--dataset", required=True, help="dataset 目录，含 full_body_img/ 和 landmarks/")
    parser.add_argument("--audio_feat", help="*_wenet.npy 路径")
    parser.add_argument("--audio_bin", help="wenet_feat_stream.bin 路径，与 Android 完全一致时用")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径，如 unet_wenet_160.onnx")
    parser.add_argument("-o", "--out", default="server_onnx_frame0.png", help="输出 PNG 路径")
    parser.add_argument("--frame", type=int, default=0, help="帧索引，默认 0")
    parser.add_argument("--raw", action="store_true", help="额外导出原始模型输出 160x160，用于与 Android raw 对比")
    parser.add_argument("--dump_inputs", action="store_true", help="导出 img/audio 输入到 .npy，用于排查预处理差异")
    parser.add_argument("--dump_output", action="store_true", help="导出模型输出到 .npy，与 Android output 对比")
    args = parser.parse_args()
    out_path = Path(args.out)

    try:
        import onnxruntime as ort
    except ImportError:
        print("错误: 需要 onnxruntime，安装: pip install onnxruntime")
        sys.exit(1)

    dataset = Path(args.dataset)
    img_dir = dataset / "full_body_img"
    lms_dir = dataset / "landmarks"
    if not img_dir.exists() or not lms_dir.exists():
        print(f"错误: dataset 需包含 full_body_img/ 和 landmarks/")
        sys.exit(1)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"错误: ONNX 模型不存在: {onnx_path}")
        sys.exit(1)

    if args.audio_bin:
        with open(args.audio_bin, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)
            T, C, H, W = header.tolist()
            frame_size = C * H * W
            if args.frame >= T:
                raise RuntimeError(f"bin 帧数不足: T={T}, 需要帧 {args.frame}")
            f.seek(16 + args.frame * frame_size * 4)
            data = np.fromfile(f, dtype=np.float32, count=frame_size)
            audio_feat = data.reshape(1, C, H, W).astype(np.float32)
    else:
        if not args.audio_feat:
            print("错误: 需指定 --audio_feat 或 --audio_bin")
            sys.exit(1)
        feats = np.load(args.audio_feat).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    img_shape = session.get_inputs()[0].shape
    input_size = int(img_shape[2]) if len(img_shape) >= 4 else 160

    i = args.frame
    img_idx = min(i, len(list(img_dir.glob("*.jpg"))) - 1)
    img_idx = max(0, img_idx)

    img_path = img_dir / f"{img_idx}.jpg"
    lms_path = lms_dir / f"{img_idx}.lms"
    if not img_path.exists() or not lms_path.exists():
        print(f"错误: 未找到 {img_path} 或 {lms_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    lms = parse_landmarks(str(lms_path))
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width

    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), interpolation=cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()

    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked_np = cv2.rectangle(img_real_ex_ori.copy(), (5, 5, 150, 145), (0, 0, 0), -1)

    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_masked = img_masked_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_concat = np.concatenate([img_real_ex, img_masked], axis=0)[np.newaxis, ...]

    if input_size != 160:
        resized = np.zeros((1, 6, input_size, input_size), dtype=np.float32)
        for c in range(6):
            resized[0, c] = cv2.resize(
                img_concat[0, c], (input_size, input_size), interpolation=cv2.INTER_AREA
            )
        img_concat = resized

    if not args.audio_bin:
        audio_feat = get_audio_features(feats, i)[np.newaxis, ...]

    if args.dump_inputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.parent / "debug_img_input.npy", img_concat)
        np.save(out_path.parent / "debug_audio_input.npy", audio_feat)
        # 保存 raw 160x160 便于与 Android 原图对比（mask 前/后）
        cv2.imwrite(str(out_path.parent / "debug_patch_real.png"), img_real_ex_ori)
        cv2.imwrite(str(out_path.parent / "debug_patch_masked.png"), img_masked_np)
        print(f"已导出输入: debug_img_input.npy, debug_audio_input.npy, debug_patch_*.png")

    inputs = {input_names[0]: img_concat, input_names[1]: audio_feat}
    pred = session.run(None, inputs)[0]

    if args.dump_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.parent / "debug_output.npy", pred)
        print(f"已导出输出: debug_output.npy")

    pred_raw = pred[0].transpose(1, 2, 0) * 255
    pred_raw = np.clip(pred_raw, 0, 255).astype(np.uint8)
    if pred_raw.shape[0] != 160:
        pred_raw = cv2.resize(pred_raw, (160, 160), interpolation=cv2.INTER_LINEAR)
    pred = pred_raw

    if args.raw:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = out_path.parent / (out_path.stem + "_raw.png")
        cv2.imwrite(str(raw_path), pred_raw)
        print(f"已保存 raw 输出: {raw_path}")

    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
