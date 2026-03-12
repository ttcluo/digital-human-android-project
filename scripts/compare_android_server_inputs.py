#!/usr/bin/env python3
"""
对比 Android 与服务器推理的输入/输出，排查画质差异根因。
用法:
  1. 服务器导出: python scripts/inference_onnx_frame0.py --dataset data/raw --audio_bin android/app/src/main/assets/wenet_feat_stream.bin --onnx android/app/src/main/assets/unet_wenet_160.onnx -o data/server_onnx_frame0.png --raw --dump_inputs --dump_output

  2. Android 运行 FullInference（选 Ultralight 160），adb pull:
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_img_input.bin data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_audio_input.bin data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_output.bin data/

  3. 对比:
     python scripts/compare_android_server_inputs.py --data_dir data
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_bin(path: Path, shape: tuple) -> np.ndarray:
    """加载 little-endian float32 二进制文件。"""
    data = np.fromfile(path, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"{path}: 期望 {np.prod(shape)} 元素，实际 {data.size}")
    return data.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="对比 Android 与服务器输入/输出")
    parser.add_argument("--data_dir", default="data", help="存放 debug 文件的目录")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 服务器文件
    server_img = data_dir / "debug_img_input.npy"
    server_audio = data_dir / "debug_audio_input.npy"

    # Android 文件
    android_img = data_dir / "android_debug_img_input.bin"
    android_audio = data_dir / "android_debug_audio_input.bin"
    android_out = data_dir / "android_debug_output.bin"

    # 服务器 output 需单独推理得到，此处用 server_onnx_frame0_raw.png 的像素值反推
    # 为简化，先只对比输入；若需对比 output，可让服务器也 dump output
    server_out_npy = data_dir / "debug_output.npy"  # 服务器暂未 dump，需加

    if not android_img.exists() or not android_audio.exists():
        print("错误: 缺少 Android dump 文件，请先 adb pull 到 data/")
        sys.exit(1)

    def compare(name: str, a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape:
            print(f"  {name}: 形状不一致 {a.shape} vs {b.shape}")
            return
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f"  {name}: shape={a.shape}")
        print(f"     max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
        if diff.max() > 1e-5:
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"     最大差异位置 {idx}: {a.ravel()[np.argmax(diff)]:.6f} vs {b.ravel()[np.argmax(diff)]:.6f}")

    # 对比 img
    if server_img.exists():
        s_img = np.load(server_img)
        a_img = load_bin(android_img, s_img.shape)
        print("=== img 输入对比 ===")
        compare("img", s_img, a_img)
    else:
        print("提示: 无 debug_img_input.npy，运行 inference_onnx_frame0.py --dump_inputs")

    # 对比 audio
    if server_audio.exists():
        s_aud = np.load(server_audio)
        a_aud = load_bin(android_audio, s_aud.shape)
        print("\n=== audio 输入对比 ===")
        compare("audio", s_aud, a_aud)
    else:
        print("提示: 无 debug_audio_input.npy")

    # 对比 output（若服务器有 dump）
    if android_out.exists():
        n = np.fromfile(android_out, dtype=np.float32).size
        out_size = 160 if n == 3 * 160 * 160 else (128 if n == 3 * 128 * 128 else int((n // 3) ** 0.5))
        a_out = load_bin(android_out, (3, out_size, out_size))
        if server_out_npy.exists():
            s_out = np.load(server_out_npy)
            s_out = s_out[0] if s_out.ndim == 4 else s_out
            print("\n=== output 对比 ===")
            compare("output", s_out, a_out)
        else:
            print(f"\noutput: Android 已 dump (shape [3,{out_size},{out_size}])，服务器加 --dump_output 后可对比")


if __name__ == "__main__":
    main()
