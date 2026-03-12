#!/usr/bin/env python3
"""
对比 Android 与服务器推理的输入/输出，排查画质差异根因。
用法:
  1. 服务器导出: python scripts/inference_onnx_frame0.py --dataset data/raw --audio_bin android/app/src/main/assets/wenet_feat_stream.bin --onnx android/app/src/main/assets/unet_wenet_160.onnx -o data/server_onnx_frame0.png --raw --dump_inputs --dump_output

  2. Android 运行 FullInference（选 Ultralight 160），adb pull:
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_img_input.bin data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_audio_input.bin data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_debug_output.bin data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_patch_real.png data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_patch_masked.png data/
     adb pull /sdcard/Android/data/com.digitalhuman.app/files/Movies/android_crop168.png data/

  3b. 用 Android crop 跑服务器（排除 decode+resize）:
     python scripts/inference_onnx_frame0.py ... --use_crop data/android_crop168.png --dump_inputs --dump_output

  3. 对比:
     python scripts/compare_android_server_inputs.py --data_dir data
     python scripts/compare_android_server_inputs.py --data_dir data --verbose  # 详细诊断
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# mask 区域与 OpenCV rectangle(5,5,150,145) 一致
MASK_X1, MASK_Y1, MASK_X2, MASK_Y2 = 5, 5, 150, 145


def in_mask(y: int, x: int) -> bool:
    return MASK_X1 <= x <= MASK_X2 and MASK_Y1 <= y <= MASK_Y2


def load_bin(path: Path, shape: tuple) -> np.ndarray:
    """加载 little-endian float32 二进制文件。"""
    data = np.fromfile(path, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"{path}: 期望 {np.prod(shape)} 元素，实际 {data.size}")
    return data.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="对比 Android 与服务器输入/输出")
    parser.add_argument("--data_dir", default="data", help="存放 debug 文件的目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细诊断：列出差异像素、保存可视化")
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

        if args.verbose:
            _diagnose_img_diff(s_img, a_img, data_dir)
    else:
        print("提示: 无 debug_img_input.npy，运行 inference_onnx_frame0.py --dump_inputs")

    _compare_patch_pngs(data_dir)

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


def _compare_patch_pngs(data_dir: Path) -> None:
    """比对 raw patch PNG，定位是 crop/resize 还是 mask 导致差异。"""
    s_real = data_dir / "debug_patch_real.png"
    s_masked = data_dir / "debug_patch_masked.png"
    a_real = data_dir / "android_patch_real.png"
    a_masked = data_dir / "android_patch_masked.png"
    if not s_real.exists() or not a_real.exists():
        return
    try:
        import cv2
        sr = cv2.imread(str(s_real))
        ar = cv2.imread(str(a_real))
        if sr is None or ar is None:
            return
        # Android Bitmap 存 PNG 为 RGB，cv2 读入后 ch0=R ch2=B，转为 BGR 便于与 server 对比
        ar = ar[:, :, [2, 1, 0]].copy()
        if sr.shape != ar.shape:
            print(f"\n=== patch real PNG 尺寸不一致: {sr.shape} vs {ar.shape}")
            return
        diff_r = np.abs(sr.astype(np.float64) - ar.astype(np.float64))
        print(f"\n=== patch real PNG 对比 (crop+resize 后原图) ===")
        print(f"  max_diff={diff_r.max():.0f}, mean_diff={diff_r.mean():.2f}")
        if diff_r.max() > 10:
            yy, xx = np.unravel_index(np.argmax(diff_r.sum(axis=2)), diff_r.shape[:2])
            print(f"  最大差异位置 (y={yy},x={xx}): server={sr[yy,xx]}, android={ar[yy,xx]}")

        if s_masked.exists() and a_masked.exists():
            sm = cv2.imread(str(s_masked))
            am = cv2.imread(str(a_masked))
            if am is not None:
                am = am[:, :, [2, 1, 0]].copy()
            if sm is not None and am is not None and sm.shape == am.shape:
                diff_m = np.abs(sm.astype(np.float64) - am.astype(np.float64))
                print(f"\n=== patch masked PNG 对比 ===")
                print(f"  max_diff={diff_m.max():.0f}, mean_diff={diff_m.mean():.2f}")
    except ImportError:
        pass


def _diagnose_img_diff(s_img: np.ndarray, a_img: np.ndarray, data_dir: Path) -> None:
    """详细诊断 img 差异：差异像素分布、real/masked 分离、可视化。"""
    # shape (1,6,160,160): ch0-2=real BGR, ch3-5=masked BGR
    s = s_img[0]
    a = a_img[0]
    diff = np.abs(s.astype(np.float64) - a.astype(np.float64))

    # 按通道统计
    for ch, name in [(0, "real_B"), (1, "real_G"), (2, "real_R"), (3, "mask_B"), (4, "mask_G"), (5, "mask_R")]:
        d = diff[ch]
        n_bad = np.sum(d > 0.01)
        if n_bad > 0:
            in_mask_cnt = sum(1 for y in range(160) for x in range(160) if d[y, x] > 0.01 and in_mask(y, x))
            out_mask_cnt = n_bad - in_mask_cnt
            print(f"  {name}: {n_bad} 像素 diff>0.01 (mask内 {in_mask_cnt}, mask外 {out_mask_cnt})")

    # 列出 diff>0.5 的像素（显著差异）
    thresh = 0.5
    bad = np.where(diff > thresh)
    if len(bad[0]) > 0:
        print(f"\n  差异>{thresh} 的像素 (最多 20 个):")
        order = np.argsort(-diff.ravel())[:20]
        for idx in order:
            flat = diff.ravel()[idx]
            if flat < thresh:
                break
            c, y, x = np.unravel_index(idx, diff.shape)
            ch_name = ["real_B", "real_G", "real_R", "mask_B", "mask_G", "mask_R"][c]
            loc = "mask内" if in_mask(y, x) else "mask外"
            print(f"    ({c},{y},{x}) {ch_name} {loc}: s={s[c,y,x]:.4f} a={a[c,y,x]:.4f}")

    # 保存 real/masked 为 PNG 便于肉眼对比
    try:
        import cv2
        # real: ch0-2 BGR, 反归一化
        s_real = (np.clip(s[:3].transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        a_real = (np.clip(a[:3].transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        s_masked = (np.clip(s[3:].transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        a_masked = (np.clip(a[3:].transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        cv2.imwrite(str(data_dir / "compare_server_real.png"), s_real)
        cv2.imwrite(str(data_dir / "compare_android_real.png"), a_real)
        cv2.imwrite(str(data_dir / "compare_server_masked.png"), s_masked)
        cv2.imwrite(str(data_dir / "compare_android_masked.png"), a_masked)
        # 差异热力图：mask 外差异
        diff_out = np.zeros((160, 160), dtype=np.float32)
        for y in range(160):
            for x in range(160):
                if not in_mask(y, x):
                    diff_out[y, x] = diff[:, y, x].max()
        if diff_out.max() > 0:
            heat = (np.clip(diff_out * 255, 0, 255)).astype(np.uint8)
            cv2.imwrite(str(data_dir / "compare_diff_mask_out.png"), heat)
        print(f"\n  已保存: compare_*_real.png, compare_*_masked.png, compare_diff_mask_out.png")
    except ImportError:
        print("  提示: 安装 cv2 可保存可视化 PNG")


if __name__ == "__main__":
    main()
