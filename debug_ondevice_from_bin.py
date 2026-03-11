import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def build_unet_input_from_ref(ref_png: str, size: int = 128) -> np.ndarray:
    """与 debug_ondevice_android_like.py 相同的图像前处理，返回 [1,6,H,W] float32。"""
    img = cv2.imread(ref_png)
    if img is None:
        raise RuntimeError(f"cv2 读取失败: {ref_png}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ref_bgr = img.astype(np.float32) / 255.0  # [H,W,3]

    masked = ref_bgr.copy()
    margin = int(size * 0.08)
    masked[margin:size - margin, margin:size - margin, :] = 0.0

    b_ref, g_ref, r_ref = ref_bgr[:, :, 0], ref_bgr[:, :, 1], ref_bgr[:, :, 2]
    b_mask, g_mask, r_mask = masked[:, :, 0], masked[:, :, 1], masked[:, :, 2]

    stacked = np.stack(
        [b_ref, g_ref, r_ref, b_mask, g_mask, r_mask],
        axis=0,
    ).astype(np.float32)  # [6,H,W]
    return stacked[None, ...]  # [1,6,H,W]


def load_first_frame_from_bin(bin_path: Path) -> np.ndarray:
    """按照 Android 端同样的格式，从 wenet_feat_stream.bin 里取第 0 帧特征 [1,128,16,32]。"""
    with bin_path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size != 4:
            raise RuntimeError(f"读取头部失败: {header}")
        T, C, H, W = header.tolist()
        if C != 128 or H != 16 or W != 32:
            raise RuntimeError(f"特征维度不匹配: C={C},H={H},W={W}, 期望 128,16,32")
        frame_size = C * H * W
        # 只读第一帧，避免一次性载入过大文件
        data = np.fromfile(f, dtype=np.float32, count=frame_size)
        if data.size != frame_size:
            raise RuntimeError(
                f"读取第一帧失败: 期望 {frame_size} float32, 实际 {data.size}"
            )
    feats = data.reshape(1, C, H, W).astype(np.float32)  # [1,128,16,32]
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(
        "在服务器上用 wenet_feat_stream.bin + ref_face_128.png 做 ONNX 推理，对比 Android 行为"
    )
    parser.add_argument(
        "--ref_png",
        type=str,
        required=True,
        help="参考人脸 patch PNG 路径，例如 ./onnx/ref_face_128.png",
    )
    parser.add_argument(
        "--bin",
        type=str,
        required=True,
        help="export_wenet_feat_for_android.py 生成的 wenet_feat_stream.bin 路径",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="./onnx/unet_ondevice_128.onnx",
        help="OnDeviceUNet ONNX 模型路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./onnx/debug_ondevice_from_bin.png",
        help="输出 PNG 路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    ref_png = Path(args.ref_png)
    bin_path = Path(args.bin)
    onnx_path = Path(args.onnx)

    if not ref_png.is_file():
        raise FileNotFoundError(f"ref_png 不存在: {ref_png}")
    if not bin_path.is_file():
        raise FileNotFoundError(f"bin 不存在: {bin_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"onnx 不存在: {onnx_path}")

    img_np = build_unet_input_from_ref(str(ref_png), size=128)  # [1,6,128,128]
    audio_np = load_first_frame_from_bin(bin_path)  # [1,128,16,32]

    print(f"[INFO] img_np shape: {img_np.shape}, audio_np shape: {audio_np.shape}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in0_name = sess.get_inputs()[0].name
    in1_name = sess.get_inputs()[1].name
    ort_out = sess.run(
        None,
        {
            in0_name: img_np,
            in1_name: audio_np,
        },
    )[0]  # [1,3,128,128]
    ort_out = ort_out[0]
    print(
        f"[ONNX-from-bin] stats: min={ort_out.min():.6f}, "
        f"max={ort_out.max():.6f}, mean={ort_out.mean():.6f}"
    )

    out_img = (ort_out * 255.0).astype(np.uint8)  # [3,H,W]
    out_img = np.transpose(out_img, (1, 2, 0))  # [H,W,3] RGB
    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_path = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img_bgr)
    print(f"[INFO] 已保存 ONNX+bin 预测图像到: {out_path}")


if __name__ == "__main__":
    main()

