import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def get_audio_features(features: np.ndarray, index: int) -> torch.Tensor:
    """与 datasetsss.py / inference.py 中的逻辑保持一致：前后各取4帧."""
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
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
    return auds


def build_unet_input_from_ref(ref_png: str, size: int = 128) -> torch.Tensor:
    """模拟 Android 端的 loadRefImageToTensor，构造 [1,6,H,W] BGR 输入."""
    img = cv2.imread(ref_png)
    if img is None:
        raise RuntimeError(f"cv2 读取失败: {ref_png}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # 参考图 BGR
    ref_bgr = img.astype(np.float32) / 255.0  # [H,W,3]

    # 中间区域涂黑的 masked 图
    masked = ref_bgr.copy()
    margin = int(size * 0.08)  # 近似 Android 端的 8% 边距
    masked[margin : size - margin, margin : size - margin, :] = 0.0

    # 通道顺序：[参考 B,G,R, 被遮挡 B,G,R]，与 ondevice 训练时 BGR*2 对齐
    b_ref, g_ref, r_ref = (
        ref_bgr[:, :, 0],
        ref_bgr[:, :, 1],
        ref_bgr[:, :, 2],
    )
    b_mask, g_mask, r_mask = (
        masked[:, :, 0],
        masked[:, :, 1],
        masked[:, :, 2],
    )

    # [6,H,W]
    stacked = np.stack(
        [b_ref, g_ref, r_ref, b_mask, g_mask, r_mask],
        axis=0,
    ).astype(np.float32)
    # [1,6,H,W]
    return torch.from_numpy(stacked).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        "在服务器上模拟 Android 端 OnDeviceUNet 输入，导出一帧预测结果 PNG"
    )
    parser.add_argument(
        "--ref_png",
        type=str,
        required=True,
        help="参考人脸 patch（128x128），例如 ./onnx/ref_face_128.png",
    )
    parser.add_argument(
        "--audio_npy",
        type=str,
        required=True,
        help="对应音频的 *_wenet.npy 路径，例如 data/preview_wenet.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Ultralight-Digital-Human/checkpoint_ondevice/105.pth",
        help="OnDeviceUNet 训练好的权重路径",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="使用第几帧音频特征做调试，默认 0",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./onnx/debug_ondevice_android_like.png",
        help="输出 PNG 路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.models.unet_ondevice_light import OnDeviceUNet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # 构造图像输入
    img_t = build_unet_input_from_ref(args.ref_png, size=128).to(device)  # [1,6,128,128]

    # 加载 WeNet 特征并构造 [1,128,16,32]
    feats = np.load(args.audio_npy).astype(np.float32)
    print(f"[INFO] audio feats shape: {feats.shape}")
    idx = max(0, min(args.frame_idx, feats.shape[0] - 1))
    aud = get_audio_features(feats, idx)  # [8, D]
    aud = aud.reshape(128, 16, 32)[None]  # [1,128,16,32]
    audio_t = torch.from_numpy(aud).to(device)

    # 加载模型
    net = OnDeviceUNet(6).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=False)
    net.eval()

    with torch.no_grad():
        out = net(img_t, audio_t)[0]  # [3,128,128], sigmoid 后在 [0,1]
    out_np = out.cpu().numpy()
    print(
        f"[INFO] U-Net output stats: min={out_np.min():.6f}, max={out_np.max():.6f}, mean={out_np.mean():.6f}"
    )

    # 转成 BGR 图片保存
    out_img = (out_np * 255.0).astype(np.uint8)  # [3,H,W]
    out_img = np.transpose(out_img, (1, 2, 0))  # [H,W,3] RGB
    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img_bgr)
    print(f"[INFO] 已保存 U-Net 预测图像到: {out_path}")


if __name__ == "__main__":
    main()

