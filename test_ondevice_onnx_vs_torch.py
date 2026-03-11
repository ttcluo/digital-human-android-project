import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


def get_audio_features(features: np.ndarray, index: int) -> torch.Tensor:
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
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right]), auds], dim=0)
    return auds


def build_unet_input_from_ref(ref_png: str, size: int = 128) -> torch.Tensor:
    img = cv2.imread(ref_png)
    if img is None:
        raise RuntimeError(f"cv2 读取失败: {ref_png}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ref_bgr = img.astype(np.float32) / 255.0

    masked = ref_bgr.copy()
    margin = int(size * 0.08)
    masked[margin:size - margin, margin:size - margin, :] = 0.0

    b_ref, g_ref, r_ref = ref_bgr[:, :, 0], ref_bgr[:, :, 1], ref_bgr[:, :, 2]
    b_mask, g_mask, r_mask = masked[:, :, 0], masked[:, :, 1], masked[:, :, 2]

    stacked = np.stack([b_ref, g_ref, r_ref, b_mask, g_mask, r_mask], axis=0).astype(
        np.float32
    )
    return torch.from_numpy(stacked).unsqueeze(0)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    ultra = project_root / "Ultralight-Digital-Human"

    ref_png = project_root / "onnx" / "ref_face_128.png"
    audio_npy = (
        project_root / "data" / "preview_wenet.npy"
    )  # 按你的路径结构，可自行调整
    ckpt = ultra / "checkpoint_ondevice" / "105.pth"
    onnx_path = project_root / "onnx" / "unet_ondevice_128.onnx"

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.models.unet_ondevice_light import OnDeviceUNet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    img_t = build_unet_input_from_ref(str(ref_png), size=128).to(device)
    feats = np.load(audio_npy).astype(np.float32)
    print(f"[INFO] audio feats shape: {feats.shape}")
    idx = 0
    aud = get_audio_features(feats, idx)
    aud = aud.reshape(128, 16, 32)[None]
    audio_t = aud.to(device)

    net = OnDeviceUNet(6).to(device)
    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=False)
    net.eval()
    with torch.no_grad():
        out_torch = net(img_t, audio_t)[0]
    out_np_torch = out_torch.cpu().numpy()
    print(
        f"[TORCH] stats: min={out_np_torch.min():.6f}, "
        f"max={out_np_torch.max():.6f}, mean={out_np_torch.mean():.6f}"
    )

    img_np = img_t.cpu().numpy()
    audio_np = audio_t.cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in0_name = sess.get_inputs()[0].name
    in1_name = sess.get_inputs()[1].name
    ort_out = sess.run(
        None,
        {
            in0_name: img_np,
            in1_name: audio_np,
        },
    )[0]
    ort_out = ort_out[0]
    print(
        f"[ONNX] stats: min={ort_out.min():.6f}, "
        f"max={ort_out.max():.6f}, mean={ort_out.mean():.6f}"
    )

    out_img = (ort_out * 255.0).astype(np.uint8)
    out_img = np.transpose(out_img, (1, 2, 0))
    out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    ort_png = project_root / "onnx" / "debug_ondevice_android_like_onnx.png"
    ort_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ort_png), out_img_bgr)
    print(f"[INFO] 已保存 ONNX 预测图像到: {ort_png}")


if __name__ == "__main__":
    main()

