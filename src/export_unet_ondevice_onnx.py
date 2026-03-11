import argparse
from pathlib import Path

import torch

from src.models.unet_ondevice_light import OnDeviceUNet


def export_fp32(model: torch.nn.Module, out_path: str) -> None:
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    img = torch.randn(1, 6, 128, 128, device=device)
    audio = torch.randn(1, 128, 16, 32, device=device)

    torch.onnx.export(
        model,
        (img, audio),
        out_path,
        input_names=["input", "audio"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[INFO] 导出 OnDeviceUNet FP32 ONNX 完成: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        "导出端侧轻量 U-Net OnDeviceUNet 为 ONNX，用于 Android 端 FPS benchmark"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="训练好的 OnDeviceUNet 权重路径（.pth 或 .pth.tar），默认不加载，导出随机初始化模型",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./onnx",
        help="ONNX 输出目录，默认 ./onnx",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="unet_ondevice_128",
        help="导出 ONNX 文件基础名，默认 unet_ondevice_128",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = str(out_dir / f"{args.name}.onnx")

    net = OnDeviceUNet(6)
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"未找到 checkpoint 文件: {ckpt_path}")
        print(f"[INFO] 从 checkpoint 加载权重: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        # 兼容两种保存方式：直接 state_dict 或 包含在字典的 'state_dict' / 'model' 字段中
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] 缺失权重: {missing}")
        if unexpected:
            print(f"[WARN] 多余权重: {unexpected}")

    export_fp32(net, out_path)


if __name__ == "__main__":
    main()

