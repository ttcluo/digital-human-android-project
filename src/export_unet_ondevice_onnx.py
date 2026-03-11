import argparse
from pathlib import Path

import torch

from models.unet_ondevice_light import OnDeviceUNet


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
    export_fp32(net, out_path)


if __name__ == "__main__":
    main()

