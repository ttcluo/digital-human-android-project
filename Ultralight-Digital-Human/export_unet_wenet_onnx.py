import argparse
import os

import torch

from unet import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 Ultralight U-Net（WeNet 模式）为 ONNX，用于端侧推理基准测试"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的 U-Net 权重路径，例如 ./checkpoint_wenet/195.pth",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./unet_wenet_160.onnx",
        help="导出的 ONNX 文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    out_path = os.path.abspath(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"loading checkpoint: {checkpoint_path}")

    net = Model(6, mode="wenet")
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state)
    net.to(device).eval()

    # 与训练/推理保持一致的 dummy 输入形状
    dummy_img = torch.randn(1, 6, 160, 160, device=device)
    dummy_audio = torch.randn(1, 128, 16, 32, device=device)

    input_names = ["input_image", "input_audio"]
    output_names = ["output_image"]

    print(f"exporting to ONNX: {out_path}")
    torch.onnx.export(
        net,
        (dummy_img, dummy_audio),
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print("export done.")


if __name__ == "__main__":
    main()

