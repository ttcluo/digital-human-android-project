import argparse
import os
from pathlib import Path

import torch

from unet import Model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # 兼容可能存在的 "module." 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v
    model.load_state_dict(new_state, strict=True)


def export_fp32(model: torch.nn.Module, out_path: str) -> None:
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    img = torch.randn(1, 6, 160, 160, device=device)
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
    print(f"[INFO] 导出 FP32 ONNX 完成: {out_path}")


def export_fp16(fp32_path: str, fp16_path: str) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("[WARN] 未安装 onnx / onnxconverter-common，跳过 FP16 导出")
        return

    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fp16_path)
    print(f"[INFO] 导出 FP16 ONNX 完成: {fp16_path}")


def main() -> None:
    parser = argparse.ArgumentParser("导出 Ultralight U-Net (WeNet 模式) ONNX，包含 FP32 / FP16")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="训练好的 U-Net checkpoint 路径，例如: ./checkpoint_wenet/net_epoch_200.pth",
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
        default="unet_wenet_160",
        help="导出 ONNX 文件基础名，默认 unet_wenet_160",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = str(out_dir / f"{args.name}.onnx")
    fp16_path = str(out_dir / f"{args.name}_fp16.onnx")

    net = Model(6, mode="wenet")
    load_checkpoint(net, args.ckpt)

    export_fp32(net, fp32_path)
    export_fp16(fp32_path, fp16_path)


if __name__ == "__main__":
    main()

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

