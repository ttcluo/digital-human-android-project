import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from datasetsss import MyDataset


def load_ondevice_unet(ckpt_path: str, asr: str = "wenet") -> torch.nn.Module:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    from src.models.unet_ondevice_light import OnDeviceUNet  # type: ignore

    if asr != "wenet":
        raise ValueError("OnDeviceUNet 当前只支持 asr=wenet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OnDeviceUNet(6).to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def save_pair(pred: torch.Tensor, gt: torch.Tensor, out_dir: Path, idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # [C,H,W] -> [H,W,C], [0,1] -> [0,255]
    pred_np = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
        np.uint8
    )
    gt_np = (gt.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    concat = np.concatenate([gt_np, pred_np], axis=1)
    cv2.imwrite(str(out_dir / f"sample_{idx:04d}_gt.png"), gt_np[:, :, ::-1])
    cv2.imwrite(str(out_dir / f"sample_{idx:04d}_pred.png"), pred_np[:, :, ::-1])
    cv2.imwrite(str(out_dir / f"sample_{idx:04d}_concat.png"), concat[:, :, ::-1])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="可视化 OnDeviceUNet 推理效果（保存若干帧的 GT / Pred 对比图）"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="训练使用的数据目录，例如 /data/.../data/raw",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="OnDeviceUNet 训练好的权重路径，例如 checkpoint_ondevice/105.pth",
    )
    parser.add_argument(
        "--asr",
        type=str,
        default="wenet",
        help="asr 模式（当前只支持 wenet）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="随机可视化的样本数量",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./vis_ondevice",
        help="输出图片目录",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_ondevice_unet(args.checkpoint, args.asr)

    dataset = MyDataset(args.dataset_dir, args.asr)
    n = len(dataset)

    out_dir = Path(args.out_dir)
    print(f"[INFO] dataset_len={n}, save_dir={out_dir}")

    indices = random.sample(range(n), k=min(args.num_samples, n))
    for i, idx in enumerate(indices):
        img_concat_T, img_real_T, audio_feat = dataset.__getitem__(idx)
        img_concat = img_concat_T[None].to(device)  # [1,6,H,W]
        audio = audio_feat[None].to(device)  # [1,128,16,32]
        with torch.no_grad():
            pred = net(img_concat, audio)[0]  # [3,H,W]
        save_pair(pred, img_real_T, out_dir, idx)
        print(f"[INFO] saved sample {i+1}/{len(indices)} (idx={idx})")


if __name__ == "__main__":
    main()

