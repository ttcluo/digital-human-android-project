import argparse
from pathlib import Path

import numpy as np
import torch


def get_audio_features(features: np.ndarray, index: int) -> torch.Tensor:
    """与 datasetsss.py / inference.py 中的逻辑保持一致。"""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        "将 *_wenet.npy 音频特征序列转换为 Android 端可直接读取的 float32 流文件"
    )
    parser.add_argument(
        "--audio_npy",
        type=str,
        required=True,
        help="由 wenet_infer.py 生成的 *_wenet.npy 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出二进制文件路径，例如 ./onnx/wenet_feat_stream.bin",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_npy)
    if not audio_path.is_file():
        raise FileNotFoundError(f"未找到音频特征文件: {audio_path}")

    feats = np.load(audio_path).astype(np.float32)
    print(f"[INFO] 加载 WeNet 特征: {feats.shape}, dtype={feats.dtype}")

    num_frames = feats.shape[0]
    out_frames = np.zeros((num_frames, 128, 16, 32), dtype=np.float32)

    for i in range(num_frames):
        aud = get_audio_features(feats, i)
        aud = aud.reshape(128, 16, 32)
        out_frames[i] = aud

    T, C, H, W = out_frames.shape
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 导出形状为 (T={T}, C={C}, H={H}, W={W}) 的特征序列到: {out_path}")
    header = np.array([T, C, H, W], dtype=np.int32)
    with open(out_path, "wb") as f:
        header.tofile(f)
        out_frames.astype(np.float32).tofile(f)


if __name__ == "__main__":
    main()

