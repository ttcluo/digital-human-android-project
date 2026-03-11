"""
端侧轻量版 U-Net 模型（面向 128x128 / 移动端）

设计目标：
- 相比 Ultralight 原始 U-Net（160x160, ch=[32,64,128,256,512]）大幅减小参数量和 FLOPs
- 保持相同的调用接口：forward(x, audio_feat)
- 仅修改通道数与分辨率假设，不改变整体拓扑结构，方便后续替换训练脚本

注意：
- 默认假设输入图像大小为 128x128，输入通道为 6（两帧 BGR 拼接）
- 音频特征默认使用 WeNet 20fps 提取后的特征，形状 [B, 128, 16, 32]
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_light import InvertedResidual


class DoubleConvDW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                use_res_connect=False,
                expand_ratio=2,
            ),
            InvertedResidual(
                out_channels,
                out_channels,
                stride=1,
                use_res_connect=True,
                expand_ratio=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.inconv = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=1,
                use_res_connect=False,
                expand_ratio=2,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AudioConvWenetLite(nn.Module):
    """
    轻量版音频分支，输入为 WeNet 特征 [B, 128, 16, 32]
    通道配置整体减半，最终输出通道与图像 bottleneck 对齐。
    """

    def __init__(self) -> None:
        super().__init__()
        # 更激进的通道压缩：整体再砍一半
        ch = [8, 16, 32, 48, 64]

        # 128 -> 96
        self.conv1 = InvertedResidual(
            128, ch[3], stride=1, use_res_connect=False, expand_ratio=2
        )
        self.conv2 = InvertedResidual(
            ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2
        )

        self.conv3 = nn.Conv2d(
            ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2)
        )
        self.bn3 = nn.BatchNorm2d(ch[3])

        self.conv4 = InvertedResidual(
            ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2
        )

        self.conv5 = nn.Conv2d(
            ch[3], ch[4], kernel_size=3, padding=3, stride=2
        )
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()

        self.conv6 = InvertedResidual(
            ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2
        )
        self.conv7 = InvertedResidual(
            ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class OnDeviceUNet(nn.Module):
    """
    端侧轻量 U-Net：
    - 输入：图像 [B, 6, H, W]，音频特征 [B, 128, 16, 32]
    - 默认分辨率：H=W=128
    - 输出：图像 [B, 3, H, W]，范围 [0,1]
    """

    def __init__(self, n_channels: int = 6) -> None:
        super().__init__()
        self.n_channels = n_channels

        # 与 AudioConvWenetLite 保持一致的轻量通道配置
        ch = [8, 16, 32, 48, 64]

        self.audio_model = AudioConvWenetLite()
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1),
        )

        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        # 注意：上采样阶段的 in_channels 需要精确等于 concat 之后的通道数
        # up1: concat(x5: ch3, x4: ch3) -> 2*ch3
        self.up1 = Up(ch[3] * 2, ch[3] // 2)
        # up2: concat(up1_out: ch3//2, x3: ch2) -> ch3//2 + ch2
        self.up2 = Up(ch[3] // 2 + ch[2], ch[2] // 2)
        # up3: concat(up2_out: ch2//2, x2: ch1) -> ch2//2 + ch1
        self.up3 = Up(ch[2] // 2 + ch[1], ch[1] // 2)
        # up4: concat(up3_out: ch1//2, x1: ch0) -> ch1//2 + ch0
        self.up4 = Up(ch[1] // 2 + ch[0], ch[0])

        self.outc = OutConv(ch[0], 3)

    def forward(self, x: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        audio_feat = self.audio_model(audio_feat)
        # 对齐空间尺寸，避免下采样路径与音频分支步长差异导致的形状不一致
        if audio_feat.shape[2:] != x5.shape[2:]:
            audio_feat = F.interpolate(
                audio_feat, size=x5.shape[2:], mode="bilinear", align_corners=False
            )
        x5 = torch.cat([x5, audio_feat], dim=1)
        x5 = self.fuse_conv(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = torch.sigmoid(out)
        return out

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _bench_once(device: torch.device) -> None:
    model = OnDeviceUNet(6).to(device).eval()
    img = torch.randn(1, 6, 128, 128, device=device)
    audio = torch.randn(1, 128, 16, 32, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(img, audio)

        iters = 100
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(img, audio)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / iters
    fps = 1000.0 / avg_ms
    print(f"device: {device}")
    print(f"OnDeviceUNet avg_ms_per_frame: {avg_ms:.4f} ({fps:.2f} FPS)")


if __name__ == "__main__":
    import time

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bench_once(dev)

