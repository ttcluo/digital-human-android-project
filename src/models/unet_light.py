"""
轻量化U-Net模型
基于Ultralight-Digital-Human项目的改进，使用倒残差块和深度可分离卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InvertedResidual(nn.Module):
    """倒残差块，来自MobileNetV2"""
    def __init__(self, inp: int, oup: int, stride: int, 
                 use_res_connect: bool, expand_ratio: int = 6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = use_res_connect
        
        hidden_dim = int(round(inp * expand_ratio))
        
        self.conv = nn.Sequential(
            # 逐点卷积
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 深度可分离卷积
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 逐点卷积
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LightUNet(nn.Module):
    """轻量化U-Net模型，用于数字人生成"""
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 audio_feature_dim: int = 256):
        super(LightUNet, self).__init__()
        
        self.audio_feature_dim = audio_feature_dim
        self.depth = depth
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # 构建编码器
        in_ch = base_channels
        for i in range(depth):
            out_ch = in_ch * 2 if i < depth - 1 else in_ch
            # 倒残差块编码器
            encoder = nn.Sequential(
                InvertedResidual(in_ch, out_ch, 1, False, expand_ratio=4),
                InvertedResidual(out_ch, out_ch, 1, True, expand_ratio=4),
            )
            self.encoders.append(encoder)
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # 音频特征融合
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, in_ch * 4 * 4),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            if i == depth - 1:
                # 最底层，融合音频特征
                in_ch = in_ch + in_ch  # 拼接音频特征
            else:
                in_ch = in_ch * 2 + out_ch
            
            out_ch = max(base_channels, in_ch // 2)
            
            decoder = nn.Sequential(
                InvertedResidual(in_ch, out_ch, 1, False, expand_ratio=4),
                InvertedResidual(out_ch, out_ch, 1, True, expand_ratio=4),
            )
            self.decoders.append(decoder)
            
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(out_ch, out_ch, 2, 2))
            in_ch = out_ch
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 1),
            nn.Tanh()  # 输出在[-1, 1]范围
        )
        
    def forward(self, x: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            audio_features: 音频特征 [B, T, D]
        
        Returns:
            output: 生成的图像 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 编码器路径
        encoder_features = []
        x = self.init_conv(x)
        
        for i in range(self.depth):
            x = self.encoders[i](x)
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        
        # 处理音频特征
        audio_proj = self.audio_projection(audio_features.mean(dim=1))  # [B, audio_feature_dim] -> [B, in_ch*4*4]
        audio_proj = audio_proj.view(B, -1, 4, 4)  # 重塑为空间特征
        
        # 在底层融合音频特征
        x = torch.cat([x, audio_proj], dim=1)
        
        # 解码器路径
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            
            if i < len(self.upsamples):
                # 上采样
                x = self.upsamples[i](x)
                # 跳跃连接
                skip_idx = self.depth - 2 - i
                skip_feature = encoder_features[skip_idx]
                x = torch.cat([x, skip_feature], dim=1)
        
        # 最终输出
        output = self.final_conv(x)
        return output
    
    def get_num_params(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_size: Tuple[int, int] = (256, 256)) -> float:
        """估算模型FLOPs（近似值）"""
        # 简化的FLOPs估算
        h, w = input_size
        total_flops = 0
        
        # 初始卷积
        total_flops += h * w * 3 * self.encoders[0][0].conv[0].in_channels * 9
        
        # 编码器FLOPs
        for encoder in self.encoders:
            for layer in encoder:
                if isinstance(layer, InvertedResidual):
                    # 简化的FLOPs计算
                    total_flops += h * w * layer.conv[0].in_channels * layer.conv[0].out_channels
                    total_flops += h * w * layer.conv[3].in_channels * 9  # 深度卷积
                    total_flops += h * w * layer.conv[6].in_channels * layer.conv[6].out_channels
            h //= 2
            w //= 2
        
        # 解码器FLOPs（类似）
        h, w = input_size[0] // (2 ** (self.depth - 1)), input_size[1] // (2 ** (self.depth - 1))
        for decoder in self.decoders:
            for layer in decoder:
                if isinstance(layer, InvertedResidual):
                    total_flops += h * w * layer.conv[0].in_channels * layer.conv[0].out_channels
                    total_flops += h * w * layer.conv[3].in_channels * 9
                    total_flops += h * w * layer.conv[6].in_channels * layer.conv[6].out_channels
            h *= 2
            w *= 2
        
        return total_flops


if __name__ == "__main__":
    # 测试模型
    model = LightUNet(base_channels=32, depth=4)
    
    # 创建测试输入
    batch_size = 2
    input_image = torch.randn(batch_size, 3, 256, 256)
    audio_features = torch.randn(batch_size, 50, 256)  # 50帧音频特征
    
    # 前向传播
    output = model(input_image, audio_features)
    
    print(f"模型参数量: {model.get_num_params():,}")
    print(f"输入形状: {input_image.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")