"""
改进的同步网络（唇形同步）
基于Ultralight-Digital-Human项目的简化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSyncNet(nn.Module):
    """改进的唇形同步网络，用于音频-视频对齐"""
    
    def __init__(self, audio_dim: int = 256, image_dim: int = 256, hidden_dim: int = 512):
        """
        初始化同步网络
        
        Args:
            audio_dim: 音频特征维度
            image_dim: 图像特征维度
            hidden_dim: 隐藏层维度
        """
        super(ImprovedSyncNet, self).__init__()
        
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear((hidden_dim // 2) + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # 输出同步分数
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, audio_features, images):
        """
        前向传播
        
        Args:
            audio_features: 音频特征 [B, T, D]
            images: 图像 [B, C, H, W]
        
        Returns:
            同步分数 [B, 1]
        """
        # 音频编码
        audio_encoded = self.encode_audio(audio_features)
        
        # 图像编码
        image_encoded = self.encode_image(images)
        
        # 融合特征
        combined = torch.cat([audio_encoded, image_encoded], dim=1)
        sync_score = self.fusion(combined)
        
        return sync_score
    
    def encode_audio(self, audio_features):
        """编码音频特征"""
        # [B, T, D] -> [B, D, T]
        audio_features = audio_features.transpose(1, 2)
        return self.audio_encoder(audio_features)
    
    def encode_image(self, images):
        """编码图像"""
        return self.image_encoder(images)
    
    def get_sync_loss(self, audio_features, images, target_sync=1.0):
        """
        计算同步损失
        
        Args:
            audio_features: 音频特征
            images: 图像
            target_sync: 目标同步分数（1.0表示同步）
        
        Returns:
            同步损失
        """
        sync_scores = self(audio_features, images)
        target = torch.full_like(sync_scores, target_sync)
        return F.mse_loss(sync_scores, target)


if __name__ == "__main__":
    # 测试同步网络
    model = ImprovedSyncNet()
    
    # 创建测试数据
    batch_size = 4
    audio_features = torch.randn(batch_size, 50, 256)  # 50帧音频特征
    images = torch.randn(batch_size, 3, 256, 256)      # 256x256图像
    
    # 前向传播
    sync_scores = model(audio_features, images)
    sync_loss = model.get_sync_loss(audio_features, images)
    
    print(f"同步网络测试:")
    print(f"  输入音频形状: {audio_features.shape}")
    print(f"  输入图像形状: {images.shape}")
    print(f"  同步分数形状: {sync_scores.shape}")
    print(f"  同步损失: {sync_loss.item():.4f}")
    print(f"  同步分数范围: [{sync_scores.min():.3f}, {sync_scores.max():.3f}]")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")