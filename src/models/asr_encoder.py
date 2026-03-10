"""
ASR音频编码器
支持HuBERT和WeNet两种音频特征提取器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import numpy as np


class ASREncoder(nn.Module):
    """音频特征编码器，支持多种ASR模型"""
    
    def __init__(self, 
                 encoder_type: str = "hubert",
                 feature_dim: int = 256,
                 sample_rate: int = 16000,
                 frame_length: int = 25):
        """
        Args:
            encoder_type: 编码器类型，'hubert' 或 'wenet'
            feature_dim: 输出特征维度
            sample_rate: 音频采样率
            frame_length: 视频帧率（fps）
        """
        super(ASREncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        
        # 音频预处理层
        self.audio_preprocess = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # 根据编码器类型选择不同的特征提取器
        if encoder_type == "hubert":
            self.feature_extractor = self._create_hubert_like_encoder()
        elif encoder_type == "wenet":
            self.feature_extractor = self._create_wenet_like_encoder()
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 时间对齐层（将音频特征对齐到视频帧率）
        self.temporal_align = nn.Conv1d(feature_dim, feature_dim, 
                                       kernel_size=3, padding=1, stride=1)
        
    def _create_hubert_like_encoder(self) -> nn.Module:
        """创建HuBERT风格的编码器"""
        return nn.Sequential(
            # 卷积特征提取
            nn.Conv1d(128, 256, kernel_size=10, stride=5, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Transformer编码器
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=4
            ),
            
            # 全局池化
            nn.AdaptiveAvgPool1d(1),
        )
    
    def _create_wenet_like_encoder(self) -> nn.Module:
        """创建WeNet风格的编码器（更轻量）"""
        return nn.Sequential(
            # 卷积特征提取（更轻量）
            nn.Conv1d(128, 192, kernel_size=10, stride=5, padding=5),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            
            # 轻量Transformer
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=192,
                    nhead=6,
                    dim_feedforward=768,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=3
            ),
            
            # 全局池化
            nn.AdaptiveAvgPool1d(1),
        )
    
    def forward(self, audio_waveform: torch.Tensor, 
                video_frames: Optional[int] = None) -> torch.Tensor:
        """
        提取音频特征
        
        Args:
            audio_waveform: 音频波形 [B, T_audio]
            video_frames: 视频帧数，用于对齐特征
        
        Returns:
            audio_features: 音频特征 [B, T_video, D]
        """
        B, T_audio = audio_waveform.shape
        
        # 添加通道维度
        audio_waveform = audio_waveform.unsqueeze(1)  # [B, 1, T_audio]
        
        # 音频预处理
        x = self.audio_preprocess(audio_waveform)  # [B, 128, T_audio/4]
        
        # 调整维度用于Transformer
        x = x.transpose(1, 2)  # [B, T_audio/4, 128]
        
        # 特征提取
        if self.encoder_type == "hubert":
            # HuBERT风格：先卷积再Transformer
            conv_features = self.feature_extractor[0](x.transpose(1, 2))
            conv_features = conv_features.transpose(1, 2)  # [B, T', 256]
            transformer_features = self.feature_extractor[1](conv_features)
            pooled_features = self.feature_extractor[2](transformer_features.transpose(1, 2))
            features = pooled_features.squeeze(-1)  # [B, 256]
        else:
            # WeNet风格
            conv_features = self.feature_extractor[0](x.transpose(1, 2))
            conv_features = conv_features.transpose(1, 2)
            transformer_features = self.feature_extractor[1](conv_features)
            pooled_features = self.feature_extractor[2](transformer_features.transpose(1, 2))
            features = pooled_features.squeeze(-1)  # [B, 192]
        
        # 特征投影
        audio_features = self.feature_projection(features)  # [B, feature_dim]
        
        # 如果需要对齐到视频帧率
        if video_frames is not None:
            # 重复特征以匹配视频帧数
            audio_features = audio_features.unsqueeze(1)  # [B, 1, feature_dim]
            audio_features = audio_features.repeat(1, video_frames, 1)  # [B, T_video, feature_dim]
            
            # 时间对齐（添加时间依赖性）
            audio_features = audio_features.transpose(1, 2)  # [B, D, T_video]
            audio_features = self.temporal_align(audio_features)
            audio_features = audio_features.transpose(1, 2)  # [B, T_video, D]
        
        return audio_features
    
    def extract_features_offline(self, audio_path: str, 
                                device: torch.device = None) -> torch.Tensor:
        """
        离线提取音频特征（简化实现）
        
        Args:
            audio_path: 音频文件路径
            device: 计算设备
        
        Returns:
            features: 音频特征 [1, T, D]
        """
        # 这里简化实现，实际应该使用librosa加载音频
        # 返回随机特征用于测试
        import librosa
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 转换为张量
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        
        # 计算音频时长对应的视频帧数
        duration = len(audio) / sr
        video_frames = int(duration * self.frame_length)
        
        # 提取特征
        with torch.no_grad():
            features = self.forward(audio_tensor, video_frames)
        
        return features
    
    def get_frame_features(self, audio_features: torch.Tensor, 
                          frame_idx: int) -> torch.Tensor:
        """
        获取特定帧的音频特征
        
        Args:
            audio_features: 完整音频特征 [B, T, D]
            frame_idx: 帧索引
        
        Returns:
            frame_feature: 单帧音频特征 [B, D]
        """
        B, T, D = audio_features.shape
        
        if frame_idx < 0 or frame_idx >= T:
            # 使用最近的帧
            frame_idx = max(0, min(frame_idx, T - 1))
        
        return audio_features[:, frame_idx, :]
    
    def compute_audio_energy(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        计算音频能量（用于语音活动检测）
        
        Args:
            audio_waveform: 音频波形 [B, T]
        
        Returns:
            energy: 音频能量 [B, T_energy]
        """
        # 简化的能量计算
        window_size = 160  # 10ms窗口（16kHz采样率）
        hop_size = 80
        
        B, T = audio_waveform.shape
        
        # 分帧
        frames = audio_waveform.unfold(1, window_size, hop_size)
        B, num_frames, window_size = frames.shape
        
        # 计算每帧能量
        energy = (frames ** 2).mean(dim=2)  # [B, num_frames]
        
        return energy


class MultiModalFusion(nn.Module):
    """多模态特征融合模块（音频+视觉）"""
    
    def __init__(self, 
                 audio_dim: int = 256,
                 visual_dim: int = 256,
                 hidden_dim: int = 128,
                 fusion_type: str = "concatenate"):
        """
        Args:
            audio_dim: 音频特征维度
            visual_dim: 视觉特征维度
            hidden_dim: 融合后隐藏维度
            fusion_type: 融合类型，'concatenate', 'add', 'attention'
        """
        super(MultiModalFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concatenate":
            self.fusion = nn.Sequential(
                nn.Linear(audio_dim + visual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
        elif fusion_type == "add":
            # 确保维度相同
            assert audio_dim == visual_dim, "Add fusion requires same dimensions"
            self.fusion = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "attention":
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, audio_features: torch.Tensor, 
                visual_features: torch.Tensor) -> torch.Tensor:
        """
        融合音频和视觉特征
        
        Args:
            audio_features: 音频特征 [B, T, D_audio] 或 [B, D_audio]
            visual_features: 视觉特征 [B, T, D_visual] 或 [B, D_visual]
        
        Returns:
            fused_features: 融合特征 [B, T, D_hidden] 或 [B, D_hidden]
        """
        if self.fusion_type == "concatenate":
            fused = torch.cat([audio_features, visual_features], dim=-1)
            return self.fusion(fused)
        
        elif self.fusion_type == "add":
            return self.fusion(audio_features + visual_features)
        
        elif self.fusion_type == "attention":
            # 投影到相同维度
            audio_proj = self.audio_proj(audio_features)
            visual_proj = self.visual_proj(visual_features)
            
            # 注意力融合
            attn_output, _ = self.attention(visual_proj, audio_proj, audio_proj)
            
            # 拼接和融合
            combined = torch.cat([attn_output, visual_proj], dim=-1)
            return self.fusion(combined)


if __name__ == "__main__":
    # 测试ASR编码器
    print("测试HuBERT编码器...")
    hubert_encoder = ASREncoder(encoder_type="hubert")
    
    # 创建测试音频
    batch_size = 2
    audio_length = 16000  # 1秒音频，16kHz采样率
    test_audio = torch.randn(batch_size, audio_length)
    
    # 提取特征
    features = hubert_encoder(test_audio, video_frames=25)
    print(f"HuBERT特征形状: {features.shape}")
    print(f"HuBERT参数量: {sum(p.numel() for p in hubert_encoder.parameters() if p.requires_grad):,}")
    
    print("\n测试WeNet编码器...")
    wenet_encoder = ASREncoder(encoder_type="wenet")
    wenet_features = wenet_encoder(test_audio, video_frames=25)
    print(f"WeNet特征形状: {wenet_features.shape}")
    print(f"WeNet参数量: {sum(p.numel() for p in wenet_encoder.parameters() if p.requires_grad):,}")
    
    # 测试多模态融合
    print("\n测试多模态融合...")
    fusion = MultiModalFusion(fusion_type="attention")
    audio_feat = torch.randn(batch_size, 25, 256)
    visual_feat = torch.randn(batch_size, 25, 256)
    fused = fusion(audio_feat, visual_feat)
    print(f"融合特征形状: {fused.shape}")