"""
损失函数模块
包含各种用于数字人训练的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionLoss(nn.Module):
    """重建损失 - 用于图像重建任务"""
    
    def __init__(self, loss_type: str = "mse"):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class SyncLoss(nn.Module):
    """
    唇形同步损失
    基于SyncNet的音频-视频同步损失
    """
    
    def __init__(self, margin: float = 0.5):
        super(SyncLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算音频-视频同步损失
        
        Args:
            audio_features: 音频特征 [B, T, D]
            video_features: 视频特征 [B, T, D]
        
        Returns:
            同步损失
        """
        # 计算余弦相似度
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(audio_features, video_features.transpose(1, 2))
        
        # 创建标签（对角线为正样本）
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)
        
        # 对比损失
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失
    使用预训练的VGG网络提取特征
    """
    
    def __init__(self, use_cuda: bool = True):
        super(PerceptualLoss, self).__init__()
        
        # 加载VGG19
        from torchvision import models
        vgg = models.vgg19(pretrained=True)
        
        # 只使用特征提取部分
        self.vgg_layers = vgg.features
        
        # 冻结参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        if use_cuda:
            self.vgg_layers = self.vgg_layers.cuda()
        
        # 定义用于计算损失的层
        self.layers = [3, 8, 17, 26, 35]
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算感知损失"""
        loss = 0.0
        
        x = pred
        y = target
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            
            if i in self.layers:
                # 计算特征差异
                loss += F.mse_loss(x, y)
        
        return loss


class GANLoss(nn.Module):
    """GAN损失"""
    
    def __init__(self, loss_type: str = "wgan-gp", use_cuda: bool = True):
        super(GANLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == "wgan-gp":
            self.real_label = 1.0
            self.fake_label = -1.0
        elif loss_type == "hinge":
            self.real_label = 1.0
            self.fake_label = -1.0
        else:
            self.real_label = 1.0
            self.fake_label = 0.0
        
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """获取目标张量"""
        if target_is_real:
            target_tensor = torch.full_like(prediction, self.real_label)
        else:
            target_tensor = torch.full_like(prediction, self.fake_label)
        return target_tensor
    
    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """计算GAN损失"""
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        
        if self.loss_type == "wgan-gp":
            return F.binary_cross_entropy_with_logits(prediction, target_tensor)
        elif self.loss_type == "hinge":
            return F.hinge_embedding_loss(prediction, target_tensor)
        else:
            return F.binary_cross_entropy(prediction, target_tensor)


class CombinedLoss(nn.Module):
    """
    组合损失
    组合多个损失函数
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        sync_weight: float = 0.5,
        gan_weight: float = 0.01,
    ):
        super(CombinedLoss, self).__init__()
        
        # 损失函数
        self.reconstruction_loss = ReconstructionLoss(loss_type="l1")
        self.perceptual_loss = PerceptualLoss(use_cuda=torch.cuda.is_available())
        
        # 损失权重
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.sync_weight = sync_weight
        self.gan_weight = gan_weight
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        pred_audio_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算组合损失
        
        Returns:
            包含各损失分量的字典
        """
        losses = {}
        
        # 重建损失
        if self.reconstruction_weight > 0:
            recon_loss = self.reconstruction_loss(pred, target)
            losses['reconstruction'] = recon_loss.item()
            total_loss = recon_loss * self.reconstruction_weight
        else:
            total_loss = 0
        
        # 感知损失
        if self.perceptual_weight > 0:
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss.item()
            total_loss = total_loss + perc_loss * self.perceptual_weight
        
        # 同步损失
        if self.sync_weight > 0 and audio_features is not None and pred_audio_features is not None:
            sync_loss = F.mse_loss(pred_audio_features, audio_features)
            losses['sync'] = sync_loss.item()
            total_loss = total_loss + sync_loss * self.sync_weight
        
        losses['total'] = total_loss.item()
        
        return losses


class TemporalLoss(nn.Module):
    """
    时序一致性损失
    确保生成的视频帧之间的时间一致性
    """
    
    def __init__(self):
        super(TemporalLoss, self).__init__()
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        计算时序损失
        
        Args:
            frames: 视频帧序列 [B, T, C, H, W]
        
        Returns:
            时序损失
        """
        # 计算相邻帧的差异
        diff = frames[:, 1:] - frames[:, :-1]
        
        # L2范数
        loss = torch.mean(diff ** 2)
        
        return loss


class EdgeLoss(nn.Module):
    """
    边缘损失
    保留图像边缘细节
    """
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边缘损失"""
        # 确保是4D张量
        if pred.dim() == 5:
            # [B, T, C, H, W] -> [B*T, C, H, W]
            pred = pred.view(-1, *pred.shape[2:])
            target = target.view(-1, *target.shape[2:])
        
        # 应用Sobel算子
        sobel_x = self.sobel_x.repeat(pred.size(1), 1, 1, 1)
        sobel_y = self.sobel_y.repeat(pred.size(1), 1, 1, 1)
        
        pred_edge_x = F.conv2d(pred, sobel_x, groups=pred.size(1), padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, groups=pred.size(1), padding=1)
        
        target_edge_x = F.conv2d(target, sobel_x, groups=target.size(1), padding=1)
        target_edge_y = F.conv2d(target, sobel_y, groups=target.size(1), padding=1)
        
        # 计算差异
        loss = F.mse_loss(pred_edge_x, target_edge_x) + F.mse_loss(pred_edge_y, target_edge_y)
        
        return loss


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    # 创建测试数据
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    # 测试重建损失
    recon_loss = ReconstructionLoss("mse")
    print(f"重建损失(MSE): {recon_loss(pred, target).item():.4f}")
    
    recon_loss_l1 = ReconstructionLoss("l1")
    print(f"重建损失(L1): {recon_loss_l1(pred, target).item():.4f}")
    
    # 测试组合损失
    combined = CombinedLoss()
    losses = combined(pred, target)
    print(f"组合损失: {losses}")
    
    # 测试时序损失
    temporal = TemporalLoss()
    frames = torch.randn(2, 10, 3, 64, 64)
    print(f"时序损失: {temporal(frames).item():.4f}")
    
    # 测试边缘损失
    edge = EdgeLoss()
    print(f"边缘损失: {edge(pred, target).item():.4f}")
    
    print("损失函数测试完成!")
