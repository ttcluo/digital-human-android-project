"""
评估指标模块
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class PSNR:
    """峰值信噪比"""
    
    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range
    
    def compute(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算PSNR"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 确保形状一致
        if pred.shape != target.shape:
            raise ValueError(f"形状不匹配: {pred.shape} vs {target.shape}")
        
        # 处理维度
        if pred.ndim == 4:
            # [B, C, H, W] -> [B, H, W, C]
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
            
            psnr_values = []
            for i in range(pred.shape[0]):
                p = psnr(target[i], pred[i], data_range=self.data_range)
                psnr_values.append(p)
            return np.mean(psnr_values)
        
        return psnr(target, pred, data_range=self.data_range)
    
    def __call__(self, pred, target):
        return self.compute(pred, target)


class SSIM:
    """结构相似性指数"""
    
    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range
    
    def compute(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算SSIM"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 确保形状一致
        if pred.shape != target.shape:
            raise ValueError(f"形状不匹配: {pred.shape} vs {target.shape}")
        
        # 处理维度
        if pred.ndim == 4:
            # [B, C, H, W] -> [B, H, W, C]
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
            
            ssim_values = []
            for i in range(pred.shape[0]):
                s = ssim(target[i], pred[i], data_range=self.data_range, channel_axis=2)
                ssim_values.append(s)
            return np.mean(ssim_values)
        
        return ssim(target, pred, data_range=self.data_range, channel_axis=2)
    
    def __call__(self, pred, target):
        return self.compute(pred, target)


class LPIPS:
    """感知相似性指数"""
    
    def __init__(self, net: str = 'alex'):
        self.net = net
        self.feature_extractor = None
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """初始化特征提取器"""
        try:
            import lpips
            self.feature_extractor = lpips.LPIPS(net=self.net)
        except ImportError:
            print("警告: lpips未安装，使用简化版LPIPS")
            self.feature_extractor = None
    
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算LPIPS"""
        if self.feature_extractor is None:
            # 简化版：使用MSE
            return F.mse_loss(pred, target).item()
        
        # 转换到[-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1
        
        # 计算距离
        distance = self.feature_extractor(pred, target)
        return distance.mean().item()
    
    def __call__(self, pred, target):
        return self.compute(pred, target)


class FID:
    """Frechet Inception Distance"""
    
    def __init__(self):
        self.inception = None
        self._init_inception()
    
    def _init_inception(self):
        """初始化Inception网络"""
        try:
            from torchvision.models import inception_v3
            self.inception = inception_v3(pretrained=True, transform_input=False)
            self.inception.fc = torch.nn.Identity()
            self.inception.eval()
        except ImportError:
            print("警告: FID计算需要安装torchvision")
            self.inception = None
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """提取特征"""
        if self.inception is None:
            return np.random.randn(len(images), 2048)
        
        with torch.no_grad():
            features = self.inception(images)
            return features.cpu().numpy()
    
    def compute(self, pred_features: np.ndarray, target_features: np.ndarray) -> float:
        """计算FID"""
        # 计算均值和协方差
        mu1, sigma1 = pred_features.mean(axis=0), np.cov(pred_features, rowvar=False)
        mu2, sigma2 = target_features.mean(axis=0), np.cov(target_features, rowvar=False)
        
        # 计算FID
        diff = mu1 - mu2
        covmean = np.sqrt(sigma1 @ sigma2)
        
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def __call__(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = self.extract_features(pred)
        if isinstance(target, torch.Tensor):
            target = self.extract_features(target)
        return self.compute(pred, target)


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        # self.fid = FID()
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """计算所有指标"""
        metrics = {}
        
        # PSNR
        metrics['psnr'] = self.psnr(pred, target)
        
        # SSIM
        metrics['ssim'] = self.ssim(pred, target)
        
        # LPIPS
        metrics['lpips'] = self.lpips(pred, target)
        
        # MSE
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
        else:
            pred_np = pred
            target_np = target
        
        metrics['mse'] = np.mean((pred_np - target_np) ** 2)
        
        return metrics


if __name__ == "__main__":
    # 测试评估指标
    print("测试评估指标...")
    
    # 创建测试数据
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    # 测试
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(pred, target)
    
    print("评估指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("评估指标测试完成!")
