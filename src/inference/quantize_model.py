"""
模型量化工具
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class ModelQuantizer:
    """模型量化工具"""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8):
        """
        动态量化
        
        Args:
            model: 待量化模型
            dtype: 量化数据类型
        
        Returns:
            量化后的模型
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype,
        )
        return quantized_model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        静态量化
        
        Args:
            model: 待量化模型
            calibration_data: 校准数据 [(image, audio), ...]
        
        Returns:
            量化后的模型
        """
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        torch.quantization.prepare(model, inplace=True)
        
        # 校准
        for img, audio in calibration_data:
            model(img, audio)
        
        # 转换
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        return quantized_model
    
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3):
        """
        模型剪枝
        
        Args:
            model: 待剪枝模型
            amount: 剪枝比例
        
        Returns:
            剪枝后的模型
        """
        from torch.nn.utils import prune
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                if module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=amount)
        
        return model
    
    @staticmethod
    def knowledge_distillation(
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader,
        device: str = "cuda",
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """
        知识蒸馏
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            train_loader: 训练数据
            device: 设备
            temperature: 温度参数
            alpha: 蒸馏损失权重
        """
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            audio_features = batch['audio_features'].to(device)
            
            # 教师模型预测
            with torch.no_grad():
                teacher_output = teacher_model(images, audio_features)
            
            # 学生模型预测
            student_output = student_model(images, audio_features)
            
            # 蒸馏损失
            distill_loss = criterion(
                F.log_softmax(student_output / temperature, dim=-1),
                F.softmax(teacher_output / temperature, dim=-1)
            )
            
            # 原始损失
            original_loss = F.mse_loss(student_output, targets)
            
            # 总损失
            loss = alpha * distill_loss + (1 - alpha) * original_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return student_model
    
    @staticmethod
    def get_model_size(model: nn.Module) -> dict:
        """获取模型大小信息"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': size_mb,
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }


if __name__ == "__main__":
    # 测试量化
    from src.models.unet_light import LightUNet
    
    model = LightUNet(base_channels=32, depth=4)
    
    # 动态量化
    quantized = ModelQuantizer.quantize_dynamic(model)
    
    # 获取大小
    size_info = ModelQuantizer.get_model_size(quantized)
    print(f"量化后模型大小: {size_info['total_size_mb']:.2f} MB")
    
    print("量化测试完成!")
