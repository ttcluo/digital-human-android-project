"""
模型导出工具
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import os


class ModelExporter:
    """模型导出工具"""
    
    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
    ):
        """导出为TorchScript格式"""
        model.eval()
        
        # 创建示例输入
        dummy_image = torch.randn(1, 3, *input_size)
        dummy_audio = torch.randn(1, 50, audio_feature_dim)
        
        # 追踪
        traced_model = torch.jit.trace(model, (dummy_image, dummy_audio))
        
        # 保存
        traced_model.save(output_path)
        
        print(f"TorchScript模型已导出: {output_path}")
        return output_path
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
        opset_version: int = 11,
    ):
        """导出为ONNX格式"""
        model.eval()
        
        # 创建示例输入
        dummy_image = torch.randn(1, 3, *input_size)
        dummy_audio = torch.randn(1, 50, audio_feature_dim)
        
        torch.onnx.export(
            model,
            (dummy_image, dummy_audio),
            output_path,
            input_names=['image', 'audio_features'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch'},
                'audio_features': {0: 'batch', 1: 'timestep'},
                'output': {0: 'batch'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        print(f"ONNX模型已导出: {output_path}")
        return output_path
    
    @staticmethod
    def export_to_android(
        model: nn.Module,
        output_dir: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
    ):
        """导出为Android格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出ONNX
        onnx_path = os.path.join(output_dir, "digital_human.onnx")
        ModelExporter.export_to_onnx(model, onnx_path, input_size, audio_feature_dim)
        
        print(f"Android模型已导出到: {output_dir}")
        return onnx_path
    
    @staticmethod
    def export_with_quantization(
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
    ):
        """导出量化模型"""
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        
        # 导出
        ModelExporter.export_to_torchscript(
            quantized_model,
            output_path,
            input_size,
            audio_feature_dim,
        )
        
        print(f"量化模型已导出: {output_path}")
        return output_path


if __name__ == "__main__":
    # 测试导出
    from src.models.unet_light import LightUNet
    
    model = LightUNet(base_channels=32, depth=4)
    
    # 导出ONNX
    ModelExporter.export_to_onnx(model, "test_model.onnx")
    print("导出测试完成!")
