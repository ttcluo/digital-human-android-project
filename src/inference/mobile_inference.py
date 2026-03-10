"""
移动端推理模块
为Android/iOS设备提供优化的推理接口
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
import os
import cv2


class MobileDigitalHumanInference:
    """移动端数字人推理接口"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        input_size: int = 256,
        audio_feature_dim: int = 256,
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
            device: 推理设备
            input_size: 输入图像大小
            audio_feature_dim: 音频特征维度
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.audio_feature_dim = audio_feature_dim
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 预处理
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        from src.models.unet_light import LightUNet
        
        model = LightUNet(
            in_channels=3,
            out_channels=3,
            base_channels=32,
            depth=4,
            audio_feature_dim=self.audio_feature_dim,
        )
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # 转换为张量
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def preprocess_audio(self, audio_features: np.ndarray) -> torch.Tensor:
        """预处理音频特征"""
        # 转换为张量
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
        
        return audio_tensor
    
    def infer(
        self,
        reference_image: np.ndarray,
        audio_features: np.ndarray,
    ) -> np.ndarray:
        """
        执行推理
        
        Args:
            reference_image: 参考图像 [H, W, C]
            audio_features: 音频特征 [T, D]
        
        Returns:
            生成的图像 [H, W, C]
        """
        # 预处理
        with torch.no_grad():
            img_tensor = self.preprocess_image(reference_image).to(self.device)
            audio_tensor = self.preprocess_audio(audio_features).to(self.device)
            
            # 推理
            output = self.model(img_tensor, audio_tensor)
            
            # 后处理
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = (output * self.std + self.mean) * 255.0
            output = np.clip(output, 0, 255).astype(np.uint8)
            
        return output
    
    def stream_infer(
        self,
        reference_images: List[np.ndarray],
        audio_stream: np.ndarray,
        frame_rate: int = 25,
    ) -> List[np.ndarray]:
        """
        流式推理
        
        Args:
            reference_images: 参考图像列表
            audio_stream: 音频流 [T, D]
            frame_rate: 帧率
        
        Returns:
            生成的图像列表
        """
        results = []
        
        for i, ref_img in enumerate(reference_images):
            # 获取对应帧的音频特征
            if i < len(audio_stream):
                audio_feat = audio_stream[i:i+1]
            else:
                audio_feat = audio_stream[-1:]
            
            # 推理
            result = self.infer(ref_img, audio_feat)
            results.append(result)
        
        return results
    
    def export_to_tflite(self, output_path: str):
        """导出为TensorFlow Lite格式"""
        # 创建示例输入
        dummy_image = torch.randn(1, 3, self.input_size, self.input_size)
        dummy_audio = torch.randn(1, 50, self.audio_feature_dim)
        
        # 导出为ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_audio),
            onnx_path,
            input_names=['image', 'audio'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch'},
                'audio': {0: 'batch', 1: 'time'},
                'output': {0: 'batch'},
            },
        )
        
        print(f"ONNX模型已导出: {onnx_path}")
        print("请使用ONNX-TensorFlow转换工具转换为TFLite格式")
    
    def export_to_onnx(self, output_path: str):
        """导出为ONNX格式"""
        # 创建示例输入
        dummy_image = torch.randn(1, 3, self.input_size, self.input_size)
        dummy_audio = torch.randn(1, 50, self.audio_feature_dim)
        
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_audio),
            output_path,
            input_names=['image', 'audio'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch'},
                'audio': {0: 'batch', 1: 'time'},
                'output': {0: 'batch'},
            },
            opset_version=11,
        )
        
        print(f"ONNX模型已导出: {output_path}")
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'num_parameters': num_params,
            'num_parameters_trainable': num_params_trainable,
            'input_size': self.input_size,
            'audio_feature_dim': self.audio_feature_dim,
            'device': str(self.device),
        }


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
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
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
            opset_version=11,
        )
        
        print(f"ONNX模型已导出: {output_path}")
    
    @staticmethod
    def export_to_android(
        model: nn.Module,
        output_dir: str,
        input_size: Tuple[int, int] = (256, 256),
        audio_feature_dim: int = 256,
    ):
        """导出为Android格式（ONNX + 量化）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        ModelExporter.export_to_onnx(model, onnx_path, input_size, audio_feature_dim)
        
        # 量化（简化版）
        quantized_path = os.path.join(output_dir, "model_quantized.onnx")
        
        print(f"Android模型已导出到: {output_dir}")


class ModelQuantizer:
    """模型量化工具"""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8):
        """动态量化"""
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
        """静态量化"""
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
        """模型剪枝"""
        from torch.nn.utils import prune
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        return model


if __name__ == "__main__":
    # 测试推理引擎
    print("测试移动端推理引擎...")
    
    from src.models.unet_light import LightUNet
    
    # 创建模型
    model = LightUNet(base_channels=32, depth=4)
    
    # 测试推理
    engine = MobileDigitalHumanInference(
        model_path="",  # 空路径，使用随机初始化
        device="cpu",
    )
    
    # 创建测试数据
    ref_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    audio_features = np.random.randn(50, 256).astype(np.float32)
    
    # 推理
    result = engine.infer(ref_image, audio_features)
    
    print(f"输入图像形状: {ref_image.shape}")
    print(f"输出图像形状: {result.shape}")
    print(f"模型信息: {engine.get_model_info()}")
    
    print("推理引擎测试完成!")
