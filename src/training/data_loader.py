"""
数据加载器模块
负责数据集的加载和预处理
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import librosa
from scipy import signal


class DigitalHumanDataset(Dataset):
    """数字人数据集"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        audio_sample_rate: int = 16000,
        video_fps: int = 25,
        mode: str = "train",
        transform: Optional[Callable] = None,
        asr_encoder: str = "hubert",
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            image_size: 图像大小
            audio_sample_rate: 音频采样率
            video_fps: 视频帧率
            mode: 'train' 或 'val'
            transform: 数据增强
            asr_encoder: ASR编码器类型
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        self.mode = mode
        self.transform = transform
        self.asr_encoder = asr_encoder
        
        # 加载数据列表
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        # 数据目录结构: data_dir/video_id/{images,audio,landmarks}
        for video_dir in self.data_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            images_dir = video_dir / "images"
            audio_file = video_dir / "audio.wav"
            landmarks_file = video_dir / "landmarks.npy"
            
            if not images_dir.exists() or not audio_file.exists():
                continue
            
            # 获取图像列表
            image_files = sorted(list(images_dir.glob("*.jpg")))
            
            # 加载关键点
            landmarks = None
            if landmarks_file.exists():
                landmarks = np.load(landmarks_file)
            
            # 添加样本
            for i, img_file in enumerate(image_files):
                sample = {
                    'video_id': video_dir.name,
                    'image_path': str(img_file),
                    'audio_path': str(audio_file),
                    'frame_idx': i,
                    'total_frames': len(image_files),
                }
                
                if landmarks is not None and i < len(landmarks):
                    sample['landmarks'] = landmarks[i]
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # 归一化到[-1, 1]
        image = torch.FloatTensor(image).permute(2, 0, 1)  # [C, H, W]
        
        # 加载音频特征
        audio_features = self._load_audio_features(
            sample['audio_path'],
            sample['frame_idx'],
            sample['total_frames']
        )
        
        # 目标图像（与输入相同，用于自监督训练）
        target = image.clone()
        
        # 加载关键点
        landmarks = None
        if 'landmarks' in sample:
            landmarks = torch.FloatTensor(sample['landmarks'])
        
        return {
            'image': image,
            'target': target,
            'audio_features': audio_features,
            'landmarks': landmarks,
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
        }
    
    def _load_audio_features(
        self,
        audio_path: str,
        frame_idx: int,
        total_frames: int,
    ) -> torch.Tensor:
        """加载音频特征"""
        # 简化实现：使用随机特征
        # 实际应该使用HuBERT或WeNet提取
        audio_feature_dim = 256
        
        # 为每一帧生成音频特征
        audio_features = torch.randn(total_frames, audio_feature_dim)
        
        # 使用高斯平滑使特征更平滑
        kernel_size = 5
        kernel = np.exp(-np.arange(kernel_size) ** 2 / (2 * (kernel_size // 2) ** 2))
        kernel = kernel / kernel.sum()
        
        for dim in range(audio_feature_dim):
            audio_features[:, dim] = torch.from_numpy(
                np.convolve(audio_features[:, dim].numpy(), kernel, mode='same')
            )
        
        return audio_features


class StreamDataset(Dataset):
    """流式推理数据集"""
    
    def __init__(
        self,
        image_dir: str,
        audio_dir: str,
        image_size: int = 256,
    ):
        self.image_dir = Path(image_dir)
        self.audio_dir = Path(audio_dir)
        self.image_size = image_size
        
        # 加载图像列表
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载参考图像
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return {
            'reference_image': image,
            'image_path': str(self.image_files[idx]),
        }


class DataLoaderWrapper:
    """数据加载器包装器"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: int = 256,
        audio_sample_rate: int = 16000,
        video_fps: int = 25,
        train_split: float = 0.9,
        asr_encoder: str = "hubert",
    ):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            num_workers: 工作进程数
            image_size: 图像大小
            audio_sample_rate: 音频采样率
            video_fps: 视频帧率
            train_split: 训练集比例
            asr_encoder: ASR编码器类型
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 创建完整数据集
        full_dataset = DigitalHumanDataset(
            data_dir=data_dir,
            image_size=image_size,
            audio_sample_rate=audio_sample_rate,
            video_fps=video_fps,
            asr_encoder=asr_encoder,
        )
        
        # 划分训练集和验证集
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * train_split)
        val_size = dataset_size - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def get_val_loader(self, shuffle: bool = False) -> DataLoader:
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 256,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        image_size: 图像大小
        train_split: 训练集比例
    
    Returns:
        train_loader, val_loader
    """
    wrapper = DataLoaderWrapper(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        train_split=train_split,
    )
    
    return wrapper.get_train_loader(), wrapper.get_val_loader()


if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")
    
    # 创建虚拟数据集
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 256, 256),
                'target': torch.randn(3, 256, 256),
                'audio_features': torch.randn(50, 256),
            }
    
    # 创建DataLoader
    loader = DataLoader(DummyDataset(50), batch_size=4, shuffle=True)
    
    # 测试
    for batch in loader:
        print(f"批次形状: image={batch['image'].shape}, target={batch['target'].shape}")
        break
    
    print("数据加载器测试完成!")
