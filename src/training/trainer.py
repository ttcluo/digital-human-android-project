"""
训练器模块
负责模型训练的主要流程
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any
from tqdm import tqdm
import numpy as np


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: nn.Module = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 训练设备
            save_dir: 模型保存目录
            log_dir: 日志目录
            use_tensorboard: 是否使用TensorBoard
            use_wandb: 是否使用WandB
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # 损失函数
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # 优化器
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            model.parameters(), lr=0.001
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 日志记录
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        if use_wandb:
            import wandb
            wandb.init(project="digital-human")
            wandb.watch(model)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备上
            inputs = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs, audio_features)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            # TensorBoard记录
            if self.use_tensorboard:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            
            # WandB记录
            if self.use_wandb:
                import wandb
                wandb.log({'train_loss': loss.item()})
        
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_metrics['loss'] = avg_loss
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                
                outputs = self.model(inputs, audio_features)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        # 记录验证损失
        if self.use_tensorboard:
            self.writer.add_scalar('Val/Loss', avg_val_loss, self.current_epoch)
        
        if self.use_wandb:
            import wandb
            wandb.log({'val_loss': avg_val_loss})
        
        return {'val_loss': avg_val_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
        
        # 每10个epoch保存一个检查点
        if self.current_epoch % 10 == 0:
            epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"从检查点恢复: epoch {self.current_epoch}, best_loss: {self.best_loss:.4f}")
    
    def train(self, num_epochs: int, start_epoch: int = 0, resume_from: Optional[str] = None):
        """
        开始训练
        
        Args:
            num_epochs: 训练轮数
            start_epoch: 起始epoch
            resume_from: 从哪个检查点恢复
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])
                
                # 更新学习率
                self.scheduler.step(val_metrics['val_loss'])
                
                # 检查是否为最佳模型
                is_best = val_metrics['val_loss'] < self.best_loss
                if is_best:
                    self.best_loss = val_metrics['val_loss']
            else:
                is_best = train_metrics['loss'] < self.best_loss
                if is_best:
                    self.best_loss = train_metrics['loss']
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 打印epoch信息
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Best Loss: {self.best_loss:.4f}")
            print("-" * 50)
        
        # 训练完成
        if self.use_tensorboard:
            self.writer.close()
        
        print("训练完成!")
        print(f"最佳损失: {self.best_loss:.4f}")
        print(f"模型保存位置: {self.save_dir}")


class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        use_sync_batch_norm: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.save_dir = save_dir
        
        # 分布式训练设置
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = torch.cuda.device_count()
        
        # 设置CUDA设备
        torch.cuda.set_device(self.rank)
        
        # 分布式训练
        if use_sync_batch_norm:
            self.model = nn.SyncBatchNorm.convert_sync_batch_norm(model)
        
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank
        )
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self, num_epochs: int):
        """分布式训练"""
        for epoch in range(num_epochs):
            self.model.train()
            
            # 训练一个epoch
            for batch in self.train_loader:
                inputs = batch['image'].cuda()
                targets = batch['target'].cuda()
                audio_features = batch['audio_features'].cuda()
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs, audio_features)
                loss = nn.functional.mse_loss(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
            
            # 保存检查点
            if self.rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, f'{self.save_dir}/checkpoint_epoch_{epoch}.pth')
                
                print(f"Epoch {epoch} 完成")


if __name__ == "__main__":
    # 测试训练器
    from src.models.unet_light import LightUNet
    
    # 创建模型
    model = LightUNet(base_channels=32, depth=4)
    
    # 创建虚拟数据加载器
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(2, 3, 256, 256),
                'target': torch.randn(2, 3, 256, 256),
                'audio_features': torch.randn(2, 50, 256),
            }
    
    train_loader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device="cpu",  # 使用CPU测试
        save_dir="./test_checkpoints",
        use_tensorboard=False,
    )
    
    # 测试训练一个epoch
    print("测试训练器...")
    metrics = trainer.train_epoch()
    print(f"训练指标: {metrics}")
    print("训练器测试完成!")
