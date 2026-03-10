"""
视频处理工具
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
import subprocess


def resize_video(
    video_path: str,
    output_path: str,
    target_size: Tuple[int, int],
    fps: Optional[int] = None,
):
    """调整视频大小
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        target_size: 目标尺寸 (width, height)
        fps: 输出视频帧率，如果为None则使用原视频帧率
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = original_fps
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    if not out.isOpened():
        raise ValueError(f"无法创建输出视频: {output_path}")
    
    # 处理每一帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整帧大小
        resized_frame = cv2.resize(frame, target_size)
        
        # 写入输出视频
        out.write(resized_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"处理进度: {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"视频调整完成: {output_path} ({frame_count} 帧)")


class VideoProcessor:
    """视频处理工具类"""
    
    def __init__(self, fps: int = 25):
        self.fps = fps
        
    def read_video(self, video_path: str) -> Tuple[List[np.ndarray], int]:
        """读取视频"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, int(cap.get(cv2.CAP_PROP_FPS))
    
    def write_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[int] = None,
    ):
        """写入视频"""
        if fps is None:
            fps = self.fps
        
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算起止帧
        start_frame = 0
        if start_time is not None:
            start_frame = int(start_time * video_fps)
        
        end_frame = total_frames
        if end_time is not None:
            end_frame = int(end_time * video_fps)
        
        # 提取帧
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while cap.isOpened() and frame_idx < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = output_path / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            frame_idx += 1
        
        cap.release()
        print(f"提取了 {frame_idx} 帧到 {output_dir}")
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
    ):
        """提取音频"""
        command = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'copy',
            '-y', output_path
        ]
        subprocess.run(command, check=True)
    
    def combine_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ):
        """合并音视频"""
        command = [
            'ffmpeg', '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y', output_path
        ]
        subprocess.run(command, check=True)
    
    def resize_frame(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """调整帧大小"""
        return cv2.resize(frame, target_size)
    
    def crop_frame(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """裁剪帧"""
        return frame[y:y+height, x:x+width]
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """归一化帧"""
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def denormalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """反归一化帧"""
        frame = (frame * 255.0).astype(np.uint8)
        return frame
    
    def apply_gaussian_blur(
        self,
        frame: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """高斯模糊"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """人脸检测（简化版）"""
        # 这里应该使用人脸检测模型
        # 简化实现：返回整个图像
        h, w = frame.shape[:2]
        return (0, 0, w, h)
    
    def align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """人脸对齐"""
        # 简化实现：不做对齐
        return frame
    
    def apply_color_correction(
        self,
        frame: np.ndarray,
        target_frame: np.ndarray,
    ) -> np.ndarray:
        """颜色校正"""
        # 简化实现：直方图匹配
        result = np.zeros_like(frame)
        
        for i in range(3):
            hist, bins = np.histogram(target_frame[:, :, i].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()
            
            lookup = np.zeros(256)
            result_hist, result_bins = np.histogram(frame[:, :, i].flatten(), 256, [0, 256])
            result_cdf = result_hist.cumsum()
            result_cdf_normalized = result_cdf / result_cdf.max()
            
            for j in range(256):
                diff = abs(result_cdf_normalized[j] - cdf_normalized)
                lookup[j] = np.argmin(diff)
            
            result[:, :, i] = lookup[frame[:, :, i]].astype(np.uint8)
        
        return result


if __name__ == "__main__":
    # 测试视频处理
    print("测试视频处理...")
    
    processor = VideoProcessor()
    
    # 创建测试帧
    test_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 测试功能
    resized = processor.resize_frame(test_frame, (128, 128))
    print(f"调整大小后形状: {resized.shape}")
    
    normalized = processor.normalize_frame(test_frame)
    print(f"归一化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    print("视频处理测试完成!")
