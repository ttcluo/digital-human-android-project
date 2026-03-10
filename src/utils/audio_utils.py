"""
音频处理工具
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import torch


class AudioProcessor:
    """音频处理工具类"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """保存音频文件"""
        sf.write(output_path, audio, self.sample_rate)
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return mfcc
    
    def extract_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ) -> np.ndarray:
        """提取频谱图"""
        spec = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        return np.abs(spec)
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """提取梅尔频谱图"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return librosa.power_to_db(mel_spec)
    
    def extract_hubert_features(self, audio: np.ndarray) -> np.ndarray:
        """提取HuBERT特征（简化版）"""
        # 这里应该是使用HuBERT模型提取特征
        # 简化实现：使用MFCC作为替代
        mfcc = self.extract_mfcc(audio, n_mfcc=13)
        
        # 扩展维度
        features = np.repeat(mfcc, 20, axis=1)  # 扩展到更长的序列
        
        return features
    
    def compute_energy(self, audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
        """计算音频能量"""
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        return energy
    
    def detect_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 1024,
        hop_length: int = 512,
    ) -> np.ndarray:
        """检测静音段"""
        energy = self.compute_energy(audio, frame_length, hop_length)
        silence_mask = energy < threshold
        return silence_mask
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
    ) -> np.ndarray:
        """去除静音"""
        audio, _ = librosa.effects.trim(audio, top_db=20)
        return audio
    
    def change_speed(
        self,
        audio: np.ndarray,
        speed_factor: float,
    ) -> np.ndarray:
        """改变音频速度"""
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def change_pitch(
        self,
        audio: np.ndarray,
        n_steps: float,
    ) -> np.ndarray:
        """改变音调"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_level: float = 0.005,
    ) -> np.ndarray:
        """添加噪声"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """归一化音频"""
        return librosa.util.normalize(audio)
    
    def resample(self, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """重采样"""
        return librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)


if __name__ == "__main__":
    # 测试音频处理
    print("测试音频处理...")
    
    processor = AudioProcessor()
    
    # 创建测试音频
    test_audio = np.random.randn(16000)  # 1秒音频
    
    # 测试各种功能
    mfcc = processor.extract_mfcc(test_audio)
    print(f"MFCC形状: {mfcc.shape}")
    
    spec = processor.extract_spectrogram(test_audio)
    print(f"频谱图形状: {spec.shape}")
    
    energy = processor.compute_energy(test_audio)
    print(f"能量形状: {energy.shape}")
    
    print("音频处理测试完成!")
