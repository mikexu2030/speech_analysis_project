"""
数据增强工具
支持: 音频增强 (速度/音调/噪声/音量) 和 频谱增强 (时域/频域掩码)
"""

import numpy as np
import librosa
import torch
from typing import Optional, Tuple


class AudioAugmentor:
    """音频增强器"""
    
    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        noise_level: float = 0.005,
        volume_range: Tuple[float, float] = (0.8, 1.2),
        sample_rate: int = 16000
    ):
        self.speed_range = speed_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_level = noise_level
        self.volume_range = volume_range
        self.sample_rate = sample_rate
    
    def augment(self, waveform: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        应用随机增强
        
        Args:
            waveform: 音频波形
            prob: 每种增强的概率
        
        Returns:
            增强后的波形
        """
        augmented = waveform.copy()
        
        # 速度变化
        if np.random.random() < prob:
            augmented = self.speed_change(augmented)
        
        # 音调变化
        if np.random.random() < prob:
            augmented = self.pitch_shift(augmented)
        
        # 添加噪声
        if np.random.random() < prob:
            augmented = self.add_noise(augmented)
        
        # 音量变化
        if np.random.random() < prob:
            augmented = self.volume_change(augmented)
        
        return augmented
    
    def speed_change(self, waveform: np.ndarray) -> np.ndarray:
        """速度变化 (保持音调)"""
        factor = np.random.uniform(*self.speed_range)
        return librosa.effects.time_stretch(waveform, rate=factor)
    
    def pitch_shift(self, waveform: np.ndarray) -> np.ndarray:
        """音调变化"""
        n_steps = np.random.randint(*self.pitch_shift_range)
        return librosa.effects.pitch_shift(
            waveform, sr=self.sample_rate, n_steps=n_steps
        )
    
    def add_noise(self, waveform: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_level, waveform.shape)
        return waveform + noise
    
    def volume_change(self, waveform: np.ndarray) -> np.ndarray:
        """音量变化"""
        factor = np.random.uniform(*self.volume_range)
        return waveform * factor


class SpecAugment:
    """频谱增强 (SpecAugment)"""
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        prob: float = 0.5
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.prob = prob
    
    def augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        应用频谱增强
        
        Args:
            spectrogram: 频谱图 (n_mels, time)
        
        Returns:
            增强后的频谱图
        """
        if np.random.random() > self.prob:
            return spectrogram
        
        augmented = spectrogram.copy()
        
        # 频率掩码
        for _ in range(self.n_freq_masks):
            augmented = self._freq_mask(augmented)
        
        # 时间掩码
        for _ in range(self.n_time_masks):
            augmented = self._time_mask(augmented)
        
        return augmented
    
    def _freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """频率域掩码"""
        n_mels = spec.shape[0]
        f = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        
        spec[f0:f0+f, :] = 0
        return spec
    
    def _time_mask(self, spec: np.ndarray) -> np.ndarray:
        """时间域掩码"""
        n_frames = spec.shape[1]
        t = np.random.randint(0, min(self.time_mask_param, n_frames))
        t0 = np.random.randint(0, n_frames - t)
        
        spec[:, t0:t0+t] = 0
        return spec


class CombinedAugmentor:
    """组合增强器"""
    
    def __init__(
        self,
        audio_aug: Optional[AudioAugmentor] = None,
        spec_aug: Optional[SpecAugment] = None,
        audio_prob: float = 0.5,
        spec_prob: float = 0.5
    ):
        self.audio_aug = audio_aug or AudioAugmentor()
        self.spec_aug = spec_aug or SpecAugment()
        self.audio_prob = audio_prob
        self.spec_prob = spec_prob
    
    def augment_audio(self, waveform: np.ndarray) -> np.ndarray:
        """音频增强"""
        if np.random.random() < self.audio_prob:
            return self.audio_aug.augment(waveform, prob=0.5)
        return waveform
    
    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """频谱增强"""
        if np.random.random() < self.spec_prob:
            return self.spec_aug.augment(spectrogram)
        return spectrogram


if __name__ == "__main__":
    # 测试
    print("Data augmentation module loaded successfully")
    
    # 创建测试数据
    sr = 16000
    duration = 2
    waveform = np.random.randn(sr * duration)
    spectrogram = np.random.randn(80, 100)
    
    # 测试音频增强
    audio_aug = AudioAugmentor()
    aug_waveform = audio_aug.augment(waveform)
    print(f"Original waveform shape: {waveform.shape}")
    print(f"Augmented waveform shape: {aug_waveform.shape}")
    
    # 测试频谱增强
    spec_aug = SpecAugment()
    aug_spec = spec_aug.augment(spectrogram)
    print(f"Original spectrogram shape: {spectrogram.shape}")
    print(f"Augmented spectrogram shape: {aug_spec.shape}")
