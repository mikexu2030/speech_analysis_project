"""
音频处理工具函数
支持: 音频加载、Mel谱图提取、MFCC提取、特征标准化
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Tuple, Optional, Union


def load_audio(path: str, sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并重新采样
    
    Args:
        path: 音频文件路径
        sr: 目标采样率
        mono: 是否转换为单声道
    
    Returns:
        (waveform, sample_rate)
    """
    try:
        waveform, orig_sr = sf.read(path, dtype='float32')
        
        # 转换为单声道
        if mono and len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # 重采样
        if orig_sr != sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sr)
        
        return waveform, sr
    
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None


def extract_melspectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> np.ndarray:
    """
    提取Mel频谱图
    
    Args:
        waveform: 音频波形
        sr: 采样率
        n_mels: Mel滤波器数量
        n_fft: FFT大小
        hop_length: 帧移
        win_length: 窗长
        f_min: 最小频率
        f_max: 最大频率
    
    Returns:
        Mel频谱图 (n_mels, time)
    """
    if f_max is None:
        f_max = sr // 2
    
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0
    )
    
    # 转换为对数刻度 (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> np.ndarray:
    """
    提取MFCC特征
    
    Args:
        waveform: 音频波形
        sr: 采样率
        n_mfcc: MFCC系数数量
        n_fft: FFT大小
        hop_length: 帧移
        n_mels: Mel滤波器数量
    
    Returns:
        MFCC (n_mfcc, time)
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    return mfcc


def pad_or_truncate(
    feature: np.ndarray,
    target_length: int,
    axis: int = -1,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    填充或截断特征到目标长度
    
    Args:
        feature: 输入特征
        target_length: 目标长度
        axis: 操作的轴
        pad_value: 填充值
    
    Returns:
        处理后的特征
    """
    current_length = feature.shape[axis]
    
    if current_length == target_length:
        return feature
    
    if current_length > target_length:
        # 截断
        slices = [slice(None)] * feature.ndim
        slices[axis] = slice(0, target_length)
        return feature[tuple(slices)]
    
    # 填充
    pad_width = [(0, 0)] * feature.ndim
    pad_width[axis] = (0, target_length - current_length)
    
    return np.pad(feature, pad_width, mode='constant', constant_values=pad_value)


def normalize_feature(
    feature: np.ndarray,
    method: str = 'global',
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    标准化特征
    
    Args:
        feature: 输入特征
        method: 'global' 或 'instance'
        mean: 预计算的均值 (global模式)
        std: 预计算的标准差 (global模式)
    
    Returns:
        (标准化后的特征, mean, std)
    """
    if method == 'global':
        if mean is None:
            mean = np.mean(feature)
        if std is None:
            std = np.std(feature) + 1e-8
        
        normalized = (feature - mean) / std
        return normalized, mean, std
    
    elif method == 'instance':
        mean = np.mean(feature)
        std = np.std(feature) + 1e-8
        normalized = (feature - mean) / std
        return normalized, mean, std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_spectrogram_length(audio_length: float, sr: int = 16000, hop_length: int = 256) -> int:
    """
    计算给定音频长度的频谱图帧数
    
    Args:
        audio_length: 音频时长(秒)
        sr: 采样率
        hop_length: 帧移
    
    Returns:
        频谱图帧数
    """
    n_samples = int(audio_length * sr)
    n_frames = 1 + (n_samples - 1) // hop_length
    return n_frames


def audio_to_model_input(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 80,
    target_length: int = 300,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    将音频文件转换为模型输入
    
    Args:
        audio_path: 音频文件路径
        sr: 采样率
        n_mels: Mel滤波器数量
        target_length: 目标帧数
        normalize: 是否标准化
    
    Returns:
        模型输入特征 (n_mels, target_length) 或 None
    """
    # 加载音频
    waveform, _ = load_audio(audio_path, sr=sr)
    if waveform is None:
        return None
    
    # 提取Mel谱图
    mel_spec = extract_melspectrogram(waveform, sr=sr, n_mels=n_mels)
    
    # 填充/截断
    mel_spec = pad_or_truncate(mel_spec, target_length, axis=-1)
    
    # 标准化
    if normalize:
        mel_spec, _, _ = normalize_feature(mel_spec, method='instance')
    
    return mel_spec


def batch_audio_to_model_input(
    audio_paths: list,
    sr: int = 16000,
    n_mels: int = 80,
    target_length: int = 300,
    normalize: bool = True
) -> Tuple[np.ndarray, list]:
    """
    批量转换音频为模型输入
    
    Args:
        audio_paths: 音频文件路径列表
        sr: 采样率
        n_mels: Mel滤波器数量
        target_length: 目标帧数
        normalize: 是否标准化
    
    Returns:
        (batch_features, valid_paths)
    """
    features = []
    valid_paths = []
    
    for path in audio_paths:
        feat = audio_to_model_input(path, sr, n_mels, target_length, normalize)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)
    
    if len(features) == 0:
        return np.array([]), []
    
    batch = np.stack(features, axis=0)
    return batch, valid_paths


if __name__ == "__main__":
    # 测试
    print("Audio utils loaded successfully")
    print(f"Functions: load_audio, extract_melspectrogram, extract_mfcc, pad_or_truncate, normalize_feature")
