"""
多任务语音分析模型 - 骨干网络
频谱学习CNN + 三维注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // len(kernel_sizes), 
                             kernel_size=k, padding=padding, bias=False),
                    nn.BatchNorm2d(out_channels // len(kernel_sizes)),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 多尺度特征
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        x = self.conv1x1(x)
        return x


class ChannelAttention(nn.Module):
    """通道注意力 (SE-like)"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    """卷积块注意力模块 (Channel + Spatial)"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.attention = CBAMBlock(channels) if use_attention else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class SpectralBackbone(nn.Module):
    """
    频谱学习骨干网络
    输入: (batch, 1, n_mels, time)
    输出: (batch, channels, 1, time)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        channels: list = [32, 64, 128, 256],
        n_residual_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.n_mels = n_mels
        
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # 下采样阶段
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        
        for out_ch in channels[1:]:
            stage = nn.Sequential()
            
            # 下采样卷积
            stage.add_module('downsample', nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            
            # 残差块
            for i in range(n_residual_blocks):
                stage.add_module(f'resblock_{i}', ResidualBlock(out_ch, use_attention))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # 频率池化 (将频率维度压缩到1)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # 最终通道数
        self.out_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time)
        
        Returns:
            (batch, out_channels, 1, time)
        """
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # 频率池化
        x = self.freq_pool(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """获取输出特征维度"""
        return self.out_channels


class LightweightBackbone(nn.Module):
    """
    轻量级骨干网络 (端侧优化)
    使用深度可分离卷积减少参数量
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        channels: list = [16, 32, 64, 128],
        n_residual_blocks: int = 1
    ):
        super().__init__()
        
        self.n_mels = n_mels
        
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # 下采样阶段 (使用标准卷积，更稳定)
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        
        for out_ch in channels[1:]:
            stage = nn.Sequential()
            
            # 下采样
            stage.add_module('downsample', nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            
            # 残差块
            for i in range(n_residual_blocks):
                stage.add_module(f'resblock_{i}', ResidualBlock(out_ch, use_attention=True))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # 频率池化
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.out_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.freq_pool(x)
        return x
    
    def get_output_dim(self) -> int:
        return self.out_channels


if __name__ == "__main__":
    # 测试
    print("Backbone module loaded successfully")
    
    # 测试标准骨干
    model = SpectralBackbone(n_mels=80, channels=[32, 64, 128, 256])
    x = torch.randn(2, 1, 80, 300)
    y = model(x)
    print(f"Standard Backbone: input {x.shape} -> output {y.shape}")
    print(f"Output channels: {model.get_output_dim()}")
    
    # 测试轻量级骨干
    model_light = LightweightBackbone(n_mels=80, channels=[16, 32, 64, 128])
    y_light = model_light(x)
    print(f"Lightweight Backbone: input {x.shape} -> output {y_light.shape}")
    print(f"Output channels: {model_light.get_output_dim()}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Standard params: {count_params(model):,}")
    print(f"Lightweight params: {count_params(model_light):,}")
