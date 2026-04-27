"""
训练后量化 (Post-Training Quantization, PTQ)
支持: INT8量化、校准数据集
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.data_loader import load_split_data
from utils.audio_utils import audio_to_model_input


def collect_calibration_data(
    data_list: List[Dict],
    n_samples: int = 500,
    n_mels: int = 80,
    target_length: int = 300
) -> torch.Tensor:
    """
    收集校准数据
    
    Args:
        data_list: 数据列表
        n_samples: 校准样本数
        n_mels: Mel滤波器数量
        target_length: 目标长度
    
    Returns:
        校准数据 (n_samples, 1, n_mels, target_length)
    """
    print(f"Collecting {n_samples} calibration samples...")
    
    # 随机采样
    indices = np.random.choice(len(data_list), min(n_samples, len(data_list)), replace=False)
    
    calibration_data = []
    
    for idx in tqdm(indices, desc="Loading calibration data"):
        item = data_list[idx]
        
        mel_spec = audio_to_model_input(
            item['audio_path'],
            sr=16000,
            n_mels=n_mels,
            target_length=target_length,
            normalize=True
        )
        
        if mel_spec is not None:
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            calibration_data.append(mel_tensor)
    
    # 合并
    calibration_data = torch.stack(calibration_data)
    
    print(f"Collected {len(calibration_data)} valid calibration samples")
    
    return calibration_data


def apply_ptq(
    model: nn.Module,
    calibration_data: torch.Tensor,
    device: str = 'cuda'
) -> nn.Module:
    """
    应用PTQ量化
    
    使用PyTorch的FX Graph Mode Quantization
    """
    print("\nApplying PTQ quantization...")
    
    model.eval()
    model.cpu()
    
    # 准备量化配置
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 准备模型
    model_prepared = torch.quantization.prepare(model)
    
    # 校准
    print("Calibrating...")
    with torch.no_grad():
        for i in tqdm(range(0, len(calibration_data), 10), desc="Calibration"):
            batch = calibration_data[i:i+10]
            _ = model_prepared(batch)
    
    # 转换为量化模型
    model_quantized = torch.quantization.convert(model_prepared)
    
    print("PTQ quantization completed")
    
    return model_quantized


def quantize_with_torch_ao(
    model: nn.Module,
    calibration_data: torch.Tensor
) -> nn.Module:
    """
    使用torch.ao.quantization进行量化 (推荐)
    """
    print("\nApplying PTQ with torch.ao.quantization...")
    
    model.eval()
    model.cpu()
    
    # 配置
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    
    # 准备
    model_prepared = torch.ao.quantization.prepare(model)
    
    # 校准
    print("Calibrating...")
    with torch.no_grad():
        for i in tqdm(range(0, len(calibration_data), 10), desc="Calibration"):
            batch = calibration_data[i:i+10]
            _ = model_prepared(batch)
    
    # 转换
    model_quantized = torch.ao.quantization.convert(model_prepared)
    
    print("PTQ quantization completed")
    
    return model_quantized


def save_quantized_model(model: nn.Module, output_path: str):
    """保存量化模型"""
    torch.save(model.state_dict(), output_path)
    print(f"Quantized model saved to: {output_path}")


def compare_model_sizes(model_fp32: nn.Module, model_int8: nn.Module):
    """比较模型大小"""
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    size_fp32 = get_model_size(model_fp32)
    size_int8 = get_model_size(model_int8)
    
    print("\n" + "=" * 60)
    print("Model Size Comparison")
    print("=" * 60)
    print(f"FP32 Model: {size_fp32 / 1024 / 1024:.2f} MB")
    print(f"INT8 Model: {size_int8 / 1024 / 1024:.2f} MB")
    print(f"Compression Ratio: {size_fp32 / size_int8:.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Post-Training Quantization')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to FP32 model')
    parser.add_argument('--data', type=str, default='data/splits/train.json',
                       help='Path to calibration data')
    parser.add_argument('--output', type=str, default='checkpoints/model_int8_ptq.pt',
                       help='Output path for quantized model')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of calibration samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for loading FP32 model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Post-Training Quantization (PTQ)")
    print("=" * 60)
    
    # 加载FP32模型
    print(f"\nLoading FP32 model from: {args.model}")
    
    model_fp32 = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()
    
    print("FP32 model loaded")
    
    # 加载校准数据
    print(f"\nLoading calibration data from: {args.data}")
    data_list = load_split_data(Path(args.data).parent, Path(args.data).stem)
    
    calibration_data = collect_calibration_data(
        data_list,
        n_samples=args.n_samples
    )
    
    # 应用PTQ
    try:
        model_int8 = quantize_with_torch_ao(model_fp32, calibration_data)
    except Exception as e:
        print(f"torch.ao.quantization failed: {e}")
        print("Trying legacy quantization...")
        model_int8 = apply_ptq(model_fp32, calibration_data)
    
    # 比较大小
    compare_model_sizes(model_fp32, model_int8)
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_quantized_model(model_int8, args.output)
    
    print("\nPTQ completed!")


if __name__ == '__main__':
    main()
