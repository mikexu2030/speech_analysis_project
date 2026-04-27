"""
导出TFLite格式
支持: FP32/INT8量化、动态范围量化
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.data_loader import load_split_data
from utils.audio_utils import audio_to_model_input


def export_pytorch_to_onnx(model: torch.nn.Module, onnx_path: str, input_shape: tuple):
    """PyTorch -> ONNX"""
    print(f"Converting PyTorch to ONNX: {onnx_path}")
    
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(*input_shape)
    
    input_names = ['input']
    output_names = ['output']
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
        do_constant_folding=True
    )
    
    print(f"ONNX model saved to: {onnx_path}")


def export_onnx_to_tflite(
    onnx_path: str,
    tflite_path: str,
    quantize: bool = False,
    calibration_data: list = None
):
    """
    ONNX -> TFLite
    
    使用onnx-tf或tf-lite converter
    """
    print(f"Converting ONNX to TFLite: {tflite_path}")
    
    try:
        import tensorflow as tf
        import onnx
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 使用tf.lite转换器
        converter = tf.lite.TFLiteConverter.from_saved_model(onnx_path)
        
        if quantize:
            print("Applying INT8 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if calibration_data:
                # 全整数量化
                def representative_dataset():
                    for data in calibration_data:
                        yield [data.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.float32
        
        # 转换
        tflite_model = converter.convert()
        
        # 保存
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to: {tflite_path}")
        
        # 打印大小
        size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"Model size: {size_mb:.2f} MB")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install tensorflow onnx")
        raise


def export_direct_tflite(
    model: torch.nn.Module,
    tflite_path: str,
    input_shape: tuple = (1, 1, 80, 300)
):
    """
    直接导出TFLite (使用ai-edge-torch)
    """
    print(f"Exporting directly to TFLite: {tflite_path}")
    
    try:
        import ai_edge_torch
        
        # 准备示例输入
        sample_args = (torch.randn(*input_shape),)
        
        # 转换
        edge_model = ai_edge_torch.convert(model.eval(), sample_args)
        
        # 导出
        edge_model.export(tflite_path)
        
        print(f"TFLite model saved to: {tflite_path}")
        
    except ImportError:
        print("ai-edge-torch not installed, falling back to ONNX route")
        print("Install with: pip install ai-edge-torch")
        return False
    
    return True


def collect_calibration_data(data_list: list, n_samples: int = 100) -> list:
    """收集校准数据"""
    print(f"Collecting {n_samples} calibration samples...")
    
    indices = np.random.choice(len(data_list), min(n_samples, len(data_list)), replace=False)
    
    calibration_data = []
    for idx in indices:
        item = data_list[idx]
        mel_spec = audio_to_model_input(
            item['audio_path'],
            sr=16000,
            n_mels=80,
            target_length=300,
            normalize=True
        )
        if mel_spec is not None:
            # (n_mels, time) -> (1, 1, n_mels, time)
            mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]
            calibration_data.append(mel_spec)
    
    print(f"Collected {len(calibration_data)} calibration samples")
    return calibration_data


def main():
    parser = argparse.ArgumentParser(description='Export to TFLite')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PyTorch model')
    parser.add_argument('--output', type=str, default='checkpoints/model.tflite',
                       help='Output TFLite path')
    parser.add_argument('--onnx_path', type=str, default='checkpoints/temp_model.onnx',
                       help='Temporary ONNX path')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization')
    parser.add_argument('--data', type=str, default='data/splits/train.json',
                       help='Calibration data')
    parser.add_argument('--n_calib', type=int, default=100,
                       help='Number of calibration samples')
    parser.add_argument('--method', type=str, default='direct',
                       choices=['direct', 'onnx'],
                       help='Export method')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Exporting to TFLite")
    print("=" * 60)
    
    # 加载模型
    print(f"\nLoading model from: {args.model}")
    
    model = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 导出
    input_shape = (1, 1, 80, 300)
    
    if args.method == 'direct':
        success = export_direct_tflite(model, args.output, input_shape)
        if not success:
            print("Falling back to ONNX method...")
            args.method = 'onnx'
    
    if args.method == 'onnx':
        # PyTorch -> ONNX -> TFLite
        export_pytorch_to_onnx(model, args.onnx_path, input_shape)
        
        # 收集校准数据
        calibration_data = None
        if args.quantize:
            data_list = load_split_data(Path(args.data).parent, Path(args.data).stem)
            calibration_data = collect_calibration_data(data_list, args.n_calib)
        
        # ONNX -> TFLite
        export_onnx_to_tflite(
            args.onnx_path,
            args.output,
            quantize=args.quantize,
            calibration_data=calibration_data
        )
        
        # 清理临时文件
        if os.path.exists(args.onnx_path):
            os.remove(args.onnx_path)
            print(f"Cleaned up: {args.onnx_path}")
    
    print("\nExport completed!")


if __name__ == '__main__':
    main()
