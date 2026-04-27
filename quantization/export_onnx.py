"""
导出ONNX格式
支持: FP32和INT8模型导出
"""

import os
import sys
import argparse
import torch
import torch.onnx
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 1, 80, 300),
    opset_version: int = 13
):
    """
    导出模型到ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: 输出路径
        input_shape: 输入形状 (batch, channel, n_mels, time)
        opset_version: ONNX opset版本
    """
    print(f"Exporting to ONNX: {output_path}")
    
    model.eval()
    model.cpu()
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 定义输入输出名称
    input_names = ['mel_spectrogram']
    output_names = [
        'speaker_embedding',
        'speaker_logits',
        'age_logits',
        'age_value',
        'gender_logits',
        'emotion_logits'
    ]
    
    # 动态轴 (支持变长输入)
    dynamic_axes = {
        'mel_spectrogram': {0: 'batch_size', 3: 'time'},
        'speaker_embedding': {0: 'batch_size'},
        'speaker_logits': {0: 'batch_size'},
        'age_logits': {0: 'batch_size'},
        'age_value': {0: 'batch_size'},
        'gender_logits': {0: 'batch_size'},
        'emotion_logits': {0: 'batch_size'}
    }
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"ONNX model exported to: {output_path}")
    
    # 验证
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except ImportError:
        print("onnx package not installed, skipping validation")
    except Exception as e:
        print(f"ONNX validation warning: {e}")


def simplify_onnx(onnx_path: str):
    """简化ONNX模型"""
    try:
        import onnx
        from onnxsim import simplify
        
        print(f"Simplifying ONNX model: {onnx_path}")
        
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        
        if check:
            onnx.save(model_simp, onnx_path)
            print("Simplification successful")
        else:
            print("Simplification failed, using original model")
    
    except ImportError:
        print("onnx-simplifier not installed, skipping simplification")
    except Exception as e:
        print(f"Simplification error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export to ONNX')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PyTorch model')
    parser.add_argument('--output', type=str, default='checkpoints/model.onnx',
                       help='Output ONNX path')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--n_mels', type=int, default=80,
                       help='Number of mel bins')
    parser.add_argument('--time_steps', type=int, default=300,
                       help='Number of time steps')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Exporting to ONNX")
    print("=" * 60)
    
    # 加载模型
    print(f"\nLoading model from: {args.model}")
    
    model = MultiTaskSpeechModel(
        n_mels=args.n_mels,
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
    
    # 导出
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    input_shape = (args.batch_size, 1, args.n_mels, args.time_steps)
    export_to_onnx(model, args.output, input_shape)
    
    # 简化
    if args.simplify:
        simplify_onnx(args.output)
    
    print("\nExport completed!")


if __name__ == '__main__':
    main()
