import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_model import MultiTaskSpeechModel


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: tuple = (1, 1, 80, 300),
    opset_version: int = 11
):
    """
    导出模型为ONNX格式
    
    Args:
        model_path: 模型检查点路径
        output_path: 输出ONNX路径
        input_shape: 输入形状 (batch, 1, n_mels, time)
        opset_version: ONNX opset版本
    """
    print(f"Loading model from: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型实例 (使用与训练时相同的配置)
    model = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7,
        use_attention=True,
        lightweight=False
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 导出ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    # 使用旧版torch.onnx.export (dynamo=False)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['speaker_embedding', 'speaker_logits', 
                           'age_logits', 'age_value',
                           'gender_logits', 'emotion_logits'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch_size', 3: 'time'},
                'speaker_embedding': {0: 'batch_size'},
                'speaker_logits': {0: 'batch_size'},
                'age_logits': {0: 'batch_size'},
                'age_value': {0: 'batch_size'},
                'gender_logits': {0: 'batch_size'},
                'emotion_logits': {0: 'batch_size'}
            },
            dynamo=False
        )
    
    print(f"ONNX model exported successfully!")
    
    # 验证模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # 获取模型大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    return output_path


def quantize_onnx_model(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = 'dynamic'
):
    """
    量化ONNX模型
    
    Args:
        onnx_path: 输入ONNX模型路径
        output_path: 输出量化模型路径
        quantization_mode: 量化模式 ('dynamic', 'static', 'qat')
    """
    print(f"Quantizing ONNX model: {onnx_path}")
    print(f"Mode: {quantization_mode}")
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantization_mode == 'dynamic':
            # 动态量化 (INT8权重, FP32激活)
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8
            )
            
        elif quantization_mode == 'static':
            # 静态量化 (需要校准数据)
            print("Static quantization requires calibration data.")
            print("Please provide a representative dataset.")
            return None
            
        elif quantization_mode == 'qat':
            # 量化感知训练 (需要在训练时进行)
            print("QAT should be done during training.")
            return None
        
        print(f"Quantized model saved: {output_path}")
        
        # 比较大小
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\nSize comparison:")
        print(f"  Original:   {original_size:.2f} MB")
        print(f"  Quantized:  {quantized_size:.2f} MB")
        print(f"  Reduction:  {(1 - quantized_size/original_size)*100:.1f}%")
        
        return output_path
        
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        return None


def export_for_tflite(
    onnx_path: str,
    output_path: str
):
    """
    导出为TFLite格式 (通过ONNX转换)
    
    Args:
        onnx_path: ONNX模型路径
        output_path: 输出TFLite路径
    """
    print(f"Converting ONNX to TFLite: {output_path}")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 转换为TensorFlow
        tf_rep = prepare(onnx_model)
        
        # 保存TensorFlow模型
        tf_path = output_path.replace('.tflite', '_tf')
        tf_rep.export_graph(tf_path)
        
        # 转换为TFLite
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        
        # 优化设置
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # 转换
        tflite_model = converter.convert()
        
        # 保存
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved: {output_path}")
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
        return output_path
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install onnx-tf tensorflow")
        return None


def benchmark_model(
    model_path: str,
    input_shape: tuple = (1, 1, 80, 300),
    num_runs: int = 100
):
    """
    基准测试模型推理速度
    
    Args:
        model_path: 模型路径 (.pt 或 .onnx)
        input_shape: 输入形状
        num_runs: 测试次数
    """
    import time
    
    print(f"Benchmarking: {model_path}")
    print(f"Input shape: {input_shape}")
    print(f"Runs: {num_runs}")
    
    dummy_input = torch.randn(*input_shape)
    
    if model_path.endswith('.pt'):
        # PyTorch模型
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict):
            # 加载检查点
            model_obj = MultiTaskSpeechModel()
            model_obj.load_state_dict(model['model_state_dict'])
            model = model_obj
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测试
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
    elif model_path.endswith('.onnx'):
        # ONNX模型
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # 预热
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input.numpy()})
        
        # 测试
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = session.run(None, {input_name: dummy_input.numpy()})
            times.append(time.time() - start)
    
    # 统计结果
    times_ms = [t * 1000 for t in times]
    
    print(f"\nResults:")
    print(f"  Mean:   {sum(times_ms)/len(times_ms):.2f} ms")
    print(f"  Min:    {min(times_ms):.2f} ms")
    print(f"  Max:    {max(times_ms):.2f} ms")
    print(f"  Median: {sorted(times_ms)[len(times_ms)//2]:.2f} ms")
    
    # 估算MT9655性能 (假设比CPU慢5-10倍)
    print(f"\nEstimated MT9655 performance:")
    print(f"  ~{sum(times_ms)/len(times_ms)*5:.0f} - {sum(times_ms)/len(times_ms)*10:.0f} ms")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export and quantize model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pt)')
    parser.add_argument('--output_dir', type=str, default='outputs/exported',
                       help='Output directory')
    parser.add_argument('--export_onnx', action='store_true',
                       help='Export to ONNX')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize ONNX model')
    parser.add_argument('--export_tflite', action='store_true',
                       help='Export to TFLite')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark model')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出ONNX
    if args.export_onnx:
        onnx_path = output_dir / 'model.onnx'
        export_to_onnx(args.model, str(onnx_path))
        
        # 量化
        if args.quantize:
            quantized_path = output_dir / 'model_int8.onnx'
            quantize_onnx_model(str(onnx_path), str(quantized_path), 'dynamic')
        
        # TFLite
        if args.export_tflite:
            tflite_path = output_dir / 'model.tflite'
            export_for_tflite(str(onnx_path), str(tflite_path))
    
    # 基准测试
    if args.benchmark:
        if args.export_onnx:
            benchmark_model(str(output_dir / 'model.onnx'))
        else:
            benchmark_model(args.model)
    
    print(f"\nExport complete! Files saved to: {output_dir}")


if __name__ == '__main__':
    main()
