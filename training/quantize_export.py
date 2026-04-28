"""
模型量化与导出脚本
支持 ONNX 和 INT8 量化导出
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
sys.path.insert(0, '/data/mikexu/speech_analysis_project')

from training.finetune_pretrained import PretrainedSpeechModel


class QuantizedSpeechModel(nn.Module):
    """量化后的语音分析模型"""
    
    def __init__(self, model_path, pretrained_path='models/pretrained/hubert_base_ls960'):
        super().__init__()
        # 加载原始模型
        self.model = PretrainedSpeechModel(
            pretrained_model_name=pretrained_path,
            num_speakers=1000,
            freeze_backbone=True
        )
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 只对任务头进行动态量化
        self.quantized_age_head = torch.quantization.quantize_dynamic(
            self.model.age_head, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized_gender_head = torch.quantization.quantize_dynamic(
            self.model.gender_head, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized_emotion_head = torch.quantization.quantize_dynamic(
            self.model.emotion_head, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized_speaker_embedding = torch.quantization.quantize_dynamic(
            self.model.speaker_embedding, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized_speaker_classifier = torch.quantization.quantize_dynamic(
            self.model.speaker_classifier, {nn.Linear}, dtype=torch.qint8
        )
    
    def forward(self, input_values):
        """前向传播"""
        with torch.no_grad():
            outputs = self.model.backbone(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
            
            # 使用量化后的头
            speaker_embedding = self.quantized_speaker_embedding(pooled)
            speaker_embedding = nn.functional.normalize(speaker_embedding, p=2, dim=1)
            speaker_logits = self.quantized_speaker_classifier(speaker_embedding)
            
            age_logits = self.quantized_age_head(pooled)
            gender_logits = self.quantized_gender_head(pooled)
            emotion_logits = self.quantized_emotion_head(pooled)
            
            return {
                'speaker_embedding': speaker_embedding,
                'speaker_logits': speaker_logits,
                'age_logits': age_logits,
                'gender_logits': gender_logits,
                'emotion_logits': emotion_logits
            }


def export_to_onnx(
    model_path='models/pretrained_finetuned/hubert_multitask/best_model.pt',
    output_path='models/quantized/hubert_multitask.onnx',
    pretrained_path='models/pretrained/hubert_base_ls960'
):
    """导出模型到 ONNX 格式"""
    
    print("Loading model...")
    # 从训练日志推断说话人数量（训练集25人）
    num_speakers = 25
    model = PretrainedSpeechModel(
        pretrained_model_name=pretrained_path,
        num_speakers=num_speakers,
        freeze_backbone=True
    )
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # 创建示例输入 (batch=1, seq_len=48000)
    dummy_input = torch.randn(1, 48000)
    
    # 导出 ONNX
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_values'],
        output_names=['speaker_logits', 'age_logits', 'gender_logits', 'emotion_logits'],
        dynamic_axes={
            'input_values': {0: 'batch_size', 1: 'sequence_length'},
            'speaker_logits': {0: 'batch_size'},
            'age_logits': {0: 'batch_size'},
            'gender_logits': {0: 'batch_size'},
            'emotion_logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # 获取模型大小
    original_size = os.path.getsize(model_path) / 1024 / 1024
    onnx_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"ONNX export complete!")
    print(f"  Original model: {original_size:.1f} MB")
    print(f"  ONNX model: {onnx_size:.1f} MB")
    print(f"  Size ratio: {onnx_size/original_size*100:.1f}%")
    
    return output_path


def quantize_and_save(
    model_path='models/pretrained_finetuned/hubert_multitask/best_model.pt',
    output_path='models/quantized/hubert_multitask_int8.pt',
    pretrained_path='models/pretrained/hubert_base_ls960'
):
    """量化并保存模型"""
    
    print("Creating quantized model...")
    # 从训练日志推断说话人数量（训练集25人）
    num_speakers = 25
    model = PretrainedSpeechModel(
        pretrained_model_name=pretrained_path,
        num_speakers=num_speakers,
        freeze_backbone=True
    )
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # 只对任务头进行动态量化
    quantized_age_head = torch.quantization.quantize_dynamic(
        model.age_head, {nn.Linear}, dtype=torch.qint8
    )
    quantized_gender_head = torch.quantization.quantize_dynamic(
        model.gender_head, {nn.Linear}, dtype=torch.qint8
    )
    quantized_emotion_head = torch.quantization.quantize_dynamic(
        model.emotion_head, {nn.Linear}, dtype=torch.qint8
    )
    quantized_speaker_embedding = torch.quantization.quantize_dynamic(
        model.speaker_embedding, {nn.Linear}, dtype=torch.qint8
    )
    quantized_speaker_classifier = torch.quantization.quantize_dynamic(
        model.speaker_classifier, {nn.Linear}, dtype=torch.qint8
    )
    
    # 保存量化后的state_dict（完整模型无法序列化parametrized modules）
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quantized_state = {
        'age_head': quantized_age_head.state_dict(),
        'gender_head': quantized_gender_head.state_dict(),
        'emotion_head': quantized_emotion_head.state_dict(),
        'speaker_embedding': quantized_speaker_embedding.state_dict(),
        'speaker_classifier': quantized_speaker_classifier.state_dict(),
        'backbone': model.backbone.state_dict()
    }
    torch.save(quantized_state, output_path)
    
    # 获取模型大小
    original_size = os.path.getsize(model_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"Quantization complete!")
    print(f"  Original model: {original_size:.1f} MB")
    print(f"  Quantized model: {quantized_size:.1f} MB")
    print(f"  Size ratio: {quantized_size/original_size*100:.1f}%")
    
    return output_path


def benchmark_model(model_path, num_runs=10):
    """基准测试模型推理速度"""
    import time
    
    # 从训练日志推断说话人数量（训练集25人）
    num_speakers = 25
    model = PretrainedSpeechModel(
        pretrained_model_name='models/pretrained/hubert_base_ls960',
        num_speakers=num_speakers,
        freeze_backbone=True
    )
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # 预热
    dummy_input = torch.randn(1, 48000)
    for _ in range(3):
        _ = model(dummy_input)
    
    # 测试
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time*1000:.1f} ms")
    print(f"Throughput: {1/avg_time:.1f} samples/sec")
    
    return avg_time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/pretrained_finetuned/hubert_multitask/best_model.pt')
    parser.add_argument('--output-dir', default='models/quantized')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--export-int8', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.export_onnx:
        onnx_path = os.path.join(args.output_dir, 'hubert_multitask.onnx')
        export_to_onnx(args.model, onnx_path)
    
    if args.export_int8:
        int8_path = os.path.join(args.output_dir, 'hubert_multitask_int8.pt')
        quantize_and_save(args.model, int8_path)
    
    if args.benchmark:
        benchmark_model(args.model)
    
    # 默认执行所有操作
    if not (args.export_onnx or args.export_int8 or args.benchmark):
        print("Exporting ONNX...")
        onnx_path = os.path.join(args.output_dir, 'hubert_multitask.onnx')
        export_to_onnx(args.model, onnx_path)
        
        print("\nExporting INT8...")
        int8_path = os.path.join(args.output_dir, 'hubert_multitask_int8.pt')
        quantize_and_save(args.model, int8_path)
        
        print("\nBenchmarking...")
        benchmark_model(args.model)
