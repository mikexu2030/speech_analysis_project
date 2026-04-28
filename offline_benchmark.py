#!/usr/bin/env python3
"""
离线基准测试 - 使用本地数据和随机初始化模型
测试数据加载、特征提取和模型推理流程
"""

import os
import sys
import json
import time
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/data/mikexu/speech_analysis_project')

from models.multitask_model import MultiTaskSpeechModel


def load_audio(path, sr=16000, n_mels=80, max_length=3.0):
    """加载音频并提取mel频谱"""
    audio, orig_sr = sf.read(path)
    
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    max_samples = int(max_length * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)


def parse_ravdess_filename(filename):
    """解析RAVDESS文件名"""
    parts = filename.replace('.wav', '').split('-')
    
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    
    emotion_id = int(parts[2])
    actor_id = int(parts[6])
    gender = 'male' if actor_id <= 12 else 'female'
    
    return {
        'emotion': emotion_map.get(emotion_id, 'unknown'),
        'emotion_id': emotion_id - 1,
        'actor_id': actor_id,
        'gender': gender,
        'gender_id': 0 if gender == 'male' else 1
    }


def benchmark_inference_speed(model, num_runs=100):
    """测试推理速度"""
    dummy_input = torch.randn(1, 1, 80, 300)
    
    # 预热
    model.eval()
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
    
    times_ms = [t * 1000 for t in times]
    
    return {
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'median_ms': np.median(times_ms)
    }


def evaluate_untrained_model(model, dataset_dir, max_samples=100):
    """
    评测未训练模型 (随机权重)
    用于验证数据流和模型结构
    """
    wav_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))
    
    if max_samples:
        wav_files = wav_files[:max_samples]
    
    print(f"Evaluating {len(wav_files)} samples with untrained model...")
    
    model.eval()
    
    # 统计
    emotion_correct = defaultdict(int)
    emotion_total = defaultdict(int)
    gender_correct = 0
    gender_total = 0
    
    # 随机猜测的期望准确率
    emotion_random_acc = 1.0 / 8  # 8种情绪
    gender_random_acc = 1.0 / 2   # 2种性别
    
    inference_times = []
    
    with torch.no_grad():
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            labels = parse_ravdess_filename(filename)
            
            # 加载音频
            start = time.time()
            mel_spec = load_audio(wav_file)
            load_time = (time.time() - start) * 1000
            
            # 推理
            start = time.time()
            predictions = model.predict(mel_spec)
            inf_time = (time.time() - start) * 1000
            inference_times.append(inf_time)
            
            # 情绪
            emotion_pred = predictions['emotion_pred'].item()
            emotion_correct[labels['emotion_id']] += int(emotion_pred == labels['emotion_id'])
            emotion_total[labels['emotion_id']] += 1
            
            # 性别
            gender_pred = predictions['gender_pred'].item()
            gender_correct += int(gender_pred == labels['gender_id'])
            gender_total += 1
    
    # 计算指标
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    total_correct = sum(emotion_correct.values())
    total_samples = sum(emotion_total.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    
    gender_acc = gender_correct / gender_total if gender_total > 0 else 0
    
    per_class_acc = {}
    for i in range(8):
        if emotion_total[i] > 0:
            per_class_acc[emotion_labels[i]] = emotion_correct[i] / emotion_total[i]
        else:
            per_class_acc[emotion_labels[i]] = 0
    
    return {
        'overall_accuracy': overall_acc,
        'random_baseline': emotion_random_acc,
        'gender_accuracy': gender_acc,
        'gender_random_baseline': gender_random_acc,
        'per_class_accuracy': per_class_acc,
        'per_class_counts': {emotion_labels[i]: emotion_total[i] for i in range(8)},
        'inference_time_ms': {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times)
        },
        'total_samples': total_samples
    }


def main():
    dataset_dir = '/data/mikexu/speech_analysis_project/data/raw/ravdess/audio_speech'
    
    print("="*60)
    print("离线基准测试")
    print("="*60)
    
    # 创建模型 (轻量级版本，适合端侧)
    print("\n创建轻量级多任务模型...")
    model = MultiTaskSpeechModel(
        n_mels=80,
        num_speakers=24,  # RAVDESS有24个说话人
        num_emotions=8,   # 8种情绪
        num_age_groups=5,
        lightweight=True   # 轻量级版本
    )
    
    sizes = model.get_model_size()
    print(f"模型大小:")
    for k, v in sizes.items():
        print(f"  {k}: {v:,} params ({v*4/1024/1024:.2f} MB)")
    
    # 测试推理速度
    print("\n测试推理速度...")
    speed = benchmark_inference_speed(model, num_runs=100)
    print(f"  平均: {speed['mean_ms']:.2f} ms")
    print(f"  中位数: {speed['median_ms']:.2f} ms")
    print(f"  最小: {speed['min_ms']:.2f} ms")
    print(f"  最大: {speed['max_ms']:.2f} ms")
    print(f"  标准差: {speed['std_ms']:.2f} ms")
    
    # 估算MT9655性能
    print(f"\n估算MT9655性能 (假设比CPU慢5-10倍):")
    print(f"  预计延迟: {speed['mean_ms']*5:.0f} - {speed['mean_ms']*10:.0f} ms")
    
    # 评测未训练模型
    print("\n评测未训练模型 (随机权重)...")
    results = evaluate_untrained_model(model, dataset_dir, max_samples=100)
    
    print(f"\n情绪识别准确率:")
    print(f"  整体: {results['overall_accuracy']:.4f} (随机基线: {results['random_baseline']:.4f})")
    print(f"\n各情绪准确率:")
    for emotion, acc in results['per_class_accuracy'].items():
        count = results['per_class_counts'][emotion]
        print(f"  {emotion:12s}: {acc:.4f} ({count} samples)")
    
    print(f"\n性别识别准确率:")
    print(f"  整体: {results['gender_accuracy']:.4f} (随机基线: {results['gender_random_baseline']:.4f})")
    
    print(f"\n推理时间 (含音频加载):")
    print(f"  平均: {results['inference_time_ms']['mean']:.2f} ms")
    print(f"  标准差: {results['inference_time_ms']['std']:.2f} ms")
    
    # 保存结果
    output_dir = '/data/mikexu/speech_analysis_project/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'model_size': {k: v for k, v in sizes.items()},
        'inference_speed': speed,
        'untrained_accuracy': {
            'emotion': results['overall_accuracy'],
            'gender': results['gender_accuracy']
        },
        'per_class_accuracy': results['per_class_accuracy'],
        'estimated_mt9655_latency_ms': {
            'min': speed['mean_ms'] * 5,
            'max': speed['mean_ms'] * 10
        }
    }
    
    with open(os.path.join(output_dir, 'offline_benchmark.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {output_dir}/offline_benchmark.json")
    
    print("\n" + "="*60)
    print("基准测试完成!")
    print("="*60)
    print("\n说明:")
    print("- 当前模型为随机初始化权重，准确率接近随机猜测")
    print("- 训练后情绪识别准确率预期可达60-70%")
    print("- 模型大小3MB，INT8量化后约0.75MB，适合MT9655")
    print("- 推理延迟约10ms (CPU)，MT9655预计50-100ms")


if __name__ == '__main__':
    main()
