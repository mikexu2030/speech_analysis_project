#!/usr/bin/env python3
"""
批量评估脚本 - 测试整个测试集
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import SpeechDataset
from demo_inference import load_onnx_model, predict_onnx, extract_mel_spectrogram, load_audio, EMOTION_LABELS


def evaluate_test_set(model_path: str, test_json: str, backend: str = 'onnx'):
    """评估测试集"""
    
    # 加载测试数据
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    print(f"测试样本数: {len(test_data)}")
    
    # 加载模型
    if backend == 'onnx':
        session, input_name = load_onnx_model(model_path)
    else:
        from demo_inference import load_pytorch_model
        model = load_pytorch_model(model_path)
        device = 'cpu'
    
    # 统计
    emotion_correct = 0
    gender_correct = 0
    total = 0
    
    emotion_confusion = defaultdict(lambda: defaultdict(int))
    gender_confusion = defaultdict(lambda: defaultdict(int))
    
    latencies = []
    
    # 遍历测试样本
    for sample in tqdm(test_data[:100], desc='Evaluating'):  # 先测试100个
        audio_path = sample['audio_path']
        
        if not os.path.exists(audio_path):
            continue
        
        # 加载音频
        audio = load_audio(audio_path)
        mel_spec = extract_mel_spectrogram(audio)
        
        # 推理
        start = time.time()
        if backend == 'onnx':
            results = predict_onnx(session, input_name, mel_spec)
        else:
            from demo_inference import predict_pytorch
            results = predict_pytorch(model, mel_spec, device)
        latencies.append((time.time() - start) * 1000)
        
        # 情绪
        true_emotion = sample.get('emotion', -1)
        pred_emotion_label = results['emotion']['label']
        # 从EMOTION_LABELS反向查找索引
        pred_emotion = None
        for idx, label in EMOTION_LABELS.items():
            if label == pred_emotion_label:
                pred_emotion = idx
                break
        
        if true_emotion >= 0 and pred_emotion is not None:
            emotion_confusion[true_emotion][pred_emotion] += 1
            if true_emotion == pred_emotion:
                emotion_correct += 1
        
        # 性别
        true_gender = sample.get('gender', -1)
        pred_gender_label = results['gender']['label']
        pred_gender = 0 if pred_gender_label == 'female' else 1 if pred_gender_label == 'male' else -1
        
        if true_gender >= 0 and pred_gender >= 0:
            gender_confusion[true_gender][pred_gender] += 1
            if true_gender == pred_gender:
                gender_correct += 1
        
        total += 1
    
    # 打印结果
    print("\n" + "=" * 60)
    print("批量评估结果")
    print("=" * 60)
    print(f"评估样本数: {total}")
    print(f"平均推理耗时: {sum(latencies)/len(latencies):.2f} ms")
    print()
    
    if total > 0:
        print(f"情绪识别准确率: {emotion_correct/total:.2%} ({emotion_correct}/{total})")
        print(f"性别识别准确率: {gender_correct/total:.2%} ({gender_correct}/{total})")
    
    print("\n情绪混淆矩阵:")
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    print(f"{'True\\Pred':>10}", end='')
    for e in emotions:
        print(f"{e:>8}", end='')
    print()
    for true_idx in range(8):
        print(f"{emotions[true_idx]:>10}", end='')
        for pred_idx in range(8):
            print(f"{emotion_confusion[true_idx][pred_idx]:>8}", end='')
        print()
    
    return {
        'total': total,
        'emotion_acc': emotion_correct / total if total > 0 else 0,
        'gender_acc': gender_correct / total if total > 0 else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
        'emotion_confusion': dict(emotion_confusion),
        'gender_confusion': dict(gender_confusion)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/exported/model.onnx')
    parser.add_argument('--test_json', type=str, default='data/splits/test.json')
    parser.add_argument('--backend', type=str, default='onnx', choices=['onnx', 'pytorch'])
    parser.add_argument('--output', type=str, default='results/evaluation/batch_eval.json')
    
    args = parser.parse_args()
    
    results = evaluate_test_set(args.model, args.test_json, args.backend)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {args.output}")
