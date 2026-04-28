#!/usr/bin/env python3
"""
使用预训练模型进行实际评测
评测开源模型在RAVDESS上的情绪识别效果
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
from tqdm import tqdm

sys.path.insert(0, '/data/mikexu/speech_analysis_project')


def load_audio(path, sr=16000, max_length=3.0):
    """加载音频"""
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
    
    return audio


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
    intensity = 'normal' if int(parts[3]) == 1 else 'strong'
    
    return {
        'emotion': emotion_map.get(emotion_id, 'unknown'),
        'emotion_id': emotion_id - 1,
        'actor_id': actor_id,
        'gender': gender,
        'gender_id': 0 if gender == 'male' else 1,
        'intensity': intensity
    }


class Wav2Vec2EmotionEvaluator:
    """使用wav2vec 2.0进行情绪识别评测"""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        
        print(f"Loading {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # 由于预训练模型没有情绪分类头，我们需要添加一个
        # 这里使用基础模型提取特征，然后用简单的分类器
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=8,  # 8种情绪
            ignore_mismatched_sizes=True
        )
        self.model.eval()
        
        self.emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    def predict(self, audio_path):
        """预测单个音频"""
        audio = load_audio(audio_path)
        
        # 预处理
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
        
        return {
            'emotion_id': pred,
            'emotion': self.emotion_labels[pred],
            'confidence': probs[0][pred].item(),
            'all_probs': probs[0].numpy().tolist()
        }


class Emotion2VecEvaluator:
    """使用Emotion2Vec进行情绪识别"""
    
    def __init__(self):
        # Emotion2Vec需要通过funasr加载
        # 这里先实现一个占位符，等funasr安装后再完善
        self.emotion_labels = ['neutral', 'happy', 'angry', 'sad', 'disgust', 'fear', 'surprise', 'calm']
        print("Emotion2Vec evaluator placeholder (requires funasr)")
    
    def predict(self, audio_path):
        """预测单个音频"""
        # 占位符实现
        return {
            'emotion_id': 0,
            'emotion': 'unknown',
            'confidence': 0.0,
            'all_probs': [0.0] * 8
        }


def evaluate_model(evaluator, dataset_dir, max_samples=None):
    """
    评测模型在数据集上的性能
    
    Args:
        evaluator: 模型评测器
        dataset_dir: 数据集目录
        max_samples: 最大评测样本数 (None=全部)
    
    Returns:
        评测结果字典
    """
    # 收集所有音频文件
    wav_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))
    
    if max_samples:
        wav_files = wav_files[:max_samples]
    
    print(f"\nEvaluating {len(wav_files)} samples...")
    
    # 评测
    correct = 0
    total = 0
    emotion_correct = {i: 0 for i in range(8)}
    emotion_total = {i: 0 for i in range(8)}
    
    results = []
    
    for wav_file in tqdm(wav_files):
        filename = os.path.basename(wav_file)
        labels = parse_ravdess_filename(filename)
        
        try:
            # 预测
            pred = evaluator.predict(wav_file)
            
            # 统计
            is_correct = pred['emotion_id'] == labels['emotion_id']
            correct += int(is_correct)
            total += 1
            
            emotion_correct[labels['emotion_id']] += int(is_correct)
            emotion_total[labels['emotion_id']] += 1
            
            results.append({
                'file': filename,
                'true_emotion': labels['emotion'],
                'true_emotion_id': labels['emotion_id'],
                'pred_emotion': pred['emotion'],
                'pred_emotion_id': pred['emotion_id'],
                'confidence': pred['confidence'],
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    per_class_acc = {}
    for i in range(8):
        if emotion_total[i] > 0:
            per_class_acc[i] = emotion_correct[i] / emotion_total[i]
        else:
            per_class_acc[i] = 0
    
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct': correct,
        'per_class_accuracy': {emotion_labels[i]: per_class_acc[i] for i in range(8)},
        'per_class_counts': {emotion_labels[i]: emotion_total[i] for i in range(8)},
        'results': results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models on RAVDESS')
    parser.add_argument('--model', type=str, default='wav2vec2',
                       choices=['wav2vec2', 'emotion2vec', 'hubert'],
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, 
                       default='/data/mikexu/speech_analysis_project/data/raw/ravdess/audio_speech',
                       help='Dataset directory')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json',
                       help='Output file')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Max samples to evaluate')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Evaluating: {args.model}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    # 创建评测器
    if args.model == 'wav2vec2':
        evaluator = Wav2Vec2EmotionEvaluator()
    elif args.model == 'emotion2vec':
        evaluator = Emotion2VecEvaluator()
    else:
        print(f"Model {args.model} not yet implemented")
        return
    
    # 评测
    results = evaluate_model(evaluator, args.dataset, args.max_samples)
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total_samples']})")
    print(f"\nPer-class accuracy:")
    for emotion, acc in results['per_class_accuracy'].items():
        count = results['per_class_counts'][emotion]
        print(f"  {emotion:12s}: {acc:.4f} ({count} samples)")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
