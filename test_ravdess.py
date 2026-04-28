#!/usr/bin/env python3
"""
RAVDESS数据集快速测试脚本
验证数据加载和模型推理
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

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
    
    # 添加batch和channel维度: (1, 1, n_mels, time)
    mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
    
    return mel_spec


def parse_ravdess_filename(filename):
    """解析RAVDESS文件名获取标签"""
    # 格式: 03-01-01-01-01-01-01.wav
    # modality-vocal_channel-emotion-intensity-statement-repetition-actor
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


def test_model_inference():
    """测试模型推理"""
    print("="*60)
    print("测试模型推理")
    print("="*60)
    
    # 创建模型
    model = MultiTaskSpeechModel(
        n_mels=80,
        num_speakers=24,
        num_emotions=8,
        num_age_groups=5,
        lightweight=True
    )
    model.eval()
    
    # 获取模型大小
    sizes = model.get_model_size()
    print(f"\n模型大小:")
    for k, v in sizes.items():
        print(f"  {k}: {v:,} params ({v*4/1024/1024:.2f} MB)")
    
    # 查找音频文件
    ravdess_dir = '/data/mikexu/speech_analysis_project/data/raw/ravdess/audio_speech'
    wav_files = []
    for root, dirs, files in os.walk(ravdess_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))
    
    print(f"\n找到 {len(wav_files)} 个音频文件")
    
    # 测试5个样本
    test_files = wav_files[:5]
    
    print("\n推理结果:")
    for wav_file in test_files:
        filename = os.path.basename(wav_file)
        labels = parse_ravdess_filename(filename)
        
        # 加载音频
        mel_spec = load_audio(wav_file)
        
        # 推理
        with torch.no_grad():
            predictions = model.predict(mel_spec)
        
        emotion_pred = predictions['emotion_pred'].item()
        gender_pred = predictions['gender_pred'].item()
        
        emotion_map = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        print(f"\n  文件: {filename}")
        print(f"  真实情绪: {labels['emotion']} (id={labels['emotion_id']})")
        print(f"  预测情绪: {emotion_map[emotion_pred]} (id={emotion_pred})")
        print(f"  真实性别: {labels['gender']}")
        print(f"  预测性别: {'male' if gender_pred == 0 else 'female'}")
        print(f"  情绪概率: {predictions['emotion_probs'][0].numpy().round(3)}")
    
    print("\n✅ 模型推理测试完成!")


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("测试数据加载")
    print("="*60)
    
    ravdess_dir = '/data/mikexu/speech_analysis_project/data/raw/ravdess/audio_speech'
    
    # 统计
    actor_counts = {}
    emotion_counts = {}
    gender_counts = {}
    
    for root, dirs, files in os.walk(ravdess_dir):
        for f in files:
            if f.endswith('.wav'):
                labels = parse_ravdess_filename(f)
                
                actor_counts[labels['actor_id']] = actor_counts.get(labels['actor_id'], 0) + 1
                emotion_counts[labels['emotion']] = emotion_counts.get(labels['emotion'], 0) + 1
                gender_counts[labels['gender']] = gender_counts.get(labels['gender'], 0) + 1
    
    print(f"\n数据集统计:")
    print(f"  总样本数: {sum(actor_counts.values())}")
    print(f"  说话人数: {len(actor_counts)}")
    print(f"  情绪分布:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"    {emotion}: {count}")
    print(f"  性别分布:")
    for gender, count in sorted(gender_counts.items()):
        print(f"    {gender}: {count}")
    
    print("\n✅ 数据加载测试完成!")


if __name__ == '__main__':
    test_data_loading()
    test_model_inference()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
