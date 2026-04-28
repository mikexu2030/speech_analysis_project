#!/usr/bin/env python3
"""
语音四合一识别演示脚本
支持: 声纹识别 + 年龄估计 + 性别分类 + 情绪分类

用法:
    python3 demo_inference.py --model models/exported/model.onnx --audio sample.wav
    python3 demo_inference.py --model checkpoints/ravdess_multitask/checkpoints/best_model.pt --audio sample.wav --backend pytorch
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_model import MultiTaskSpeechModel


# 标签映射
EMOTION_LABELS = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}

GENDER_LABELS = {0: 'female', 1: 'male'}

AGE_LABELS = {
    0: 'child (0-12)', 1: 'teen (13-19)', 2: 'young adult (20-29)',
    3: 'adult (30-49)', 4: 'senior (50+)'
}


def load_audio(audio_path: str, sr: int = 16000, max_length: float = 3.0) -> np.ndarray:
    """加载音频并预处理"""
    audio, orig_sr = sf.read(audio_path)
    
    # 重采样
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    
    # 转单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 截断/填充到固定长度
    max_samples = int(max_length * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    
    return audio


def extract_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """提取Mel频谱图"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def load_pytorch_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """加载PyTorch模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model


def load_onnx_model(model_path: str):
    """加载ONNX模型"""
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    return session, input_name


def predict_pytorch(model: nn.Module, mel_spec: np.ndarray, device: str = 'cpu') -> dict:
    """PyTorch推理"""
    # 准备输入 (1, 1, n_mels, time)
    x = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.predict(x)
    
    # 提取结果
    results = {}
    
    # 情绪
    if 'emotion_probs' in outputs:
        probs = outputs['emotion_probs'].cpu().numpy()[0]
        pred = int(outputs['emotion_pred'].cpu().numpy()[0])
        results['emotion'] = {
            'label': EMOTION_LABELS.get(pred, 'unknown'),
            'confidence': float(probs[pred]),
            'all_probs': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)}
        }
    
    # 性别
    if 'gender_probs' in outputs:
        probs = outputs['gender_probs'].cpu().numpy()[0]
        pred = int(outputs['gender_pred'].cpu().numpy()[0])
        results['gender'] = {
            'label': GENDER_LABELS.get(pred, 'unknown'),
            'confidence': float(probs[pred]),
            'all_probs': {GENDER_LABELS[i]: float(p) for i, p in enumerate(probs)}
        }
    
    # 年龄
    if 'age_probs' in outputs:
        probs = outputs['age_probs'].cpu().numpy()[0]
        pred = int(np.argmax(probs))
        results['age'] = {
            'label': AGE_LABELS.get(pred, 'unknown'),
            'confidence': float(probs[pred]),
            'all_probs': {AGE_LABELS[i]: float(p) for i, p in enumerate(probs)}
        }
    
    # 声纹嵌入
    if 'speaker_embedding' in outputs:
        emb = outputs['speaker_embedding'].cpu().numpy()[0]
        results['speaker_embedding'] = {
            'dim': len(emb),
            'norm': float(np.linalg.norm(emb))
        }
    
    return results


def predict_onnx(session, input_name: str, mel_spec: np.ndarray) -> dict:
    """ONNX推理"""
    # 准备输入 (1, 1, n_mels, time)
    x = mel_spec.astype(np.float32)
    x = np.expand_dims(np.expand_dims(x, 0), 0)
    
    # 推理
    outputs = session.run(None, {input_name: x})
    
    # 输出顺序: speaker_embedding, speaker_logits, age_logits, age_value, gender_logits, emotion_logits
    speaker_emb = outputs[0][0]
    speaker_logits = outputs[1][0]
    age_logits = outputs[2][0]
    age_value = outputs[3][0]
    gender_logits = outputs[4][0]
    emotion_logits = outputs[5][0]
    
    # 计算概率
    emotion_probs = softmax(emotion_logits)
    gender_probs = softmax(gender_logits)
    age_probs = softmax(age_logits)
    
    results = {}
    
    # 情绪
    pred = int(np.argmax(emotion_probs))
    results['emotion'] = {
        'label': EMOTION_LABELS.get(pred, 'unknown'),
        'confidence': float(emotion_probs[pred]),
        'all_probs': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(emotion_probs)}
    }
    
    # 性别
    pred = int(np.argmax(gender_probs))
    results['gender'] = {
        'label': GENDER_LABELS.get(pred, 'unknown'),
        'confidence': float(gender_probs[pred]),
        'all_probs': {GENDER_LABELS[i]: float(p) for i, p in enumerate(gender_probs)}
    }
    
    # 年龄
    pred = int(np.argmax(age_probs))
    results['age'] = {
        'label': AGE_LABELS.get(pred, 'unknown'),
        'confidence': float(age_probs[pred]),
        'all_probs': {AGE_LABELS[i]: float(p) for i, p in enumerate(age_probs)}
    }
    
    # 声纹嵌入
    results['speaker_embedding'] = {
        'dim': len(speaker_emb),
        'norm': float(np.linalg.norm(speaker_emb))
    }
    
    return results


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax计算"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def print_results(results: dict, latency_ms: float):
    """打印推理结果"""
    print("\n" + "=" * 60)
    print("语音四合一识别结果")
    print("=" * 60)
    print(f"推理耗时: {latency_ms:.2f} ms")
    print()
    
    # 情绪
    if 'emotion' in results:
        emotion = results['emotion']
        print(f"[情绪识别]")
        print(f"  预测: {emotion['label']} (置信度: {emotion['confidence']:.2%})")
        print(f"  详细分布:")
        for label, prob in sorted(emotion['all_probs'].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {label:12s}: {prob:.2%} {bar}")
        print()
    
    # 性别
    if 'gender' in results:
        gender = results['gender']
        print(f"[性别识别]")
        print(f"  预测: {gender['label']} (置信度: {gender['confidence']:.2%})")
        print()
    
    # 年龄
    if 'age' in results:
        age = results['age']
        print(f"[年龄段识别]")
        print(f"  预测: {age['label']} (置信度: {age['confidence']:.2%})")
        print()
    
    # 声纹
    if 'speaker_embedding' in results:
        emb = results['speaker_embedding']
        print(f"[声纹嵌入]")
        print(f"  维度: {emb['dim']}")
        print(f"  L2范数: {emb['norm']:.4f}")
        print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='语音四合一识别演示')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pt 或 .onnx)')
    parser.add_argument('--audio', type=str, required=True,
                       help='音频文件路径 (.wav)')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'pytorch', 'onnx'],
                       help='推理后端')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径')
    parser.add_argument('--sr', type=int, default=16000,
                       help='采样率')
    parser.add_argument('--max_length', type=float, default=3.0,
                       help='最大音频长度(秒)')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 自动选择后端
    if args.backend == 'auto':
        if args.model.endswith('.onnx'):
            args.backend = 'onnx'
        else:
            args.backend = 'pytorch'
    
    print(f"模型: {args.model}")
    print(f"音频: {args.audio}")
    print(f"后端: {args.backend}")
    
    # 加载模型
    print("\n加载模型...")
    if args.backend == 'pytorch':
        model = load_pytorch_model(args.model, args.device)
    else:
        session, input_name = load_onnx_model(args.model)
    
    # 加载音频
    print("加载音频...")
    audio = load_audio(args.audio, args.sr, args.max_length)
    mel_spec = extract_mel_spectrogram(audio, args.sr)
    print(f"Mel频谱形状: {mel_spec.shape}")
    
    # 推理
    print("推理中...")
    start = time.time()
    
    if args.backend == 'pytorch':
        results = predict_pytorch(model, mel_spec, args.device)
    else:
        results = predict_onnx(session, input_name, mel_spec)
    
    latency_ms = (time.time() - start) * 1000
    
    # 打印结果
    print_results(results, latency_ms)
    
    # 保存JSON
    if args.output:
        results['latency_ms'] = latency_ms
        results['audio_path'] = args.audio
        results['model_path'] = args.model
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
