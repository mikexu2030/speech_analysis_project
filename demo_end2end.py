#!/usr/bin/env python3
"""
端到端语音四合一识别演示
用法: python3 demo_end2end.py --audio <wav_file>
"""

import argparse
import time
import numpy as np
import soundfile as sf
import librosa

from demo_inference import load_onnx_model, predict_onnx, EMOTION_LABELS, GENDER_LABELS
from utils.audio_utils import extract_melspectrogram, pad_or_truncate, normalize_feature


def preprocess_audio(audio_path: str, sr: int = 16000, n_mels: int = 80, target_length: int = 300):
    """音频预处理 (与训练一致，优化速度)"""
    # 加载音频 (使用soundfile直接读取)
    audio, orig_sr = sf.read(audio_path, dtype='float32')
    
    # 重采样 (如果需要)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    
    # 转单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 提取Mel频谱图 (与训练相同参数)
    mel_spec = extract_melspectrogram(
        waveform=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256
    )
    
    # 填充/截断
    mel_spec = pad_or_truncate(mel_spec, target_length, axis=-1)
    
    # 实例标准化
    mel_spec, _, _ = normalize_feature(mel_spec, method='instance')
    
    return mel_spec


def main():
    parser = argparse.ArgumentParser(description='语音四合一识别演示')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', type=str, default='models/exported/model_int8.onnx',
                       help='ONNX模型路径')
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    session, input_name = load_onnx_model(args.model)
    
    # 预处理音频
    print(f"处理音频: {args.audio}")
    start = time.time()
    mel_spec = preprocess_audio(args.audio)
    preprocess_time = (time.time() - start) * 1000
    
    # 推理
    start = time.time()
    results = predict_onnx(session, input_name, mel_spec)
    inference_time = (time.time() - start) * 1000
    
    # 打印结果
    print("\n" + "=" * 60)
    print("语音四合一识别结果")
    print("=" * 60)
    print(f"预处理耗时: {preprocess_time:.2f} ms")
    print(f"推理耗时: {inference_time:.2f} ms")
    print(f"总耗时: {preprocess_time + inference_time:.2f} ms")
    print()
    
    # 情绪
    emotion = results['emotion']
    print(f"[情绪识别]")
    print(f"  预测: {emotion['label']} (置信度: {emotion['confidence']:.2%})")
    print(f"  详细分布:")
    for label, prob in sorted(emotion['all_probs'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {label:12s}: {prob:.2%} {bar}")
    print()
    
    # 性别
    gender = results['gender']
    print(f"[性别识别]")
    print(f"  预测: {gender['label']} (置信度: {gender['confidence']:.2%})")
    print()
    
    # 年龄
    age = results['age']
    print(f"[年龄估计]")
    print(f"  预测: {age['label']} (置信度: {age['confidence']:.2%})")
    print()
    
    # 声纹
    speaker = results['speaker_embedding']
    print(f"[声纹嵌入]")
    print(f"  维度: {speaker['dim']}")
    print(f"  L2范数: {speaker['norm']:.4f}")
    print()


if __name__ == '__main__':
    main()
