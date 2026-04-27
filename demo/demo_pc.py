#!/usr/bin/env python3
"""
PC端Demo
支持: 单音频推理、实时麦克风输入
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.audio_utils import audio_to_model_input, load_audio, extract_melspectrogram, pad_or_truncate


# 情绪标签
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
GENDER_LABELS = ['female', 'male']
AGE_GROUPS = ['child/teen', 'young adult', 'middle age', 'senior', 'elderly']


class SpeechAnalyzer:
    """语音分析器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        n_mels: int = 80,
        target_length: int = 300
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.n_mels = n_mels
        self.target_length = target_length
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded. Device: {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型"""
        model = MultiTaskSpeechModel(
            n_mels=self.n_mels,
            backbone_channels=[32, 64, 128, 256],
            embedding_dim=192,
            num_speakers=1000,
            num_age_groups=5,
            num_emotions=7
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def analyze(self, audio_path: str) -> Dict:
        """
        分析音频
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            分析结果字典
        """
        # 预处理
        start_time = time.time()
        
        mel_spec = audio_to_model_input(
            audio_path,
            sr=16000,
            n_mels=self.n_mels,
            target_length=self.target_length,
            normalize=True
        )
        
        if mel_spec is None:
            return {'error': 'Failed to load audio'}
        
        preprocess_time = time.time() - start_time
        
        # 推理
        start_time = time.time()
        
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(mel_tensor, return_probs=True)
        
        inference_time = time.time() - start_time
        
        # 解析结果
        results = {
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'total_time': preprocess_time + inference_time
        }
        
        # 情绪
        if 'emotion_probs' in predictions:
            emotion_probs = predictions['emotion_probs'].cpu().numpy()[0]
            emotion_idx = int(predictions['emotion_pred'].cpu().numpy()[0])
            
            results['emotion'] = {
                'label': EMOTION_LABELS[emotion_idx],
                'index': emotion_idx,
                'confidence': float(emotion_probs[emotion_idx]),
                'probabilities': {
                    EMOTION_LABELS[i]: float(emotion_probs[i])
                    for i in range(len(EMOTION_LABELS))
                }
            }
        
        # 性别
        if 'gender_probs' in predictions:
            gender_probs = predictions['gender_probs'].cpu().numpy()[0]
            gender_idx = int(predictions['gender_pred'].cpu().numpy()[0])
            
            results['gender'] = {
                'label': GENDER_LABELS[gender_idx],
                'index': gender_idx,
                'confidence': float(gender_probs[gender_idx]),
                'probabilities': {
                    GENDER_LABELS[i]: float(gender_probs[i])
                    for i in range(len(GENDER_LABELS))
                }
            }
        
        # 年龄
        if 'age_pred' in predictions:
            age_value = float(predictions['age_pred'].cpu().numpy()[0])
            age_group_idx = min(int(age_value / 20), 4)
            
            results['age'] = {
                'estimated_years': age_value,
                'group': AGE_GROUPS[age_group_idx],
                'group_index': age_group_idx
            }
        
        # 说话人嵌入
        if 'speaker_embedding' in predictions:
            embedding = predictions['speaker_embedding'].cpu().numpy()[0]
            results['speaker'] = {
                'embedding_dim': len(embedding),
                'embedding_norm': float(np.linalg.norm(embedding))
            }
        
        return results
    
    def print_results(self, results: Dict):
        """打印结果"""
        print("\n" + "=" * 60)
        print("Speech Analysis Results")
        print("=" * 60)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # 情绪
        if 'emotion' in results:
            emotion = results['emotion']
            print(f"\n🎭 Emotion: {emotion['label'].upper()}")
            print(f"   Confidence: {emotion['confidence']:.2%}")
            print(f"   Top 3:")
            sorted_emotions = sorted(
                emotion['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for label, prob in sorted_emotions:
                bar = '█' * int(prob * 20)
                print(f"     {label:12s} {bar} {prob:.2%}")
        
        # 性别
        if 'gender' in results:
            gender = results['gender']
            print(f"\n👤 Gender: {gender['label'].upper()}")
            print(f"   Confidence: {gender['confidence']:.2%}")
        
        # 年龄
        if 'age' in results:
            age = results['age']
            print(f"\n📅 Age: ~{age['estimated_years']:.0f} years")
            print(f"   Group: {age['group']}")
        
        # 性能
        print(f"\n⚡ Performance:")
        print(f"   Preprocess: {results['preprocess_time']*1000:.1f}ms")
        print(f"   Inference:  {results['inference_time']*1000:.1f}ms")
        print(f"   Total:      {results['total_time']*1000:.1f}ms")
        print("=" * 60)


def analyze_microphone(analyzer: SpeechAnalyzer, duration: float = 3.0):
    """从麦克风分析"""
    try:
        import sounddevice as sd
        import soundfile as sf
        import tempfile
        
        print(f"\nRecording for {duration} seconds...")
        
        # 录音
        sr = 16000
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        sf.write(temp_path, recording, sr)
        
        # 分析
        results = analyzer.analyze(temp_path)
        analyzer.print_results(results)
        
        # 清理
        os.remove(temp_path)
        
    except ImportError:
        print("Error: sounddevice not installed. Install with: pip install sounddevice")


def main():
    parser = argparse.ArgumentParser(description='PC Demo for Speech Analysis')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--audio', type=str,
                       help='Audio file path')
    parser.add_argument('--mic', action='store_true',
                       help='Use microphone input')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Recording duration (seconds)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SpeechAnalyzer(
        model_path=args.model,
        device=args.device
    )
    
    # 分析音频文件
    if args.audio:
        results = analyzer.analyze(args.audio)
        analyzer.print_results(results)
    
    # 麦克风输入
    elif args.mic:
        analyze_microphone(analyzer, duration=args.duration)
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print(f"  python {sys.argv[0]} --model checkpoints/best_model.pt --audio test.wav")
        print(f"  python {sys.argv[0]} --model checkpoints/best_model.pt --mic --duration 5")


if __name__ == '__main__':
    main()
