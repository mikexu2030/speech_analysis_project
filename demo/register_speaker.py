#!/usr/bin/env python3
"""
声纹注册工具
支持: 批量注册、声纹库管理
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.audio_utils import audio_to_model_input


class SpeakerRegistry:
    """
    声纹注册库
    管理已注册说话人的嵌入向量
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.speakers: Dict[str, np.ndarray] = {}  # name -> embedding
    
    def register(self, name: str, audio_paths: List[str]) -> bool:
        """
        注册说话人
        
        Args:
            name: 说话人名称
            audio_paths: 音频文件路径列表 (用于平均嵌入)
        
        Returns:
            是否成功
        """
        embeddings = []
        
        for audio_path in audio_paths:
            # 提取特征
            mel_spec = audio_to_model_input(
                audio_path,
                sr=16000,
                n_mels=80,
                target_length=300,
                normalize=True
            )
            
            if mel_spec is None:
                print(f"  Warning: Failed to load {audio_path}")
                continue
            
            # 转换为tensor
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(self.device)
            
            # 提取嵌入
            with torch.no_grad():
                embedding = self.model.extract_embedding(mel_tensor)
                embedding = embedding.cpu().numpy()[0]
            
            embeddings.append(embedding)
        
        if len(embeddings) == 0:
            print(f"Error: No valid embeddings for {name}")
            return False
        
        # 平均嵌入
        avg_embedding = np.mean(embeddings, axis=0)
        # L2归一化
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        self.speakers[name] = avg_embedding
        print(f"Registered: {name} (from {len(embeddings)} samples)")
        
        return True
    
    def verify(self, audio_path: str, threshold: float = 0.5) -> Optional[Dict]:
        """
        验证说话人
        
        Args:
            audio_path: 待验证音频
            threshold: 相似度阈值
        
        Returns:
            最匹配的说话人信息 或 None
        """
        # 提取嵌入
        mel_spec = audio_to_model_input(audio_path, sr=16000, n_mels=80, target_length=300)
        if mel_spec is None:
            return None
        
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)
        
        with torch.no_grad():
            embedding = self.model.extract_embedding(mel_tensor)
            embedding = embedding.cpu().numpy()[0]
        
        # 计算与所有注册说话人的相似度
        best_match = None
        best_score = -1.0
        
        for name, registered_emb in self.speakers.items():
            score = np.dot(embedding, registered_emb)
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold:
            return {
                'name': best_match,
                'score': float(best_score),
                'is_match': True
            }
        else:
            return {
                'name': best_match,
                'score': float(best_score),
                'is_match': False
            }
    
    def save(self, path: str):
        """保存声纹库"""
        data = {
            'speakers': self.speakers,
            'n_speakers': len(self.speakers)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Speaker registry saved to: {path}")
    
    def load(self, path: str):
        """加载声纹库"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.speakers = data['speakers']
        print(f"Loaded {data['n_speakers']} speakers from {path}")
    
    def list_speakers(self) -> List[str]:
        """列出所有注册说话人"""
        return list(self.speakers.keys())
    
    def remove_speaker(self, name: str) -> bool:
        """移除说话人"""
        if name in self.speakers:
            del self.speakers[name]
            print(f"Removed speaker: {name}")
            return True
        print(f"Speaker not found: {name}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Speaker Registration Tool')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--action', type=str, required=True,
                       choices=['register', 'verify', 'list', 'save', 'load'],
                       help='Action to perform')
    parser.add_argument('--name', type=str,
                       help='Speaker name')
    parser.add_argument('--audio', type=str, nargs='+',
                       help='Audio file path(s)')
    parser.add_argument('--registry', type=str, default='speaker_registry.pkl',
                       help='Registry file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Verification threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from: {args.model}")
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    model = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建注册库
    registry = SpeakerRegistry(model, device=device)
    
    # 加载已有注册库
    if os.path.exists(args.registry):
        registry.load(args.registry)
    
    # 执行操作
    if args.action == 'register':
        if not args.name or not args.audio:
            print("Error: --name and --audio required for registration")
            return
        
        # 支持通配符
        audio_paths = []
        for pattern in args.audio:
            if '*' in pattern:
                import glob
                audio_paths.extend(glob.glob(pattern))
            else:
                audio_paths.append(pattern)
        
        registry.register(args.name, audio_paths)
        registry.save(args.registry)
    
    elif args.action == 'verify':
        if not args.audio:
            print("Error: --audio required for verification")
            return
        
        result = registry.verify(args.audio[0], threshold=args.threshold)
        if result:
            print(f"Verification result:")
            print(f"  Best match: {result['name']}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Match: {'YES' if result['is_match'] else 'NO'}")
        else:
            print("Verification failed")
    
    elif args.action == 'list':
        speakers = registry.list_speakers()
        print(f"Registered speakers ({len(speakers)}):")
        for name in speakers:
            print(f"  - {name}")
    
    elif args.action == 'save':
        registry.save(args.registry)
    
    elif args.action == 'load':
        registry.load(args.registry)


if __name__ == '__main__':
    main()
