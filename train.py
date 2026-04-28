#!/usr/bin/env python3
"""
训练脚本 - 多任务语音分析模型
支持: 说话人识别 + 年龄估计 + 性别分类 + 情绪分类
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_model import MultiTaskSpeechModel
from models.losses import MultiTaskLoss


class SpeechDataset(Dataset):
    """
    语音数据集
    支持多种数据格式
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        sr: int = 16000,
        n_mels: int = 80,
        max_length: int = 3.0,  # 最大时长(秒)
        task_labels: List[str] = ['speaker', 'age', 'gender', 'emotion']
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.sr = sr
        self.n_mels = n_mels
        self.max_length = max_length
        self.task_labels = task_labels
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        # 标签映射
        self._build_label_mappings()
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        # 根据数据集类型加载
        if self.dataset_name == 'ravdess':
            samples = self._load_ravdess()
        elif self.dataset_name == 'cremad':
            samples = self._load_cremad()
        elif self.dataset_name == 'esd':
            samples = self._load_esd()
        elif self.dataset_name == 'iemocap':
            samples = self._load_iemocap()
        else:
            # 通用格式
            samples = self._load_generic()
        
        return samples
    
    def _load_ravdess(self) -> List[Dict]:
        """加载RAVDESS数据集"""
        samples = []
        ravdess_dir = self.data_dir / 'ravdess'
        
        # 查找所有wav文件
        for wav_file in ravdess_dir.rglob('*.wav'):
            # RAVDESS文件名格式: 03-01-01-01-01-01-01.wav
            # 格式: modality-vocal_channel-emotion-intensity-statement-repetition-actor
            parts = wav_file.stem.split('-')
            if len(parts) >= 7:
                emotion_id = int(parts[2])
                intensity = int(parts[3])
                actor_id = int(parts[6])
                
                # 情绪映射
                emotion_map = {
                    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                }
                
                # 性别 (actor_id: 1-12男, 13-24女)
                gender = 'male' if actor_id <= 12 else 'female'
                
                # 年龄组 (假设)
                age_group = 'adult'  # RAVDESS都是成年人
                
                samples.append({
                    'path': str(wav_file),
                    'speaker_id': actor_id,
                    'emotion': emotion_map.get(emotion_id, 'unknown'),
                    'emotion_id': emotion_id - 1,  # 0-based
                    'gender': gender,
                    'gender_id': 0 if gender == 'male' else 1,
                    'age_group': age_group,
                    'age_id': 2,  # adult
                    'dataset': 'ravdess'
                })
        
        return samples
    
    def _load_cremad(self) -> List[Dict]:
        """加载CREMA-D数据集"""
        samples = []
        cremad_dir = self.data_dir / 'cremad'
        
        for wav_file in cremad_dir.rglob('*.wav'):
            # CREMA-D格式: 1001_DFA_ANG_XX.wav
            parts = wav_file.stem.split('_')
            if len(parts) >= 3:
                actor_id = parts[0]
                emotion_code = parts[2]
                
                emotion_map = {
                    'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad',
                    'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'
                }
                
                samples.append({
                    'path': str(wav_file),
                    'speaker_id': int(actor_id),
                    'emotion': emotion_map.get(emotion_code, 'unknown'),
                    'emotion_id': list(emotion_map.keys()).index(emotion_code) if emotion_code in emotion_map else -1,
                    'gender': 'unknown',
                    'gender_id': -1,
                    'age_group': 'unknown',
                    'age_id': -1,
                    'dataset': 'cremad'
                })
        
        return samples
    
    def _load_esd(self) -> List[Dict]:
        """加载ESD数据集"""
        samples = []
        esd_dir = self.data_dir / 'esd'
        
        for wav_file in esd_dir.rglob('*.wav'):
            # ESD格式: 0011_000001.wav (在emotion子目录中)
            emotion_dir = wav_file.parent.name
            
            emotion_map = {
                'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'surprise': 4
            }
            
            if emotion_dir in emotion_map:
                samples.append({
                    'path': str(wav_file),
                    'speaker_id': int(wav_file.parent.parent.name),
                    'emotion': emotion_dir,
                    'emotion_id': emotion_map[emotion_dir],
                    'gender': 'unknown',
                    'gender_id': -1,
                    'age_group': 'unknown',
                    'age_id': -1,
                    'dataset': 'esd'
                })
        
        return samples
    
    def _load_iemocap(self) -> List[Dict]:
        """加载IEMOCAP数据集"""
        # IEMOCAP需要特殊处理
        return []
    
    def _load_generic(self) -> List[Dict]:
        """通用格式加载"""
        samples = []
        
        for wav_file in self.data_dir.rglob('*.wav'):
            samples.append({
                'path': str(wav_file),
                'speaker_id': 0,
                'emotion': 'unknown',
                'emotion_id': -1,
                'gender': 'unknown',
                'gender_id': -1,
                'age_group': 'unknown',
                'age_id': -1,
                'dataset': 'generic'
            })
        
        return samples
    
    def _build_label_mappings(self):
        """构建标签映射"""
        # 情绪标签
        emotions = sorted(set(s['emotion'] for s in self.samples if s['emotion'] != 'unknown'))
        self.emotion_to_id = {e: i for i, e in enumerate(emotions)}
        self.id_to_emotion = {i: e for e, i in self.emotion_to_id.items()}
        
        # 性别标签
        genders = sorted(set(s['gender'] for s in self.samples if s['gender'] != 'unknown'))
        self.gender_to_id = {g: i for i, g in enumerate(genders)}
        
        # 年龄标签
        ages = sorted(set(s['age_group'] for s in self.samples if s['age_group'] != 'unknown'))
        self.age_to_id = {a: i for i, a in enumerate(ages)}
        
        # 说话人标签
        speakers = sorted(set(s['speaker_id'] for s in self.samples))
        self.speaker_to_id = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)
        
        print(f"Dataset: {self.dataset_name}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Emotions: {len(self.emotion_to_id)} - {list(self.emotion_to_id.keys())}")
        print(f"  Genders: {len(self.gender_to_id)} - {list(self.gender_to_id.keys())}")
        print(f"  Ages: {len(self.age_to_id)} - {list(self.age_to_id.keys())}")
        print(f"  Speakers: {self.num_speakers}")
    
    def _load_audio(self, path: str) -> np.ndarray:
        """加载音频并提取mel频谱"""
        # 加载音频
        audio, sr = sf.read(path)
        
        # 重采样
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 截断或填充
        max_samples = int(self.max_length * self.sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # 提取mel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # 转换为对数尺度
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频特征
        mel_spec = self._load_audio(sample['path'])
        
        # 转换为tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)
        
        # 构建标签
        labels = {}
        
        if 'speaker' in self.task_labels:
            labels['speaker'] = self.speaker_to_id.get(sample['speaker_id'], 0)
        
        if 'emotion' in self.task_labels:
            labels['emotion'] = sample['emotion_id']
        
        if 'gender' in self.task_labels:
            labels['gender'] = sample['gender_id']
        
        if 'age' in self.task_labels:
            labels['age'] = sample['age_id']
        
        return mel_spec, labels, sample['path']


class Trainer:
    """
    训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        lr: float = 1e-3,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 损失函数
        self.criterion = MultiTaskLoss(task_weights=task_weights)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': {},
            'val_acc': {}
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        task_correct = {}
        task_total = {}
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (mel_specs, labels, paths) in enumerate(pbar):
            mel_specs = mel_specs.to(self.device)
            
            # 前向传播
            outputs = self.model(mel_specs)
            
            # 计算损失
            loss, task_losses = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算各任务准确率
            for task in ['emotion', 'gender', 'age']:
                if task in labels and task in outputs:
                    preds = torch.argmax(outputs[f'{task}_logits'], dim=1)
                    correct = (preds == labels[task].to(self.device)).sum().item()
                    
                    task_correct[task] = task_correct.get(task, 0) + correct
                    task_total[task] = task_total.get(task, 0) + len(labels[task])
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                **{f'{k}_loss': f'{v:.4f}' for k, v in task_losses.items()}
            })
        
        # 计算平均
        avg_loss = total_loss / len(self.train_loader)
        acc = {task: task_correct[task] / task_total[task] 
               for task in task_correct}
        
        return {'loss': avg_loss, 'acc': acc}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        task_correct = {}
        task_total = {}
        
        with torch.no_grad():
            for mel_specs, labels, paths in tqdm(self.val_loader, desc='Validation'):
                mel_specs = mel_specs.to(self.device)
                
                outputs = self.model(mel_specs)
                loss, _ = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                for task in ['emotion', 'gender', 'age']:
                    if task in labels and task in outputs:
                        preds = torch.argmax(outputs[f'{task}_logits'], dim=1)
                        correct = (preds == labels[task].to(self.device)).sum().item()
                        
                        task_correct[task] = task_correct.get(task, 0) + correct
                        task_total[task] = task_total.get(task, 0) + len(labels[task])
        
        avg_loss = total_loss / len(self.val_loader)
        acc = {task: task_correct[task] / task_total[task] 
               for task in task_correct}
        
        return {'loss': avg_loss, 'acc': acc}
    
    def train(self, epochs: int, save_dir: str):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_metrics = self.train_epoch()
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}")
            for task, acc in train_metrics['acc'].items():
                print(f"  {task}_acc: {acc:.4f}")
            
            # 验证
            if self.val_loader:
                val_metrics = self.validate()
                print(f"\nVal - Loss: {val_metrics['loss']:.4f}")
                for task, acc in val_metrics['acc'].items():
                    print(f"  {task}_acc: {acc:.4f}")
                
                # 学习率调整
                self.scheduler.step(val_metrics['loss'])
                
                # 保存最佳模型
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                    }, save_dir / 'best_model.pt')
                    print(f"Saved best model (val_loss: {best_val_loss:.4f})")
            
            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train multi-task speech model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Data directory')
    parser.add_argument('--dataset', type=str, default='ravdess',
                       choices=['ravdess', 'cremad', 'esd', 'iemocap', 'all'],
                       help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default='outputs/training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight model')
    
    args = parser.parse_args()
    
    # 检查数据
    data_path = Path(args.data_dir) / args.dataset
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Please download the dataset first.")
        sys.exit(1)
    
    # 创建设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 创建数据集
    print("\nLoading dataset...")
    dataset = SpeechDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        task_labels=['speaker', 'emotion', 'gender', 'age']
    )
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print("\nCreating model...")
    model = MultiTaskSpeechModel(
        n_mels=80,
        num_speakers=dataset.num_speakers,
        num_emotions=len(dataset.emotion_to_id),
        num_age_groups=len(dataset.age_to_id),
        lightweight=args.lightweight
    )
    
    sizes = model.get_model_size()
    print(f"Model size: {sizes['total']:,} params ({sizes['total']*4/1024/1024:.2f} MB)")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        task_weights={'emotion': 1.0, 'gender': 1.0, 'age': 1.0, 'speaker': 1.0}
    )
    
    # 训练
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, save_dir=args.output_dir)
    
    print(f"\nTraining complete! Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
