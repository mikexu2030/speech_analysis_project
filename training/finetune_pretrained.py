"""
预训练模型微调脚本
使用HuBERT/WavLM作为backbone，训练多任务头
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, '/data/mikexu/speech_analysis_project')

from transformers import HubertModel, WavLMModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor
from utils.audio_utils import load_audio

from models.heads import SpeakerHead, AgeHead, GenderHead, EmotionHead
from utils.audio_utils import load_audio

class SimpleHead(nn.Module):
    """简单分类头，适配2D输入 (batch, hidden)"""
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class PretrainedSpeechModel(nn.Module):
    """基于预训练模型的多任务语音分析模型"""
    
    def __init__(
        self,
        pretrained_model_name: str,
        embedding_dim: int = 192,
        num_speakers: int = 1000,
        num_age_groups: int = 5,
        num_emotions: int = 7,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
        
        # 加载预训练backbone
        if 'hubert' in pretrained_model_name.lower():
            self.backbone = HubertModel.from_pretrained(pretrained_model_name)
        elif 'wavlm' in pretrained_model_name.lower():
            self.backbone = WavLMModel.from_pretrained(pretrained_model_name)
        elif 'wav2vec' in pretrained_model_name.lower():
            self.backbone = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        else:
            raise ValueError(f"Unknown model: {pretrained_model_name}")
        
        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 获取backbone输出维度
        backbone_dim = self.backbone.config.hidden_size
        
        # 简单任务头 (适配2D输入)
        self.speaker_embedding = nn.Linear(backbone_dim, embedding_dim)
        self.speaker_classifier = nn.Linear(embedding_dim, num_speakers)
        self.age_head = SimpleHead(backbone_dim, num_age_groups)
        self.gender_head = SimpleHead(backbone_dim, 2)
        self.emotion_head = SimpleHead(backbone_dim, num_emotions)
    
    def forward(self, input_values, task='all'):
        """
        Args:
            input_values: 预处理后的音频输入 (batch, seq_len)
            task: 'all', 'speaker', 'age', 'gender', 'emotion'
        """
        # 提取特征
        outputs = self.backbone(input_values)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # 池化
        pooled = hidden_states.mean(dim=1)  # (batch, hidden)
        
        results = {}
        
        if task in ['all', 'speaker']:
            embedding = self.speaker_embedding(pooled)
            embedding = F.normalize(embedding, p=2, dim=1)
            results['speaker_embedding'] = embedding
            results['speaker_logits'] = self.speaker_classifier(embedding)
        
        if task in ['all', 'age']:
            results['age_logits'] = self.age_head(pooled)
        
        if task in ['all', 'gender']:
            results['gender_logits'] = self.gender_head(pooled)
        
        if task in ['all', 'emotion']:
            results['emotion_logits'] = self.emotion_head(pooled)
        
        return results


class PretrainedDataset(torch.utils.data.Dataset):
    """预训练模型数据集"""
    
    def __init__(self, data_list, feature_extractor, max_length=16000*3):
        self.data = data_list
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 加载音频
        waveform, sr = load_audio(sample['audio_path'], sr=16000)
        if waveform is None:
            waveform = np.zeros(self.max_length)
        
        # 截断或填充
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        
        # 预处理
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'speaker_id': torch.tensor(sample['speaker_id'], dtype=torch.long),
            'age_group': torch.tensor(sample.get('age_group', sample.get('age', 0)), dtype=torch.long),
            'gender': torch.tensor(sample['gender'], dtype=torch.long),
            'emotion': torch.tensor(sample['emotion'], dtype=torch.long),
            'age_value': torch.tensor(sample.get('age_value', sample.get('age', 0)), dtype=torch.float32)
        }


def train_pretrained_model(
    pretrained_path='models/pretrained/hubert_base_ls960',
    output_dir='models/pretrained_finetuned/hubert_multitask',
    epochs=10,
    batch_size=8,
    lr=1e-4,
    freeze_backbone=True
):
    """训练基于预训练模型的多任务模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    with open('data/splits/train.json', 'r') as f:
        train_data = json.load(f)
    with open('data/splits/val.json', 'r') as f:
        val_data = json.load(f)
    
    # 特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
    
    # 数据集
    train_dataset = PretrainedDataset(train_data, feature_extractor)
    val_dataset = PretrainedDataset(val_data, feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    # 模型
    num_speakers = max([s['speaker_id'] for s in train_data]) + 1
    model = PretrainedSpeechModel(
        pretrained_model_name=pretrained_path,
        num_speakers=num_speakers,
        freeze_backbone=freeze_backbone
    )
    model.to(device)
    
    # 优化器 - 使用更大学习率
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    # 训练循环
    best_val_metric = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = {'speaker': 0, 'gender': 0, 'emotion': 0}
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            input_values = batch['input_values'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            
            # 计算损失
            loss = 0
            
            # 说话人损失
            if 'speaker_logits' in outputs:
                speaker_loss = criterion_ce(outputs['speaker_logits'], batch['speaker_id'].to(device))
                loss += speaker_loss
                train_correct['speaker'] += (outputs['speaker_logits'].argmax(dim=1) == batch['speaker_id'].to(device)).sum().item()
            
            # 性别损失
            if 'gender_logits' in outputs:
                gender_loss = criterion_ce(outputs['gender_logits'], batch['gender'].to(device))
                loss += gender_loss
                train_correct['gender'] += (outputs['gender_logits'].argmax(dim=1) == batch['gender'].to(device)).sum().item()
            
            # 情绪损失 - 增加权重
            if 'emotion_logits' in outputs:
                emotion_loss = criterion_ce(outputs['emotion_logits'], batch['emotion'].to(device))
                loss += emotion_loss * 2.0  # 情绪任务权重更高
                train_correct['emotion'] += (outputs['emotion_logits'].argmax(dim=1) == batch['emotion'].to(device)).sum().item()
            
            # 年龄损失
            if 'age_logits' in outputs:
                age_loss = criterion_ce(outputs['age_logits'], batch['age_group'].to(device))
                loss += age_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_total += input_values.size(0)
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = {'speaker': 0, 'gender': 0, 'emotion': 0}
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_values = batch['input_values'].to(device)
                outputs = model(input_values)
                
                loss = 0
                if 'speaker_logits' in outputs:
                    loss += criterion_ce(outputs['speaker_logits'], batch['speaker_id'].to(device))
                    val_correct['speaker'] += (outputs['speaker_logits'].argmax(dim=1) == batch['speaker_id'].to(device)).sum().item()
                
                if 'gender_logits' in outputs:
                    loss += criterion_ce(outputs['gender_logits'], batch['gender'].to(device))
                    val_correct['gender'] += (outputs['gender_logits'].argmax(dim=1) == batch['gender'].to(device)).sum().item()
                
                if 'emotion_logits' in outputs:
                    loss += criterion_ce(outputs['emotion_logits'], batch['emotion'].to(device)) * 2.0
                    val_correct['emotion'] += (outputs['emotion_logits'].argmax(dim=1) == batch['emotion'].to(device)).sum().item()
                
                if 'age_logits' in outputs:
                    loss += criterion_ce(outputs['age_logits'], batch['age_group'].to(device))
                
                val_loss += loss.item()
                val_total += input_values.size(0)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Train Acc: Speaker={train_correct['speaker']/train_total*100:.1f}%, Gender={train_correct['gender']/train_total*100:.1f}%, Emotion={train_correct['emotion']/train_total*100:.1f}%")
        print(f"  Val Acc: Speaker={val_correct['speaker']/val_total*100:.1f}%, Gender={val_correct['gender']/val_total*100:.1f}%, Emotion={val_correct['emotion']/val_total*100:.1f}%")
        
        # 保存最佳模型
        val_metric = val_correct['emotion'] / val_total  # 基于情绪准确率
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            print(f"  ✅ Saved best model (val emotion acc: {val_metric*100:.1f}%)")
    
    print(f"\nTraining complete! Best val emotion acc: {best_val_metric*100:.1f}%")
    return model


if __name__ == '__main__':
    # 使用HuBERT Base训练 - 减少epoch，增加学习率
    model = train_pretrained_model(
        pretrained_path='models/pretrained/hubert_base_ls960',
        output_dir='models/pretrained_finetuned/hubert_multitask',
        epochs=5,
        batch_size=16,
        lr=5e-4,
        freeze_backbone=True
    )
