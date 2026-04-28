#!/usr/bin/env python3
"""
评测微调后的HuBERT模型
对比微调前后的性能
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import sys
sys.path.insert(0, '/data/mikexu/speech_analysis_project')

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from utils.audio_utils import load_audio

class SimpleHead(nn.Module):
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
    def __init__(self, pretrained_model_name, embedding_dim=192, num_speakers=25, num_age_groups=5, num_emotions=7):
        super().__init__()
        self.backbone = HubertModel.from_pretrained(pretrained_model_name)
        backbone_dim = self.backbone.config.hidden_size
        
        self.speaker_embedding = nn.Linear(backbone_dim, embedding_dim)
        self.speaker_classifier = nn.Linear(embedding_dim, num_speakers)
        self.age_head = SimpleHead(backbone_dim, num_age_groups)
        self.gender_head = SimpleHead(backbone_dim, 2)
        self.emotion_head = SimpleHead(backbone_dim, num_emotions)
    
    def forward(self, input_values):
        outputs = self.backbone(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        
        embedding = self.speaker_embedding(pooled)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return {
            'speaker_embedding': embedding,
            'speaker_logits': self.speaker_classifier(embedding),
            'age_logits': self.age_head(pooled),
            'gender_logits': self.gender_head(pooled),
            'emotion_logits': self.emotion_head(pooled)
        }

def evaluate_model(model_path, test_data, feature_extractor, device):
    """评测模型"""
    model = PretrainedSpeechModel('models/pretrained/hubert_base_ls960', num_speakers=25)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    results = {
        'speaker_true': [], 'speaker_pred': [],
        'gender_true': [], 'gender_pred': [],
        'emotion_true': [], 'emotion_pred': [],
        'age_true': [], 'age_pred': []
    }
    
    for sample in tqdm(test_data, desc="Evaluating"):
        waveform, sr = load_audio(sample['audio_path'], sr=16000)
        if waveform is None:
            continue
        
        # 截断或填充到3秒
        max_length = 16000 * 3
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)))
        
        inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        with torch.no_grad():
            outputs = model(input_values)
        
        results['speaker_true'].append(sample['speaker_id'])
        results['speaker_pred'].append(outputs['speaker_logits'].argmax(dim=1).cpu().item())
        
        results['gender_true'].append(sample['gender'])
        results['gender_pred'].append(outputs['gender_logits'].argmax(dim=1).cpu().item())
        
        results['emotion_true'].append(sample['emotion'])
        results['emotion_pred'].append(outputs['emotion_logits'].argmax(dim=1).cpu().item())
        
        results['age_true'].append(sample.get('age_group', sample.get('age', 0)))
        results['age_pred'].append(outputs['age_logits'].argmax(dim=1).cpu().item())
    
    # 计算指标
    metrics = {}
    for task in ['speaker', 'gender', 'emotion', 'age']:
        true = results[f'{task}_true']
        pred = results[f'{task}_pred']
        metrics[f'{task}_acc'] = accuracy_score(true, pred)
        metrics[f'{task}_uar'] = np.mean([
            accuracy_score(np.array(true)[np.array(true)==cls], 
                          np.array(pred)[np.array(true)==cls]) 
            for cls in np.unique(true)
        ])
    
    return metrics, results

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载测试数据
    with open('data/splits/test.json', 'r') as f:
        test_data = json.load(f)
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('models/pretrained/hubert_base_ls960')
    
    # 评测微调后的模型
    model_path = 'models/pretrained_finetuned/hubert_multitask/best_model.pt'
    if os.path.exists(model_path):
        print("\n" + "="*60)
        print("评测微调后的模型")
        print("="*60)
        metrics, results = evaluate_model(model_path, test_data, feature_extractor, device)
        
        print(f"\n说话人识别: Acc={metrics['speaker_acc']*100:.1f}%, UAR={metrics['speaker_uar']*100:.1f}%")
        print(f"性别识别: Acc={metrics['gender_acc']*100:.1f}%, UAR={metrics['gender_uar']*100:.1f}%")
        print(f"情绪识别: Acc={metrics['emotion_acc']*100:.1f}%, UAR={metrics['emotion_uar']*100:.1f}%")
        print(f"年龄分组: Acc={metrics['age_acc']*100:.1f}%, UAR={metrics['age_uar']*100:.1f}%")
        
        # 保存结果
        with open('results/evaluation/finetuned_hubert_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        print(f"模型文件不存在: {model_path}")
        print("等待训练完成...")
