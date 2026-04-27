"""
多任务语音分析模型 - 任务头
说话人嵌入、年龄回归、性别分类、情绪分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpeakerHead(nn.Module):
    """
    说话人嵌入头
    输出: 归一化的说话人嵌入向量
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 192,
        num_speakers: int = 1000
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )
        
        # 分类层 (用于训练时的辅助监督)
        self.classifier = nn.Linear(embedding_dim, num_speakers)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, input_dim, 1, time) 或 (batch, input_dim)
        
        Returns:
            (embedding, logits)
            embedding: (batch, embedding_dim) - 归一化
            logits: (batch, num_speakers)
        """
        # 处理输入形状
        if x.dim() == 4:
            # (batch, input_dim, 1, time) -> (batch, input_dim, time)
            x = x.squeeze(2)
            # 时间平均池化
            x = x.mean(dim=-1)
        
        # 投影
        embedding = self.projection(x)
        
        # L2归一化
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        
        # 分类
        logits = self.classifier(embedding_norm)
        
        return embedding_norm, logits
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """仅提取嵌入 (推理时使用)"""
        if x.dim() == 4:
            x = x.squeeze(2)
            x = x.mean(dim=-1)
        
        embedding = self.projection(x)
        return F.normalize(embedding, p=2, dim=1)


class AgeHead(nn.Module):
    """
    年龄估计头
    输出: 年龄段分类 + 细粒度回归
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_age_groups: int = 5,
        min_age: int = 0,
        max_age: int = 100
    ):
        super().__init__()
        
        self.num_age_groups = num_age_groups
        self.min_age = min_age
        self.max_age = max_age
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 年龄段分类
        self.age_classifier = nn.Linear(256, num_age_groups)
        
        # 年龄回归
        self.age_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, input_dim, 1, time) 或 (batch, input_dim)
        
        Returns:
            (age_logits, age_value)
            age_logits: (batch, num_age_groups)
            age_value: (batch, 1)
        """
        if x.dim() == 4:
            x = x.squeeze(2)
            x = x.mean(dim=-1)
        
        shared_feat = self.shared(x)
        
        # 分类
        age_logits = self.age_classifier(shared_feat)
        
        # 回归
        age_value = self.age_regressor(shared_feat)
        age_value = torch.clamp(age_value, self.min_age, self.max_age)
        
        return age_logits, age_value


class GenderHead(nn.Module):
    """
    性别分类头
    输出: 男女二分类
    """
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim, 1, time) 或 (batch, input_dim)
        
        Returns:
            (batch, 2) - logits
        """
        if x.dim() == 4:
            x = x.squeeze(2)
            x = x.mean(dim=-1)
        
        return self.classifier(x)


class EmotionHead(nn.Module):
    """
    情绪分类头
    输出: 7类情绪分类
    
    情绪类别:
    0: neutral (中性)
    1: happy (快乐)
    2: sad (悲伤)
    3: angry (愤怒)
    4: fear (恐惧)
    5: disgust (厌恶)
    6: surprise (惊讶)
    """
    
    def __init__(self, input_dim: int = 256, num_emotions: int = 7):
        super().__init__()
        
        self.num_emotions = num_emotions
        
        # 时间注意力聚合
        self.temporal_attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim, 1, time) 或 (batch, input_dim, time)
        
        Returns:
            (batch, num_emotions) - logits
        """
        if x.dim() == 4:
            x = x.squeeze(2)  # (batch, input_dim, time)
        
        # 时间注意力
        # x: (batch, input_dim, time)
        x_t = x.transpose(1, 2)  # (batch, time, input_dim)
        attn_weights = self.temporal_attn(x_t)  # (batch, time, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权聚合
        x_pooled = torch.sum(x_t * attn_weights, dim=1)  # (batch, input_dim)
        
        # 分类
        logits = self.classifier(x_pooled)
        
        return logits


if __name__ == "__main__":
    # 测试
    print("Heads module loaded successfully")
    
    batch_size = 2
    input_dim = 256
    time = 100
    
    # 测试输入 (模拟backbone输出)
    x = torch.randn(batch_size, input_dim, 1, time)
    
    # 说话人头
    speaker_head = SpeakerHead(input_dim, embedding_dim=192, num_speakers=100)
    emb, logits = speaker_head(x)
    print(f"Speaker: embedding {emb.shape}, logits {logits.shape}")
    print(f"  Embedding norm: {torch.norm(emb, dim=1)}")
    
    # 年龄头
    age_head = AgeHead(input_dim, num_age_groups=5)
    age_logits, age_val = age_head(x)
    print(f"Age: logits {age_logits.shape}, value {age_val.shape}")
    
    # 性别头
    gender_head = GenderHead(input_dim)
    gender_logits = gender_head(x)
    print(f"Gender: logits {gender_logits.shape}")
    
    # 情绪头
    emotion_head = EmotionHead(input_dim, num_emotions=7)
    emotion_logits = emotion_head(x)
    print(f"Emotion: logits {emotion_logits.shape}")
