"""
多任务损失函数
支持: AAMSoftmax (说话人), CE (分类), MSE (回归), 联合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class AAMSoftmaxLoss(nn.Module):
    """
    Additive Angular Margin Softmax Loss
    用于说话人识别，增强类内紧凑性和类间分离度
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
        easy_margin: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # 类别权重 (可学习)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embedding: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (batch, embedding_dim) - 归一化后的嵌入
            label: (batch,) - 类别标签
        
        Returns:
            loss
        """
        # 归一化权重
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(embedding, weight_norm)  # (batch, num_classes)
        
        # 计算正弦
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)
        
        # phi = cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # one-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # 应用margin
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        
        # 交叉熵损失
        loss = F.cross_entropy(output, label)
        
        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失
    支持: 说话人、年龄、性别、情绪
    """
    
    def __init__(
        self,
        embedding_dim: int = 192,
        num_speakers: int = 1000,
        num_age_groups: int = 5,
        num_emotions: int = 7,
        weights: Optional[Dict[str, float]] = None,
        speaker_margin: float = 0.2,
        speaker_scale: float = 30.0
    ):
        super().__init__()
        
        # 损失权重
        default_weights = {
            'emotion': 1.0,
            'speaker': 0.8,
            'age_reg': 0.3,
            'age_cls': 0.3,
            'gender': 0.5
        }
        self.weights = weights or default_weights
        
        # 说话人损失 (AAMSoftmax)
        self.speaker_loss = AAMSoftmaxLoss(
            embedding_dim=embedding_dim,
            num_classes=num_speakers,
            margin=speaker_margin,
            scale=speaker_scale
        )
        
        # 年龄损失
        self.age_cls_loss = nn.CrossEntropyLoss()
        self.age_reg_loss = nn.MSELoss()
        
        # 性别损失
        self.gender_loss = nn.CrossEntropyLoss()
        
        # 情绪损失
        self.emotion_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
        
        Returns:
            损失字典，包含 'total' 和各项损失
        """
        losses = {}
        total_loss = 0.0
        
        # 说话人损失
        if 'speaker_logits' in outputs and 'speaker_id' in targets:
            speaker_loss = self.speaker_loss(
                outputs['speaker_embedding'],
                targets['speaker_id']
            )
            losses['speaker'] = speaker_loss
            total_loss += self.weights.get('speaker', 0.8) * speaker_loss
        
        # 年龄分类损失
        if 'age_logits' in outputs and 'age_group' in targets:
            age_cls_loss = self.age_cls_loss(outputs['age_logits'], targets['age_group'])
            losses['age_cls'] = age_cls_loss
            total_loss += self.weights.get('age_cls', 0.3) * age_cls_loss
        
        # 年龄回归损失
        if 'age_value' in outputs and 'age' in targets:
            age_reg_loss = self.age_reg_loss(
                outputs['age_value'].squeeze(),
                targets['age'].float()
            )
            losses['age_reg'] = age_reg_loss
            total_loss += self.weights.get('age_reg', 0.3) * age_reg_loss
        
        # 性别损失
        if 'gender_logits' in outputs and 'gender' in targets:
            gender_loss = self.gender_loss(outputs['gender_logits'], targets['gender'])
            losses['gender'] = gender_loss
            total_loss += self.weights.get('gender', 0.5) * gender_loss
        
        # 情绪损失
        if 'emotion_logits' in outputs and 'emotion' in targets:
            emotion_loss = self.emotion_loss(outputs['emotion_logits'], targets['emotion'])
            losses['emotion'] = emotion_loss
            total_loss += self.weights.get('emotion', 1.0) * emotion_loss
        
        losses['total'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return loss


if __name__ == "__main__":
    # 测试
    print("Losses module loaded successfully")
    
    batch_size = 4
    embedding_dim = 192
    num_speakers = 100
    
    # 测试AAMSoftmax
    aam_loss = AAMSoftmaxLoss(embedding_dim, num_speakers)
    embedding = torch.randn(batch_size, embedding_dim)
    embedding = F.normalize(embedding, p=2, dim=1)
    speaker_labels = torch.randint(0, num_speakers, (batch_size,))
    
    loss = aam_loss(embedding, speaker_labels)
    print(f"AAMSoftmax loss: {loss.item():.4f}")
    
    # 测试多任务损失
    multi_loss = MultiTaskLoss(
        embedding_dim=embedding_dim,
        num_speakers=num_speakers,
        num_age_groups=5,
        num_emotions=7
    )
    
    outputs = {
        'speaker_embedding': embedding,
        'speaker_logits': torch.randn(batch_size, num_speakers),
        'age_logits': torch.randn(batch_size, 5),
        'age_value': torch.randn(batch_size, 1),
        'gender_logits': torch.randn(batch_size, 2),
        'emotion_logits': torch.randn(batch_size, 7)
    }
    
    targets = {
        'speaker_id': speaker_labels,
        'age_group': torch.randint(0, 5, (batch_size,)),
        'age': torch.randint(0, 100, (batch_size,)).float(),
        'gender': torch.randint(0, 2, (batch_size,)),
        'emotion': torch.randint(0, 7, (batch_size,))
    }
    
    losses = multi_loss(outputs, targets)
    print("\nMulti-task losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\nAll tests passed!")
