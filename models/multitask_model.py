"""
多任务语音分析模型 - 完整模型组装
整合骨干网络和任务头，支持多任务训练和推理
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import SpectralBackbone, LightweightBackbone
from .heads import SpeakerHead, AgeHead, GenderHead, EmotionHead


class MultiTaskSpeechModel(nn.Module):
    """
    多任务语音分析模型
    
    同时支持:
    - 说话人识别 (嵌入 + 分类)
    - 年龄估计 (分类 + 回归)
    - 性别分类
    - 情绪分类
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        backbone_channels: list = [32, 64, 128, 256],
        embedding_dim: int = 192,
        num_speakers: int = 1000,
        num_age_groups: int = 5,
        num_emotions: int = 7,
        use_attention: bool = True,
        lightweight: bool = False
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
        
        # 骨干网络
        if lightweight:
            self.backbone = LightweightBackbone(
                n_mels=n_mels,
                channels=[16, 32, 64, 128],
                n_residual_blocks=1
            )
        else:
            self.backbone = SpectralBackbone(
                n_mels=n_mels,
                channels=backbone_channels,
                n_residual_blocks=2,
                use_attention=use_attention
            )
        
        backbone_dim = self.backbone.get_output_dim()
        
        # 任务头
        self.speaker_head = SpeakerHead(
            input_dim=backbone_dim,
            embedding_dim=embedding_dim,
            num_speakers=num_speakers
        )
        
        self.age_head = AgeHead(
            input_dim=backbone_dim,
            num_age_groups=num_age_groups
        )
        
        self.gender_head = GenderHead(
            input_dim=backbone_dim
        )
        
        self.emotion_head = EmotionHead(
            input_dim=backbone_dim,
            num_emotions=num_emotions
        )
    
    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入频谱图 (batch, 1, n_mels, time)
            task: 指定任务 (None=所有任务, 'speaker'/'age'/'gender'/'emotion')
        
        Returns:
            包含各任务输出的字典
        """
        # 骨干网络特征提取
        features = self.backbone(x)
        
        outputs = {}
        
        # 说话人识别
        if task is None or task == 'speaker':
            speaker_emb, speaker_logits = self.speaker_head(features)
            outputs['speaker_embedding'] = speaker_emb
            outputs['speaker_logits'] = speaker_logits
        
        # 年龄估计
        if task is None or task == 'age':
            age_logits, age_value = self.age_head(features)
            outputs['age_logits'] = age_logits
            outputs['age_value'] = age_value
        
        # 性别分类
        if task is None or task == 'gender':
            gender_logits = self.gender_head(features)
            outputs['gender_logits'] = gender_logits
        
        # 情绪分类
        if task is None or task == 'emotion':
            emotion_logits = self.emotion_head(features)
            outputs['emotion_logits'] = emotion_logits
        
        return outputs
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取说话人嵌入 (用于声纹注册和验证)
        
        Args:
            x: 输入频谱图 (batch, 1, n_mels, time)
        
        Returns:
            说话人嵌入 (batch, embedding_dim)
        """
        features = self.backbone(x)
        return self.speaker_head.extract_embedding(features)
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        推理预测 (带softmax概率)
        
        Args:
            x: 输入频谱图 (batch, 1, n_mels, time)
            return_probs: 是否返回概率
        
        Returns:
            预测结果字典
        """
        outputs = self.forward(x)
        predictions = {}
        
        # 说话人
        if 'speaker_embedding' in outputs:
            predictions['speaker_embedding'] = outputs['speaker_embedding']
        
        # 年龄
        if 'age_logits' in outputs:
            if return_probs:
                predictions['age_probs'] = torch.softmax(outputs['age_logits'], dim=1)
            predictions['age_pred'] = outputs['age_value']
        
        # 性别
        if 'gender_logits' in outputs:
            if return_probs:
                predictions['gender_probs'] = torch.softmax(outputs['gender_logits'], dim=1)
            predictions['gender_pred'] = torch.argmax(outputs['gender_logits'], dim=1)
        
        # 情绪
        if 'emotion_logits' in outputs:
            if return_probs:
                predictions['emotion_probs'] = torch.softmax(outputs['emotion_logits'], dim=1)
            predictions['emotion_pred'] = torch.argmax(outputs['emotion_logits'], dim=1)
        
        return predictions
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型各部分的参数量"""
        sizes = {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'speaker_head': sum(p.numel() for p in self.speaker_head.parameters()),
            'age_head': sum(p.numel() for p in self.age_head.parameters()),
            'gender_head': sum(p.numel() for p in self.gender_head.parameters()),
            'emotion_head': sum(p.numel() for p in self.emotion_head.parameters()),
        }
        sizes['total'] = sum(sizes.values())
        return sizes


if __name__ == "__main__":
    # 测试
    print("MultiTaskSpeechModel loaded successfully")
    
    batch_size = 2
    n_mels = 80
    time = 300
    
    # 测试输入
    x = torch.randn(batch_size, 1, n_mels, time)
    
    # 标准模型
    model = MultiTaskSpeechModel(
        n_mels=n_mels,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=100,
        num_age_groups=5,
        num_emotions=7
    )
    
    print("\n=== Standard Model ===")
    sizes = model.get_model_size()
    for k, v in sizes.items():
        print(f"  {k}: {v:,} params ({v * 4 / 1024 / 1024:.2f} MB)")
    
    # 多任务前向
    outputs = model(x)
    print("\nMulti-task outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    # 单任务前向
    speaker_out = model(x, task='speaker')
    print("\nSpeaker-only output:")
    for k, v in speaker_out.items():
        print(f"  {k}: {v.shape}")
    
    # 推理预测
    predictions = model.predict(x)
    print("\nPredictions:")
    for k, v in predictions.items():
        if v.dim() > 0:
            print(f"  {k}: {v.shape}")
    
    # 轻量级模型
    model_light = MultiTaskSpeechModel(
        n_mels=n_mels,
        lightweight=True,
        embedding_dim=192,
        num_speakers=100,
        num_age_groups=5,
        num_emotions=7
    )
    
    print("\n=== Lightweight Model ===")
    sizes_light = model_light.get_model_size()
    for k, v in sizes_light.items():
        print(f"  {k}: {v:,} params ({v * 4 / 1024 / 1024:.2f} MB)")
    
    print("\nAll tests passed!")
