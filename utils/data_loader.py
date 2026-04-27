"""
PyTorch Dataset 和 DataLoader
支持: 多任务数据加载、数据增强、LOSO划分
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Callable

from .audio_utils import audio_to_model_input, batch_audio_to_model_input
from .data_augmentation import CombinedAugmentor


class SpeechDataset(Dataset):
    """
    多任务语音数据集
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        n_mels: int = 80,
        target_length: int = 300,
        augment: bool = False,
        augment_prob: float = 0.5,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_list: 数据列表，每项包含:
                - 'audio_path': 音频路径
                - 'speaker_id': 说话人ID
                - 'emotion': 情绪标签
                - 'age': 年龄
                - 'gender': 性别 (0=female, 1=male)
                - 'age_group': 年龄段 (可选)
            n_mels: Mel滤波器数量
            target_length: 目标帧数
            augment: 是否启用数据增强
            augment_prob: 增强概率
            cache_dir: 缓存目录
        """
        self.data_list = data_list
        self.n_mels = n_mels
        self.target_length = target_length
        self.augment = augment
        self.cache_dir = cache_dir
        
        # 数据增强器
        if augment:
            self.augmentor = CombinedAugmentor(audio_prob=augment_prob, spec_prob=augment_prob)
        else:
            self.augmentor = None
        
        # 缓存
        self.cache = {}
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        # 检查缓存
        cache_key = item['audio_path']
        if cache_key in self.cache:
            mel_spec = self.cache[cache_key].copy()
        else:
            # 加载音频并提取特征
            mel_spec = audio_to_model_input(
                item['audio_path'],
                sr=16000,
                n_mels=self.n_mels,
                target_length=self.target_length,
                normalize=True
            )
            
            if mel_spec is None:
                # 返回零特征 (处理损坏文件)
                mel_spec = np.zeros((self.n_mels, self.target_length))
            
            # 存入缓存
            if len(self.cache) < 1000:  # 限制缓存大小
                self.cache[cache_key] = mel_spec.copy()
        
        # 数据增强 (频谱)
        if self.augmentor is not None:
            mel_spec = self.augmentor.augment_spectrogram(mel_spec)
        
        # 转换为tensor
        mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, n_mels, time)
        
        # 构建输出
        output = {
            'mel_spec': mel_spec,
            'audio_path': item['audio_path']
        }
        
        # 标签
        if 'speaker_id' in item:
            output['speaker_id'] = torch.tensor(item['speaker_id'], dtype=torch.long)
        
        if 'emotion' in item:
            output['emotion'] = torch.tensor(item['emotion'], dtype=torch.long)
        
        if 'age' in item:
            output['age'] = torch.tensor(item['age'], dtype=torch.float32)
        
        if 'age_group' in item:
            output['age_group'] = torch.tensor(item['age_group'], dtype=torch.long)
        elif 'age' in item:
            # 自动计算年龄段
            age = item['age']
            if age < 20:
                age_group = 0  # child/teen
            elif age < 35:
                age_group = 1  # young adult
            elif age < 50:
                age_group = 2  # middle age
            elif age < 65:
                age_group = 3  # senior
            else:
                age_group = 4  # elderly
            output['age_group'] = torch.tensor(age_group, dtype=torch.long)
        
        if 'gender' in item:
            output['gender'] = torch.tensor(item['gender'], dtype=torch.long)
        
        return output


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次合并函数
    """
    # 合并mel_spec
    mel_specs = torch.stack([item['mel_spec'] for item in batch])
    
    output = {'mel_spec': mel_specs}
    
    # 合并标签
    keys = ['speaker_id', 'emotion', 'age', 'age_group', 'gender']
    for key in keys:
        values = [item[key] for item in batch if key in item]
        if values:
            output[key] = torch.stack(values)
    
    return output


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: Optional[List[Dict]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    n_mels: int = 80,
    target_length: int = 300,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建DataLoader
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据 (可选)
        batch_size: 批次大小
        num_workers: 工作进程数
        n_mels: Mel滤波器数量
        target_length: 目标帧数
        augment_train: 是否增强训练数据
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = SpeechDataset(
        train_data,
        n_mels=n_mels,
        target_length=target_length,
        augment=augment_train
    )
    
    val_dataset = SpeechDataset(
        val_data,
        n_mels=n_mels,
        target_length=target_length,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = None
    if test_data is not None:
        test_dataset = SpeechDataset(
            test_data,
            n_mels=n_mels,
            target_length=target_length,
            augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def load_split_data(split_dir: str, split_name: str) -> List[Dict]:
    """
    加载划分好的数据
    
    Args:
        split_dir: 划分文件目录
        split_name: 划分名称 (train/val/test)
    
    Returns:
        数据列表
    """
    json_path = os.path.join(split_dir, f"{split_name}.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


if __name__ == "__main__":
    # 测试
    print("Data loader module loaded successfully")
    
    # 创建测试数据
    test_data = [
        {
            'audio_path': f'test_audio_{i}.wav',
            'speaker_id': i % 10,
            'emotion': i % 7,
            'age': 20 + i * 5,
            'gender': i % 2
        }
        for i in range(20)
    ]
    
    # 创建数据集
    dataset = SpeechDataset(test_data, n_mels=80, target_length=300, augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"mel_spec shape: {sample['mel_spec'].shape}")
    print(f"speaker_id: {sample['speaker_id']}")
    print(f"emotion: {sample['emotion']}")
    print(f"age: {sample['age']}")
    print(f"gender: {sample['gender']}")
    print(f"age_group: {sample['age_group']}")
    
    # 测试DataLoader
    train_data = test_data[:16]
    val_data = test_data[16:]
    
    train_loader, val_loader, _ = create_dataloaders(
        train_data, val_data,
        batch_size=4,
        num_workers=0,
        augment_train=True
    )
    
    print(f"\nTrain loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"\nBatch mel_spec shape: {batch['mel_spec'].shape}")
    print(f"Batch speaker_id shape: {batch['speaker_id'].shape}")
    
    print("\nAll tests passed!")
