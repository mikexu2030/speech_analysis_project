"""
数据集划分脚本
LOSO (Leave-One-Speaker-Out) 划分，确保同一说话人不出现在多个集合中
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def loso_split(
    data_list: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    LOSO (Leave-One-Speaker-Out) 划分
    
    按说话人ID划分，确保同一说话人不出现在多个集合
    
    Args:
        data_list: 数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        (train_data, val_data, test_data)
    """
    random.seed(seed)
    
    # 按说话人分组
    speaker_groups = defaultdict(list)
    for item in data_list:
        speaker_id = item.get('speaker_id', 'unknown')
        speaker_groups[speaker_id].append(item)
    
    # 获取所有说话人
    speakers = list(speaker_groups.keys())
    random.shuffle(speakers)
    
    n_speakers = len(speakers)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)
    
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]
    
    # 收集数据
    train_data = []
    for s in train_speakers:
        train_data.extend(speaker_groups[s])
    
    val_data = []
    for s in val_speakers:
        val_data.extend(speaker_groups[s])
    
    test_data = []
    for s in test_speakers:
        test_data.extend(speaker_groups[s])
    
    print(f"LOSO split:")
    print(f"  Total speakers: {n_speakers}")
    print(f"  Train: {len(train_speakers)} speakers, {len(train_data)} samples")
    print(f"  Val:   {len(val_speakers)} speakers, {len(val_data)} samples")
    print(f"  Test:  {len(test_speakers)} speakers, {len(test_data)} samples")
    
    return train_data, val_data, test_data


def random_split(
    data_list: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    随机划分 (用于不需要LOSO的情况)
    """
    random.seed(seed)
    data_copy = data_list.copy()
    random.shuffle(data_copy)
    
    n_total = len(data_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = data_copy[:n_train]
    val_data = data_copy[n_train:n_train + n_val]
    test_data = data_copy[n_train + n_val:]
    
    print(f"Random split:")
    print(f"  Total samples: {n_total}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    return train_data, val_data, test_data


def stratified_split(
    data_list: List[Dict],
    stratify_key: str = 'emotion',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层划分 (按某个字段分层)
    """
    random.seed(seed)
    
    # 按指定字段分组
    groups = defaultdict(list)
    for item in data_list:
        key = item.get(stratify_key, 'unknown')
        groups[key].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for key, items in groups.items():
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    print(f"Stratified split (by {stratify_key}):")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    return train_data, val_data, test_data


def merge_datasets(json_paths: List[str]) -> List[Dict]:
    """合并多个数据集的JSON文件"""
    all_data = []
    
    # 用于重新映射speaker_id避免冲突
    speaker_offset = 0
    
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping")
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 重映射speaker_id
        max_speaker_id = 0
        for item in data:
            old_id = item.get('speaker_id', 0)
            item['speaker_id'] = old_id + speaker_offset
            max_speaker_id = max(max_speaker_id, item['speaker_id'])
            all_data.append(item)
        
        speaker_offset = max_speaker_id + 1
        print(f"  Loaded {len(data)} samples from {json_path}")
    
    print(f"Total: {len(all_data)} samples from {len(json_paths)} datasets")
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description='Create dataset splits')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                       help='Input JSON file(s)')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='loso',
                       choices=['loso', 'random', 'stratified'],
                       help='Splitting method')
    parser.add_argument('--stratify_key', type=str, default='emotion',
                       help='Key for stratified split')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: ratios must sum to 1.0, got {total_ratio}")
        return
    
    # 加载数据
    print("Loading data...")
    data_list = merge_datasets(args.input)
    
    if len(data_list) == 0:
        print("No data loaded!")
        return
    
    # 划分
    print(f"\nSplitting with method: {args.method}")
    if args.method == 'loso':
        train_data, val_data, test_data = loso_split(
            data_list, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    elif args.method == 'random':
        train_data, val_data, test_data = random_split(
            data_list, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    elif args.method == 'stratified':
        train_data, val_data, test_data = stratified_split(
            data_list, args.stratify_key,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    
    # 保存
    prefix = f"{args.prefix}_" if args.prefix else ""
    
    splits = {
        f'{prefix}train': train_data,
        f'{prefix}val': val_data,
        f'{prefix}test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(args.output_dir, f'{split_name}.json')
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2, default=str)
        print(f"Saved {split_name}: {output_path}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)
    
    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {len(split_data)}")
        
        # 情绪分布
        if any('emotion' in item for item in split_data):
            emotions = defaultdict(int)
            for item in split_data:
                if 'emotion' in item:
                    emotions[item['emotion']] += 1
            print(f"  Emotions: {dict(emotions)}")
        
        # 性别分布
        if any('gender' in item for item in split_data):
            genders = defaultdict(int)
            for item in split_data:
                if 'gender' in item:
                    genders[item['gender']] += 1
            print(f"  Genders: {dict(genders)}")
        
        # 说话人数量
        speakers = set()
        for item in split_data:
            if 'speaker_id' in item:
                speakers.add(item['speaker_id'])
        print(f"  Unique speakers: {len(speakers)}")
    
    print("\nSplit creation completed!")


if __name__ == '__main__':
    main()
