#!/usr/bin/env python3
"""
多数据集预处理器 - 整合RAVDESS + TESS + SAVEE + EMODB + JL-Corpus + CREMA-D
统一情绪标签映射，生成标准格式的训练数据
"""

import os
import sys
import json
import re
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import warnings

# 统一情绪标签映射 (7类标准情绪)
EMOTION_MAPPING = {
    # RAVDESS (0-7)
    'ravdess': {
        0: 0,   # neutral -> neutral
        1: 0,   # calm -> neutral (合并到neutral)
        2: 2,   # happy -> happy
        3: 3,   # sad -> sad
        4: 4,   # angry -> angry
        5: 5,   # fearful -> fear
        6: 6,   # disgust -> disgust
        7: 7,   # surprised -> surprise
    },
    # TESS
    'tess': {
        'neutral': 0,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fear': 5,
        'disgust': 6,
        'surprise': 7,
        'ps': 7,  # pleasant surprise -> surprise
    },
    # SAVEE
    'savee': {
        'n': 0,   # neutral
        'h': 2,   # happy
        'sa': 3,  # sad
        'a': 4,   # angry
        'f': 5,   # fear
        'd': 6,   # disgust
        'su': 7,  # surprise
    },
    # EMO-DB (文件名编码)
    'emodb': {
        'N': 0,   # neutral
        'F': 2,   # happy (Freude)
        'T': 3,   # sad (Trauer)
        'W': 4,   # angry (Wut)
        'A': 5,   # fear (Angst)
        'E': 6,   # disgust (Ekel)
        'L': 0,   # boredom -> neutral (或丢弃)
    },
    # JL-Corpus
    'jl-corpus': {
        'neutral': 0,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fear': 5,
        'disgust': 6,
        'surprise': 7,
    },
    # CREMA-D
    'cremad': {
        'NEU': 0,   # neutral
        'HAP': 2,   # happy
        'SAD': 3,   # sad
        'ANG': 4,   # angry
        'FEA': 5,   # fear
        'DIS': 6,   # disgust
    }
}

# 情绪名称
EMOTION_NAMES = {
    0: 'neutral',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fear',
    6: 'disgust',
    7: 'surprise',
}

# 性别映射
GENDER_MAP = {
    'M': 1, 'm': 1, 'male': 1, 'man': 1, 'boy': 1,
    'F': 0, 'f': 0, 'female': 0, 'woman': 0, 'girl': 0,
}


def parse_ravdess_filename(filename: str) -> Optional[Dict]:
    """解析RAVDESS文件名"""
    # 格式: 03-01-02-02-02-01-07.wav
    # 模态-声道-情绪-强度-语句-重复-说话人
    match = re.match(r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', filename)
    if not match:
        return None
    
    parts = match.groups()
    return {
        'modality': int(parts[0]),
        'vocal_channel': int(parts[1]),
        'emotion': int(parts[2]) - 1,  # 1-based to 0-based
        'intensity': int(parts[3]),
        'statement': int(parts[4]),
        'repetition': int(parts[5]),
        'speaker_id': int(parts[6]),
    }


def parse_tess_filename(filename: str) -> Optional[Dict]:
    """解析TESS文件名"""
    # 格式: OAF_angry/IEO_angry.wav 或 YAF_disgust/OAF_disgust.wav
    # 或: OAF_back_angry.wav
    match = re.match(r'([OY]AF)_(.+?)_(\w+)\.wav', filename)
    if match:
        speaker_code, sentence, emotion = match.groups()
        speaker_id = 100 + (1 if speaker_code == 'OAF' else 2)
        return {
            'speaker_id': speaker_id,
            'emotion': emotion.lower(),
            'sentence': sentence,
        }
    
    # 另一种格式
    match = re.match(r'([OY]AF)_(\w+)\.wav', filename)
    if match:
        speaker_code, emotion = match.groups()
        speaker_id = 100 + (1 if speaker_code == 'OAF' else 2)
        return {
            'speaker_id': speaker_id,
            'emotion': emotion.lower(),
        }
    
    return None


def parse_savee_filename(filename: str) -> Optional[Dict]:
    """解析SAVEE文件名"""
    # 格式: DC_a01.wav, JE_n01.wav, JK_sa01.wav, KL_su01.wav
    match = re.match(r'([A-Z]{2})_([a-z]+)(\d{2})\.wav', filename)
    if not match:
        return None
    
    speaker_code, emotion_code, number = match.groups()
    
    # 说话人映射
    speaker_map = {'DC': 200, 'JE': 201, 'JK': 202, 'KL': 203}
    
    return {
        'speaker_id': speaker_map.get(speaker_code, 200),
        'emotion': emotion_code,
        'utterance': int(number),
    }


def parse_emodb_filename(filename: str) -> Optional[Dict]:
    """解析EMO-DB文件名"""
    # 格式: 03a01Fa.wav, 11b01Lb.wav
    # 说话人(2位) + 文本(1位) + 编号(2位) + 情绪(1位) + 版本(1位) + .wav
    match = re.match(r'(\d{2})([ab])(\d{2})([A-Z])([ab])\.wav', filename)
    if not match:
        return None
    
    speaker, text, number, emotion, version = match.groups()
    speaker_id = 300 + int(speaker)
    
    # 性别判断: 说话人编号 03-10
    # 03, 08, 09, 10 = male; 11, 12, 13, 14 = female
    speaker_num = int(speaker)
    gender = 1 if speaker_num in [3, 8, 9, 10, 15] else 0
    
    return {
        'speaker_id': speaker_id,
        'emotion': emotion,
        'gender': gender,
        'text': text,
        'number': number,
    }


def parse_jl_corpus_filename(filename: str) -> Optional[Dict]:
    """解析JL-Corpus文件名"""
    # 格式: angry_1-28_0001.wav, neutral_1-28_0001.wav
    match = re.match(r'(\w+)_([\d-]+)_(\d{4})\.wav', filename)
    if not match:
        return None
    
    emotion, speaker_info, number = match.groups()
    
    # 从speaker_info提取说话人ID
    speaker_parts = speaker_info.split('-')
    speaker_id = 400 + int(speaker_parts[0])
    
    return {
        'speaker_id': speaker_id,
        'emotion': emotion.lower(),
        'utterance': int(number),
    }


def parse_cremad_filename(filename: str) -> Optional[Dict]:
    """解析CREMA-D文件名"""
    # 格式: 1001_DFA_ANG_XX.wav
    # 说话人ID_句子_情绪_强度
    match = re.match(r'(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav', filename)
    if not match:
        return None
    
    speaker_id, sentence, emotion, intensity = match.groups()
    
    # 性别判断: 1001-1048 = male, 1049-1091 = female (大致)
    speaker_num = int(speaker_id)
    gender = 1 if speaker_num <= 1048 else 0
    
    return {
        'speaker_id': 500 + speaker_num,
        'emotion': emotion,
        'gender': gender,
        'sentence': sentence,
        'intensity': intensity,
    }


def process_ravdess(data_dir: str) -> List[Dict]:
    """处理RAVDESS数据集"""
    samples = []
    
    # 查找音频文件
    audio_dirs = [
        Path(data_dir) / 'audio_speech',
        Path(data_dir) / 'audio',
    ]
    
    for audio_dir in audio_dirs:
        if not audio_dir.exists():
            continue
        
        for actor_dir in sorted(audio_dir.glob('Actor_*')):
            speaker_match = re.search(r'Actor_(\d+)', actor_dir.name)
            if not speaker_match:
                continue
            
            speaker_id = int(speaker_match.group(1))
            gender = 1 if speaker_id % 2 == 1 else 0  # 奇数男，偶数女
            
            for wav_file in sorted(actor_dir.glob('*.wav')):
                parsed = parse_ravdess_filename(wav_file.name)
                if not parsed:
                    continue
                
                emotion_code = parsed['emotion']
                mapped_emotion = EMOTION_MAPPING['ravdess'].get(emotion_code)
                
                if mapped_emotion is None:
                    continue
                
                samples.append({
                    'audio_path': str(wav_file),
                    'dataset': 'ravdess',
                    'emotion': mapped_emotion,
                    'speaker_id': speaker_id,
                    'gender': gender,
                    'language': 'en',
                    'metadata': {
                        'original_emotion': emotion_code,
                        'intensity': parsed['intensity'],
                        'statement': parsed['statement'],
                        'repetition': parsed['repetition'],
                    }
                })
    
    return samples


def process_tess(data_dir: str) -> List[Dict]:
    """处理TESS数据集"""
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return samples
    
    # TESS may be under 'audio/' subdirectory
    search_paths = [data_path, data_path / 'audio']
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        for emotion_dir in sorted(search_path.glob('*/')):
            if not emotion_dir.is_dir():
                continue
            
            # 从目录名提取情绪
            dir_name = emotion_dir.name
            
            # Parse folder name like OAF_angry, YAF_happy, OAF_Pleasant_surprise, etc.
            emotion_match = re.match(r'[OY]AF[_-]([\w_]+)', dir_name, re.IGNORECASE)
            if emotion_match:
                emotion = emotion_match.group(1).lower()
            else:
                emotion = dir_name.lower()
            
            # Normalize emotion names
            emotion = emotion.replace('pleasant_surprise', 'ps').replace('pleasant_surprised', 'ps')
            
            mapped_emotion = EMOTION_MAPPING['tess'].get(emotion)
            if mapped_emotion is None:
                # Try without underscore variations
                emotion_alt = emotion.replace('_', '')
                mapped_emotion = EMOTION_MAPPING['tess'].get(emotion_alt)
                if mapped_emotion is None:
                    continue
            
            # 判断说话人和性别
            speaker_id = 101 if 'OAF' in dir_name.upper() else 102
            gender = 0  # TESS全是女性
            
            for wav_file in sorted(emotion_dir.glob('*.wav')):
                samples.append({
                    'audio_path': str(wav_file),
                    'dataset': 'tess',
                    'emotion': mapped_emotion,
                    'speaker_id': speaker_id,
                    'gender': gender,
                    'language': 'en',
                    'metadata': {
                        'original_emotion': emotion,
                    }
                })
    
    return samples


def process_savee(data_dir: str) -> List[Dict]:
    """处理SAVEE数据集"""
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return samples
    
    for wav_file in sorted(data_path.glob('*.wav')):
        parsed = parse_savee_filename(wav_file.name)
        if not parsed:
            continue
        
        mapped_emotion = EMOTION_MAPPING['savee'].get(parsed['emotion'])
        if mapped_emotion is None:
            continue
        
        samples.append({
            'audio_path': str(wav_file),
            'dataset': 'savee',
            'emotion': mapped_emotion,
            'speaker_id': parsed['speaker_id'],
            'gender': 1,  # SAVEE全是男性
            'language': 'en',
            'metadata': {
                'original_emotion': parsed['emotion'],
                'utterance': parsed['utterance'],
            }
        })
    
    return samples


def process_emodb(data_dir: str) -> List[Dict]:
    """处理EMO-DB数据集"""
    samples = []
    
    # 可能的音频目录
    wav_dirs = [
        Path(data_dir) / 'wav',
        Path(data_dir) / 'extracted' / 'wav',
        Path(data_dir),
    ]
    
    for wav_dir in wav_dirs:
        if not wav_dir.exists():
            continue
        
        for wav_file in sorted(wav_dir.glob('*.wav')):
            parsed = parse_emodb_filename(wav_file.name)
            if not parsed:
                continue
            
            mapped_emotion = EMOTION_MAPPING['emodb'].get(parsed['emotion'])
            if mapped_emotion is None:
                continue
            
            samples.append({
                'audio_path': str(wav_file),
                'dataset': 'emodb',
                'emotion': mapped_emotion,
                'speaker_id': parsed['speaker_id'],
                'gender': parsed['gender'],
                'language': 'de',
                'metadata': {
                    'original_emotion': parsed['emotion'],
                    'text': parsed['text'],
                }
            })
    
    return samples


def process_jl_corpus(data_dir: str) -> List[Dict]:
    """处理JL-Corpus数据集"""
    samples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return samples
    
    # 查找所有wav文件
    for wav_file in sorted(data_path.rglob('*.wav')):
        parsed = parse_jl_corpus_filename(wav_file.name)
        if not parsed:
            continue
        
        mapped_emotion = EMOTION_MAPPING['jl-corpus'].get(parsed['emotion'])
        if mapped_emotion is None:
            continue
        
        # 性别判断: 说话人1-3为男，4-5为女
        gender = 1 if parsed['speaker_id'] <= 403 else 0
        
        samples.append({
            'audio_path': str(wav_file),
            'dataset': 'jl-corpus',
            'emotion': mapped_emotion,
            'speaker_id': parsed['speaker_id'],
            'gender': gender,
            'language': 'en',
            'metadata': {
                'original_emotion': parsed['emotion'],
                'utterance': parsed['utterance'],
            }
        })
    
    return samples


def process_cremad(data_dir: str) -> List[Dict]:
    """处理CREMA-D数据集"""
    samples = []
    
    # 可能的音频目录
    wav_dirs = [
        Path(data_dir) / 'AudioWAV',
        Path(data_dir) / 'wav',
        Path(data_dir),
    ]
    
    for wav_dir in wav_dirs:
        if not wav_dir.exists():
            continue
        
        for wav_file in sorted(wav_dir.glob('*.wav')):
            parsed = parse_cremad_filename(wav_file.name)
            if not parsed:
                continue
            
            mapped_emotion = EMOTION_MAPPING['cremad'].get(parsed['emotion'])
            if mapped_emotion is None:
                continue
            
            samples.append({
                'audio_path': str(wav_file),
                'dataset': 'cremad',
                'emotion': mapped_emotion,
                'speaker_id': parsed['speaker_id'],
                'gender': parsed['gender'],
                'language': 'en',
                'metadata': {
                    'original_emotion': parsed['emotion'],
                    'sentence': parsed['sentence'],
                    'intensity': parsed['intensity'],
                }
            })
    
    return samples


def process_all_datasets(
    raw_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    datasets: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    处理所有数据集
    
    Args:
        raw_dir: 原始数据目录
        output_dir: 输出目录
        datasets: 要处理的数据集列表
    
    Returns:
        各数据集样本字典
    """
    if datasets is None:
        datasets = ['ravdess', 'tess', 'savee', 'emodb', 'jl-corpus', 'cremad']
    
    all_samples = {}
    
    processors = {
        'ravdess': process_ravdess,
        'tess': process_tess,
        'savee': process_savee,
        'emodb': process_emodb,
        'jl-corpus': process_jl_corpus,
        'cremad': process_cremad,
    }
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_dir = Path(raw_dir) / dataset_name
        
        if not dataset_dir.exists():
            print(f"Directory not found: {dataset_dir}")
            all_samples[dataset_name] = []
            continue
        
        processor = processors.get(dataset_name)
        if not processor:
            print(f"No processor for: {dataset_name}")
            continue
        
        samples = processor(str(dataset_dir))
        all_samples[dataset_name] = samples
        
        print(f"Found {len(samples)} valid samples")
        
        if samples:
            # 统计
            emotions = Counter([s['emotion'] for s in samples])
            genders = Counter([s['gender'] for s in samples])
            speakers = len(set([s['speaker_id'] for s in samples]))
            
            print(f"  Speakers: {speakers}")
            print(f"  Emotions: {dict(sorted(emotions.items()))}")
            print(f"  Gender: Female={genders.get(0,0)}, Male={genders.get(1,0)}")
    
    return all_samples


def merge_datasets(
    all_samples: Dict[str, List[Dict]],
    output_path: str,
    min_samples_per_emotion: int = 10
) -> List[Dict]:
    """
    合并所有数据集并保存
    
    Args:
        all_samples: 各数据集样本
        output_path: 输出文件路径
        min_samples_per_emotion: 每情绪最少样本数
    
    Returns:
        合并后的样本列表
    """
    merged = []
    
    for dataset_name, samples in all_samples.items():
        print(f"Adding {len(samples)} samples from {dataset_name}")
        merged.extend(samples)
    
    if not merged:
        print("No samples to merge!")
        return []
    
    # 统计合并后数据
    print(f"\n{'='*60}")
    print(f"Merged Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(merged)}")
    
    emotions = Counter([s['emotion'] for s in merged])
    genders = Counter([s['gender'] for s in merged])
    speakers = len(set([s['speaker_id'] for s in merged]))
    datasets = Counter([s['dataset'] for s in merged])
    
    print(f"\nDataset sources:")
    for ds, count in sorted(datasets.items()):
        print(f"  {ds}: {count}")
    
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(emotions.items()):
        name = EMOTION_NAMES.get(emotion, f'unknown_{emotion}')
        print(f"  {emotion} ({name}): {count}")
    
    print(f"\nGender distribution:")
    print(f"  Female: {genders.get(0, 0)}")
    print(f"  Male: {genders.get(1, 0)}")
    
    print(f"\nTotal speakers: {speakers}")
    
    # 检查每情绪最少样本
    for emotion in range(8):
        if emotion == 1:  # skip calm (merged to neutral)
            continue
        count = emotions.get(emotion, 0)
        if count < min_samples_per_emotion:
            print(f"WARNING: Emotion {emotion} has only {count} samples (min: {min_samples_per_emotion})")
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\nSaved merged dataset to: {output_path}")
    
    return merged


def create_train_val_test_split(
    samples: List[Dict],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    按说话人划分训练/验证/测试集
    确保同一说话人只出现在一个集合中
    """
    import random
    random.seed(seed)
    
    # 按说话人分组
    speaker_samples = defaultdict(list)
    for sample in samples:
        speaker_samples[sample['speaker_id']].append(sample)
    
    speakers = list(speaker_samples.keys())
    random.shuffle(speakers)
    
    # 划分说话人
    n_train = int(len(speakers) * train_ratio)
    n_val = int(len(speakers) * val_ratio)
    
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]
    
    # 收集样本
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for speaker in train_speakers:
        splits['train'].extend(speaker_samples[speaker])
    for speaker in val_speakers:
        splits['val'].extend(speaker_samples[speaker])
    for speaker in test_speakers:
        splits['test'].extend(speaker_samples[speaker])
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_samples in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.json')
        with open(output_path, 'w') as f:
            json.dump(split_samples, f, indent=2)
        
        print(f"\n{split_name.upper()} split:")
        print(f"  Samples: {len(split_samples)}")
        print(f"  Speakers: {len(set(s['speaker_id'] for s in split_samples))}")
        emotions = Counter([s['emotion'] for s in split_samples])
        print(f"  Emotions: {dict(sorted(emotions.items()))}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Process and merge multiple speech datasets')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='Raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--datasets', nargs='+',
                       default=['ravdess', 'tess', 'savee', 'emodb', 'jl-corpus', 'cremad'],
                       help='Datasets to process')
    parser.add_argument('--split', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    
    args = parser.parse_args()
    
    # 处理所有数据集
    all_samples = process_all_datasets(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        datasets=args.datasets
    )
    
    # 合并数据集
    merged_path = os.path.join(args.output_dir, 'merged_dataset.json')
    merged = merge_datasets(all_samples, merged_path)
    
    # 划分训练/验证/测试
    if args.split and merged:
        splits_dir = os.path.join(args.output_dir, 'splits')
        create_train_val_test_split(merged, splits_dir, seed=args.seed)
    
    print("\n" + "=" * 60)
    print("Dataset processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
