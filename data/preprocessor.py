"""
数据预处理器
统一各数据集格式，提取特征，保存为标准格式
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


# 统一情绪映射
UNIFIED_EMOTION_MAP = {
    'neutral': 0, 'calm': 0, 'cal': 0, 'NEU': 0, 'N': 0,
    'happy': 1, 'happiness': 1, 'joy': 1, 'HAP': 1, 'H': 1,
    'sad': 2, 'sadness': 2, 'SAD': 2, 'S': 2,
    'angry': 3, 'anger': 3, 'ANG': 3, 'A': 3,
    'fear': 4, 'fearful': 4, 'FEA': 4, 'F': 4,
    'disgust': 5, 'disgusted': 5, 'DIS': 5, 'D': 5,
    'surprise': 6, 'surprised': 6, 'SUR': 6
}

# 性别映射
GENDER_MAP = {
    'female': 0, 'F': 0, 'f': 0, 'woman': 0,
    'male': 1, 'M': 1, 'm': 1, 'man': 1,
    'other': 2  # 不使用，但保留
}


def parse_ravdess_filename(filename: str) -> Optional[Dict]:
    """
    解析RAVDESS文件名
    格式: 03-01-01-01-01-01-01.wav
    
    位置说明:
    1. Modality (01=AV, 02=video, 03=audio)
    2. Vocal channel (01=speech, 02=song)
    3. Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    4. Emotional intensity (01=normal, 02=strong)
    5. Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    6. Repetition (01=1st, 02=2nd)
    7. Actor (01-24, 奇数=男, 偶数=女)
    """
    # 移除扩展名
    name = os.path.splitext(filename)[0]
    parts = name.split('-')
    
    if len(parts) != 7:
        return None
    
    try:
        emotion_code = int(parts[2])
        # RAVDESS情绪映射
        ravdess_emotion = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
        }
        
        actor_id = int(parts[6])
        gender = 'male' if actor_id % 2 == 1 else 'female'
        
        return {
            'emotion': UNIFIED_EMOTION_MAP[ravdess_emotion[emotion_code]],
            'speaker_id': actor_id,
            'gender': GENDER_MAP[gender],
            'intensity': int(parts[3]),
            'statement': int(parts[4]),
            'repetition': int(parts[5])
        }
    except (KeyError, ValueError) as e:
        return None


def parse_cremad_filename(filename: str) -> Optional[Dict]:
    """
    解析CREMA-D文件名
    格式: 1001_DFA_ANG_XX.wav
    
    位置说明:
    1. Actor ID (1001-1091)
    2. Sentence (DFA, IEO, IOM, ITH, ITS, IWL, IWW, MTI, TAI, TIE, TSI, WSI)
    3. Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
    4. Intensity (LO, MD, HI, XX)
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    
    if len(parts) != 4:
        return None
    
    try:
        actor_id = int(parts[0])
        emotion_code = parts[2]
        
        if emotion_code not in UNIFIED_EMOTION_MAP:
            return None
        
        # CREMA-D actor性别 (1001-1043=female, 1044-1091=male)
        # 实际需要查询CREMA-D元数据，这里用近似
        gender = 'female' if actor_id <= 1043 else 'male'
        
        return {
            'emotion': UNIFIED_EMOTION_MAP[emotion_code],
            'speaker_id': actor_id,
            'gender': GENDER_MAP[gender],
            'sentence': parts[1],
            'intensity': parts[3]
        }
    except (KeyError, ValueError):
        return None


def parse_esd_path(audio_path: str) -> Optional[Dict]:
    """
    解析ESD文件路径
    格式: ESD/0011/Angry/0011_000001.wav
    
    Speaker IDs:
    - 0001-0010: 中文说话人
    - 0011-0020: 英文说话人
    """
    parts = audio_path.replace('\\', '/').split('/')
    
    try:
        # 找到说话人ID
        speaker_id = None
        emotion = None
        
        for i, part in enumerate(parts):
            if re.match(r'^\d{4}$', part):
                speaker_id = int(part)
            if part.lower() in UNIFIED_EMOTION_MAP:
                emotion = UNIFIED_EMOTION_MAP[part.lower()]
        
        if speaker_id is None or emotion is None:
            return None
        
        # ESD语言
        language = 'zh' if speaker_id <= 10 else 'en'
        
        return {
            'emotion': emotion,
            'speaker_id': speaker_id,
            'language': language
        }
    except (ValueError, IndexError):
        return None


def process_ravdess(raw_dir: str, output_path: str) -> List[Dict]:
    """
    处理RAVDESS数据集
    """
    print(f"\nProcessing RAVDESS from: {raw_dir}")
    
    data_list = []
    
    # 查找所有wav文件
    raw_path = Path(raw_dir)
    wav_files = list(raw_path.rglob('*.wav'))
    
    print(f"Found {len(wav_files)} wav files")
    
    for wav_file in tqdm(wav_files, desc="Processing RAVDESS"):
        filename = wav_file.name
        info = parse_ravdess_filename(filename)
        
        if info is None:
            continue
        
        data_list.append({
            'audio_path': str(wav_file),
            'dataset': 'ravdess',
            'emotion': info['emotion'],
            'speaker_id': info['speaker_id'],
            'gender': info['gender'],
            'language': 'en',
            'metadata': info
        })
    
    print(f"Processed {len(data_list)} valid samples")
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    
    return data_list


def process_cremad(raw_dir: str, output_path: str) -> List[Dict]:
    """
    处理CREMA-D数据集
    """
    print(f"\nProcessing CREMA-D from: {raw_dir}")
    
    data_list = []
    
    raw_path = Path(raw_dir)
    wav_files = list(raw_path.rglob('*.wav'))
    
    print(f"Found {len(wav_files)} wav files")
    
    for wav_file in tqdm(wav_files, desc="Processing CREMA-D"):
        filename = wav_file.name
        info = parse_cremad_filename(filename)
        
        if info is None:
            continue
        
        data_list.append({
            'audio_path': str(wav_file),
            'dataset': 'cremad',
            'emotion': info['emotion'],
            'speaker_id': info['speaker_id'],
            'gender': info['gender'],
            'language': 'en',
            'metadata': info
        })
    
    print(f"Processed {len(data_list)} valid samples")
    
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    
    return data_list


def process_esd(raw_dir: str, output_path: str) -> List[Dict]:
    """
    处理ESD数据集
    """
    print(f"\nProcessing ESD from: {raw_dir}")
    
    data_list = []
    
    raw_path = Path(raw_dir)
    wav_files = list(raw_path.rglob('*.wav'))
    
    print(f"Found {len(wav_files)} wav files")
    
    for wav_file in tqdm(wav_files, desc="Processing ESD"):
        rel_path = str(wav_file)
        info = parse_esd_path(rel_path)
        
        if info is None:
            continue
        
        data_list.append({
            'audio_path': str(wav_file),
            'dataset': 'esd',
            'emotion': info['emotion'],
            'speaker_id': info['speaker_id'],
            'language': info['language'],
            'metadata': info
        })
    
    print(f"Processed {len(data_list)} valid samples")
    
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    
    return data_list


def process_common_voice(raw_dir: str, output_path: str, language: str = 'en') -> List[Dict]:
    """
    处理Common Voice数据集
    用于年龄和性别识别
    """
    print(f"\nProcessing Common Voice ({language}) from: {raw_dir}")
    
    # 查找TSV文件
    tsv_path = os.path.join(raw_dir, 'validated.tsv')
    if not os.path.exists(tsv_path):
        tsv_path = os.path.join(raw_dir, 'train.tsv')
    
    if not os.path.exists(tsv_path):
        print(f"No TSV file found in {raw_dir}")
        return []
    
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} entries from {tsv_path}")
    
    # 过滤有年龄和性别的条目
    df = df.dropna(subset=['age', 'gender'])
    print(f"Entries with age and gender: {len(df)}")
    
    data_list = []
    clips_dir = os.path.join(raw_dir, 'clips')
    
    age_map = {
        'teens': 15, 'twenties': 25, 'thirties': 35,
        'fourties': 45, 'fifties': 55, 'sixties': 65,
        'seventies': 75, 'eighties': 85, 'nineties': 95
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Common Voice"):
        audio_path = os.path.join(clips_dir, row['path'])
        
        if not os.path.exists(audio_path):
            continue
        
        # 年龄
        age = age_map.get(row['age'], None)
        if age is None:
            continue
        
        # 性别
        gender = GENDER_MAP.get(row['gender'], None)
        if gender is None or gender == 2:  # 跳过other
            continue
        
        data_list.append({
            'audio_path': audio_path,
            'dataset': 'common_voice',
            'language': language,
            'age': age,
            'gender': gender,
            'speaker_id': hash(row.get('client_id', '')) % 100000,
        })
    
    print(f"Processed {len(data_list)} valid samples")
    
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    
    return data_list


def main():
    parser = argparse.ArgumentParser(description='Preprocess speech datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ravdess', 'cremad', 'esd', 'common_voice', 'all'],
                       help='Dataset to preprocess')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='Raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--language', type=str, default='en',
                       help='Language for Common Voice')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'all' or args.dataset == 'ravdess':
        ravdess_dir = os.path.join(args.raw_dir, 'ravdess')
        if os.path.exists(ravdess_dir):
            process_ravdess(ravdess_dir, os.path.join(args.output_dir, 'ravdess.json'))
        else:
            print(f"RAVDESS directory not found: {ravdess_dir}")
    
    if args.dataset == 'all' or args.dataset == 'cremad':
        cremad_dir = os.path.join(args.raw_dir, 'cremad')
        if os.path.exists(cremad_dir):
            process_cremad(cremad_dir, os.path.join(args.output_dir, 'cremad.json'))
        else:
            print(f"CREMA-D directory not found: {cremad_dir}")
    
    if args.dataset == 'all' or args.dataset == 'esd':
        esd_dir = os.path.join(args.raw_dir, 'esd')
        if os.path.exists(esd_dir):
            process_esd(esd_dir, os.path.join(args.output_dir, 'esd.json'))
        else:
            print(f"ESD directory not found: {esd_dir}")
    
    if args.dataset == 'all' or args.dataset == 'common_voice':
        cv_dir = os.path.join(args.raw_dir, f'common_voice_{args.language}')
        if os.path.exists(cv_dir):
            process_common_voice(
                cv_dir,
                os.path.join(args.output_dir, f'common_voice_{args.language}.json'),
                language=args.language
            )
        else:
            print(f"Common Voice directory not found: {cv_dir}")
    
    print("\nPreprocessing completed!")


if __name__ == '__main__':
    main()
