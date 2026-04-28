#!/usr/bin/env python3
"""
下载Common Voice数据集
支持: 英语、西班牙语、法语、德语、意大利语、日语
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

# 语言代码映射
LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese'
}


def download_common_voice(
    language: str = 'en',
    output_dir: str = 'data/raw/common_voice',
    max_samples: int = 10000,
    splits: List[str] = ['train', 'validation', 'test']
):
    """
    下载Common Voice数据集
    
    Args:
        language: 语言代码 (en, es, fr, de, it, ja)
        output_dir: 输出目录
        max_samples: 每个split最大样本数
        splits: 要下载的数据集划分
    """
    output_path = Path(output_dir) / language
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Common Voice 11.0 for {LANGUAGE_MAP.get(language, language)}...")
    print(f"Output: {output_path}")
    
    all_samples = []
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        try:
            # 使用streaming模式避免内存问题
            ds = load_dataset(
                'mozilla-foundation/common_voice_11_0',
                language,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            
            split_samples = []
            count = 0
            
            for sample in ds:
                # 提取有用信息
                sample_info = {
                    'audio_path': str(output_path / f"{split}_{count}.mp3"),
                    'dataset': 'common_voice',
                    'language': language,
                    'gender': sample.get('gender', ''),
                    'age': sample.get('age', ''),
                    'speaker_id': sample.get('client_id', ''),
                    'sentence': sample.get('sentence', ''),
                    'sample_rate': sample.get('audio', {}).get('sampling_rate', 48000)
                }
                
                # 保存音频文件
                if 'audio' in sample and 'array' in sample['audio']:
                    import soundfile as sf
                    audio_array = sample['audio']['array']
                    sr = sample['audio']['sampling_rate']
                    sf.write(sample_info['audio_path'], audio_array, sr)
                
                split_samples.append(sample_info)
                count += 1
                
                if count >= max_samples:
                    break
                
                if count % 100 == 0:
                    print(f"  Downloaded {count} samples...")
            
            # 保存split的metadata
            split_file = output_path / f"{split}.json"
            with open(split_file, 'w') as f:
                json.dump(split_samples, f, indent=2)
            
            all_samples.extend(split_samples)
            print(f"  Saved {count} samples to {split_file}")
            
        except Exception as e:
            print(f"  Error processing {split}: {e}")
            continue
    
    # 保存总metadata
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Download Common Voice dataset')
    parser.add_argument('--language', type=str, default='en',
                       choices=list(LANGUAGE_MAP.keys()),
                       help='Language code')
    parser.add_argument('--output_dir', type=str, default='data/raw/common_voice',
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Max samples per split')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Dataset splits to download')
    
    args = parser.parse_args()
    
    download_common_voice(
        language=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        splits=args.splits
    )


if __name__ == '__main__':
    main()
