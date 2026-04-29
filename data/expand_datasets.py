#!/usr/bin/env python3
"""
数据扩充脚本 - 下载并整合多个开源语音数据集
支持: Common Voice, TESS, SAVEE, EMODB, JL-Corpus, CREMA-D
"""

import os
import sys
import json
import argparse
import zipfile
import tarfile
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from collections import Counter

# 数据集配置
DATASET_CONFIGS = {
    'tess': {
        'name': 'TESS (Toronto Emotional Speech Set)',
        'url': 'https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/download',
        'size': '~500MB',
        'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'ps'],
        'speakers': 2,  # 2位女性说话人 (OAF, YAF)
        'language': 'en',
        'format': 'kaggle',
        'note': '需要从Kaggle手动下载'
    },
    'savee': {
        'name': 'SAVEE (Surrey Audio-Visual Expressed Emotion)',
        'url': 'http://kahlan.eps.surrey.ac.uk/savee/Download.html',
        'size': '~250MB',
        'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
        'speakers': 4,  # 4位男性说话人
        'language': 'en',
        'format': 'direct',
        'note': '需要手动下载并同意使用条款'
    },
    'emodb': {
        'name': 'EMO-DB (Berlin Emotional Speech Database)',
        'url': 'http://emodb.bilderbar.info/download/',
        'size': '~45MB',
        'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'boredom'],
        'speakers': 10,  # 5男5女
        'language': 'de',
        'format': 'direct',
        'note': '可直接下载'
    },
    'jl-corpus': {
        'name': 'JL-Corpus (Emotional Speech Dataset)',
        'url': 'https://www.kaggle.com/datasets/tli7544/jl-corpus-emotional-speech-dataset/download',
        'size': '~1GB',
        'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
        'speakers': 5,  # 3男2女
        'language': 'en',
        'format': 'kaggle',
        'note': '需要从Kaggle手动下载'
    },
    'cremad': {
        'name': 'CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)',
        'url': 'https://github.com/CheyneyComputerScience/CREMA-D',
        'size': '~1.5GB',
        'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust'],
        'speakers': 91,  # 48男43女
        'language': 'en',
        'format': 'github',
        'note': '需要从GitHub/Kaggle下载'
    },
    'common_voice': {
        'name': 'Mozilla Common Voice',
        'url': 'https://commonvoice.mozilla.org/en/datasets',
        'size': '~10GB+',
        'emotions': None,  # 无情绪标签，用于说话人/性别/年龄
        'speakers': 'thousands',
        'language': 'multi',
        'format': 'direct',
        'note': '多语言，包含年龄/性别元数据'
    }
}


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str = None) -> bool:
    """下载文件"""
    try:
        print(f"Downloading: {url}")
        desc = desc or os.path.basename(output_path)
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print(f"Download completed: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """解压压缩包"""
    try:
        os.makedirs(extract_to, exist_ok=True)
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path}")
            return False
        print(f"Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def setup_dataset_directory(dataset_name: str, output_dir: str) -> Path:
    """创建数据集目录结构"""
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def create_readme(dataset_name: str, output_dir: str, config: Dict) -> None:
    """创建数据集README说明文件"""
    dataset_dir = Path(output_dir) / dataset_name
    readme_path = dataset_dir / 'README.txt'
    
    with open(readme_path, 'w') as f:
        f.write(f"{config['name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Download URL: {config['url']}\n")
        f.write(f"Approximate Size: {config['size']}\n")
        f.write(f"Language: {config['language']}\n")
        f.write(f"Speakers: {config['speakers']}\n")
        if config['emotions']:
            f.write(f"Emotions: {', '.join(config['emotions'])}\n")
        f.write(f"\nNote: {config['note']}\n")
        f.write(f"Format: {config['format']}\n\n")
        f.write("Expected directory structure after extraction:\n")
        
        if dataset_name == 'tess':
            f.write("  tess/\n    OAF_angry/\n    OAF_disgust/\n    ...\n    YAF_angry/\n    ...\n")
        elif dataset_name == 'savee':
            f.write("  savee/\n    DC/\n    JE/\n    JK/\n    KL/\n    ...\n")
        elif dataset_name == 'emodb':
            f.write("  emodb/\n    wav/\n      03a01Fa.wav\n      ...\n")
        elif dataset_name == 'jl-corpus':
            f.write("  jl-corpus/\n    angry/\n    happy/\n    ...\n")
        elif dataset_name == 'cremad':
            f.write("  cremad/\n    AudioWAV/\n      1001_DFA_ANG_XX.wav\n      ...\n")
        elif dataset_name == 'common_voice':
            f.write("  common_voice/\n    clips/\n    train.tsv\n    dev.tsv\n    test.tsv\n")
    
    print(f"Created README: {readme_path}")


def download_emodb(output_dir: str) -> bool:
    """下载EMO-DB数据集 (可直接下载)"""
    print("\n" + "=" * 60)
    print("Downloading EMO-DB Dataset")
    print("=" * 60)
    
    emodb_dir = setup_dataset_directory('emodb', output_dir)
    
    # EMO-DB直接下载链接
    urls = [
        "http://emodb.bilderbar.info/download/download.zip",
        "http://www.emodb.bilderbar.info/download/emodb.zip"
    ]
    
    zip_path = emodb_dir / 'emodb.zip'
    
    for url in urls:
        if download_file(url, str(zip_path), desc='emodb.zip'):
            break
    else:
        print("Failed to download EMO-DB from all URLs")
        print("Please download manually from: http://emodb.bilderbar.info/download/")
        create_readme('emodb', output_dir, DATASET_CONFIGS['emodb'])
        return False
    
    # 解压
    extract_dir = emodb_dir / 'extracted'
    if extract_archive(str(zip_path), str(extract_dir)):
        print(f"EMO-DB extracted to: {extract_dir}")
        return True
    
    return False


def setup_all_datasets(output_dir: str, datasets: List[str] = None) -> Dict[str, bool]:
    """
    设置所有数据集目录和README
    
    Args:
        output_dir: 输出目录
        datasets: 要设置的数据集列表，None表示全部
    
    Returns:
        各数据集状态字典
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    
    results = {}
    
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {dataset_name}")
            results[dataset_name] = False
            continue
        
        print(f"\n{'='*60}")
        print(f"Setting up: {DATASET_CONFIGS[dataset_name]['name']}")
        print(f"{'='*60}")
        
        dataset_dir = setup_dataset_directory(dataset_name, output_dir)
        create_readme(dataset_name, output_dir, DATASET_CONFIGS[dataset_name])
        
        # 尝试自动下载（仅对支持直接下载的数据集）
        if dataset_name == 'emodb':
            results[dataset_name] = download_emodb(output_dir)
        else:
            print(f"Dataset '{dataset_name}' requires manual download.")
            print(f"Please download from: {DATASET_CONFIGS[dataset_name]['url']}")
            print(f"Extract to: {dataset_dir}")
            results[dataset_name] = None  # None表示需要手动下载
    
    return results


def print_summary(results: Dict[str, bool], output_dir: str) -> None:
    """打印数据集设置摘要"""
    print("\n" + "=" * 60)
    print("Dataset Setup Summary")
    print("=" * 60)
    
    for dataset_name, status in results.items():
        config = DATASET_CONFIGS[dataset_name]
        if status is True:
            status_str = "✅ Downloaded"
        elif status is False:
            status_str = "❌ Failed"
        else:
            status_str = "⏳ Manual download required"
        
        print(f"\n{config['name']}")
        print(f"  Status: {status_str}")
        print(f"  Path: {Path(output_dir) / dataset_name}")
        print(f"  Size: {config['size']}")
        print(f"  URL: {config['url']}")
    
    print(f"\n{'='*60}")
    print(f"All datasets configured in: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download and setup speech datasets')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                       choices=list(DATASET_CONFIGS.keys()) + ['all'],
                       help='Datasets to setup')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory for datasets')
    parser.add_argument('--auto_download', action='store_true',
                       help='Attempt automatic download for supported datasets')
    
    args = parser.parse_args()
    
    # 确定要处理的数据集
    if 'all' in args.datasets:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置数据集
    results = setup_all_datasets(args.output_dir, datasets)
    
    # 打印摘要
    print_summary(results, args.output_dir)
    
    # 生成下载指南
    guide_path = Path(args.output_dir) / 'DOWNLOAD_GUIDE.md'
    with open(guide_path, 'w') as f:
        f.write("# 数据集下载指南\n\n")
        f.write("## 自动下载\n\n")
        f.write("部分数据集支持自动下载:\n")
        f.write("```bash\n")
        f.write("python data/expand_datasets.py --auto_download --datasets emodb\n")
        f.write("```\n\n")
        f.write("## 手动下载\n\n")
        for dataset_name in datasets:
            config = DATASET_CONFIGS[dataset_name]
            f.write(f"### {config['name']}\n\n")
            f.write(f"- **下载链接**: {config['url']}\n")
            f.write(f"- **大小**: {config['size']}\n")
            f.write(f"- **格式**: {config['format']}\n")
            f.write(f"- **说明**: {config['note']}\n\n")
            f.write(f"下载后解压到: `{args.output_dir}/{dataset_name}/`\n\n")
    
    print(f"\nDownload guide saved to: {guide_path}")


if __name__ == '__main__':
    main()
