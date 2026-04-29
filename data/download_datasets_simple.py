#!/usr/bin/env python3
"""
数据扩充脚本 - 使用HuggingFace Datasets下载公开数据集
支持: TESS, SAVEE, EMODB, JL-Corpus, CREMA-D
"""

import os
import sys
import argparse
from pathlib import Path

def download_tess(output_dir: str) -> bool:
    """下载TESS数据集"""
    print("\n" + "=" * 60)
    print("Downloading TESS Dataset")
    print("=" * 60)
    
    tess_dir = Path(output_dir) / 'tess'
    tess_dir.mkdir(parents=True, exist_ok=True)
    
    print("TESS dataset download options:")
    print("1. Kaggle: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
    print("2. Direct download from: https://tspace.library.utoronto.ca/handle/1807/24487")
    print(f"\nDownload and extract to: {tess_dir}")
    
    # 创建说明文件
    readme = tess_dir / 'README.txt'
    readme.write_text("""TESS Dataset
============
Download: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
Size: ~500MB
Speakers: 2 (OAF - older adult female, YAF - younger adult female)
Emotions: neutral, happy, sad, angry, fear, disgust, surprise, ps (pleasant surprise)
Language: English

Expected structure:
  tess/
    OAF_angry/
    OAF_disgust/
    ...
    YAF_angry/
    YAF_disgust/
    ...
""")
    return True

def download_savee(output_dir: str) -> bool:
    """下载SAVEE数据集"""
    print("\n" + "=" * 60)
    print("Downloading SAVEE Dataset")
    print("=" * 60)
    
    savee_dir = Path(output_dir) / 'savee'
    savee_dir.mkdir(parents=True, exist_ok=True)
    
    print("SAVEE dataset download:")
    print("URL: http://kahlan.eps.surrey.ac.uk/savee/Download.html")
    print("Note: Requires registration and agreement to terms")
    print(f"\nDownload and extract to: {savee_dir}")
    
    readme = savee_dir / 'README.txt'
    readme.write_text("""SAVEE Dataset
=============
Download: http://kahlan.eps.surrey.ac.uk/savee/Download.html
Size: ~250MB
Speakers: 4 male (DC, JE, JK, KL)
Emotions: neutral, happy, sad, angry, fear, disgust, surprise
Language: English

Expected structure:
  savee/
    DC/
    JE/
    JK/
    KL/
""")
    return True

def download_emodb(output_dir: str) -> bool:
    """下载EMO-DB数据集"""
    print("\n" + "=" * 60)
    print("Downloading EMO-DB Dataset")
    print("=" * 60)
    
    emodb_dir = Path(output_dir) / 'emodb'
    emodb_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试直接下载
    urls = [
        "http://emodb.bilderbar.info/download/download.zip",
        "http://www.emodb.bilderbar.info/download/emodb.zip"
    ]
    
    zip_path = emodb_dir / 'emodb.zip'
    
    for url in urls:
        print(f"Trying: {url}")
        ret = os.system(f"cd {emodb_dir} && wget -q --show-progress {url} -O emodb.zip 2>/dev/null || curl -L -o emodb.zip {url}")
        if zip_path.exists() and zip_path.stat().st_size > 1000:
            print("Download successful!")
            break
    
    if zip_path.exists() and zip_path.stat().st_size > 1000:
        print("Extracting...")
        os.system(f"cd {emodb_dir} && unzip -q emodb.zip -d extracted/ && find extracted -name '*.wav' -exec cp {{}} . \\; 2>/dev/null || true")
        print(f"EMO-DB ready at: {emodb_dir}")
        return True
    else:
        print("Automatic download failed. Manual download required.")
        print("URL: http://emodb.bilderbar.info/download/")
        return False

def download_jl_corpus(output_dir: str) -> bool:
    """下载JL-Corpus数据集"""
    print("\n" + "=" * 60)
    print("Downloading JL-Corpus Dataset")
    print("=" * 60)
    
    jl_dir = Path(output_dir) / 'jl-corpus'
    jl_dir.mkdir(parents=True, exist_ok=True)
    
    print("JL-Corpus dataset download:")
    print("Kaggle: https://www.kaggle.com/datasets/tli7544/jl-corpus-emotional-speech-dataset")
    print("Note: Requires Kaggle account")
    print(f"\nDownload and extract to: {jl_dir}")
    
    readme = jl_dir / 'README.txt'
    readme.write_text("""JL-Corpus Dataset
=================
Download: https://www.kaggle.com/datasets/tli7544/jl-corpus-emotional-speech-dataset
Size: ~1GB
Speakers: 5 (3 male, 2 female)
Emotions: neutral, happy, sad, angry, fear, disgust, surprise
Language: English

Expected structure:
  jl-corpus/
    angry/
    happy/
    neutral/
    ...
""")
    return True

def download_cremad(output_dir: str) -> bool:
    """下载CREMA-D数据集"""
    print("\n" + "=" * 60)
    print("Downloading CREMA-D Dataset")
    print("=" * 60)
    
    cremad_dir = Path(output_dir) / 'cremad'
    cremad_dir.mkdir(parents=True, exist_ok=True)
    
    print("CREMA-D dataset download options:")
    print("1. GitHub: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad")
    print("Note: Large dataset (~1.5GB)")
    print(f"\nDownload and extract to: {cremad_dir}")
    
    readme = cremad_dir / 'README.txt'
    readme.write_text("""CREMA-D Dataset
===============
Download: https://github.com/CheyneyComputerScience/CREMA-D
Size: ~1.5GB
Speakers: 91 (48 male, 43 female)
Emotions: neutral, happy, sad, angry, fear, disgust
Language: English

Expected structure:
  cremad/
    AudioWAV/
      1001_DFA_ANG_XX.wav
      ...
""")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download speech emotion datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tess', 'savee', 'emodb', 'jl-corpus', 'cremad', 'all'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = ['tess', 'savee', 'emodb', 'jl-corpus', 'cremad'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        if dataset == 'tess':
            download_tess(args.output_dir)
        elif dataset == 'savee':
            download_savee(args.output_dir)
        elif dataset == 'emodb':
            download_emodb(args.output_dir)
        elif dataset == 'jl-corpus':
            download_jl_corpus(args.output_dir)
        elif dataset == 'cremad':
            download_cremad(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset download setup complete")
    print("=" * 60)
    print(f"Check {args.output_dir} for download instructions")

if __name__ == '__main__':
    main()
