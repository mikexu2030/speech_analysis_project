"""
数据下载脚本 - 情绪数据集
支持: RAVDESS, CREMA-D, ESD
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str) -> bool:
    """
    下载文件并显示进度
    
    Args:
        url: 下载链接
        output_path: 保存路径
    
    Returns:
        是否成功
    """
    try:
        print(f"Downloading: {url}")
        print(f"Saving to: {output_path}")
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
        print(f"Download completed: {output_path}")
        return True
    
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """
    解压压缩包
    
    Args:
        archive_path: 压缩包路径
        extract_to: 解压目录
    
    Returns:
        是否成功
    """
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


def download_ravdess(output_dir: str) -> bool:
    """
    下载RAVDESS数据集
    
    RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
    - 24名专业演员 (12男12女)
    - 8种情绪: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    - 2种强度: normal, strong
    - 英语
    
    下载链接: Zenodo
    """
    print("\n" + "=" * 60)
    print("Downloading RAVDESS Dataset")
    print("=" * 60)
    
    ravdess_dir = os.path.join(output_dir, 'ravdess')
    os.makedirs(ravdess_dir, exist_ok=True)
    
    # RAVDESS有多个部分
    base_url = "https://zenodo.org/record/1188976/files/"
    
    files = [
        "Audio_Speech_Actors_01-24.zip",
        "Audio_Song_Actors_01-24.zip"
    ]
    
    for filename in files:
        url = base_url + filename
        output_path = os.path.join(ravdess_dir, filename)
        
        if os.path.exists(output_path):
            print(f"File already exists: {filename}")
        else:
            if not download_file(url, output_path):
                print(f"Failed to download {filename}")
                continue
        
        # 解压
        extract_dir = os.path.join(ravdess_dir, 'audio')
        if not os.path.exists(extract_dir):
            extract_archive(output_path, extract_dir)
    
    print(f"RAVDESS dataset ready at: {ravdess_dir}")
    return True


def download_cremad(output_dir: str) -> bool:
    """
    下载CREMA-D数据集
    
    CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
    - 91名演员 (48男43女)
    - 6种情绪: neutral, happy, sad, angry, fearful, disgust
    - 英语
    
    需要通过GitHub下载
    """
    print("\n" + "=" * 60)
    print("Downloading CREMA-D Dataset")
    print("=" * 60)
    
    cremad_dir = os.path.join(output_dir, 'cremad')
    os.makedirs(cremad_dir, exist_ok=True)
    
    print("CREMA-D requires manual download from:")
    print("  https://github.com/CheyneyComputerScience/CREMA-D")
    print("  or https://www.kaggle.com/datasets/ejlok1/cremad")
    print(f"\nPlease download and extract to: {cremad_dir}")
    
    # 创建说明文件
    readme_path = os.path.join(cremad_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("CREMA-D Dataset\n")
        f.write("===============\n\n")
        f.write("Download from:\n")
        f.write("  https://github.com/CheyneyComputerScience/CREMA-D\n")
        f.write("  https://www.kaggle.com/datasets/ejlok1/cremad\n\n")
        f.write("Expected structure:\n")
        f.write("  cremad/\n")
        f.write("    AudioWAV/\n")
        f.write("      1001_DFA_ANG_XX.wav\n")
        f.write("      ...\n")
    
    return True


def download_esd(output_dir: str) -> bool:
    """
    下载ESD数据集 (Emotional Speech Dataset)
    
    ESD: 包含英语和中文的情绪语音
    - 10名英语说话人 (5男5女)
    - 5种情绪: neutral, happy, angry, sad, surprise
    - 英语 + 中文
    
    下载链接: GitHub
    """
    print("\n" + "=" * 60)
    print("Downloading ESD Dataset")
    print("=" * 60)
    
    esd_dir = os.path.join(output_dir, 'esd')
    os.makedirs(esd_dir, exist_ok=True)
    
    print("ESD requires manual download from:")
    print("  https://github.com/HLTSingapore/Emotional-Speech-Data")
    print(f"\nPlease download and extract to: {esd_dir}")
    
    # 创建说明文件
    readme_path = os.path.join(esd_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("ESD Dataset\n")
        f.write("===========\n\n")
        f.write("Download from:\n")
        f.write("  https://github.com/HLTSingapore/Emotional-Speech-Data\n\n")
        f.write("Expected structure:\n")
        f.write("  esd/\n")
        f.write("    0011/\n")
        f.write("      angry/\n")
        f.write("      happy/\n")
        f.write("      ...\n")
    
    return True


def download_iemocap(output_dir: str) -> bool:
    """
    下载IEMOCAP数据集
    
    IEMOCAP: Interactive Emotional Dyadic Motion Capture
    - 10名演员
    - 多种情绪标签
    - 需要申请获取
    """
    print("\n" + "=" * 60)
    print("IEMOCAP Dataset")
    print("=" * 60)
    
    iemocap_dir = os.path.join(output_dir, 'iemocap')
    os.makedirs(iemocap_dir, exist_ok=True)
    
    print("IEMOCAP requires registration and approval:")
    print("  https://sail.usc.edu/iemocap/iemocap_release.htm")
    print(f"\nPlease request access and extract to: {iemocap_dir}")
    
    readme_path = os.path.join(iemocap_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("IEMOCAP Dataset\n")
        f.write("===============\n\n")
        f.write("Request access at:\n")
        f.write("  https://sail.usc.edu/iemocap/iemocap_release.htm\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Download emotion datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ravdess', 'cremad', 'esd', 'iemocap', 'all'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--extract', action='store_true',
                       help='Extract archives after download')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载数据集
    if args.dataset == 'all' or args.dataset == 'ravdess':
        download_ravdess(args.output_dir)
    
    if args.dataset == 'all' or args.dataset == 'cremad':
        download_cremad(args.output_dir)
    
    if args.dataset == 'all' or args.dataset == 'esd':
        download_esd(args.output_dir)
    
    if args.dataset == 'all' or args.dataset == 'iemocap':
        download_iemocap(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Download process completed")
    print("=" * 60)
    print(f"Data saved to: {args.output_dir}")
    print("\nNote: Some datasets require manual download.")
    print("Please check the README files in each dataset directory.")


if __name__ == '__main__':
    main()
