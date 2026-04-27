"""
数据下载脚本 - 说话人/年龄/性别数据集
支持: VoxCeleb, Common Voice
"""

import os
import sys
import argparse
import zipfile
import tarfile
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
    """下载文件"""
    try:
        print(f"Downloading: {url}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
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


def download_voxceleb(output_dir: str, version: str = '1') -> bool:
    """
    下载VoxCeleb数据集
    
    VoxCeleb: 大规模说话人识别数据集
    - VoxCeleb1: 1,251名说话人，100k+ utterances
    - VoxCeleb2: 6,112名说话人，1M+ utterances
    - 多语言、多场景
    
    需要通过Google Drive或官网下载
    """
    print("\n" + "=" * 60)
    print(f"VoxCeleb {version} Dataset")
    print("=" * 60)
    
    vox_dir = os.path.join(output_dir, f'voxceleb{version}')
    os.makedirs(vox_dir, exist_ok=True)
    
    print("VoxCeleb requires manual download:")
    print("  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print(f"\nPlease download and extract to: {vox_dir}")
    
    readme_path = os.path.join(vox_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"VoxCeleb {version} Dataset\n")
        f.write("=" * 40 + "\n\n")
        f.write("Download from:\n")
        f.write("  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/\n\n")
        f.write("Expected structure:\n")
        f.write(f"  voxceleb{version}/\n")
        f.write("    wav/\n")
        f.write("      id10001/\n")
        f.write("        1zcIwhmdeo4/\n")
        f.write("          00001.wav\n")
    
    return True


def download_common_voice(output_dir: str, language: str = 'en') -> bool:
    """
    下载Common Voice数据集
    
    Common Voice: Mozilla开源语音数据集
    - 多语言
    - 包含年龄、性别元数据
    - 适合年龄/性别识别训练
    
    下载链接: https://commonvoice.mozilla.org/
    """
    print("\n" + "=" * 60)
    print(f"Common Voice Dataset ({language})")
    print("=" * 60)
    
    cv_dir = os.path.join(output_dir, f'common_voice_{language}')
    os.makedirs(cv_dir, exist_ok=True)
    
    print("Common Voice requires manual download:")
    print("  https://commonvoice.mozilla.org/en/datasets")
    print(f"\nPlease download and extract to: {cv_dir}")
    
    readme_path = os.path.join(cv_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"Common Voice ({language}) Dataset\n")
        f.write("=" * 40 + "\n\n")
        f.write("Download from:\n")
        f.write("  https://commonvoice.mozilla.org/en/datasets\n\n")
        f.write("Expected structure:\n")
        f.write(f"  common_voice_{language}/\n")
        f.write("    clips/\n")
        f.write("      common_voice_en_1.mp3\n")
        f.write("      ...\n")
        f.write("    train.tsv\n")
        f.write("    dev.tsv\n")
        f.write("    test.tsv\n")
        f.write("    validated.tsv\n")
    
    return True


def download_libri_tts(output_dir: str) -> bool:
    """
    下载LibriTTS数据集
    
    LibriTTS: 多说话人TTS数据集
    - 适合说话人识别
    - 高质量英语语音
    """
    print("\n" + "=" * 60)
    print("LibriTTS Dataset")
    print("=" * 60)
    
    libri_dir = os.path.join(output_dir, 'libritts')
    os.makedirs(libri_dir, exist_ok=True)
    
    print("LibriTTS can be downloaded from:")
    print("  http://www.openslr.org/60/")
    print(f"\nPlease download and extract to: {libri_dir}")
    
    readme_path = os.path.join(libri_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("LibriTTS Dataset\n")
        f.write("===============\n\n")
        f.write("Download from:\n")
        f.write("  http://www.openslr.org/60/\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Download speaker datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['voxceleb1', 'voxceleb2', 'common_voice', 'libritts', 'all'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--language', type=str, default='en',
                       help='Language for Common Voice (en, es, fr, de, etc.)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'all' or args.dataset == 'voxceleb1':
        download_voxceleb(args.output_dir, version='1')
    
    if args.dataset == 'all' or args.dataset == 'voxceleb2':
        download_voxceleb(args.output_dir, version='2')
    
    if args.dataset == 'all' or args.dataset == 'common_voice':
        download_common_voice(args.output_dir, language=args.language)
    
    if args.dataset == 'all' or args.dataset == 'libritts':
        download_libri_tts(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Download process completed")
    print("=" * 60)
    print(f"Data saved to: {args.output_dir}")
    print("\nNote: Most datasets require manual download.")
    print("Please check the README files in each dataset directory.")


if __name__ == '__main__':
    main()
