#!/usr/bin/env python3
"""
模型下载状态追踪脚本
网络恢复后自动继续下载
"""

import os
import json
import time
from datetime import datetime

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
STATUS_FILE = os.path.join(BASE_DIR, 'models/download_status.json')
LOG_FILE = os.path.join(BASE_DIR, 'logs/model_download.log')

# 模型配置
MODELS = {
    'emotion2vec_plus_base': {
        'repo_id': 'emotion2vec/emotion2vec_plus_base',
        'desc': 'Emotion2Vec+ Base (95M params)',
        'size_mb': 380,
        'priority': 1,
        'series': 'Emotion2Vec+ Series'
    },
    'wav2vec2_base_960h': {
        'repo_id': 'facebook/wav2vec2-base-960h',
        'desc': 'wav2vec 2.0 Base (95M params)',
        'size_mb': 380,
        'priority': 2,
        'series': 'wav2vec 2.0 Series'
    },
    'hubert_base_ls960': {
        'repo_id': 'facebook/hubert-base-ls960',
        'desc': 'HuBERT Base (95M params)',
        'size_mb': 380,
        'priority': 3,
        'series': 'HuBERT/WavLM Series'
    },
    'wavlm_base': {
        'repo_id': 'microsoft/wavlm-base',
        'desc': 'WavLM Base (95M params)',
        'size_mb': 380,
        'priority': 4,
        'series': 'HuBERT/WavLM Series'
    },
    'ecapa_tdnn': {
        'repo_id': 'speechbrain/spkrec-ecapa-voxceleb',
        'desc': 'ECAPA-TDNN (6.2M params, Speaker)',
        'size_mb': 25,
        'priority': 5,
        'series': 'Speaker Recognition'
    }
}

def check_network():
    """检查网络是否可用"""
    import urllib.request
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        return True
    except:
        return False

def get_model_dir_size(model_dir):
    """获取模型目录大小"""
    if not os.path.exists(model_dir):
        return 0
    total = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

def check_model_complete(model_dir):
    """检查模型是否完整"""
    if not os.path.exists(model_dir):
        return False
    files = os.listdir(model_dir)
    has_config = any(f == 'config.json' for f in files)
    has_model = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth')) for f in files)
    return has_config and has_model

def print_status():
    """打印当前下载状态"""
    print("="*70)
    print("Top3 Model Series Download Status")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Network: {'Available' if check_network() else 'UNAVAILABLE'}")
    print()
    
    # 加载状态
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
    
    # 按系列分组显示
    series_groups = {}
    for name, config in MODELS.items():
        series = config['series']
        if series not in series_groups:
            series_groups[series] = []
        series_groups[series].append((name, config))
    
    for series, models in series_groups.items():
        print(f"\n【{series}】")
        print("-"*70)
        
        for name, config in sorted(models, key=lambda x: x[1]['priority']):
            model_dir = os.path.join(MODEL_DIR, name)
            size = get_model_dir_size(model_dir)
            complete = check_model_complete(model_dir)
            
            if complete:
                icon = "✅"
                state = "COMPLETE"
            elif size > 0:
                icon = "⏳"
                state = f"PARTIAL ({size/1024/1024:.1f} MB)"
            else:
                icon = "❌"
                state = "NOT STARTED"
            
            # 检查之前的状态记录
            if name in status:
                if 'failed' in status[name]:
                    state += " - Network Error"
                elif status[name] == 'success':
                    state = "COMPLETE"
            
            print(f"  {icon} {name}")
            print(f"     Desc: {config['desc']}")
            print(f"     State: {state}")
            print(f"     Repo: https://huggingface.co/{config['repo_id']}")
    
    # 统计
    complete_count = sum(1 for name in MODELS if check_model_complete(os.path.join(MODEL_DIR, name)))
    total = len(MODELS)
    
    print(f"\n{'='*70}")
    print(f"Progress: {complete_count}/{total} models complete")
    print(f"{'='*70}")
    
    if complete_count < total:
        print("\nTo download missing models:")
        print("  1. Wait for network to recover")
        print("  2. Run: python3 download_all_models.py")
        print("  3. Or run: python3 download_all_models.py --model <name>")
        
        print("\nAlternative download methods:")
        print("  - Manual: https://huggingface.co/<repo_id>")
        print("  - Git: git clone https://huggingface.co/<repo_id>")
        print("  - wget: wget -r -np -nH https://huggingface.co/<repo_id>/resolve/main/")
    
    print(f"\nStatus file: {STATUS_FILE}")
    print(f"Model dir: {MODEL_DIR}")

def download_single_model(name):
    """下载单个模型"""
    from huggingface_hub import snapshot_download
    
    config = MODELS.get(name)
    if not config:
        print(f"Unknown model: {name}")
        return
    
    target_dir = os.path.join(MODEL_DIR, name)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\nDownloading: {config['desc']}")
    print(f"From: {config['repo_id']}")
    
    try:
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=target_dir,
            resume_download=True
        )
        print("Download complete!")
        
        # 更新状态
        status = {}
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        
        status[name] = 'success'
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Network may be unavailable. Check with: python3 check_model_status.py")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and download models')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--download', type=str, help='Download specific model')
    parser.add_argument('--all', action='store_true', help='Download all missing models')
    
    args = parser.parse_args()
    
    if args.download:
        download_single_model(args.download)
    elif args.all:
        print("Downloading all missing models...")
        for name, config in sorted(MODELS.items(), key=lambda x: x[1]['priority']):
            model_dir = os.path.join(MODEL_DIR, name)
            if not check_model_complete(model_dir):
                download_single_model(name)
                time.sleep(2)
    else:
        print_status()

if __name__ == '__main__':
    main()
