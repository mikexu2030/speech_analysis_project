#!/usr/bin/env python3
"""
模型下载脚本 - 支持断点续传和断网恢复
"""

import os
import sys
import time
from pathlib import Path

# 模型配置
MODELS = {
    'emotion2vec_plus_base': {
        'repo_id': 'emotion2vec/emotion2vec_plus_base',
        'desc': 'Emotion2Vec+ Base (95M params)',
        'size_mb': 380,
        'priority': 1,
    },
    'wav2vec2_base_960h': {
        'repo_id': 'facebook/wav2vec2-base-960h',
        'desc': 'wav2vec 2.0 Base (95M params)',
        'size_mb': 380,
        'priority': 2,
    },
    'hubert_base_ls960': {
        'repo_id': 'facebook/hubert-base-ls960',
        'desc': 'HuBERT Base (95M params)',
        'size_mb': 380,
        'priority': 3,
    },
    'wavlm_base': {
        'repo_id': 'microsoft/wavlm-base',
        'desc': 'WavLM Base (95M params)',
        'size_mb': 380,
        'priority': 4,
    },
    'ecapa_tdnn': {
        'repo_id': 'speechbrain/spkrec-ecapa-voxceleb',
        'desc': 'ECAPA-TDNN (6.2M params, Speaker)',
        'size_mb': 25,
        'priority': 5,
    }
}

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
STATUS_FILE = os.path.join(BASE_DIR, 'models/download_status.json')

def check_model_complete(model_dir):
    """检查模型是否完整下载"""
    if not os.path.exists(model_dir):
        return False
    
    files = os.listdir(model_dir)
    
    # 检查关键文件
    has_config = any(f == 'config.json' for f in files)
    has_model = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth')) for f in files)
    
    return has_config and has_model

def get_model_size(model_dir):
    """获取模型目录大小"""
    if not os.path.exists(model_dir):
        return 0
    
    total = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    
    return total

def download_model(name, config):
    """下载单个模型"""
    from huggingface_hub import snapshot_download
    
    target_dir = os.path.join(MODEL_DIR, name)
    
    print(f"\n{'='*60}")
    print(f"[{config['priority']}] Downloading: {config['desc']}")
    print(f"Repo: {config['repo_id']}")
    print(f"Expected size: ~{config['size_mb']} MB")
    print(f"{'='*60}")
    
    # 检查是否已完整下载
    if check_model_complete(target_dir):
        size = get_model_size(target_dir)
        print(f"Already complete: {size/1024/1024:.1f} MB")
        return 'complete'
    
    # 检查是否有部分下载
    if os.path.exists(target_dir):
        size = get_model_size(target_dir)
        print(f"Partial download: {size/1024/1024:.1f} MB")
        print("Resuming download...")
    
    try:
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # 验证
        if check_model_complete(target_dir):
            size = get_model_size(target_dir)
            print(f"Download complete: {size/1024/1024:.1f} MB")
            return 'success'
        else:
            print("Download incomplete - missing key files")
            return 'incomplete'
            
    except Exception as e:
        print(f"ERROR: {e}")
        return f'failed: {str(e)[:50]}'

def save_status(status):
    """保存下载状态"""
    import json
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def load_status():
    """加载下载状态"""
    import json
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {}

def main():
    print("="*60)
    print("Top3 Model Series Download")
    print("="*60)
    print("\nSeries 1: Emotion2Vec+")
    print("  - emotion2vec_plus_base (95M)")
    print("\nSeries 2: wav2vec 2.0")
    print("  - wav2vec2_base_960h (95M)")
    print("\nSeries 3: HuBERT / WavLM")
    print("  - hubert_base_ls960 (95M)")
    print("  - wavlm_base (95M)")
    print("\nBonus: ECAPA-TDNN (Speaker)")
    print("  - ecapa_tdnn (6.2M)")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 加载之前的状态
    status = load_status()
    
    # 按优先级下载
    for name, config in sorted(MODELS.items(), key=lambda x: x[1]['priority']):
        if name in status and status[name] == 'success':
            print(f"\n{name}: Already downloaded, skipping")
            continue
        
        result = download_model(name, config)
        status[name] = result
        
        # 保存状态
        save_status(status)
        
        # 如果失败，暂停一下再继续
        if result not in ['success', 'complete']:
            print("Pausing 5s before next model...")
            time.sleep(5)
    
    # 最终报告
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    
    for name, result in status.items():
        icon = '✅' if result in ['success', 'complete'] else '❌'
        print(f"{icon} {name}: {result}")
    
    # 统计
    success = sum(1 for r in status.values() if r in ['success', 'complete'])
    total = len(MODELS)
    print(f"\nTotal: {success}/{total} models downloaded")
    
    if success < total:
        print(f"\nTo resume: python3 {sys.argv[0]}")
        print(f"Status file: {STATUS_FILE}")

if __name__ == '__main__':
    main()
