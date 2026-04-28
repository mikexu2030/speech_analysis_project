#!/usr/bin/env python3
"""
网络恢复后自动下载所有缺失模型
用法: python3 auto_download_on_network_recovery.py
"""

import os
import sys
import time
import json
from datetime import datetime

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
STATUS_FILE = os.path.join(BASE_DIR, 'models/download_status.json')
LOG_FILE = os.path.join(BASE_DIR, 'logs/auto_download.log')

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

MODELS = {
    'emotion2vec_plus_base': {
        'repo_id': 'emotion2vec/emotion2vec_plus_base',
        'desc': 'Emotion2Vec+ Base',
        'priority': 1,
    },
    'wav2vec2_base_960h': {
        'repo_id': 'facebook/wav2vec2-base-960h',
        'desc': 'wav2vec 2.0 Base',
        'priority': 2,
    },
    'hubert_base_ls960': {
        'repo_id': 'facebook/hubert-base-ls960',
        'desc': 'HuBERT Base',
        'priority': 3,
    },
    'wavlm_base': {
        'repo_id': 'microsoft/wavlm-base',
        'desc': 'WavLM Base',
        'priority': 4,
    },
    'ecapa_tdnn': {
        'repo_id': 'speechbrain/spkrec-ecapa-voxceleb',
        'desc': 'ECAPA-TDNN',
        'priority': 5,
    }
}

def log(msg):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def check_network():
    """检查网络是否可用"""
    import urllib.request
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        return True
    except:
        return False

def check_model_complete(model_dir):
    """检查模型是否完整"""
    if not os.path.exists(model_dir):
        return False
    files = os.listdir(model_dir)
    has_config = any(f == 'config.json' for f in files)
    has_model = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth', '.ckpt')) for f in files)
    return has_config and has_model

def download_model(name, config):
    """下载单个模型"""
    from huggingface_hub import snapshot_download
    
    target_dir = os.path.join(MODEL_DIR, name)
    os.makedirs(target_dir, exist_ok=True)
    
    log(f"Downloading: {config['desc']} ({config['repo_id']})")
    
    try:
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=target_dir,
            resume_download=True
        )
        
        if check_model_complete(target_dir):
            log(f"✅ {name} download complete")
            return 'success'
        else:
            log(f"⚠️ {name} incomplete")
            return 'incomplete'
            
    except Exception as e:
        log(f"❌ {name} failed: {e}")
        return f'failed: {str(e)[:50]}'

def main():
    log("="*60)
    log("Auto Download on Network Recovery")
    log("="*60)
    
    # 检查网络
    if not check_network():
        log("❌ Network still unavailable")
        log("Please run again when network is back")
        return
    
    log("✅ Network is available!")
    
    # 加载状态
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
    
    # 检查哪些模型需要下载
    needed = []
    for name, config in sorted(MODELS.items(), key=lambda x: x[1]['priority']):
        target_dir = os.path.join(MODEL_DIR, name)
        
        if check_model_complete(target_dir):
            log(f"✅ {name} already complete, skipping")
            status[name] = 'complete'
            continue
        
        if name in status and status[name] in ['success', 'complete', 'restored_from_cache']:
            # 再次检查文件
            if not check_model_complete(target_dir):
                log(f"⚠️ {name} marked complete but files missing, re-downloading")
                needed.append((name, config))
            else:
                log(f"✅ {name} already complete, skipping")
                continue
        else:
            needed.append((name, config))
    
    if not needed:
        log("\n✅ All models are already downloaded!")
        return
    
    log(f"\nNeed to download: {len(needed)} models")
    
    # 下载缺失的模型
    for name, config in needed:
        result = download_model(name, config)
        status[name] = result
        
        # 保存状态
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        
        # 短暂暂停
        if result != 'success':
            log("Pausing 10s...")
            time.sleep(10)
    
    # 最终报告
    log(f"\n{'='*60}")
    log("Download Summary")
    log(f"{'='*60}")
    
    success = 0
    for name, result in status.items():
        if result in ['success', 'complete', 'restored_from_cache']:
            icon = '✅'
            success += 1
        else:
            icon = '❌'
        log(f"{icon} {name}: {result}")
    
    total = len(MODELS)
    log(f"\nTotal: {success}/{total} models ready")
    
    if success == total:
        log("\n🎉 All models downloaded! Run evaluation:")
        log("  python3 scripts/evaluate_models_offline.py")
    else:
        log(f"\nTo retry failed downloads:")
        log("  python3 auto_download_on_network_recovery.py")

if __name__ == '__main__':
    main()
