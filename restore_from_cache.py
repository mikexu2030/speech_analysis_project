#!/usr/bin/env python3
"""
从本地HuggingFace缓存恢复模型到项目目录
"""

import os
import shutil
import json

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
CACHE_DIR = os.path.expanduser('~/.cache/huggingface/hub')

# 缓存中的模型映射
CACHE_MODELS = {
    'wavlm_base_plus': {
        'cache_name': 'models--microsoft--wavlm-base-plus',
        'target_name': 'wavlm_base',
        'desc': 'WavLM Base Plus (从缓存恢复)'
    }
}

def restore_from_cache(cache_model_name, target_name):
    """从缓存恢复模型"""
    cache_path = os.path.join(CACHE_DIR, cache_model_name)
    target_path = os.path.join(MODEL_DIR, target_name)
    
    if not os.path.exists(cache_path):
        print(f"❌ Cache not found: {cache_path}")
        return False
    
    print(f"\n恢复模型: {cache_model_name} -> {target_name}")
    
    # 查找快照目录
    snapshots_dir = os.path.join(cache_path, 'snapshots')
    if not os.path.exists(snapshots_dir):
        print(f"❌ No snapshots found")
        return False
    
    snapshots = os.listdir(snapshots_dir)
    if not snapshots:
        print(f"❌ Empty snapshots")
        return False
    
    # 使用第一个快照
    snapshot = snapshots[0]
    snapshot_path = os.path.join(snapshots_dir, snapshot)
    
    print(f"  Snapshot: {snapshot}")
    print(f"  Files: {os.listdir(snapshot_path)}")
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 复制文件
    for f in os.listdir(snapshot_path):
        src = os.path.join(snapshot_path, f)
        dst = os.path.join(target_path, f)
        
        if os.path.isfile(src):
            if not os.path.exists(dst):
                print(f"  Copying: {f}")
                shutil.copy2(src, dst)
            else:
                print(f"  Already exists: {f}")
    
    print(f"✅ Restored to: {target_path}")
    return True

def check_model_complete(model_dir):
    """检查模型是否完整"""
    if not os.path.exists(model_dir):
        return False
    
    files = os.listdir(model_dir)
    has_config = any(f == 'config.json' for f in files)
    has_model = any(f.endswith(('.bin', '.safetensors')) for f in files)
    
    return has_config and has_model

def main():
    print("="*60)
    print("从本地缓存恢复模型")
    print("="*60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    restored = []
    failed = []
    
    for cache_name, info in CACHE_MODELS.items():
        target = info['target_name']
        target_path = os.path.join(MODEL_DIR, target)
        
        # 检查是否已完整
        if check_model_complete(target_path):
            print(f"\n✅ {target} already complete")
            restored.append(target)
            continue
        
        # 从缓存恢复
        if restore_from_cache(info['cache_name'], target):
            restored.append(target)
        else:
            failed.append(target)
    
    # 报告
    print(f"\n{'='*60}")
    print("恢复结果")
    print(f"{'='*60}")
    
    for name in restored:
        path = os.path.join(MODEL_DIR, name)
        files = os.listdir(path)
        size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
        print(f"✅ {name}: {len(files)} files, {size/1024/1024:.1f} MB")
    
    for name in failed:
        print(f"❌ {name}: Failed")
    
    print(f"\n总计: {len(restored)} 成功, {len(failed)} 失败")
    
    # 更新状态文件
    status_file = os.path.join(BASE_DIR, 'models/download_status.json')
    status = {}
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
    
    for name in restored:
        status[name] = 'restored_from_cache'
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n状态已更新: {status_file}")

if __name__ == '__main__':
    main()
