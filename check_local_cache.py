#!/usr/bin/env python3
"""
断网恢复方案 - 检查所有可能的本地缓存和镜像源
"""

import os
import sys
import glob
import json

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
CACHE_DIRS = [
    os.path.expanduser('~/.cache/huggingface/hub'),
    os.path.expanduser('~/.cache/torch/hub'),
    '/tmp/huggingface',
    '/tmp/torch',
]

# 检查本地缓存
print("="*60)
print("检查本地HuggingFace缓存...")
print("="*60)

found_models = {}

for cache_dir in CACHE_DIRS:
    if not os.path.exists(cache_dir):
        continue
    
    print(f"\n检查: {cache_dir}")
    
    # 查找模型快照
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(('.bin', '.safetensors', '.ckpt', '.pt')):
                full_path = os.path.join(root, f)
                size = os.path.getsize(full_path) / 1024 / 1024
                
                # 尝试识别模型
                model_name = None
                if 'emotion2vec' in root.lower():
                    model_name = 'emotion2vec_plus_base'
                elif 'wav2vec2' in root.lower():
                    model_name = 'wav2vec2_base_960h'
                elif 'hubert' in root.lower():
                    model_name = 'hubert_base_ls960'
                elif 'wavlm' in root.lower():
                    model_name = 'wavlm_base'
                elif 'ecapa' in root.lower():
                    model_name = 'ecapa_tdnn'
                
                if model_name:
                    if model_name not in found_models:
                        found_models[model_name] = []
                    found_models[model_name].append({
                        'path': full_path,
                        'size_mb': size,
                        'file': f
                    })
                
                print(f"  [{size:.1f}MB] {full_path}")

print(f"\n{'='*60}")
print("缓存中找到的模型:")
print(f"{'='*60}")

for model_name, files in found_models.items():
    total_size = sum(f['size_mb'] for f in files)
    print(f"\n{model_name}: {len(files)} files, {total_size:.1f} MB total")
    for f in files[:3]:
        print(f"  - {f['file']} ({f['size_mb']:.1f} MB)")

# 检查conda缓存
print(f"\n{'='*60}")
print("检查conda/pip缓存...")
print(f"{'='*60}")

conda_cache = os.path.expanduser('~/.conda/pkgs')
pip_cache = os.path.expanduser('~/.cache/pip')

for cache in [conda_cache, pip_cache]:
    if os.path.exists(cache):
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, dirs, files in os.walk(cache)
            for f in files
        ) / 1024 / 1024
        print(f"{cache}: {size:.1f} MB")

# 检查系统是否有预装模型
print(f"\n{'='*60}")
print("检查系统预装模型...")
print(f"{'='*60}")

system_paths = [
    '/usr/local/lib/python*/dist-packages/transformers/models',
    '/usr/lib/python*/site-packages/transformers/models',
]

for pattern in system_paths:
    for path in glob.glob(pattern):
        if os.path.exists(path):
            print(f"Found: {path}")
            models = os.listdir(path)
            for m in models:
                print(f"  - {m}")

print(f"\n{'='*60}")
print("总结")
print(f"{'='*60}")

if found_models:
    print(f"✅ 在缓存中找到 {len(found_models)} 个模型")
    print("\n可以将缓存模型复制到项目目录:")
    for model_name in found_models:
        print(f"  cp -r {found_models[model_name][0]['path'].rsplit('/', 1)[0]} {MODEL_DIR}/{model_name}")
else:
    print("❌ 未在本地缓存中找到任何模型")
    print("\n建议:")
    print("  1. 等待网络恢复后重新下载")
    print("  2. 从其他机器复制模型文件")
    print("  3. 使用U盘/移动硬盘导入模型")
    print("\n手动导入方法:")
    print("  1. 从 https://huggingface.co 下载模型")
    print("  2. 解压到 models/pretrained/<model_name>/")
    print("  3. 确保包含 config.json 和 pytorch_model.bin")
