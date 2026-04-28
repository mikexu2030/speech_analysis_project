#!/usr/bin/env python3
"""
下载所有预训练模型变体
- wav2vec 2.0: base, large
- HuBERT: base, large
- WavLM: base, base-plus, large
- ECAPA-TDNN variants
"""

import os
import sys
import json
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from transformers import AutoModel, AutoConfig, Wav2Vec2Model, HubertModel, WavLMModel
from transformers import AutoProcessor

def download_model(model_name, output_dir, model_type='auto'):
    """下载模型到指定目录"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        if model_type == 'wav2vec2':
            model = Wav2Vec2Model.from_pretrained(model_name)
        elif model_type == 'hubert':
            model = HubertModel.from_pretrained(model_name)
        elif model_type == 'wavlm':
            model = WavLMModel.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        # 下载processor/config
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            processor.save_pretrained(output_path)
        except:
            pass
        
        # 保存模型
        model.save_pretrained(output_path)
        
        # 获取模型信息
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        print(f"✅ Success: {model_name}")
        print(f"   Parameters: {num_params:,}")
        print(f"   Size: {size_mb:.1f} MB")
        
        return {
            'status': 'success',
            'model_name': model_name,
            'output_dir': str(output_path),
            'num_params': num_params,
            'size_mb': size_mb
        }
        
    except Exception as e:
        print(f"❌ Failed: {model_name}")
        print(f"   Error: {str(e)[:100]}")
        return {
            'status': 'failed',
            'model_name': model_name,
            'error': str(e)
        }

# 模型列表
MODELS_TO_DOWNLOAD = [
    # Wav2Vec 2.0
    ('facebook/wav2vec2-base-960h', 'models/pretrained/wav2vec2_base_960h', 'wav2vec2'),
    ('facebook/wav2vec2-large-960h', 'models/pretrained/wav2vec2_large_960h', 'wav2vec2'),
    ('facebook/wav2vec2-large-960h-lv60-self', 'models/pretrained/wav2vec2_large_lv60', 'wav2vec2'),
    
    # HuBERT
    ('facebook/hubert-base-ls960', 'models/pretrained/hubert_base_ls960', 'hubert'),
    ('facebook/hubert-large-ls960-ft', 'models/pretrained/hubert_large_ls960', 'hubert'),
    
    # WavLM
    ('microsoft/wavlm-base', 'models/pretrained/wavlm_base', 'wavlm'),
    ('microsoft/wavlm-base-plus', 'models/pretrained/wavlm_base_plus', 'wavlm'),
    ('microsoft/wavlm-large', 'models/pretrained/wavlm_large', 'wavlm'),
]

# 下载所有模型
results = []
for model_name, output_dir, model_type in MODELS_TO_DOWNLOAD:
    result = download_model(model_name, output_dir, model_type)
    results.append(result)

# 保存结果
with open('models/download_all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 打印总结
print(f"\n{'='*60}")
print("Download Summary")
print(f"{'='*60}")
success = sum(1 for r in results if r['status'] == 'success')
failed = sum(1 for r in results if r['status'] == 'failed')
print(f"Success: {success}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")

for r in results:
    status = '✅' if r['status'] == 'success' else '❌'
    print(f"{status} {r['model_name']}")

