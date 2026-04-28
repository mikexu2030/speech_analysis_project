#!/usr/bin/env python3
"""
快速下载预训练模型变体 - 使用snapshot_download直接下载文件
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download
from pathlib import Path

MODELS = [
    ('facebook/wav2vec2-large-960h', 'models/pretrained/wav2vec2_large_960h'),
    ('facebook/wav2vec2-large-960h-lv60-self', 'models/pretrained/wav2vec2_large_lv60'),
    ('facebook/hubert-large-ls960-ft', 'models/pretrained/hubert_large_ls960'),
    ('microsoft/wavlm-base-plus', 'models/pretrained/wavlm_base_plus'),
    ('microsoft/wavlm-large', 'models/pretrained/wavlm_large'),
]

for model_id, output_dir in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Success: {model_id}")
    except Exception as e:
        print(f"❌ Failed: {model_id}")
        print(f"   Error: {str(e)[:100]}")

print("\nDone!")
