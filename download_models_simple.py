#!/usr/bin/env python3
"""
简化版模型下载 - 逐个下载，带进度显示
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
from pathlib import Path

MODELS = [
    ('facebook/wav2vec2-large-960h', 'models/pretrained/wav2vec2_large_960h'),
    ('facebook/hubert-large-ls960-ft', 'models/pretrained/hubert_large_ls960'),
    ('microsoft/wavlm-large', 'models/pretrained/wavlm_large'),
]

FILES_TO_DOWNLOAD = [
    'config.json',
    'pytorch_model.bin',
    'preprocessor_config.json',
]

for model_id, output_dir in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for filename in FILES_TO_DOWNLOAD:
        try:
            print(f"  Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"  ✅ {filename}")
        except Exception as e:
            print(f"  ❌ {filename}: {str(e)[:80]}")
    
    print(f"Done: {model_id}")

print("\nAll downloads complete!")
