#!/usr/bin/env python3
"""
下载语音数据集
- Common Voice 11.0 (多语言)
- 使用streaming模式避免内存问题
"""

import os
import sys
import json
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 语言列表
LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'ja']

def download_common_voice(language='en', max_samples=5000):
    """下载Common Voice数据集"""
    print(f"\n{'='*60}")
    print(f"Downloading Common Voice 11.0 - {language}")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        output_dir = Path(f'data/raw/common_voice/{language}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用streaming模式
        ds_train = load_dataset(
            'mozilla-foundation/common_voice_11_0',
            language,
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        
        ds_test = load_dataset(
            'mozilla-foundation/common_voice_11_0',
            language,
            split='test',
            streaming=True,
            trust_remote_code=True
        )
        
        # 收集样本信息
        train_samples = []
        test_samples = []
        
        print(f"Collecting train samples...")
        for i, sample in enumerate(ds_train):
            if i >= max_samples:
                break
            train_samples.append({
                'path': sample.get('path', ''),
                'age': sample.get('age', ''),
                'gender': sample.get('gender', ''),
                'accent': sample.get('accent', ''),
                'sentence': sample.get('sentence', '')
            })
            if (i + 1) % 1000 == 0:
                print(f"  Collected {i+1} train samples...")
        
        print(f"Collecting test samples...")
        for i, sample in enumerate(ds_test):
            if i >= max_samples // 5:
                break
            test_samples.append({
                'path': sample.get('path', ''),
                'age': sample.get('age', ''),
                'gender': sample.get('gender', ''),
                'accent': sample.get('accent', ''),
                'sentence': sample.get('sentence', '')
            })
            if (i + 1) % 500 == 0:
                print(f"  Collected {i+1} test samples...")
        
        # 保存metadata
        metadata = {
            'language': language,
            'train_count': len(train_samples),
            'test_count': len(test_samples),
            'train_samples': train_samples,
            'test_samples': test_samples
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Success: {language}")
        print(f"   Train: {len(train_samples)} samples")
        print(f"   Test: {len(test_samples)} samples")
        print(f"   Metadata: {metadata_file}")
        
        return {
            'status': 'success',
            'language': language,
            'train_count': len(train_samples),
            'test_count': len(test_samples)
        }
        
    except Exception as e:
        print(f"❌ Failed: {language}")
        print(f"   Error: {str(e)[:200]}")
        return {
            'status': 'failed',
            'language': language,
            'error': str(e)
        }

# 下载所有语言
results = []
for lang in LANGUAGES:
    result = download_common_voice(lang, max_samples=5000)
    results.append(result)

# 保存结果
with open('data/raw/common_voice/download_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 总结
print(f"\n{'='*60}")
print("Dataset Download Summary")
print(f"{'='*60}")
success = sum(1 for r in results if r['status'] == 'success')
failed = sum(1 for r in results if r['status'] == 'failed')
print(f"Success: {success}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")

for r in results:
    status = '✅' if r['status'] == 'success' else '❌'
    if r['status'] == 'success':
        print(f"{status} {r['language']}: train={r['train_count']}, test={r['test_count']}")
    else:
        print(f"{status} {r['language']}: {r.get('error', 'unknown error')[:50]}")

