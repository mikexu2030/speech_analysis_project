#!/usr/bin/env python3
"""
测试运行器 - 测试项目各项能力
支持: 数据加载、模型推理、音频处理、Git上传
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def run_test(self, name: str, test_func) -> bool:
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        
        start = time.time()
        try:
            result = test_func()
            elapsed = time.time() - start
            
            if result:
                print(f"✅ PASS ({elapsed:.2f}s)")
                self.passed += 1
                self.results.append({"name": name, "status": "PASS", "time": elapsed})
                return True
            else:
                print(f"❌ FAIL ({elapsed:.2f}s)")
                self.failed += 1
                self.results.append({"name": name, "status": "FAIL", "time": elapsed})
                return False
        
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ERROR ({elapsed:.2f}s): {e}")
            self.failed += 1
            self.results.append({"name": name, "status": "ERROR", "error": str(e), "time": elapsed})
            return False
    
    def print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed/(self.passed+self.failed)*100:.1f}%")
        
        if self.failed > 0:
            print(f"\nFailed tests:")
            for r in self.results:
                if r["status"] != "PASS":
                    print(f"  ❌ {r['name']}: {r.get('error', 'FAIL')}")


# ===== 测试用例 =====

def test_data_loading():
    """测试数据加载能力"""
    print("Testing data loading...")
    
    # 测试合并数据集
    merged_path = PROJECT_ROOT / "data" / "processed" / "merged_dataset.json"
    if not merged_path.exists():
        print(f"  Merged dataset not found: {merged_path}")
        return False
    
    with open(merged_path) as f:
        data = json.load(f)
    
    print(f"  Total samples: {len(data)}")
    assert len(data) > 0, "No samples loaded"
    
    # 测试样本格式
    sample = data[0]
    required_keys = ['audio_path', 'dataset', 'emotion', 'speaker_id', 'gender', 'language']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Datasets: {set(s['dataset'] for s in data)}")
    return True


def test_splits():
    """测试数据划分"""
    print("Testing data splits...")
    
    splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    for split in ['train', 'val', 'test']:
        path = splits_dir / f"{split}.json"
        if not path.exists():
            print(f"  {split}.json not found")
            return False
        
        with open(path) as f:
            data = json.load(f)
        
        print(f"  {split}: {len(data)} samples")
        assert len(data) > 0, f"Empty {split} split"
    
    return True


def test_audio_utils():
    """测试音频工具"""
    print("Testing audio utilities...")
    
    try:
        from utils.audio_utils import load_audio
        
        # 查找一个音频文件
        audio_files = list((PROJECT_ROOT / "data" / "raw").rglob("*.wav"))
        if not audio_files:
            print("  No audio files found")
            return False
        
        test_file = audio_files[0]
        print(f"  Testing with: {test_file}")
        
        waveform, sr = load_audio(str(test_file), sr=16000)
        print(f"  Loaded: shape={waveform.shape}, sr={sr}")
        
        assert waveform is not None, "Failed to load audio"
        assert sr == 16000, f"Wrong sample rate: {sr}"
        assert len(waveform) > 0, "Empty waveform"
        
        return True
    
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    print("Testing model loading...")
    
    try:
        import torch
        from models.heads import SpeakerHead, GenderHead, EmotionHead
        
        # 测试各个head
        speaker_head = SpeakerHead(input_dim=768, embedding_dim=192, num_speakers=100)
        gender_head = GenderHead(input_dim=768)
        emotion_head = EmotionHead(input_dim=768, num_emotions=8)
        
        # 测试前向传播
        dummy_input = torch.randn(2, 768)
        
        speaker_out = speaker_head(dummy_input)
        print(f"  Speaker head: embedding shape={speaker_out['embedding'].shape}")
        
        gender_out = gender_head(dummy_input)
        print(f"  Gender head: logits shape={gender_out.shape}")
        
        emotion_out = emotion_head(dummy_input)
        print(f"  Emotion head: logits shape={emotion_out.shape}")
        
        return True
    
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_preprocessing():
    """测试数据预处理"""
    print("Testing data preprocessing...")
    
    try:
        from data.preprocess_multidata import (
            process_ravdess, process_tess, process_emodb,
            EMOTION_MAPPING, EMOTION_NAMES
        )
        
        print(f"  Emotion mapping: {EMOTION_NAMES}")
        print(f"  RAVDESS mapping: {EMOTION_MAPPING['ravdess']}")
        print(f"  TESS mapping: {EMOTION_MAPPING['tess']}")
        print(f"  EMO-DB mapping: {EMOTION_MAPPING['emodb']}")
        
        # 测试处理函数是否存在
        assert callable(process_ravdess)
        assert callable(process_tess)
        assert callable(process_emodb)
        
        return True
    
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_git_auto_push():
    """测试Git自动上传"""
    print("Testing git auto-push...")
    
    try:
        from utils.git_auto_push import test_git_connection, load_config
        
        # 测试配置加载
        config = load_config()
        print(f"  Config loaded: enabled={config.get('git', {}).get('enabled', False)}")
        
        # 测试连接
        result = test_git_connection(str(PROJECT_ROOT))
        print(f"  Git installed: {result['git_installed']}")
        print(f"  Config exists: {result['config_exists']}")
        print(f"  Credentials: {result['credentials_ok']}")
        print(f"  Has changes: {result['has_changes']}")
        if result['remote_url']:
            print(f"  Remote: {result['remote_url']}")
        
        return True
    
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_docx_reader():
    """测试docx读取能力"""
    print("Testing docx reader...")
    
    mcp_path = PROJECT_ROOT / "mcp-servers" / "docx_reader.py"
    if not mcp_path.exists():
        print(f"  docx_reader.py not found")
        return False
    
    print(f"  Found: {mcp_path}")
    
    # 检查依赖
    try:
        import docx
        print(f"  python-docx: available")
    except ImportError:
        print(f"  python-docx: NOT installed (pip install python-docx)")
    
    try:
        import mammoth
        print(f"  mammoth: available")
    except ImportError:
        print(f"  mammoth: NOT installed (pip install mammoth)")
    
    return True


def test_environment():
    """测试环境配置"""
    print("Testing environment...")
    
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查transformers
    try:
        import transformers
        print(f"  Transformers version: {transformers.__version__}")
    except ImportError:
        print(f"  Transformers: NOT installed")
    
    # 检查librosa
    try:
        import librosa
        print(f"  Librosa: available")
    except ImportError:
        print(f"  Librosa: NOT installed")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test project capabilities')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'data', 'audio', 'model', 'git', 'docx', 'env'],
                       help='Test category to run')
    parser.add_argument('--list', action='store_true',
                       help='List available tests')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        print("  data    - Data loading and splits")
        print("  audio   - Audio utilities")
        print("  model   - Model loading and inference")
        print("  git     - Git auto-push")
        print("  docx    - DOCX reader")
        print("  env     - Environment check")
        print("  all     - Run all tests")
        return
    
    runner = TestRunner()
    
    # 定义测试映射
    tests = {
        'data': [test_data_loading, test_splits, test_preprocessing],
        'audio': [test_audio_utils],
        'model': [test_model_loading],
        'git': [test_git_auto_push],
        'docx': [test_docx_reader],
        'env': [test_environment],
    }
    
    # 运行测试
    if args.test == 'all':
        for category, test_funcs in tests.items():
            for test_func in test_funcs:
                runner.run_test(f"{category}/{test_func.__name__}", test_func)
    else:
        for test_func in tests.get(args.test, []):
            runner.run_test(test_func.__name__, test_func)
    
    # 打印摘要
    runner.print_summary()
    
    # 返回退出码
    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == '__main__':
    main()
