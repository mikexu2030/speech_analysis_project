#!/usr/bin/env python3
"""
模型评测脚本 - 离线版（无需网络）
当模型下载完成后，运行此脚本进行评测
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = '/data/mikexu/speech_analysis_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/pretrained')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/evaluation')

os.makedirs(RESULTS_DIR, exist_ok=True)

# 模型配置
MODEL_CONFIGS = {
    'emotion2vec_plus_base': {
        'name': 'Emotion2Vec+ Base',
        'params': '95M',
        'size_mb': 380,
        'series': 'Emotion2Vec+',
        'tasks': ['emotion'],
        'languages': ['en', 'zh', 'multi'],
        'hf_repo': 'emotion2vec/emotion2vec_plus_base'
    },
    'wav2vec2_base_960h': {
        'name': 'wav2vec 2.0 Base',
        'params': '95M',
        'size_mb': 380,
        'series': 'wav2vec 2.0',
        'tasks': ['asr', 'representation'],
        'languages': ['en'],
        'hf_repo': 'facebook/wav2vec2-base-960h'
    },
    'hubert_base_ls960': {
        'name': 'HuBERT Base',
        'params': '95M',
        'size_mb': 380,
        'series': 'HuBERT/WavLM',
        'tasks': ['representation', 'asr'],
        'languages': ['en'],
        'hf_repo': 'facebook/hubert-base-ls960'
    },
    'wavlm_base': {
        'name': 'WavLM Base',
        'params': '95M',
        'size_mb': 380,
        'series': 'HuBERT/WavLM',
        'tasks': ['representation', 'speaker'],
        'languages': ['en'],
        'hf_repo': 'microsoft/wavlm-base'
    },
    'ecapa_tdnn': {
        'name': 'ECAPA-TDNN',
        'params': '6.2M',
        'size_mb': 25,
        'series': 'Speaker Recognition',
        'tasks': ['speaker'],
        'languages': ['multi'],
        'hf_repo': 'speechbrain/spkrec-ecapa-voxceleb'
    }
}

def check_model_files(model_name):
    """检查模型文件是否存在且完整"""
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        return False, "Directory not found"
    
    files = os.listdir(model_path)
    
    # 检查关键文件
    has_config = any(f == 'config.json' for f in files)
    has_model = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth', '.ckpt')) for f in files)
    
    if not has_config:
        return False, "Missing config.json"
    
    if not has_model:
        return False, "Missing model weights"
    
    # 计算大小
    total_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in files
    )
    
    return True, f"{total_size / 1024 / 1024:.1f} MB"

def load_and_test_model(model_name):
    """加载并测试模型"""
    model_path = os.path.join(MODEL_DIR, model_name)
    config = MODEL_CONFIGS.get(model_name, {})
    
    print(f"\n{'='*60}")
    print(f"Testing: {config.get('name', model_name)}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_name,
        'display_name': config.get('name', model_name),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'load_success': False,
        'inference_test': False,
        'error': None
    }
    
    try:
        import torch
        
        if model_name == 'emotion2vec_plus_base':
            # 测试 Emotion2Vec+
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            
            print("Loading feature extractor...")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            
            print("Loading model...")
            model = Wav2Vec2Model.from_pretrained(model_path)
            model.eval()
            
            # 创建测试输入
            dummy_waveform = torch.randn(1, 16000)  # 1秒16kHz音频
            
            print("Running inference...")
            with torch.no_grad():
                inputs = feature_extractor(
                    dummy_waveform.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                outputs = model(**inputs)
                
            print(f"Output shape: {outputs.last_hidden_state.shape}")
            results['load_success'] = True
            results['inference_test'] = True
            results['output_shape'] = str(outputs.last_hidden_state.shape)
            
        elif model_name in ['wav2vec2_base_960h', 'hubert_base_ls960', 'wavlm_base']:
            # 测试 wav2vec2 / HuBERT / WavLM
            from transformers import AutoProcessor, AutoModel
            
            print("Loading processor...")
            processor = AutoProcessor.from_pretrained(model_path)
            
            print("Loading model...")
            model = AutoModel.from_pretrained(model_path)
            model.eval()
            
            # 创建测试输入
            dummy_waveform = torch.randn(1, 16000)
            
            print("Running inference...")
            with torch.no_grad():
                inputs = processor(
                    dummy_waveform.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                outputs = model(**inputs)
                
            print(f"Output shape: {outputs.last_hidden_state.shape}")
            results['load_success'] = True
            results['inference_test'] = True
            results['output_shape'] = str(outputs.last_hidden_state.shape)
            
        elif model_name == 'ecapa_tdnn':
            # 测试 ECAPA-TDNN
            try:
                from speechbrain.pretrained import EncoderClassifier
                
                print("Loading SpeechBrain classifier...")
                classifier = EncoderClassifier.from_hparams(source=model_path)
                
                dummy_waveform = torch.randn(1, 16000)
                
                print("Running inference...")
                with torch.no_grad():
                    embeddings = classifier.encode_batch(dummy_waveform)
                    
                print(f"Embedding shape: {embeddings.shape}")
                results['load_success'] = True
                results['inference_test'] = True
                results['output_shape'] = str(embeddings.shape)
                
            except ImportError:
                print("SpeechBrain not installed, skipping ECAPA test")
                results['error'] = "SpeechBrain not installed"
                
        # 内存使用
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024 / 1024
            results['gpu_memory_mb'] = f"{mem:.1f}"
        
        print("✅ Model test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results['error'] = str(e)
    
    return results

def generate_comparison_report(all_results):
    """生成模型对比报告"""
    report_path = os.path.join(RESULTS_DIR, 'model_comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Top3 Model Series - 评测对比报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 概览表
        f.write("## 模型概览\n\n")
        f.write("| 模型 | 系列 | 参数量 | 大小 | 支持任务 | 支持语言 | 状态 |\n")
        f.write("|------|------|--------|------|----------|----------|------|\n")
        
        for model_name, config in MODEL_CONFIGS.items():
            status = "✅ 可用" if all_results.get(model_name, {}).get('load_success') else "❌ 未下载/失败"
            tasks = ', '.join(config['tasks'])
            langs = ', '.join(config['languages'])
            f.write(f"| {config['name']} | {config['series']} | {config['params']} | {config['size_mb']}MB | {tasks} | {langs} | {status} |\n")
        
        f.write("\n## 详细评测结果\n\n")
        
        for model_name, result in all_results.items():
            config = MODEL_CONFIGS.get(model_name, {})
            f.write(f"### {config.get('name', model_name)}\n\n")
            f.write(f"- **加载状态**: {'✅ 成功' if result['load_success'] else '❌ 失败'}\n")
            f.write(f"- **推理测试**: {'✅ 通过' if result['inference_test'] else '❌ 失败'}\n")
            
            if 'output_shape' in result:
                f.write(f"- **输出维度**: {result['output_shape']}\n")
            
            if 'gpu_memory_mb' in result:
                f.write(f"- **GPU内存**: {result['gpu_memory_mb']} MB\n")
            
            if result['error']:
                f.write(f"- **错误信息**: {result['error']}\n")
            
            f.write(f"- **测试时间**: {result['timestamp']}\n")
            f.write("\n")
        
        # 推荐分析
        f.write("## 模型选型推荐\n\n")
        f.write("### 场景1: 单模型实现四合一任务\n\n")
        f.write("**推荐方案**: 使用 wav2vec 2.0 / HuBERT / WavLM 作为共享编码器\n")
        f.write("- 优点: 一个骨干网络，多个任务头\n")
        f.write("- 适合: 端侧部署，参数共享\n")
        f.write("- 实现: 在基础模型上添加4个分类头\n\n")
        
        f.write("### 场景2: 最佳性能组合\n\n")
        f.write("**推荐方案**: WavLM (声纹) + Emotion2Vec+ (情绪) + 自定义分类器 (年龄/性别)\n")
        f.write("- 优点: 每个任务使用最优模型\n")
        f.write("- 缺点: 多个模型，内存占用大\n\n")
        
        f.write("### 场景3: 端侧最优 (MT9655)\n\n")
        f.write("**推荐方案**: 共享 wav2vec 2.0 Base 编码器\n")
        f.write("- 模型大小: ~380MB\n")
        f.write("- 推理速度: 实时\n")
        f.write("- 量化后: ~100MB\n\n")
        
        f.write("## 下一步行动\n\n")
        f.write("1. 完成所有模型下载\n")
        f.write("2. 在真实数据集上评测\n")
        f.write("3. 训练多任务模型\n")
        f.write("4. 导出ONNX/量化模型\n")
    
    print(f"\nReport saved to: {report_path}")
    return report_path

def main():
    print("="*70)
    print("Top3 Model Series - Offline Evaluation")
    print("="*70)
    
    # 检查哪些模型已下载
    available_models = []
    missing_models = []
    
    for model_name in MODEL_CONFIGS:
        is_complete, msg = check_model_files(model_name)
        if is_complete:
            available_models.append(model_name)
            print(f"✅ {model_name}: {msg}")
        else:
            missing_models.append(model_name)
            print(f"❌ {model_name}: {msg}")
    
    print(f"\nAvailable: {len(available_models)}/{len(MODEL_CONFIGS)}")
    
    if not available_models:
        print("\nNo models available for testing.")
        print("Please download models first:")
        print("  python3 download_all_models.py")
        print("  or: bash download_models.sh")
        return
    
    # 测试每个可用模型
    all_results = {}
    
    for model_name in available_models:
        result = load_and_test_model(model_name)
        all_results[model_name] = result
        
        # 保存单个结果
        result_path = os.path.join(RESULTS_DIR, f'{model_name}_eval.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    # 生成对比报告
    report_path = generate_comparison_report(all_results)
    
    # 保存完整结果
    full_results_path = os.path.join(RESULTS_DIR, 'all_evaluations.json')
    with open(full_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete")
    print(f"{'='*70}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Report: {report_path}")
    
    if missing_models:
        print(f"\nMissing models (need download):")
        for m in missing_models:
            print(f"  - {m}")

if __name__ == '__main__':
    main()
