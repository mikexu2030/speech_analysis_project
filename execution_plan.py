#!/usr/bin/env python3
"""
数据集下载与模型评测执行计划
用于断网续执行
"""

import os
import sys
import json
import time
from pathlib import Path

# 项目根目录
BASE_DIR = "/data/mikexu/speech_analysis_project"
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 数据集配置
DATASETS = {
    "ravdess": {
        "name": "RAVDESS",
        "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        "size_mb": 208,
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "language": "en",
        "actors": 24,
        "samples": 2452,
        "priority": 1,  # 最高优先级
        "download_method": "direct"
    },
    "cremad": {
        "name": "CREMA-D",
        "url": "https://github.com/CheyneyComputerScience/CREMA-D",
        "size_mb": 1200,
        "emotions": ["neutral", "happy", "sad", "angry", "fear", "disgust"],
        "language": "en",
        "actors": 91,
        "samples": 7442,
        "priority": 2,
        "download_method": "manual"  # 需要手动下载或Kaggle
    },
    "esd": {
        "name": "ESD",
        "url": "https://github.com/HLTSingapore/Emotional-Speech-Data",
        "size_mb": 2900,
        "emotions": ["neutral", "happy", "angry", "sad", "surprise"],
        "language": "en+zh",
        "actors": 20,
        "samples": 17500,
        "priority": 3,
        "download_method": "manual"
    },
    "iemocap": {
        "name": "IEMOCAP",
        "url": "https://sail.usc.edu/iemocap/iemocap_release.htm",
        "size_mb": 12000,
        "emotions": ["neutral", "happy", "sad", "angry", "excited", "frustrated"],
        "language": "en",
        "actors": 10,
        "samples": 10039,
        "priority": 4,
        "download_method": "request"  # 需要申请
    },
    "emodb": {
        "name": "EmoDB",
        "url": "http://emodb.bilderbar.info/download/",
        "size_mb": 30,
        "emotions": ["neutral", "happy", "sad", "angry", "fear", "disgust", "boredom"],
        "language": "de",
        "actors": 10,
        "samples": 535,
        "priority": 5,
        "download_method": "direct"
    }
}

# 模型评测配置
MODELS_TO_EVALUATE = {
    "emotion2vec_plus_large": {
        "name": "Emotion2Vec+ Large",
        "source": "modelscope",
        "model_id": "iic/emotion2vec_plus_large",
        "tasks": ["emotion"],
        "priority": 1,
        "size_mb": 316,
        "requires_gpu": True
    },
    "emotion2vec_plus_base": {
        "name": "Emotion2Vec+ Base",
        "source": "huggingface",
        "model_id": "emotion2vec/emotion2vec_plus_base",
        "tasks": ["emotion"],
        "priority": 2,
        "size_mb": 95,
        "requires_gpu": False
    },
    "wav2vec2_large": {
        "name": "wav2vec 2.0 Large",
        "source": "huggingface",
        "model_id": "facebook/wav2vec2-large-960h",
        "tasks": ["emotion"],
        "priority": 3,
        "size_mb": 315,
        "requires_gpu": True
    },
    "hubert_large": {
        "name": "HuBERT Large",
        "source": "huggingface",
        "model_id": "facebook/hubert-large-ls960-ft",
        "tasks": ["emotion"],
        "priority": 4,
        "size_mb": 316,
        "requires_gpu": True
    },
    "ecapa_tdnn": {
        "name": "ECAPA-TDNN",
        "source": "speechbrain",
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "tasks": ["speaker"],
        "priority": 5,
        "size_mb": 6.2,
        "requires_gpu": False
    },
    "wavlm_base_sv": {
        "name": "WavLM Base SV",
        "source": "huggingface",
        "model_id": "microsoft/wavlm-base-sv",
        "tasks": ["speaker"],
        "priority": 6,
        "size_mb": 95,
        "requires_gpu": False
    }
}


def check_disk_space():
    """检查磁盘空间"""
    import shutil
    stat = shutil.disk_usage(BASE_DIR)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    print(f"磁盘空间: 总计 {total_gb:.1f}GB, 可用 {free_gb:.1f}GB")
    return free_gb


def check_dataset_status():
    """检查数据集下载状态"""
    print("\n" + "="*60)
    print("数据集下载状态检查")
    print("="*60)
    
    status = {}
    for key, info in DATASETS.items():
        dataset_dir = os.path.join(DATA_DIR, key)
        exists = os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0
        
        # 检查是否有音频文件
        has_audio = False
        if exists:
            for root, dirs, files in os.walk(dataset_dir):
                if any(f.endswith(('.wav', '.mp3', '.flac')) for f in files):
                    has_audio = True
                    break
        
        status[key] = {
            "exists": exists,
            "has_audio": has_audio,
            "info": info
        }
        
        status_str = "✅ 已下载" if has_audio else "❌ 未下载"
        print(f"{info['name']:15s} {status_str} (优先级: {info['priority']}, 大小: {info['size_mb']}MB)")
    
    return status


def create_evaluation_script():
    """创建模型评测脚本"""
    script_path = os.path.join(BASE_DIR, "evaluation", "run_real_benchmark.py")
    
    script_content = '''#!/usr/bin/env python3
"""
实际模型评测脚本
使用真实模型在真实数据上运行评测
"""

import os
import sys
import json
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
BASE_DIR = "/data/mikexu/speech_analysis_project"
sys.path.insert(0, BASE_DIR)

# 评测配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000


def load_audio(path, sr=SAMPLE_RATE):
    """加载音频文件"""
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
    return waveform


def evaluate_emotion2vec(model_id, dataset_path, output_path):
    """
    评测 Emotion2Vec 模型
    
    Args:
        model_id: 模型ID (如 "iic/emotion2vec_plus_large")
        dataset_path: 数据集路径
        output_path: 输出结果路径
    """
    print(f"\\n{'='*60}")
    print(f"评测模型: {model_id}")
    print(f"数据集: {dataset_path}")
    print(f"{'='*60}")
    
    try:
        # 导入模型
        from funasr import AutoModel
        
        model = AutoModel(
            model=model_id,
            disable_update=True
        )
        
        # 查找所有音频文件
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(Path(dataset_path).rglob(ext))
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        # 运行推理
        results = []
        for audio_file in tqdm(audio_files[:100], desc="Processing"):  # 先评测100个样本
            try:
                # 加载音频
                waveform = load_audio(str(audio_file))
                
                # 推理
                result = model.generate(
                    input=str(audio_file),
                    output_dir=None,
                    granularity="utterance"
                )
                
                results.append({
                    "file": str(audio_file),
                    "result": result
                })
                
            except Exception as e:
                print(f"处理 {audio_file} 失败: {e}")
                continue
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"评测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_with_transformers(model_id, dataset_path, output_path):
    """
    使用 transformers 库评测模型
    
    Args:
        model_id: HuggingFace 模型ID
        dataset_path: 数据集路径
        output_path: 输出结果路径
    """
    print(f"\\n{'='*60}")
    print(f"评测模型: {model_id}")
    print(f"数据集: {dataset_path}")
    print(f"{'='*60}")
    
    try:
        from transformers import AutoProcessor, AutoModelForAudioClassification
        
        # 加载模型
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        model = model.to(DEVICE)
        model.eval()
        
        # 查找音频文件
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(Path(dataset_path).rglob(ext))
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        # 运行推理
        results = []
        for audio_file in tqdm(audio_files[:100], desc="Processing"):
            try:
                # 加载音频
                waveform = load_audio(str(audio_file))
                
                # 预处理
                inputs = processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt"
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=-1).item()
                    confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
                
                results.append({
                    "file": str(audio_file),
                    "predicted_class": predicted_class,
                    "confidence": confidence
                })
                
            except Exception as e:
                print(f"处理 {audio_file} 失败: {e}")
                continue
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"结果已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"评测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run real model benchmark')
    parser.add_argument('--model', type=str, required=True,
                       choices=['emotion2vec_large', 'emotion2vec_base', 'wav2vec2', 'hubert'],
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ravdess', 'cremad', 'esd', 'iemocap'],
                       help='Dataset to use')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # 数据集路径
    dataset_path = os.path.join(BASE_DIR, "data", "raw", args.dataset)
    
    if not os.path.exists(dataset_path):
        print(f"数据集不存在: {dataset_path}")
        print("请先下载数据集")
        sys.exit(1)
    
    # 输出路径
    if args.output is None:
        args.output = os.path.join(
            BASE_DIR, "outputs",
            f"{args.model}_{args.dataset}_results.json"
        )
    
    # 模型配置
    model_configs = {
        "emotion2vec_large": ("iic/emotion2vec_plus_large", evaluate_emotion2vec),
        "emotion2vec_base": ("emotion2vec/emotion2vec_plus_base", evaluate_emotion2vec),
        "wav2vec2": ("facebook/wav2vec2-large-960h", evaluate_with_transformers),
        "hubert": ("facebook/hubert-large-ls960-ft", evaluate_with_transformers)
    }
    
    model_id, evaluate_fn = model_configs[args.model]
    
    # 运行评测
    success = evaluate_fn(model_id, dataset_path, args.output)
    
    if success:
        print("\\n✅ 评测完成!")
    else:
        print("\\n❌ 评测失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"评测脚本已创建: {script_path}")
    return script_path


def print_execution_plan():
    """打印执行计划"""
    print("\n" + "="*60)
    print("语音四合一识别项目 - 执行计划")
    print("="*60)
    
    print("\n【阶段1: 数据集下载】")
    print("-" * 60)
    for key, info in sorted(DATASETS.items(), key=lambda x: x[1]['priority']):
        method_icon = {
            "direct": "🌐",
            "manual": "👤",
            "request": "📝"
        }.get(info['download_method'], "❓")
        
        print(f"{method_icon} {info['name']}")
        print(f"   大小: {info['size_mb']}MB | 语言: {info['language']} | 样本: {info['samples']}")
        print(f"   方法: {info['download_method']} | URL: {info['url']}")
        print()
    
    print("\n【阶段2: 模型评测】")
    print("-" * 60)
    for key, info in sorted(MODELS_TO_EVALUATE.items(), key=lambda x: x[1]['priority']):
        gpu_icon = "🔥" if info['requires_gpu'] else "❄️"
        print(f"{gpu_icon} {info['name']}")
        print(f"   来源: {info['source']} | 大小: {info['size_mb']}MB | 任务: {', '.join(info['tasks'])}")
        print(f"   ID: {info['model_id']}")
        print()
    
    print("\n【阶段3: 模型训练】")
    print("-" * 60)
    print("待数据集和评测完成后进行")
    
    print("\n【阶段4: 量化与导出】")
    print("-" * 60)
    print("待训练完成后进行")
    
    print("\n【阶段5: 端侧验证】")
    print("-" * 60)
    print("待量化完成后进行")
    
    print("\n" + "="*60)
    print("断网续执行指南:")
    print("="*60)
    print("1. 检查数据集状态: python3 check_status.py")
    print("2. 继续下载数据集: python3 data/download_emotion_datasets.py --dataset <name>")
    print("3. 运行模型评测: python3 evaluation/run_real_benchmark.py --model <name> --dataset <name>")
    print("4. 查看评测报告: cat outputs/detailed_model_benchmark.md")
    print("="*60)


def save_status():
    """保存当前状态到JSON"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": BASE_DIR,
        "datasets": {},
        "models": {},
        "completed_steps": [
            "项目结构搭建",
            "评测报告脚本修复",
            "添加Emotion2Vec+和GMP-ATL模型",
            "生成详细评测报告"
        ],
        "next_steps": [
            "下载RAVDESS数据集",
            "下载CREMA-D数据集",
            "安装模型依赖",
            "运行Emotion2Vec+评测",
            "运行其他模型评测"
        ]
    }
    
    # 检查数据集状态
    for key, info in DATASETS.items():
        dataset_dir = os.path.join(DATA_DIR, key)
        exists = os.path.exists(dataset_dir)
        has_audio = False
        if exists:
            for root, dirs, files in os.walk(dataset_dir):
                if any(f.endswith(('.wav', '.mp3', '.flac')) for f in files):
                    has_audio = True
                    break
        
        status["datasets"][key] = {
            "exists": exists,
            "has_audio": has_audio,
            **info
        }
    
    # 保存状态文件
    status_path = os.path.join(BASE_DIR, "project_status.json")
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    print(f"\n状态已保存: {status_path}")
    return status


if __name__ == '__main__':
    # 检查磁盘空间
    free_gb = check_disk_space()
    
    # 检查数据集状态
    dataset_status = check_dataset_status()
    
    # 创建评测脚本
    script_path = create_evaluation_script()
    
    # 保存状态
    status = save_status()
    
    # 打印执行计划
    print_execution_plan()
    
    print("\n✅ 执行计划准备完成!")
    print(f"项目目录: {BASE_DIR}")
    print(f"可用空间: {free_gb:.1f}GB")
