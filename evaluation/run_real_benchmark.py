#!/usr/bin/env python3
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
    print(f"\n{'='*60}")
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
    print(f"\n{'='*60}")
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
        print("\n✅ 评测完成!")
    else:
        print("\n❌ 评测失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()
