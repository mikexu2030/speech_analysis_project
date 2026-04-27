"""
综合评估脚本
评估说话人EER、情绪UAR/WAR、性别准确率、年龄MAE
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.data_loader import create_dataloaders, load_split_data
from utils.metrics import (
    compute_eer, compute_uar, compute_war, compute_accuracy,
    compute_mae, compute_per_class_metrics, compute_speaker_verification_metrics,
    compute_age_metrics
)


# 标签
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
GENDER_LABELS = ['female', 'male']


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str = 'cuda'
) -> Dict:
    """
    评估模型
    
    Returns:
        包含所有任务指标的字典
    """
    model.eval()
    
    # 收集预测
    all_emotion_preds = []
    all_emotion_labels = []
    
    all_gender_preds = []
    all_gender_labels = []
    
    all_age_preds = []
    all_age_labels = []
    
    all_speaker_embeddings = []
    all_speaker_labels = []
    
    print("Running inference on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            mel_spec = batch['mel_spec'].to(device)
            
            # 前向
            outputs = model(mel_spec)
            
            # 情绪
            if 'emotion' in batch and 'emotion_logits' in outputs:
                preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
                labels = batch['emotion'].cpu().numpy()
                all_emotion_preds.extend(preds)
                all_emotion_labels.extend(labels)
            
            # 性别
            if 'gender' in batch and 'gender_logits' in outputs:
                preds = torch.argmax(outputs['gender_logits'], dim=1).cpu().numpy()
                labels = batch['gender'].cpu().numpy()
                all_gender_preds.extend(preds)
                all_gender_labels.extend(labels)
            
            # 年龄
            if 'age' in batch and 'age_value' in outputs:
                preds = outputs['age_value'].squeeze().cpu().numpy()
                labels = batch['age'].cpu().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds])
                all_age_preds.extend(preds)
                all_age_labels.extend(labels)
            
            # 说话人
            if 'speaker_id' in batch and 'speaker_embedding' in outputs:
                embeddings = outputs['speaker_embedding'].cpu().numpy()
                labels = batch['speaker_id'].cpu().numpy()
                all_speaker_embeddings.extend(embeddings)
                all_speaker_labels.extend(labels)
    
    # 计算指标
    metrics = {}
    
    # 情绪
    if all_emotion_preds:
        metrics['emotion'] = {
            'uar': float(compute_uar(np.array(all_emotion_labels), np.array(all_emotion_preds))),
            'war': float(compute_war(np.array(all_emotion_labels), np.array(all_emotion_preds))),
            'accuracy': float(compute_accuracy(np.array(all_emotion_labels), np.array(all_emotion_preds))),
            'per_class': compute_per_class_metrics(
                np.array(all_emotion_labels),
                np.array(all_emotion_preds),
                EMOTION_LABELS
            )
        }
    
    # 性别
    if all_gender_preds:
        metrics['gender'] = {
            'accuracy': float(compute_accuracy(np.array(all_gender_labels), np.array(all_gender_preds))),
            'per_class': compute_per_class_metrics(
                np.array(all_gender_labels),
                np.array(all_gender_preds),
                GENDER_LABELS
            )
        }
    
    # 年龄
    if all_age_preds:
        age_metrics = compute_age_metrics(np.array(all_age_labels), np.array(all_age_preds))
        metrics['age'] = {k: float(v) for k, v in age_metrics.items()}
    
    # 说话人
    if all_speaker_embeddings:
        # 构建验证对
        embeddings = np.array(all_speaker_embeddings)
        labels = np.array(all_speaker_labels)
        
        scores, pair_labels = build_verification_pairs(embeddings, labels, n_pairs=10000)
        
        if len(scores) > 0:
            speaker_metrics = compute_speaker_verification_metrics(scores, pair_labels)
            metrics['speaker'] = {k: float(v) for k, v in speaker_metrics.items()}
    
    return metrics


def build_verification_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 10000
) -> tuple:
    """
    构建说话人验证对
    
    返回: (scores, pair_labels)
    """
    n = len(embeddings)
    
    # 生成正样本对 (同一说话人)
    pos_pairs = []
    speaker_to_indices = {}
    for i, label in enumerate(labels):
        if label not in speaker_to_indices:
            speaker_to_indices[label] = []
        speaker_to_indices[label].append(i)
    
    for label, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i+1, min(i+5, len(indices))):
                    pos_pairs.append((indices[i], indices[j]))
                    if len(pos_pairs) >= n_pairs // 2:
                        break
                if len(pos_pairs) >= n_pairs // 2:
                    break
        if len(pos_pairs) >= n_pairs // 2:
            break
    
    # 生成负样本对 (不同说话人)
    neg_pairs = []
    while len(neg_pairs) < n_pairs // 2:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if labels[i] != labels[j]:
            neg_pairs.append((i, j))
    
    # 计算分数
    scores = []
    pair_labels = []
    
    for i, j in pos_pairs:
        score = np.dot(embeddings[i], embeddings[j])
        scores.append(score)
        pair_labels.append(1)
    
    for i, j in neg_pairs:
        score = np.dot(embeddings[i], embeddings[j])
        scores.append(score)
        pair_labels.append(0)
    
    return np.array(scores), np.array(pair_labels)


def print_evaluation_report(metrics: Dict):
    """打印评估报告"""
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    # 情绪
    if 'emotion' in metrics:
        print("\n📊 EMOTION RECOGNITION")
        print("-" * 40)
        em = metrics['emotion']
        print(f"  UAR (Unweighted Average Recall): {em['uar']:.4f}")
        print(f"  WAR (Weighted Average Recall):   {em['war']:.4f}")
        print(f"  Accuracy:                        {em['accuracy']:.4f}")
        
        print("\n  Per-class metrics:")
        for label, m in em['per_class'].items():
            print(f"    {label:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} N={int(m['support'])}")
    
    # 性别
    if 'gender' in metrics:
        print("\n👤 GENDER CLASSIFICATION")
        print("-" * 40)
        g = metrics['gender']
        print(f"  Accuracy: {g['accuracy']:.4f}")
        
        print("\n  Per-class metrics:")
        for label, m in g['per_class'].items():
            print(f"    {label:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    
    # 年龄
    if 'age' in metrics:
        print("\n📅 AGE ESTIMATION")
        print("-" * 40)
        a = metrics['age']
        print(f"  MAE:           {a['mae']:.2f} years")
        print(f"  RMSE:          {a['rmse']:.2f} years")
        print(f"  Acc (±5 yrs):  {a['acc_5yr']:.4f}")
        print(f"  Acc (±10 yrs): {a['acc_10yr']:.4f}")
    
    # 说话人
    if 'speaker' in metrics:
        print("\n🎤 SPEAKER VERIFICATION")
        print("-" * 40)
        s = metrics['speaker']
        print(f"  EER:           {s['eer']:.4f}")
        print(f"  minDCF:        {s['min_dcf']:.4f}")
        print(f"  Threshold:     {s['threshold']:.4f}")
    
    print("\n" + "=" * 80)


def save_metrics(metrics: Dict, output_path: str):
    """保存评估结果"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model')
    parser.add_argument('--data_dir', type=str, default='data/splits',
                       help='Data directory')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to evaluate')
    parser.add_argument('--output', type=str, default='outputs/evaluation_metrics.json',
                       help='Output metrics file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Multi-Task Speech Model Evaluation")
    print("=" * 80)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print(f"\nLoading model from: {args.model}")
    
    model = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded")
    
    # 加载数据
    print(f"\nLoading {args.split} data from: {args.data_dir}")
    test_data = load_split_data(args.data_dir, args.split)
    print(f"  {len(test_data)} samples")
    
    # 创建DataLoader
    _, _, test_loader = create_dataloaders(
        train_data=test_data[:1],  # 占位
        val_data=test_data[:1],     # 占位
        test_data=test_data,
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=False
    )
    
    # 评估
    metrics = evaluate_model(model, test_loader, device=device)
    
    # 打印报告
    print_evaluation_report(metrics)
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metrics(metrics, args.output)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
