#!/usr/bin/env python3
"""
训练脚本入口
支持: 命令行参数、配置加载、多GPU训练
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from training.losses import MultiTaskLoss
from training.trainer import Trainer
from utils.data_loader import create_dataloaders, load_split_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Multi-Task Speech Model')
    
    # 配置
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                       help='Path to model config file')
    
    # 数据
    parser.add_argument('--data_dir', type=str, default='data/splits',
                       help='Directory containing data splits')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 模型
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight model variant')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default='multitask_speech',
                       help='Experiment name')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # 命令行参数覆盖配置
    if args.epochs:
        train_config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        train_config['training']['batch_size'] = args.batch_size
    if args.lr:
        train_config['training']['learning_rate'] = args.lr
    
    # 创建输出目录
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Task Speech Model Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Output dir: {exp_dir}")
    print(f"Device: {args.device}")
    print(f"Lightweight: {args.lightweight}")
    
    # 加载数据
    print("\nLoading data...")
    try:
        train_data = load_split_data(args.data_dir, 'train')
        val_data = load_split_data(args.data_dir, 'val')
        test_data = load_split_data(args.data_dir, 'test') if os.path.exists(
            os.path.join(args.data_dir, 'test.json')
        ) else None
        
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        if test_data:
            print(f"  Test samples: {len(test_data)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data preprocessing first!")
        return
    
    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=train_config['training']['batch_size'],
        num_workers=args.num_workers,
        n_mels=train_config['data']['n_mels'],
        target_length=train_config['data']['target_length'],
        augment_train=True
    )
    
    # 创建模型
    print("\nCreating model...")
    model = MultiTaskSpeechModel(
        n_mels=model_config['model']['backbone']['n_mels'],
        backbone_channels=model_config['model']['backbone']['channels'],
        embedding_dim=model_config['model']['speaker_head']['embedding_dim'],
        num_speakers=model_config['model']['speaker_head']['num_speakers'],
        num_age_groups=model_config['model']['age_head']['num_age_groups'],
        num_emotions=model_config['model']['emotion_head']['num_emotions'],
        lightweight=args.lightweight
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"  Trainable params: {trainable_params:,}")
    
    # 创建损失函数
    loss_fn = MultiTaskLoss(
        embedding_dim=model_config['model']['speaker_head']['embedding_dim'],
        num_speakers=model_config['model']['speaker_head']['num_speakers'],
        num_age_groups=model_config['model']['age_head']['num_age_groups'],
        num_emotions=model_config['model']['emotion_head']['num_emotions'],
        weights=train_config['training']['loss_weights']
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config['training'],
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # 加载检查点
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # 开始训练
    print("\n" + "=" * 60)
    trainer.train(num_epochs=train_config['training']['num_epochs'])
    
    # 最终测试
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("Final evaluation on test set...")
        # TODO: 实现测试评估
    
    print("\nTraining completed!")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
