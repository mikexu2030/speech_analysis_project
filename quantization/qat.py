"""
量化感知训练 (Quantization-Aware Training, QAT)
在训练过程中模拟量化，提高量化模型精度
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from training.losses import MultiTaskLoss
from training.trainer import Trainer
from utils.data_loader import create_dataloaders, load_split_data


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """
    准备QAT模型
    插入FakeQuantize层
    """
    print("Preparing model for QAT...")
    
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 准备QAT
    model_prepared = torch.quantization.prepare_qat(model)
    
    print("Model prepared for QAT")
    
    return model_prepared


def train_qat(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader,
    val_loader,
    config: Dict,
    device: str = 'cuda',
    output_dir: str = 'checkpoints'
) -> nn.Module:
    """
    QAT训练
    
    训练流程:
    1. 前几个epoch: 正常训练 (FP32)
    2. 中间阶段: 启用量化感知
    3. 最后阶段: 转换为INT8
    """
    print("\nStarting QAT training...")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=output_dir
    )
    
    # 训练
    trainer.train(num_epochs=config.get('qat_epochs', 10))
    
    return model


def convert_to_int8(model: nn.Module) -> nn.Module:
    """将QAT模型转换为INT8"""
    print("\nConverting to INT8...")
    
    model.eval()
    model.cpu()
    
    model_int8 = torch.quantization.convert(model)
    
    print("Conversion completed")
    
    return model_int8


def main():
    parser = argparse.ArgumentParser(description='Quantization-Aware Training')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pre-trained FP32 model')
    parser.add_argument('--data_dir', type=str, default='data/splits',
                       help='Data directory')
    parser.add_argument('--output', type=str, default='checkpoints/model_int8_qat.pt',
                       help='Output path')
    parser.add_argument('--qat_epochs', type=int, default=10,
                       help='Number of QAT epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for QAT')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Quantization-Aware Training (QAT)")
    print("=" * 60)
    
    # 加载预训练模型
    print(f"\nLoading pre-trained model from: {args.model}")
    
    model = MultiTaskSpeechModel(
        n_mels=80,
        backbone_channels=[32, 64, 128, 256],
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Pre-trained model loaded")
    
    # 准备QAT
    model_qat = prepare_qat_model(model)
    
    # 加载数据
    print(f"\nLoading data from: {args.data_dir}")
    train_data = load_split_data(args.data_dir, 'train')
    val_data = load_split_data(args.data_dir, 'val')
    
    train_loader, val_loader, _ = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=True
    )
    
    # 创建损失函数
    loss_fn = MultiTaskLoss(
        embedding_dim=192,
        num_speakers=1000,
        num_age_groups=5,
        num_emotions=7
    )
    
    # QAT配置
    qat_config = {
        'learning_rate': args.lr,
        'weight_decay': 0.0001,
        'num_epochs': args.qat_epochs,
        'patience': 5,
        'qat_epochs': args.qat_epochs
    }
    
    # 训练
    model_trained = train_qat(
        model=model_qat,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=qat_config,
        device=args.device,
        output_dir=os.path.dirname(args.output)
    )
    
    # 转换为INT8
    model_int8 = convert_to_int8(model_trained)
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model_int8.state_dict(), args.output)
    print(f"\nQAT model saved to: {args.output}")
    
    print("\nQAT completed!")


if __name__ == '__main__':
    main()
