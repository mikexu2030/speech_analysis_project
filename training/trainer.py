"""
训练器类
支持: 训练循环、验证循环、学习率调度、早停、检查点保存
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm

from utils.metrics import compute_uar, compute_war, compute_accuracy, compute_mae, compute_eer


class Trainer:
    """
    多任务训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # 学习率调度
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('num_epochs', 100),
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            self.scheduler = None
        
        # TensorBoard
        if HAS_TENSORBOARD and log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # 早停
        self.patience = config.get('patience', 15)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # 移动到设备
            mel_spec = batch['mel_spec'].to(self.device)
            
            # 构建目标
            targets = {}
            for key in ['speaker_id', 'emotion', 'age', 'age_group', 'gender']:
                if key in batch:
                    targets[key] = batch[key].to(self.device)
            
            # 前向传播
            outputs = self.model(mel_spec)
            
            # 计算损失
            losses = self.loss_fn(outputs, targets)
            loss = losses['total']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in task_losses:
                    task_losses[k] = 0.0
                task_losses[k] += v.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
            
            self.global_step += 1
        
        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        for k in task_losses:
            task_losses[k] /= len(self.train_loader)
        
        return {'loss': avg_loss, **task_losses}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {}
        
        # 收集预测用于计算指标
        all_emotion_preds = []
        all_emotion_labels = []
        all_gender_preds = []
        all_gender_labels = []
        all_age_preds = []
        all_age_labels = []
        all_speaker_embeddings = []
        all_speaker_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            mel_spec = batch['mel_spec'].to(self.device)
            
            targets = {}
            for key in ['speaker_id', 'emotion', 'age', 'age_group', 'gender']:
                if key in batch:
                    targets[key] = batch[key].to(self.device)
            
            # 前向
            outputs = self.model(mel_spec)
            
            # 损失
            losses = self.loss_fn(outputs, targets)
            total_loss += losses['total'].item()
            
            for k, v in losses.items():
                if k not in task_losses:
                    task_losses[k] = 0.0
                task_losses[k] += v.item()
            
            # 收集预测
            if 'emotion_logits' in outputs:
                preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
                all_emotion_preds.extend(preds)
                all_emotion_labels.extend(targets['emotion'].cpu().numpy())
            
            if 'gender_logits' in outputs:
                preds = torch.argmax(outputs['gender_logits'], dim=1).cpu().numpy()
                all_gender_preds.extend(preds)
                all_gender_labels.extend(targets['gender'].cpu().numpy())
            
            if 'age_value' in outputs and 'age' in targets:
                preds = outputs['age_value'].squeeze().cpu().numpy()
                all_age_preds.extend(preds)
                all_age_labels.extend(targets['age'].cpu().numpy())
            
            if 'speaker_embedding' in outputs and 'speaker_id' in targets:
                embeddings = outputs['speaker_embedding'].cpu().numpy()
                all_speaker_embeddings.extend(embeddings)
                all_speaker_labels.extend(targets['speaker_id'].cpu().numpy())
        
        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        for k in task_losses:
            task_losses[k] /= len(self.val_loader)
        
        metrics = {'val_loss': avg_loss, **{f'val_{k}': v for k, v in task_losses.items()}}
        
        # 计算任务指标
        if all_emotion_preds:
            metrics['val_emotion_uar'] = compute_uar(
                np.array(all_emotion_labels),
                np.array(all_emotion_preds)
            )
            metrics['val_emotion_war'] = compute_war(
                np.array(all_emotion_labels),
                np.array(all_emotion_preds)
            )
        
        if all_gender_preds:
            metrics['val_gender_acc'] = compute_accuracy(
                np.array(all_gender_labels),
                np.array(all_gender_preds)
            )
        
        if all_age_preds:
            metrics['val_age_mae'] = compute_mae(
                np.array(all_age_labels),
                np.array(all_age_preds)
            )
        
        # 说话人EER (简化计算)
        if all_speaker_embeddings and len(set(all_speaker_labels)) > 1:
            # 计算同/不同说话人的分数
            embeddings = np.array(all_speaker_embeddings)
            labels = np.array(all_speaker_labels)
            
            # 采样计算EER
            n_samples = min(len(embeddings), 1000)
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            
            scores = []
            pair_labels = []
            
            for i in range(min(100, len(indices))):
                for j in range(i+1, min(i+10, len(indices))):
                    idx_i, idx_j = indices[i], indices[j]
                    sim = np.dot(embeddings[idx_i], embeddings[idx_j])
                    scores.append(sim)
                    pair_labels.append(1 if labels[idx_i] == labels[idx_j] else 0)
            
            if scores:
                eer, _ = compute_eer(np.array(scores), np.array(pair_labels))
                metrics['val_speaker_eer'] = eer
        
        return metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        完整训练流程
        """
        num_epochs = num_epochs or self.config.get('num_epochs', 100)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 打印日志
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            
            if 'val_emotion_uar' in val_metrics:
                print(f"  Val Emotion UAR: {val_metrics['val_emotion_uar']:.4f}")
            if 'val_gender_acc' in val_metrics:
                print(f"  Val Gender Acc: {val_metrics['val_gender_acc']:.4f}")
            if 'val_age_mae' in val_metrics:
                print(f"  Val Age MAE: {val_metrics['val_age_mae']:.2f}")
            if 'val_speaker_eer' in val_metrics:
                print(f"  Val Speaker EER: {val_metrics['val_speaker_eer']:.4f}")
            
            # TensorBoard
            if self.writer:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存检查点
            save_every = self.config.get('save_every', 5)
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # 早停检查
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', is_best=True)
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        print("\nTraining completed!")
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    print("Trainer module loaded successfully")
    print("Class: Trainer - supports multi-task training with early stopping and checkpointing")
