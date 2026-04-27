# 训练指南

## 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载数据集
python data/download_emotion_datasets.py --dataset ravdess
python data/download_speaker_datasets.py --dataset common_voice

# 预处理
python data/preprocessor.py --dataset ravdess
python data/preprocessor.py --dataset common_voice

# 划分数据集
python data/create_splits.py \
    --input data/processed/ravdess.json data/processed/common_voice_en.json \
    --output_dir data/splits \
    --method loso
```

## 训练流程

### 1. 基础训练

```bash
python training/train.py \
    --config configs/train_config.yaml \
    --model_config configs/model_config.yaml \
    --data_dir data/splits \
    --output_dir checkpoints \
    --exp_name baseline
```

### 2. 轻量模型训练

```bash
python training/train.py \
    --config configs/train_config.yaml \
    --lightweight \
    --exp_name lightweight
```

### 3. 恢复训练

```bash
python training/train.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/baseline/checkpoints/checkpoint_epoch_50.pt \
    --exp_name baseline_resume
```

## 配置说明

### 训练配置 (configs/train_config.yaml)

```yaml
training:
  batch_size: 64          # 根据GPU内存调整
  num_epochs: 100         # 最大训练轮数
  learning_rate: 0.001    # 初始学习率
  weight_decay: 0.0001    # L2正则化
  warmup_epochs: 5        # 预热轮数
  scheduler: cosine       # 学习率调度: cosine/plateau
  patience: 15            # 早停耐心值
  save_every: 5           # 每N轮保存检查点

  loss_weights:           # 多任务损失权重
    emotion: 1.0
    speaker: 0.8
    age_reg: 0.3
    age_cls: 0.3
    gender: 0.5

data:
  target_length: 300      # 频谱图目标帧数
  sample_rate: 16000      # 采样率
  n_mels: 80              # Mel滤波器数量
```

### 模型配置 (configs/model_config.yaml)

```yaml
model:
  backbone:
    n_mels: 80
    channels: [32, 64, 128, 256]  # 各阶段通道数
  
  speaker_head:
    embedding_dim: 192
    num_speakers: 1000            # 根据数据集调整
  
  age_head:
    num_age_groups: 5
  
  gender_head:
    input_dim: 512
  
  emotion_head:
    num_emotions: 7
```

## 训练监控

### TensorBoard

```bash
tensorboard --logdir checkpoints/*/logs
```

### 查看训练日志

```bash
# 实时查看
tail -f checkpoints/baseline/logs/train.log
```

## 常见问题

### 1. 显存不足

**解决方案:**
- 减小batch_size
- 使用轻量模型 `--lightweight`
- 减小target_length
- 使用混合精度训练 (添加 `--amp`)

### 2. 多任务训练不稳定

**解决方案:**
- 调整损失权重
- 先单任务预训练
- 使用梯度裁剪 (已默认启用)
- 降低学习率

### 3. 情绪识别精度低

**解决方案:**
- 增加数据增强
- 使用Focal Loss处理类别不平衡
- 增加情绪数据集 (CREMA-D, ESD)
- 使用标签平滑

### 4. 说话人EER过高

**解决方案:**
- 增加说话人数据 (VoxCeleb)
- 调整AAMSoftmax margin
- 增加embedding_dim
- 使用更大的batch size

## 超参数调优

### 学习率

| 阶段 | 学习率 | 说明 |
|------|--------|------|
| 预热 | 1e-5 → 1e-3 | 线性增长 |
| 主训练 | 1e-3 | 使用cosine衰减 |
| 微调 | 1e-4 | 降低学习率 |

### 损失权重调优

```python
# 如果某个任务欠拟合，增加其权重
weights = {
    'emotion': 1.0,    # 基础权重
    'speaker': 1.2,    # 如果EER高，增加
    'age_reg': 0.5,    # 如果MAE高，增加
    'age_cls': 0.3,
    'gender': 0.8      # 如果acc低，增加
}
```

## 训练技巧

### 1. 数据增强

已自动启用:
- 音频增强: 速度变化、音调变化、噪声添加
- 频谱增强: 时域掩码、频域掩码 (SpecAugment)

### 2. 正则化

- Dropout: 0.3-0.4
- 权重衰减: 1e-4
- 标签平滑: 0.1 (可选)

### 3. 学习率调度

推荐使用cosine:
```
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / max_epochs))
```

## 预期训练时间

| 配置 | 数据集 | 单epoch时间 | 总时间 |
|------|--------|-------------|--------|
| RTX 3090 | RAVDESS+CV | ~2min | ~3h (100 epoch) |
| RTX 2080 Ti | RAVDESS+CV | ~5min | ~8h |
| CPU | RAVDESS | ~30min | ~50h |

## 检查点管理

```bash
# 列出所有检查点
ls -lh checkpoints/baseline/checkpoints/

# 查看训练状态
cat checkpoints/baseline/logs/train.log | grep "Epoch"

# 最佳模型
checkpoints/baseline/checkpoints/best_model.pt
```
