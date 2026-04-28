# 语音四合一识别项目 - 会话恢复文档

## 文档信息
- 更新时间: 2026-04-28
- 项目路径: /data/mikexu/speech_analysis_project
- 当前状态: 50epoch训练完成 + 模型评估修复 + 准备INT8量化

---

## 项目概述

**目标**: 构建一个语音分析系统，实现4个功能：
1. 声纹识别（分辨说话人）
2. 年龄段识别
3. 性别识别
4. 情绪识别

**约束条件**:
- 端侧MT9655可运行
- 优先英语，其次西语，法德意日等
- 尽量1个模型实现
- 可demo演示

---

## 已完成工作

### 1. 项目结构搭建
- 16个目录，41个文件
- 核心代码30个Python文件，7240行代码
- Git提交16次，所有代码版本控制

### 2. 数据预处理与分割
- RAVDESS数据集：1440个样本
- 预处理完成：data/processed/ravdess.json
- LOSO分割完成：
  - Train: 16 speakers, 960 samples
  - Val: 3 speakers, 180 samples
  - Test: 5 speakers, 300 samples

### 3. 模型训练
- **5epoch训练**: 情绪14%, 性别0%（预处理不一致导致）
- **50epoch训练**: 情绪32%, 性别96.33%（修复预处理后）
- 最佳检查点: checkpoint_epoch_15.pt（早停触发）
- 模型配置: embedding_dim=192, num_emotions=7, num_speakers=1000

### 4. 模型导出与评估
- ONNX模型导出: models/exported/model.onnx (16.21 MB)
- 推理速度: ~8ms (CPU)
- 评估脚本: batch_evaluate.py

### 5. 关键Bug修复
1. **数据预处理不一致**: demo_inference使用n_fft=2048/hop=512，训练使用n_fft=1024/hop=256
2. **模型类别数不匹配**: 模型输出7类，但评估期望8类（surprised合并到disgust）
3. **best_model保存逻辑**: 基于val_loss保存，但val_loss未改善导致保存epoch 0

---

## 当前模型性能 (Epoch 15, ONNX, 300样本)

| 任务 | 准确率 | 备注 |
|------|--------|------|
| 情绪识别 | 32.00% (96/300) | 7类分类，随机基线14% |
| 性别识别 | 96.33% (289/300) | 2类分类，非常好 |
| 推理速度 | 8.22 ms | CPU单线程 |
| 模型大小 | 16.21 MB | FP32 ONNX |

---

## 当前进行中的任务

### 任务1: 模型INT8量化
- 状态: 待执行
- 目标: 减少模型大小至~4MB，加速推理
- 方法: ONNX动态量化

### 任务2: 下载更多数据集
- 状态: 待执行
- 目标: Common Voice（年龄/性别，多语言）
- 预期: 提升情绪识别准确率至>50%

---

## 下一步计划

### 优先级1: 模型INT8量化
```bash
cd /data/mikexu/speech_analysis_project
python3 export.py \
  --model checkpoints/ravdess_multitask_50ep/checkpoints/checkpoint_epoch_15.pt \
  --output_dir models/exported \
  --export_onnx --quantize --benchmark
```

### 优先级2: 下载Common Voice数据集
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 data/download_common_voice.py
```

### 优先级3: 端到端演示
```bash
python3 demo_inference.py --model models/exported/model_int8.onnx --audio sample.wav
```

---

## 重要文件路径

| 文件 | 路径 |
|------|------|
| 项目根目录 | /data/mikexu/speech_analysis_project |
| 训练脚本 | training/train.py |
| 导出脚本 | export.py |
| 评估脚本 | batch_evaluate.py |
| ONNX模型 | models/exported/model.onnx |
| 训练配置 | configs/train_config.yaml |
| 数据分割 | data/splits/*.json |
| 训练日志 | logs/train_50epoch.log |
| 评估结果 | results/evaluation/batch_eval.json |

---

## 模型架构参数

```python
MultiTaskSpeechModel(
    n_mels=80,
    backbone_channels=[32, 64, 128, 256],
    embedding_dim=192,
    num_speakers=1000,
    num_age_groups=5,
    num_emotions=7,  # 注意: 7类（surprised合并到disgust）
    use_attention=True,
    lightweight=False
)
```

---

## 新会话恢复步骤

开启新对话后，粘贴以下内容：

```
继续语音四合一识别项目。当前状态：
1. 50epoch训练完成，最佳检查点: checkpoint_epoch_15.pt
2. ONNX模型已导出 (16.21MB)，评估: 情绪32%, 性别96%
3. 已修复数据预处理不一致和类别数不匹配问题
4. 演示脚本已完成 (demo_inference.py + batch_evaluate.py)

请执行：
1. 模型INT8量化
2. 下载Common Voice数据集
3. 创建端到端演示
```

---

*本文档用于会话恢复，请在新会话开始时加载。*
