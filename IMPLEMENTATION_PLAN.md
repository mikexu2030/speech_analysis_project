# 语音四合一识别系统 — 详细实施计划

> **目标**: 在 MT9655 TV SoC 端侧实现说话人识别 + 年龄识别 + 性别识别 + 情绪识别  
> **语言优先级**: 英语 > 西语 > 法/德/意/日  
> **模型约束**: 单模型多任务、端侧可运行、可 Demo  
> **规划日期**: 2026-04-27  
> **Git仓库**: 已初始化，定期提交

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [阶段规划](#2-阶段规划)
3. [详细任务清单](#3-详细任务清单)
4. [Git提交计划](#4-git提交计划)
5. [断网续执行指南](#5-断网续执行指南)
6. [风险与备选方案](#6-风险与备选方案)

---

## 1. 项目架构总览

### 1.1 统一模型架构 (多任务学习)

```
[音频输入 16kHz]
      |
      v
[特征提取层] ──→ Mel Spectrogram (80-dim)
      |
      v
[共享编码器 Backbone]
      |  频谱学习 CNN + 三维注意力 (~5-10M 参数)
      |
      +--→ [说话人嵌入头] ──→ Speaker Embedding (192-dim)
      |                        └── 余弦相似度比对注册声纹
      |
      +--→ [年龄回归头] ──→ Age (回归, 0-100岁)
      |
      +--→ [性别分类头] ──→ Gender (2类: male/female)
      |
      +--→ [情绪分类头] ──→ Emotion (7类)
```

### 1.2 技术路线决策

| 维度 | 选择 | 理由 |
|------|------|------|
| **骨干网络** | 频谱学习 CNN + 三维注意力 | 非Transformer，MT9655 CPU友好 |
| **情绪类别** | 7类 (neutral/happy/sad/angry/fear/disgust/surprise) | 覆盖基本情绪，平静↔中性合并 |
| **说话人识别** | 嵌入+余弦相似度 | 提前注册声纹库，实时比对 |
| **年龄输出** | 年龄段分类 + 细粒度回归 | 兼顾可用性与精度 |
| **量化方案** | INT8 PTQ + 部分QAT | TFLite/ONNX 端侧兼容 |
| **推理引擎** | TFLite (CPU) / ONNX Runtime | MT9655 NeuroPilot 兼容 |

### 1.3 关键性能指标 (KPI)

| 任务 | 最低要求 | 目标 | 测试集 |
|------|---------|------|--------|
| 情绪识别 (7类) | UA ≥ 55% | UA ≥ 70% | RAVDESS + CREMA-D |
| 说话人验证 | EER ≤ 5% | EER ≤ 2% | VoxCeleb1 |
| 性别识别 | 准确率 ≥ 90% | ≥ 95% | Common Voice |
| 年龄识别 | MAE ≤ 10年 | ≤ 7年 | Common Voice |
| 模型大小 (INT8) | ≤ 50MB | ≤ 10MB | — |
| 推理延迟 (1s音频) | ≤ 2s | ≤ 500ms | MT9655实测 |

---

## 2. 阶段规划

```
Phase 0: 环境搭建与基础工具 (Day 1-3)
Phase 1: 数据下载与预处理 (Day 4-7)
Phase 2: 模型训练 (Day 8-15)
Phase 3: 量化与导出 (Day 16-20)
Phase 4: 端侧验证与Demo (Day 21-25)
─────────────────────────────────────────
总计: 约 25 天 (可并行优化)
```

---

## 3. 详细任务清单

### Phase 0: 环境搭建与基础工具 (Day 1-3)

#### Task 0.1: 创建项目目录结构

**目标**: 建立标准化项目目录

**命令**:
```bash
mkdir -p speech_analysis_project/{configs,data/{raw,processed,splits},models,training,evaluation,quantization,demo,utils,notebooks,docs,pretrained,checkpoints,outputs}
touch speech_analysis_project/{models,training,evaluation,quantization,demo,utils}/__init__.py
cd speech_analysis_project
git init
```

**验证**:
```bash
ls -la speech_analysis_project/
# 应看到所有子目录
```

**提交**:
```bash
git add .
git commit -m "chore: initialize project directory structure"
```

---

#### Task 0.2: 创建 requirements.txt

**目标**: 定义所有Python依赖

**文件**: `requirements.txt`

**内容**:
```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
speechbrain>=0.5.15
funasr>=1.0.0
modelscope>=1.9.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0
onnx>=1.14.0
onnxruntime>=1.15.0
tflite-runtime>=2.13.0
accelerate>=0.20.0
peft>=0.4.0
opensmile>=2.5.0
```

**提交**:
```bash
git add requirements.txt
git commit -m "chore: add Python dependencies"
```

---

#### Task 0.3: 实现音频工具函数

**目标**: 实现基础音频处理功能

**文件**: `utils/audio_utils.py`

**核心函数**:
- `load_audio(path, sr=16000)` - 加载音频并重采样
- `extract_melspectrogram(waveform, sr=16000, n_mels=80)` - 提取Mel谱图
- `extract_mfcc(waveform, sr=16000, n_mfcc=13)` - 提取MFCC
- `pad_or_truncate(feature, target_length, axis=-1)` - 填充/截断
- `normalize_feature(feature)` - 标准化

**验证**:
```python
python -c "from utils.audio_utils import *; print('Audio utils OK')"
```

**提交**:
```bash
git add utils/audio_utils.py
git commit -m "feat(utils): add audio processing utilities"
```

---

#### Task 0.4: 实现评估指标

**目标**: 实现多任务评估指标

**文件**: `utils/metrics.py`

**核心函数**:
- `compute_eer(scores, labels)` - 等错误率
- `compute_uar(y_true, y_pred)` - 无加权平均召回率
- `compute_mae(y_true, y_pred)` - 平均绝对误差
- `compute_accuracy(y_true, y_pred)` - 准确率

**提交**:
```bash
git add utils/metrics.py
git commit -m "feat(utils): add evaluation metrics"
```

---

#### Task 0.5: 实现数据增强

**目标**: 实现音频和频谱增强

**文件**: `utils/data_augmentation.py`

**类**:
- `AudioAugmentor` - 音频增强 (速度/音调/噪声/音量)
- `SpecAugment` - 频谱增强 (时域/频域掩码)

**提交**:
```bash
git add utils/data_augmentation.py
git commit -m "feat(utils): add data augmentation"
```

---

### Phase 1: 数据下载与预处理 (Day 4-7)

#### Task 1.1: 下载情绪数据集脚本

**目标**: 创建RAVDESS/CREMA-D/ESD下载脚本

**文件**: `data/download_emotion_datasets.py`

**功能**:
- 下载RAVDESS (https://zenodo.org/record/1188976)
- 下载CREMA-D (GitHub)
- 下载ESD (GitHub)
- 自动解压到 `data/raw/`

**命令**:
```bash
python data/download_emotion_datasets.py --dataset ravdess
python data/download_emotion_datasets.py --dataset cremad
python data/download_emotion_datasets.py --dataset esd
```

**提交**:
```bash
git add data/download_emotion_datasets.py
git commit -m "feat(data): add emotion dataset download scripts"
```

---

#### Task 1.2: 下载说话人/年龄/性别数据集脚本

**目标**: 创建VoxCeleb/Common Voice下载脚本

**文件**: `data/download_speaker_datasets.py`

**功能**:
- 下载VoxCeleb1 (说话人)
- 下载Common Voice (年龄/性别)

**提交**:
```bash
git add data/download_speaker_datasets.py
git commit -m "feat(data): add speaker dataset download scripts"
```

---

#### Task 1.3: 实现数据预处理器

**目标**: 统一数据格式和预处理

**文件**: `data/preprocessor.py`

**功能**:
- 统一采样率到16kHz
- 提取Mel Spectrogram (80维)
- 标准化
- 保存为numpy格式

**标签映射**:
```python
EMOTION_MAP = {
    'neutral': 0, 'calm': 0,
    'happy': 1, 'joy': 1,
    'sad': 2, 'sadness': 2,
    'angry': 3, 'anger': 3,
    'fear': 4, 'fearful': 4,
    'disgust': 5,
    'surprise': 6, 'surprised': 6,
}
```

**提交**:
```bash
git add data/preprocessor.py
git commit -m "feat(data): add data preprocessor"
```

---

#### Task 1.4: 实现数据加载器

**目标**: PyTorch Dataset和DataLoader

**文件**: `utils/data_loader.py`

**类**: `SpeechDataset(Dataset)`

**功能**:
- 加载预处理后的数据
- 支持数据增强
- 返回 (mel_spec, labels) 元组

**提交**:
```bash
git add utils/data_loader.py
git commit -m "feat(utils): add PyTorch data loader"
```

---

#### Task 1.5: 数据集划分

**目标**: LOSO (Leave-One-Speaker-Out) 划分

**文件**: `data/create_splits.py`

**功能**:
- 按说话人划分训练/验证/测试集
- 确保同一说话人不出现在多个集合
- 输出JSON格式的索引文件

**输出**:
- `data/splits/ravdess_train.json`
- `data/splits/ravdess_val.json`
- `data/splits/ravdess_test.json`

**提交**:
```bash
git add data/create_splits.py data/splits/
git commit -m "feat(data): add LOSO dataset splitting"
```

---

### Phase 2: 模型训练 (Day 8-15)

#### Task 2.1: 实现骨干网络

**目标**: 频谱学习CNN + 三维注意力

**文件**: `models/backbone.py`

**类**:
- `MultiScaleConvBlock` - 多尺度卷积
- `ChannelAttention` - 通道注意力
- `TemporalAttention` - 时间注意力
- `FrequencyAttention` - 频率注意力
- `SpectralBackbone` - 完整骨干网络

**提交**:
```bash
git add models/backbone.py
git commit -m "feat(model): add spectral backbone with attention"
```

---

#### Task 2.2: 实现任务头

**目标**: 四个任务的预测头

**文件**: `models/heads.py`

**类**:
- `SpeakerHead` - 说话人嵌入 (192-dim)
- `AgeHead` - 年龄回归+分类
- `GenderHead` - 性别分类
- `EmotionHead` - 情绪分类

**提交**:
```bash
git add models/heads.py
git commit -m "feat(model): add task-specific heads"
```

---

#### Task 2.3: 实现多任务模型

**目标**: 组装完整模型

**文件**: `models/multitask_model.py`

**类**: `MultiTaskSpeechModel`

**功能**:
- 整合backbone和heads
- 支持多任务前向传播
- 支持单任务推理

**提交**:
```bash
git add models/multitask_model.py
git commit -m "feat(model): add multi-task model assembly"
```

---

#### Task 2.4: 实现损失函数

**目标**: 多任务联合损失

**文件**: `training/losses.py`

**类**:
- `AAMSoftmaxLoss` - 说话人损失
- `MultiTaskLoss` - 联合损失

**提交**:
```bash
git add training/losses.py
git commit -m "feat(training): add multi-task loss functions"
```

---

#### Task 2.5: 实现训练器

**目标**: 完整的训练循环

**文件**: `training/trainer.py`

**类**: `Trainer`

**功能**:
- 训练循环
- 验证循环
- 学习率调度
- 早停
- 检查点保存

**提交**:
```bash
git add training/trainer.py
git commit -m "feat(training): add trainer class"
```

---

#### Task 2.6: 实现训练脚本

**目标**: 可运行的训练入口

**文件**: `training/train.py`

**功能**:
- 解析命令行参数
- 加载配置
- 初始化模型/数据/优化器
- 启动训练

**命令**:
```bash
python training/train.py --config configs/train_config.yaml
```

**提交**:
```bash
git add training/train.py configs/train_config.yaml
git commit -m "feat(training): add training script"
```

---

#### Task 2.7: 训练模型

**目标**: 实际训练得到模型权重

**命令**:
```bash
python training/train.py \
    --config configs/train_config.yaml \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001
```

**输出**: `checkpoints/best_model.pt`

**提交**:
```bash
git add checkpoints/best_model.pt
git commit -m "model: trained multi-task model (UA=xx%, EER=xx%)"
```

---

### Phase 3: 量化与导出 (Day 16-20)

#### Task 3.1: 实现PTQ量化

**目标**: 训练后INT8量化

**文件**: `quantization/ptq.py`

**功能**:
- 加载FP32模型
- 校准数据集准备
- INT8量化
- 精度验证

**提交**:
```bash
git add quantization/ptq.py
git commit -m "feat(quant): add PTQ quantization"
```

---

#### Task 3.2: 实现QAT量化

**目标**: 量化感知训练

**文件**: `quantization/qat.py`

**功能**:
- 从FP32模型开始
- 插入FakeQuantize
- 继续训练5-10 epoch
- 导出INT8模型

**提交**:
```bash
git add quantization/qat.py
git commit -m "feat(quant): add QAT quantization"
```

---

#### Task 3.3: 实现ONNX导出

**目标**: 导出ONNX格式

**文件**: `quantization/export_onnx.py`

**功能**:
- 加载PyTorch模型
- 导出ONNX
- 验证ONNX推理

**输出**: `checkpoints/model.onnx`

**提交**:
```bash
git add quantization/export_onnx.py
git commit -m "feat(export): add ONNX export"
```

---

#### Task 3.4: 实现TFLite导出

**目标**: 导出TFLite INT8格式

**文件**: `quantization/export_tflite.py`

**功能**:
- ONNX转TFLite
- 或PyTorch直接转TFLite
- INT8量化
- 验证TFLite推理

**输出**: `checkpoints/model_int8.tflite`

**提交**:
```bash
git add quantization/export_tflite.py checkpoints/model_int8.tflite
git commit -m "feat(export): add TFLite INT8 export"
```

---

### Phase 4: 端侧验证与Demo (Day 21-25)

#### Task 4.1: 实现声纹注册工具

**目标**: 说话人注册功能

**文件**: `demo/register_speaker.py`

**功能**:
- 加载音频
- 提取嵌入
- 保存到声纹库

**命令**:
```bash
python demo/register_speaker.py --name "张三" --audio "samples/zhangsan_*.wav"
```

**提交**:
```bash
git add demo/register_speaker.py
git commit -m "feat(demo): add speaker registration tool"
```

---

#### Task 4.2: 实现PC端Demo

**目标**: 完整推理Demo

**文件**: `demo/demo_pc.py`

**功能**:
- 加载模型
- 音频输入
- 多任务推理
- 结果展示

**命令**:
```bash
python demo/demo_pc.py --model checkpoints/best_model.pt --audio test.wav
```

**提交**:
```bash
git add demo/demo_pc.py
git commit -m "feat(demo): add PC demo"
```

---

#### Task 4.3: 实现MT9655适配Demo

**目标**: 端侧推理代码

**文件**: `demo/demo_mt9655.py`

**功能**:
- TFLite加载
- 内存优化
- 多线程推理
- INT8输入处理

**提交**:
```bash
git add demo/demo_mt9655.py
git commit -m "feat(demo): add MT9655 demo"
```

---

#### Task 4.4: 实现延迟测试

**目标**: 推理性能基准

**文件**: `evaluation/benchmark_latency.py`

**功能**:
- 端到端延迟测试
- 预处理/推理/后处理分解
- 统计P50/P95/P99

**提交**:
```bash
git add evaluation/benchmark_latency.py
git commit -m "feat(eval): add latency benchmark"
```

---

#### Task 4.5: 实现综合评估

**目标**: 完整评估报告

**文件**: `evaluation/evaluate.py`

**功能**:
- 情绪UA/WA
- 说话人EER
- 性别准确率
- 年龄MAE

**提交**:
```bash
git add evaluation/evaluate.py
git commit -m "feat(eval): add comprehensive evaluation"
```

---

## 4. Git提交计划

### 提交规范

| 类型 | 用途 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(model): add attention module` |
| `fix` | 修复 | `fix(data): handle missing audio files` |
| `chore` | 杂项 | `chore: add requirements.txt` |
| `docs` | 文档 | `docs: update README` |
| `refactor` | 重构 | `refactor: simplify loss calculation` |
| `model` | 模型权重 | `model: trained v1.0` |

### 提交时间线

```
Day 1:  chore: initialize project directory structure
        chore: add Python dependencies
        feat(utils): add audio processing utilities
        feat(utils): add evaluation metrics
        feat(utils): add data augmentation

Day 4:  feat(data): add emotion dataset download scripts
        feat(data): add speaker dataset download scripts
        feat(data): add data preprocessor
        feat(utils): add PyTorch data loader
        feat(data): add LOSO dataset splitting

Day 8:  feat(model): add spectral backbone with attention
        feat(model): add task-specific heads
        feat(model): add multi-task model assembly
        feat(training): add multi-task loss functions
        feat(training): add trainer class
        feat(training): add training script
        model: trained multi-task model

Day 16: feat(quant): add PTQ quantization
        feat(quant): add QAT quantization
        feat(export): add ONNX export
        feat(export): add TFLite INT8 export

Day 21: feat(demo): add speaker registration tool
        feat(demo): add PC demo
        feat(demo): add MT9655 demo
        feat(eval): add latency benchmark
        feat(eval): add comprehensive evaluation
```

---

## 5. 断网续执行指南

### 断网前必须完成

1. **依赖包离线安装包**:
   ```bash
   pip download -r requirements.txt -d packages/
   ```

2. **预训练模型缓存**:
   ```bash
   export HF_HOME="./pretrained/hf"
   # 提前下载所有需要的模型
   ```

3. **数据集本地备份**:
   - 所有下载的数据集放入 `data/raw/`
   - 预处理后的数据放入 `data/processed/`

4. **Git提交**:
   ```bash
   git add .
   git commit -m "checkpoint: before offline execution"
   ```

### 断网后恢复

```bash
# 1. 检查环境
python -c "import torch; print(torch.__version__)"

# 2. 检查数据
ls data/raw/ravdess/
ls data/processed/

# 3. 检查模型
ls checkpoints/

# 4. 查看进度
git log --oneline -10
```

---

## 6. 风险与备选方案

### 风险矩阵

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| 多任务训练不稳定 | 中 | 高 | 调整损失权重；先单任务预训练 |
| 情绪精度不达标 | 中 | 高 | 增加数据增强；使用SSL蒸馏 |
| 模型太大 | 低 | 高 | 减小backbone通道数；知识蒸馏 |
| MT9655延迟超标 | 中 | 高 | 降低输入长度；减层；INT4量化 |
| 数据集下载失败 | 中 | 中 | 使用备用镜像；缩小数据集 |

### 备选方案

**方案B: 蒸馏SSL路线**
- 教师: HuBERT-Large
- 学生: 2层Transformer (~24M)

**方案C: 分离模型**
- 模型1: 说话人 (ECAPA-TDNN, ~6M)
- 模型2: 情绪+年龄+性别 (CNN, ~5M)

---

## 附录: 快速命令参考

```bash
# 环境搭建
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 数据下载
python data/download_emotion_datasets.py --dataset ravdess
python data/preprocessor.py --dataset ravdess

# 训练
python training/train.py --config configs/train_config.yaml

# 量化
python quantization/ptq.py --model checkpoints/best.pt
python quantization/qat.py --model checkpoints/best.pt

# 导出
python quantization/export_tflite.py --model checkpoints/model_int8_qat.pt

# Demo
python demo/demo_pc.py --model checkpoints/best.pt --audio test.wav
```

---

> **注意**: 本计划应与代码同步更新。每次重大变更后更新文档并提交Git。
