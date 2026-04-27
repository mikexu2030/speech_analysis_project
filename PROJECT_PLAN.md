# 语音四合一识别系统 — 详细实施计划

> 目标：在 MT9655 TV SoC 端侧实现说话人识别 + 年龄识别 + 性别识别 + 情绪识别
> 语言优先级：英语 > 西语 > 法/德/意/日等
> 模型约束：单模型多任务、端侧可运行、可 Demo
> 计划日期：2026-04-27

---

## 目录

1. [项目总览与架构设计](#1-项目总览与架构设计)
2. [阶段一：环境搭建与数据准备 (Day 1-3)](#2-阶段一环境搭建与数据准备-day-1-3)
3. [阶段二：模型选型与开源评测 (Day 4-7)](#3-阶段二模型选型与开源评测-day-4-7)
4. [阶段三：数据下载与制作 (Day 8-12)](#4-阶段三数据下载与制作-day-8-12)
5. [阶段四：模型训练 (Day 13-20)](#5-阶段四模型训练-day-13-20)
6. [阶段五：量化与导出 (Day 21-25)](#6-阶段五量化与导出-day-21-25)
7. [阶段六：端侧验证与Demo (Day 26-30)](#7-阶段六端侧验证与demo-day-26-30)
8. [文件清单与目录结构](#8-文件清单与目录结构)
9. [断网续执行指南](#9-断网续执行指南)
10. [风险与备选方案](#10-风险与备选方案)

---

## 1. 项目总览与架构设计

### 1.1 统一模型架构 (多任务学习)

```
[音频输入 16kHz]
      |
      v
[特征提取层] ──→ Mel Spectrogram (80-dim) 或 Raw Waveform
      |
      v
[共享编码器 Backbone]
      |  方案A: 轻量 CNN (ECAPA-TDNN 风格, ~6M 参数)
      |  方案B: DistilHuBERT 2层 (~24M 参数)
      |  方案C: 频谱学习+注意力 CNN (~5-10M 参数) ★推荐
      |
      +--→ [说话人嵌入头] ──→ Speaker Embedding (192-dim)
      |                        └── 余弦相似度比对注册声纹
      |
      +--→ [年龄回归头] ──→ Age (回归, 0-100岁)
      |
      +--→ [性别分类头] ──→ Gender (2类: male/female)
      |
      +--→ [情绪分类头] ──→ Emotion (7类: neutral/happy/sad/angry/fear/disgust/surprise)
```

### 1.2 技术路线决策

| 维度 | 选择 | 理由 |
|------|------|------|
| **骨干网络** | 频谱学习 CNN + 三维注意力 | 非Transformer，MT9655 CPU友好，RAVDESS 99.23% |
| **情绪类别** | 7类 (neutral/happy/sad/angry/fear/disgust/surprise) | 覆盖基本情绪，8类中平静↔中性易混淆，合并为7类 |
| **说话人识别** | 嵌入+余弦相似度 | 提前注册声纹库，实时比对 |
| **年龄输出** | 年龄段分类 (儿童/青年/中年/老年) + 细粒度回归 | 兼顾可用性与精度 |
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

## 2. 阶段一：环境搭建与数据准备 (Day 1-3)

### Day 1: Python 环境搭建

**目标**: 创建隔离的 conda/venv 环境，安装所有依赖

**执行步骤**:
1. 检查系统环境: `python --version`, `nvidia-smi` (如有GPU)
2. 创建虚拟环境: `python -m venv venv_speech`
3. 激活环境并安装基础包

**依赖清单 (requirements.txt)**:
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
tflite-runtime>=2.13.0  # 或 tensorflow>=2.13.0
accelerate>=0.20.0
peft>=0.4.0  # LoRA
opensmile>=2.5.0
```

**验证命令**:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import librosa; print(librosa.__version__)"
```

### Day 2: 项目目录结构创建

```
speech_analysis_project/
├── configs/                    # 配置文件
│   ├── model_config.yaml       # 模型架构配置
│   ├── train_config.yaml       # 训练超参配置
│   └── data_config.yaml        # 数据集路径配置
├── data/                       # 数据集目录
│   ├── raw/                    # 原始下载数据
│   ├── processed/              # 预处理后数据
│   └── splits/                 # 训练/验证/测试划分
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── backbone.py             # 共享编码器
│   ├── heads.py                # 各任务头
│   ├── multitask_model.py      # 多任务模型组装
│   └── speaker_encoder.py      # 说话人编码器
├── training/                   # 训练脚本
│   ├── train_multitask.py      # 主训练脚本
│   ├── train_speaker.py        # 说话人预训练
│   ├── distillation.py         # 知识蒸馏
│   └── losses.py               # 多任务损失函数
├── evaluation/                 # 评估脚本
│   ├── eval_emotion.py         # 情绪评估
│   ├── eval_speaker.py         # 说话人评估
│   ├── eval_age_gender.py      # 年龄性别评估
│   └── benchmark.py            # 综合基准测试
├── quantization/               # 量化与导出
│   ├── ptq.py                  # 训练后量化
│   ├── qat.py                  # 量化感知训练
│   ├── export_onnx.py          # ONNX导出
│   └── export_tflite.py        # TFLite导出
├── demo/                       # Demo脚本
│   ├── demo_server.py          # PC端Demo
│   ├── demo_mt9655.py          # MT9655端Demo框架
│   └── register_speaker.py     # 声纹注册工具
├── utils/                      # 工具函数
│   ├── audio_utils.py          # 音频处理
│   ├── feature_extract.py      # 特征提取
│   ├── data_loader.py          # 数据加载器
│   ├── metrics.py              # 评估指标
│   └── visualization.py        # 可视化
├── notebooks/                  # Jupyter分析笔记本
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── results_analysis.ipynb
├── docs/                       # 文档
│   ├── model_selection.md      # 模型选型报告
│   ├── training_log.md         # 训练日志
│   └── deployment_guide.md     # 部署指南
├── pretrained/                 # 预训练模型缓存
├── checkpoints/                # 训练检查点
├── outputs/                    # 输出结果
└── requirements.txt
```

**创建命令**:
```bash
mkdir -p speech_analysis_project/{configs,data/{raw,processed,splits},models,training,evaluation,quantization,demo,utils,notebooks,docs,pretrained,checkpoints,outputs}
touch speech_analysis_project/{models,training,evaluation,quantization,demo,utils}/__init__.py
```

### Day 3: 基础工具函数实现

**文件**: `utils/audio_utils.py`
**功能**:
- `load_audio(path, sr=16000)` - 加载音频并重采样
- `preprocess_audio(waveform)` - 预加重、分帧、加窗
- `extract_melspectrogram(waveform, sr=16000, n_mels=80)` - 提取Mel谱图
- `extract_mfcc(waveform, sr=16000, n_mfcc=13)` - 提取MFCC
- `apply_vad(waveform, sr=16000)` - 语音活动检测

**文件**: `utils/feature_extract.py`
**功能**:
- `compute_fbank(waveform, sr=16000)` - Filter Bank特征
- `compute_pitch(waveform, sr=16000)` - 基频提取
- `compute_energy(waveform)` - 能量特征
- `compute_zcr(waveform)` - 过零率

**文件**: `utils/metrics.py`
**功能**:
- `compute_eer(scores, labels)` - 等错误率
- `compute_uar(y_true, y_pred)` - 无加权平均召回率
- `compute_mae(y_true, y_pred)` - 平均绝对误差
- `compute_accuracy(y_true, y_pred)` - 准确率

**验证**: 用一段测试音频验证所有特征提取函数输出shape正确

---

## 3. 阶段二：模型选型与开源评测 (Day 4-7)

### Day 4: 开源情绪识别模型评测

**评测模型清单**:
1. **Emotion2Vec+ Large** (ModelScope: iic/emotion2vec_plus_large)
2. **Emotion2Vec+ Base** (HuggingFace: emotion2vec/emotion2vec_plus_base)
3. **HuBERT-Large SUPERB-ER** (HuggingFace: superb/hubert-large-superb-er)
4. **Wav2Vec2-Large SUPERB-ER** (HuggingFace: superb/wav2vec2-large-superb-er)
5. **Wav2Vec2-XLSR 英语7类** (HuggingFace: r-f/wav2vec-english-speech-emotion-recognition)
6. **DistilHuBERT** (HuggingFace: ntu-spml/distilhubert)

**评测数据集**:
- RAVDESS (8类情绪，固定文本) - 文本无关核心测试
- CREMA-D (6类情绪，固定文本) - 跨种族验证
- EmoDB (7类情绪，德语) - 跨语言验证

**脚本**: `evaluation/benchmark_emotion.py`

**输出**: `docs/emotion_model_benchmark.md`

**评测指标**:
- 每模型每数据集的: UA, WA, Macro-F1, 每类召回率
- 混淆矩阵可视化
- 推理延迟 (CPU单线程)
- 模型大小

### Day 5: 开源说话人/年龄/性别模型评测

**说话人模型**:
1. ECAPA-TDNN (SpeechBrain: speechbrain/spkrec-ecapa-voxceleb)
2. WavLM-Base-SV (microsoft/wavlm-base-sv)

**年龄/性别模型**:
1. audeering/wav2vec2-large-robust-age-gender
2. ECAPA-TDNN + ANN (griko/age_reg_ann_ecapa_timit)

**评测脚本**:
- `evaluation/benchmark_speaker.py` - 说话人验证EER
- `evaluation/benchmark_age_gender.py` - 年龄MAE/性别准确率

**输出**: `docs/speaker_age_gender_benchmark.md`

### Day 6: 模型选型决策

**决策矩阵**:

| 模型 | 情绪UA | 说话人EER | 性别Acc | 年龄MAE | 大小 | 延迟 | MT9655可行 |
|------|--------|-----------|---------|---------|------|------|-----------|
| Emotion2Vec+ L | ~78% | — | — | — | 1.2GB | 慢 | ❌ |
| WavLM-Large | ~75% | SOTA | — | — | 1.2GB | 慢 | ❌ |
| ECAPA-TDNN | — | 0.8% | — | ~5年 | 80MB | 中 | ⚠️ |
| **DistilHuBERT** | ~70% | 需微调 | 需微调 | 需微调 | 24MB | 中 | ✅ |
| **1D-CNN** | ~60% | 需训练 | 需训练 | 需训练 | 2MB | 快 | ✅ |
| **频谱CNN+Attn** | ~65%* | 需训练 | 需训练 | 需训练 | 5MB | 快 | ✅ |

*注: IEMOCAP估计值，RAVDESS上更高

**决策结论**:
- **主路线**: 频谱学习CNN + 三维注意力 (非Transformer，端侧友好)
- **备选路线**: DistilHuBERT 蒸馏 (如需更高泛化)
- **说话人**: ECAPA-TDNN小型化作为共享骨干的一部分

**输出**: `docs/model_selection.md` (详细决策文档)

### Day 7: 快速原型验证

**目标**: 用选定的骨干架构在RAVDESS上训练一个快速原型，验证 pipeline 通畅

**脚本**: `training/quick_prototype.py`
- 简化版CNN (2层Conv + 1层Attention)
- 仅在RAVDESS上训练情绪分类
- 训练10个epoch，验证pipeline

**成功标准**: 训练不报错，验证集UA > 50%

---

## 4. 阶段三：数据下载与制作 (Day 8-12)

### Day 8: 情绪数据集下载

**数据集1: RAVDESS**
- 来源: https://zenodo.org/record/1188976
- 大小: ~1GB (仅语音部分)
- 格式: 24位演员, 8类情绪, 固定2句文本
- 下载命令:
```bash
cd data/raw
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip -d ravdess/
```

**数据集2: CREMA-D**
- 来源: https://github.com/CheyneyComputerScience/CREMA-D
- 大小: ~5GB
- 格式: 91位演员, 6类情绪, 12句固定文本
- 下载:
```bash
git clone https://github.com/CheyneyComputerScience/CREMA-D.git
cd CREMA-D
# 按README下载音频文件
```

**数据集3: ESD (中英双语)**
- 来源: https://github.com/HLTSingapore/Emotional-Speech-Data
- 大小: ~3GB
- 格式: 10位说话人, 5类情绪, 中英双语
- 下载:
```bash
git clone https://github.com/HLTSingapore/Emotional-Speech-Data.git
```

**数据集4: TESS (加拿大英语)**
- 来源: Kaggle
- 格式: 2位女性, 7类情绪
- 下载:
```bash
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
```

### Day 9: 说话人/年龄/性别数据集下载

**数据集1: VoxCeleb1 (说话人)**
- 来源: https://mm.kaist.ac.kr/datasets/voxceleb/
- 需同意许可协议
- 下载脚本:
```bash
# 使用官方下载脚本
wget https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_dev_wav.zip
wget https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_test_wav.zip
```

**数据集2: Common Voice (年龄/性别/说话人)**
- 来源: https://commonvoice.mozilla.org/datasets
- 选择: English v16.1 (或最新)
- 使用HuggingFace datasets加载:
```python
from datasets import load_dataset
cv = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train")
```

**数据集3: LibriSpeech (干净语音, 辅助)**
- 来源: https://www.openslr.org/12
- 下载:
```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
```

### Day 10: 数据预处理与统一格式

**脚本**: `utils/data_loader.py` + `data/preprocess.py`

**处理流程**:
1. 统一采样率到 16kHz
2. 统一音频长度 (截断/填充到 3-5秒)
3. 提取Mel Spectrogram (80维, 25ms帧, 10ms帧移)
4. 标准化 (均值0, 方差1)
5. 标签统一映射

**情绪标签统一映射**:
```python
EMOTION_MAP = {
    # RAVDESS
    'neutral': 0, 'calm': 0,  # 合并平静到中性
    'happy': 1, 'joy': 1,
    'sad': 2, 'sadness': 2,
    'angry': 3, 'anger': 3,
    'fear': 4, 'fearful': 4, 'anxious': 4,
    'disgust': 5,
    'surprise': 6, 'surprised': 6,
}
```

**年龄分组映射**:
```python
AGE_MAP = {
    'child': 0,      # 0-12
    'teen': 1,       # 13-18
    'young_adult': 2, # 19-35
    'adult': 3,       # 36-60
    'senior': 4,      # 60+
}
```

**输出**: `data/processed/` 目录下 .npy 或 .pt 格式的特征文件

### Day 11: 数据增强管线

**脚本**: `utils/data_augmentation.py`

**增强方法**:
1. **SpecAugment**: 频域掩码 + 时域掩码
2. **噪声注入**: 添加MUSAN噪声 (babble/music/noise)
3. **速度扰动**: 0.9x, 1.0x, 1.1x
4. **音调扰动**: ±2 semitones
5. **混响**: 添加RIR (房间冲激响应)
6. **音量扰动**: ±6dB

**MUSAN数据集下载**:
```bash
wget http://www.openslr.org/resources/17/musan.tar.gz
```

**增强配置**:
```yaml
# configs/augment_config.yaml
specaug:
  freq_masks: 2
  freq_width: 15
  time_masks: 2
  time_width: 40
noise:
  snr_range: [5, 20]  # dB
speed:
  factors: [0.9, 1.0, 1.1]
pitch:
  semitones: [-2, 0, 2]
reverb:
  rir_prob: 0.5
volume:
  db_range: [-6, 6]
```

### Day 12: 数据集划分 (LOSO策略)

**脚本**: `data/create_splits.py`

**划分策略**:
- **情绪数据**: Leave-One-Speaker-Out (LOSO) 交叉验证
  - 确保训练/测试说话人不重叠
  - 验证文本无关能力
- **说话人数据**: 官方VoxCeleb1划分
- **年龄/性别数据**: 按说话人划分，确保同一说话人不出现在多个集合

**输出**:
- `data/splits/ravdess_train.json`
- `data/splits/ravdess_val.json`
- `data/splits/ravdess_test.json`
- `data/splits/crema_train.json`
- ... (类似格式)

**JSON格式**:
```json
{
  "files": [
    {
      "path": "data/processed/ravdess/Actor_01/03-01-01-01-01-01-01.wav",
      "speaker_id": "Actor_01",
      "emotion": 0,
      "gender": 0,
      "age_group": 3
    }
  ]
}
```

---

## 5. 阶段四：模型训练 (Day 13-20)

### Day 13-14: 共享编码器实现

**文件**: `models/backbone.py`

**架构**: 频谱学习 CNN + 三维注意力

```python
class SpectralBackbone(nn.Module):
    """
    输入: Mel Spectrogram [B, 1, 80, T]
    输出: 特征向量 [B, 512]
    """
    def __init__(self, n_mels=80, channels=[32, 64, 128, 256], 
                 use_channel_attn=True, use_temporal_attn=True, 
                 use_freq_attn=True):
        super().__init__()
        # 多尺度CNN
        self.conv_blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            in_ch = 1 if i == 0 else channels[i-1]
            self.conv_blocks.append(
                MultiScaleConvBlock(in_ch, ch, kernel_sizes=[3,5,7])
            )
        
        # 三维注意力
        self.channel_attn = ChannelAttention(channels[-1]) if use_channel_attn else None
        self.temporal_attn = TemporalAttention(channels[-1]) if use_temporal_attn else None
        self.freq_attn = FrequencyAttention(channels[-1]) if use_freq_attn else None
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 投影层
        self.projection = nn.Linear(channels[-1], 512)
    
    def forward(self, x):
        # x: [B, 1, 80, T]
        for block in self.conv_blocks:
            x = block(x)  # [B, C, F', T']
        
        # 注意力
        if self.channel_attn:
            x = self.channel_attn(x)
        if self.temporal_attn:
            x = self.temporal_attn(x)
        if self.freq_attn:
            x = self.freq_attn(x)
        
        # 池化
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.projection(x)  # [B, 512]
        return x
```

**关键组件**:
- `MultiScaleConvBlock`: 并行3x3, 5x5, 7x7卷积 + 拼接
- `ChannelAttention`: SE-Net风格通道注意力
- `TemporalAttention`: 时间维度注意力 (1D卷积+Sigmoid)
- `FrequencyAttention`: 频率维度注意力 (1D卷积+Sigmoid)

### Day 15: 多任务头实现

**文件**: `models/heads.py`

```python
class SpeakerHead(nn.Module):
    """说话人嵌入头"""
    def __init__(self, input_dim=512, embedding_dim=192):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)  # L2归一化

class AgeHead(nn.Module):
    """年龄回归头"""
    def __init__(self, input_dim=512, num_age_groups=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_reg = nn.Linear(128, 1)  # 回归: 具体年龄
        self.fc_cls = nn.Linear(128, num_age_groups)  # 分类: 年龄段
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        age_reg = self.fc_reg(x)  # [B, 1]
        age_cls = self.fc_cls(x)  # [B, num_groups]
        return age_reg, age_cls

class GenderHead(nn.Module):
    """性别分类头"""
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EmotionHead(nn.Module):
    """情绪分类头"""
    def __init__(self, input_dim=512, num_emotions=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_emotions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### Day 16: 多任务模型组装

**文件**: `models/multitask_model.py`

```python
class MultiTaskSpeechModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = SpectralBackbone(**config.backbone)
        self.speaker_head = SpeakerHead(**config.speaker_head)
        self.age_head = AgeHead(**config.age_head)
        self.gender_head = GenderHead(**config.gender_head)
        self.emotion_head = EmotionHead(**config.emotion_head)
    
    def forward(self, x, task=None):
        features = self.backbone(x)  # [B, 512]
        
        outputs = {}
        if task is None or task == 'speaker':
            outputs['speaker_embedding'] = self.speaker_head(features)
        if task is None or task == 'age':
            outputs['age_reg'], outputs['age_cls'] = self.age_head(features)
        if task is None or task == 'gender':
            outputs['gender'] = self.gender_head(features)
        if task is None or task == 'emotion':
            outputs['emotion'] = self.emotion_head(features)
        
        return outputs
```

### Day 17: 多任务损失函数

**文件**: `training/losses.py`

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, emotion_weight=1.0, speaker_weight=1.0, 
                 age_weight=1.0, gender_weight=1.0):
        super().__init__()
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.speaker_criterion = nn.CosineEmbeddingLoss()
        self.age_reg_criterion = nn.L1Loss()
        self.age_cls_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        
        self.weights = {
            'emotion': emotion_weight,
            'speaker': speaker_weight,
            'age_reg': age_weight,
            'age_cls': age_weight * 0.5,
            'gender': gender_weight
        }
    
    def forward(self, predictions, targets):
        loss = 0
        
        # 情绪损失
        if 'emotion' in predictions:
            loss += self.weights['emotion'] * self.emotion_criterion(
                predictions['emotion'], targets['emotion']
            )
        
        # 说话人损失 (对比学习)
        if 'speaker_embedding' in predictions:
            # AAM-Softmax 或 ArcFace
            loss += self.weights['speaker'] * compute_aam_loss(
                predictions['speaker_embedding'], targets['speaker_id']
            )
        
        # 年龄损失
        if 'age_reg' in predictions:
            loss += self.weights['age_reg'] * self.age_reg_criterion(
                predictions['age_reg'].squeeze(), targets['age']
            )
        if 'age_cls' in predictions:
            loss += self.weights['age_cls'] * self.age_cls_criterion(
                predictions['age_cls'], targets['age_group']
            )
        
        # 性别损失
        if 'gender' in predictions:
            loss += self.weights['gender'] * self.gender_criterion(
                predictions['gender'], targets['gender']
            )
        
        return loss
```

### Day 18-19: 训练脚本实现

**文件**: `training/train_multitask.py`

**训练配置** (`configs/train_config.yaml`):
```yaml
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 5
  scheduler: cosine
  
  # 多任务权重
  loss_weights:
    emotion: 1.0
    speaker: 0.8
    age: 0.5
    gender: 0.5
  
  # 数据增强
  augment_prob: 0.5
  
  # 早停
  patience: 15
  
  # 检查点
  save_every: 5
  
model:
  backbone:
    n_mels: 80
    channels: [32, 64, 128, 256]
  speaker_head:
    embedding_dim: 192
  age_head:
    num_age_groups: 5
  emotion_head:
    num_emotions: 7
```

**训练流程**:
1. 加载预训练权重 (如有)
2. 多任务数据加载 (每个batch混合不同任务的数据)
3. 前向传播
4. 计算多任务损失
5. 反向传播
6. 验证集评估
7. 保存最佳模型

**命令**:
```bash
python training/train_multitask.py --config configs/train_config.yaml --data_config configs/data_config.yaml
```

### Day 20: 训练监控与调试

**工具**:
- TensorBoard: `tensorboard --logdir checkpoints/logs/`
- 监控指标: 各任务损失、UA、EER、MAE、准确率
- 可视化: 混淆矩阵、学习曲线、特征分布

**调试检查清单**:
- [ ] 损失是否下降
- [ ] 各任务损失是否平衡 (无某个任务主导)
- [ ] 验证集是否过拟合
- [ ] 梯度是否爆炸/消失
- [ ] 学习率是否合适

---

## 6. 阶段五：量化与导出 (Day 21-25)

### Day 21: 训练后量化 (PTQ)

**脚本**: `quantization/ptq.py`

**流程**:
1. 加载训练好的FP32模型
2. 准备校准数据集 (100-500条代表性样本)
3. 插入量化观察者 (Observer)
4. 校准 (前向传播，收集统计信息)
5. 转换到量化模型
6. 验证精度损失

**PyTorch量化**:
```python
import torch.quantization

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 校准
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

torch.quantization.convert(model, inplace=True)
```

**输出**: `checkpoints/model_int8_ptq.pt`

### Day 22: 量化感知训练 (QAT)

**脚本**: `quantization/qat.py`

**流程**:
1. 从FP32模型开始
2. 插入FakeQuantize层
3. 继续训练5-10个epoch
4. 冻结量化参数
5. 导出INT8模型

```python
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# 继续训练
for epoch in range(qat_epochs):
    train(model, ...)

model.eval()
torch.quantization.convert(model, inplace=True)
```

**输出**: `checkpoints/model_int8_qat.pt`

### Day 23: ONNX导出

**脚本**: `quantization/export_onnx.py`

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 1, 80, 300)  # [B, C, F, T]

torch.onnx.export(
    model,
    dummy_input,
    "checkpoints/model.onnx",
    input_names=["mel_spectrogram"],
    output_names=["speaker_emb", "age_reg", "age_cls", "gender", "emotion"],
    dynamic_axes={
        "mel_spectrogram": {0: "batch", 3: "time"},
        "speaker_emb": {0: "batch"},
        "age_reg": {0: "batch"},
        "age_cls": {0: "batch"},
        "gender": {0: "batch"},
        "emotion": {0: "batch"}
    },
    opset_version=13
)
```

**验证**:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("checkpoints/model.onnx")
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: dummy_input.numpy()})
```

**输出**: `checkpoints/model.onnx`

### Day 24: TFLite导出

**脚本**: `quantization/export_tflite.py`

**流程**:
1. ONNX → TFLite (使用onnx-tf)
2. 或 PyTorch → TFLite (使用ai-edge-torch)

```python
# 方法1: ONNX → TFLite
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("checkpoints/model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("checkpoints/model_tf")

# 转换为TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("checkpoints/model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

with open("checkpoints/model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

**输出**: `checkpoints/model_int8.tflite`

### Day 25: 量化精度验证

**脚本**: `evaluation/eval_quantized.py`

**对比测试**:
| 模型 | 情绪UA | 说话人EER | 性别Acc | 年龄MAE | 大小 |
|------|--------|-----------|---------|---------|------|
| FP32 | — | — | — | — | — |
| PTQ-INT8 | — | — | — | — | — |
| QAT-INT8 | — | — | — | — | — |

**验收标准**:
- INT8精度损失 < 3%
- 模型大小压缩 > 3x

---

## 7. 阶段六：端侧验证与Demo (Day 26-30)

### Day 26: x86端模拟推理

**脚本**: `demo/demo_server.py`

**功能**:
- 加载TFLite/ONNX模型
- 实时音频采集 (或文件输入)
- 预处理 → 推理 → 结果输出
- 声纹注册与比对

**声纹注册流程**:
```python
# 注册说话人
def register_speaker(audio_files, speaker_name):
    embeddings = []
    for f in audio_files:
        emb = extract_embedding(f)
        embeddings.append(emb)
    
    # 平均嵌入作为声纹模板
    template = np.mean(embeddings, axis=0)
    template = template / np.linalg.norm(template)
    
    # 保存到声纹库
    speaker_db[speaker_name] = template
    np.save("speaker_db.npy", speaker_db)

# 识别说话人
def identify_speaker(audio_file, threshold=0.5):
    emb = extract_embedding(audio_file)
    
    best_match = None
    best_score = -1
    for name, template in speaker_db.items():
        score = np.dot(emb, template)  # 余弦相似度
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score > threshold:
        return best_match, best_score
    return "unknown", best_score
```

### Day 27: 推理延迟测试

**脚本**: `evaluation/benchmark_latency.py`

**测试方法**:
1. 准备100条1-3秒音频
2. 测量端到端延迟 (加载 → 预处理 → 推理 → 后处理)
3. 统计: 平均延迟、P50、P95、P99
4. 对比: CPU单核 vs 多核

**目标延迟**:
- 预处理: < 20ms
- 推理: < 300ms (1s音频)
- 后处理: < 10ms
- 端到端: < 500ms

### Day 28: MT9655适配准备

**文件**: `demo/demo_mt9655.py`

**MT9655特定适配**:
1. TFLite加载: `tf.lite.Interpreter(model_path="model_int8.tflite")`
2. 内存映射: `mmap`加载模型，减少内存占用
3. 多线程: `interpreter.set_num_threads(4)` (A73大核)
4. 输入格式: INT8输入需量化参数 (scale/zero_point)

```python
# TFLite推理示例
interpreter = tf.lite.Interpreter(
    model_path="model_int8.tflite",
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备输入 (需量化到INT8)
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
input_data = (mel_spec / input_scale + input_zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 获取输出
emotion_output = interpreter.get_tensor(output_details[4]['index'])
```

### Day 29: 综合Demo实现

**脚本**: `demo/full_pipeline_demo.py`

**功能**:
1. 音频采集 (麦克风或文件)
2. VAD检测语音段
3. 特征提取 (Mel Spectrogram)
4. 多任务推理
5. 结果展示:
   ```
   === 语音分析结果 ===
   说话人: 张三 (置信度: 0.92)
   年龄段: 青年 (19-35岁)
   性别: 男
   情绪: 快乐 (置信度: 0.85)
   ====================
   ```

### Day 30: 最终验证与报告

**验证清单**:
- [ ] 情绪识别: RAVDESS测试集UA ≥ 55%
- [ ] 说话人验证: 注册5人，EER ≤ 5%
- [ ] 性别识别: 测试集准确率 ≥ 90%
- [ ] 年龄识别: 测试集MAE ≤ 10年
- [ ] 模型大小: INT8 ≤ 50MB
- [ ] 推理延迟: x86模拟 ≤ 500ms
- [ ] 多语言: 英语测试通过

**输出报告**: `docs/final_report.md`

---

## 8. 文件清单与目录结构

### 核心代码文件 (按实现顺序)

| 序号 | 文件路径 | 功能 | 实现阶段 |
|------|----------|------|----------|
| 1 | `utils/audio_utils.py` | 音频加载/预处理 | Day 3 |
| 2 | `utils/feature_extract.py` | 特征提取 | Day 3 |
| 3 | `utils/metrics.py` | 评估指标 | Day 3 |
| 4 | `evaluation/benchmark_emotion.py` | 情绪模型评测 | Day 4 |
| 5 | `evaluation/benchmark_speaker.py` | 说话人模型评测 | Day 5 |
| 6 | `evaluation/benchmark_age_gender.py` | 年龄性别评测 | Day 5 |
| 7 | `utils/data_augmentation.py` | 数据增强 | Day 11 |
| 8 | `utils/data_loader.py` | 数据加载器 | Day 10 |
| 9 | `data/preprocess.py` | 数据预处理 | Day 10 |
| 10 | `data/create_splits.py` | 数据集划分 | Day 12 |
| 11 | `models/backbone.py` | 共享编码器 | Day 13-14 |
| 12 | `models/heads.py` | 任务头 | Day 15 |
| 13 | `models/multitask_model.py` | 多任务模型 | Day 16 |
| 14 | `training/losses.py` | 多任务损失 | Day 17 |
| 15 | `training/train_multitask.py` | 训练脚本 | Day 18-19 |
| 16 | `quantization/ptq.py` | 训练后量化 | Day 21 |
| 17 | `quantization/qat.py` | 量化感知训练 | Day 22 |
| 18 | `quantization/export_onnx.py` | ONNX导出 | Day 23 |
| 19 | `quantization/export_tflite.py` | TFLite导出 | Day 24 |
| 20 | `demo/demo_server.py` | PC端Demo | Day 26 |
| 21 | `demo/demo_mt9655.py` | MT9655 Demo | Day 28 |
| 22 | `demo/register_speaker.py` | 声纹注册 | Day 26 |
| 23 | `demo/full_pipeline_demo.py` | 完整Demo | Day 29 |

### 配置文件

| 文件 | 内容 |
|------|------|
| `configs/model_config.yaml` | 模型架构参数 |
| `configs/train_config.yaml` | 训练超参数 |
| `configs/data_config.yaml` | 数据集路径 |
| `configs/augment_config.yaml` | 数据增强参数 |
| `configs/quantize_config.yaml` | 量化参数 |

### 文档

| 文件 | 内容 |
|------|------|
| `docs/model_selection.md` | 模型选型报告 |
| `docs/emotion_model_benchmark.md` | 情绪模型评测 |
| `docs/speaker_age_gender_benchmark.md` | 说话人/年龄/性别评测 |
| `docs/training_log.md` | 训练日志 |
| `docs/deployment_guide.md` | 部署指南 |
| `docs/final_report.md` | 最终报告 |

---

## 9. 断网续执行指南

### 断网前必须完成的准备

1. **依赖包离线安装**:
   ```bash
   # 下载所有依赖到本地wheel
   pip download -r requirements.txt -d packages/
   # 离线安装时
   pip install --no-index --find-links=packages/ -r requirements.txt
   ```

2. **预训练模型缓存**:
   ```bash
   # HuggingFace模型缓存
   export HF_HOME="./pretrained/hf"
   # 提前下载所有需要的模型
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('ntu-spml/distilhubert')"
   ```

3. **数据集本地备份**:
   - 所有下载的数据集放入 `data/raw/` 并备份
   - 预处理后的数据放入 `data/processed/` 并备份

4. **代码版本控制**:
   ```bash
   git init
   git add .
   git commit -m "checkpoint: before offline execution"
   ```

### 断网后恢复执行

**检查清单**:
```bash
# 1. 检查环境
python -c "import torch; print(torch.__version__)"
python -c "import librosa; print(librosa.__version__)"

# 2. 检查数据
ls data/raw/ravdess/
ls data/processed/

# 3. 检查模型
ls pretrained/
ls checkpoints/

# 4. 从上次中断处继续
# 查看TODO列表
cat docs/progress.md
```

**各阶段断网恢复方法**:

| 中断阶段 | 恢复方法 |
|----------|----------|
| 数据下载中 | 检查已下载文件，跳过已完成的，继续剩余 |
| 数据预处理中 | 检查`data/processed/`已有文件，跳过已处理 |
| 模型训练中 | 加载最新checkpoint，`--resume`参数继续 |
| 量化中 | 重新加载FP32模型，从量化步骤继续 |
| 导出中 | 重新加载量化模型，从导出步骤继续 |

---

## 10. 风险与备选方案

### 风险矩阵

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| 多任务训练不稳定 | 中 | 高 | 调整损失权重；先单任务预训练再联合 |
| 情绪精度不达标 | 中 | 高 | 增加数据增强；使用SSL蒸馏作为教师 |
| 模型太大无法量化 | 低 | 高 | 减小backbone通道数；知识蒸馏 |
| MT9655延迟超标 | 中 | 高 | 降低输入长度；减层；INT4量化 |
| 数据集下载失败 | 中 | 中 | 使用备用镜像；缩小数据集范围 |
| 跨语言泛化差 | 高 | 中 | 增加多语言数据；使用多语言SSL |

### 备选方案

**方案B: 蒸馏SSL路线**
- 如果CNN路线精度不够，切换到DistilHuBERT
- 教师: HuBERT-Large
- 学生: 2层Transformer (~24M)
- 蒸馏后微调多任务头

**方案C: 分离模型**
- 如果单模型多任务不稳定，拆分为:
  - 模型1: 说话人识别 (ECAPA-TDNN, ~6M)
  - 模型2: 情绪+年龄+性别 (CNN, ~5M)
- 总大小仍可控

**方案D: 云端辅助**
- 端侧做简单分类 (正面/负面/中性)
- 复杂情绪走云端大模型
- 延迟取决于网络

---

## 附录A: 快速参考命令

```bash
# 环境搭建
python -m venv venv_speech
source venv_speech/bin/activate
pip install -r requirements.txt

# 数据下载
python data/download_datasets.py

# 预处理
python data/preprocess.py --config configs/data_config.yaml

# 训练
python training/train_multitask.py --config configs/train_config.yaml

# 评估
python evaluation/benchmark.py --model checkpoints/best.pt --test_set ravdess

# 量化
python quantization/ptq.py --model checkpoints/best.pt
python quantization/qat.py --model checkpoints/best.pt

# 导出
python quantization/export_tflite.py --model checkpoints/model_int8_qat.pt

# Demo
python demo/full_pipeline_demo.py --model checkpoints/model_int8.tflite
```

## 附录B: 关键超参数参考

| 参数 | 推荐值 | 范围 |
|------|--------|------|
| 学习率 | 1e-3 | 1e-4 ~ 1e-2 |
| Batch Size | 64 | 32 ~ 128 |
| Epochs | 100 | 50 ~ 200 |
| Dropout | 0.3 | 0.2 ~ 0.5 |
| 权重衰减 | 1e-4 | 1e-5 ~ 1e-3 |
| Mel维度 | 80 | 40 ~ 128 |
| 嵌入维度 | 192 | 128 ~ 256 |
| 注意力头数 | 4 | 2 ~ 8 |

---

> 本计划覆盖完整项目周期约30天，可根据实际情况调整各阶段时间分配。
> 关键路径: 数据准备 → 模型训练 → 量化导出 → 端侧验证
