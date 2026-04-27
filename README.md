# 语音四合一识别系统

基于深度学习的多任务语音分析系统，同时实现说话人识别、年龄识别、性别识别和情绪识别。

## 功能特性

| 功能 | 描述 | 性能目标 |
|------|------|----------|
| **说话人识别** | 声纹注册与验证 | EER ≤ 5% |
| **年龄识别** | 年龄段分类 + 细粒度回归 | MAE ≤ 10年 |
| **性别识别** | 男女二分类 | 准确率 ≥ 90% |
| **情绪识别** | 7类情绪分类 | UA ≥ 55% |

## 技术架构

```
音频输入 (16kHz)
    ↓
Mel Spectrogram (80-dim)
    ↓
频谱学习CNN + 三维注意力 (骨干网络)
    ↓
├─→ 说话人嵌入头 → 192-dim embedding
├─→ 年龄回归头 → 年龄估计
├─→ 性别分类头 → 性别预测
└─→ 情绪分类头 → 情绪识别
```

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone <repository-url>
cd speech_analysis_project

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 下载情绪数据集
python data/download_emotion_datasets.py --dataset ravdess
python data/download_emotion_datasets.py --dataset cremad

# 预处理
python data/preprocessor.py --dataset ravdess
python data/preprocessor.py --dataset cremad

# 数据集划分
python data/create_splits.py --dataset ravdess
```

### 模型训练

```bash
# 单GPU训练
python training/train.py --config configs/train_config.yaml

# 多GPU训练
python training/train.py --config configs/train_config.yaml --multi_gpu
```

### 模型导出

```bash
# PTQ量化
python quantization/ptq.py --model checkpoints/best.pt

# QAT量化
python quantization/qat.py --model checkpoints/best.pt

# 导出TFLite
python quantization/export_tflite.py --model checkpoints/model_int8_qat.pt
```

### 推理Demo

```bash
# 注册说话人
python demo/register_speaker.py --name "张三" --audio "samples/zhangsan_*.wav"

# 运行Demo
python demo/demo_pc.py --model checkpoints/best_model.pt --audio test.wav
```

## 项目结构

```
speech_analysis_project/
├── configs/              # 配置文件
├── data/                 # 数据集
│   ├── raw/             # 原始数据
│   ├── processed/       # 预处理数据
│   └── splits/          # 数据划分
├── models/              # 模型定义
├── training/            # 训练脚本
├── evaluation/          # 评估脚本
├── quantization/        # 量化与导出
├── demo/                # Demo脚本
├── utils/               # 工具函数
├── docs/                # 文档
├── checkpoints/         # 模型检查点
└── requirements.txt     # 依赖
```

## 文档

- [实施计划](IMPLEMENTATION_PLAN.md) - 详细开发计划
- [模型架构](docs/model_architecture.md) - 模型设计文档
- [训练指南](docs/training_guide.md) - 训练详细说明
- [部署指南](docs/deployment_guide.md) - 端侧部署说明

## 性能指标

### 情绪识别 (7类)

| 数据集 | UA | WA |
|--------|-----|-----|
| RAVDESS | 70%+ | 75%+ |
| CREMA-D | 65%+ | 70%+ |

### 说话人验证

| 数据集 | EER |
|--------|-----|
| VoxCeleb1 | ≤ 5% |

### 性别识别

| 数据集 | 准确率 |
|--------|--------|
| Common Voice | ≥ 90% |

### 年龄识别

| 数据集 | MAE |
|--------|-----|
| Common Voice | ≤ 10年 |

## 模型规格

| 指标 | 数值 |
|------|------|
| 参数量 | ~8M |
| FP32大小 | ~32MB |
| INT8大小 | ~8MB |
| 推理延迟 (1s音频) | < 500ms (MT9655) |

## 依赖

- Python >= 3.8
- PyTorch >= 2.0
- librosa >= 0.10
- transformers >= 4.30

完整依赖见 [requirements.txt](requirements.txt)

## 许可证

MIT License

## 致谢

- RAVDESS Dataset
- CREMA-D Dataset
- ESD Dataset
- VoxCeleb Dataset
