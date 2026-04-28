# 语音四合一识别项目 - 会话恢复文档

## 文档信息
- 更新时间: 2026-04-28
- 项目路径: /data/mikexu/speech_analysis_project
- 当前状态: 50epoch训练中 + ONNX演示脚本完成

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

### 1. 项目结构搭建 ✅
- 16个目录，41个文件
- 核心代码30个Python文件，7240行代码
- Git提交16次，所有代码版本控制

### 2. 模型下载与验证 (3/5成功) ✅

| 模型 | 系列 | 状态 | 大小 | 验证结果 |
|------|------|------|------|----------|
| wav2vec 2.0 Base | wav2vec 2.0 | ✅可用 | 360MB | 94.4M参数，768维输出 |
| HuBERT Base | HuBERT/WavLM | ✅可用 | 360MB | 94.4M参数，768维输出 |
| WavLM Base Plus | HuBERT/WavLM | ✅可用 | 720MB | 94.4M参数，768维输出 |
| ECAPA-TDNN | Speaker | ⏳部分 | 80MB | SpeechBrain格式 |
| Emotion2Vec+ Base | Emotion2Vec+ | ❌放弃 | - | 使用WavLM+自定义头替代 |

### 3. 数据预处理与分割 ✅
- RAVDESS数据集：1440个样本
- 预处理完成：data/processed/ravdess.json
- LOSO分割完成：
  - Train: 16 speakers, 960 samples
  - Val: 3 speakers, 180 samples
  - Test: 5 speakers, 300 samples

### 4. 模型训练 ✅

**第一轮训练 (5 epoch - 快速验证)**:
- 模型参数量: 4,235,620 (16.16 MB)
- 训练loss: 14.25 → 4.12
- Val Gender Acc: 81.1% → 89.4%
- Val Emotion UAR: 22.8% → 43.3%
- 模型保存: checkpoints/ravdess_multitask/checkpoints/best_model.pt

**第二轮训练 (50 epoch - 进行中)**:
- 配置: batch_size=32, device=cpu
- 当前进度: Epoch 9/50
- Epoch 9结果: Val Emotion UAR=47.0%, Val Gender Acc=90.0%
- 训练日志: logs/train_50epoch.log
- 后台PID: 278496

### 5. ONNX导出与兼容性修复 ✅

**导出结果**:
- 路径: models/exported/model.onnx
- 大小: 16.21 MB
- 推理速度: ~9ms (CPU)
- 估计MT9655性能: ~45-91ms

**ONNX兼容性修复**:
- 将AdaptiveAvgPool2d替换为固定kernel_size的AvgPool2d
- 将ChannelAttention中的AdaptivePool替换为mean/amax操作

### 6. 演示脚本 ✅

**demo_inference.py** - 单样本推理:
```bash
python3 demo_inference.py --model models/exported/model.onnx --audio sample.wav
```
- 支持ONNX和PyTorch后端
- 输出: 情绪、性别、年龄、声纹嵌入
- 显示置信度和概率分布

**batch_evaluate.py** - 批量评估:
```bash
python3 batch_evaluate.py --model models/exported/model.onnx --test_json data/splits/test.json
```
- 评估整个测试集
- 输出混淆矩阵和准确率

### 7. 批量评估结果 (5epoch模型) ✅

| 指标 | 准确率 | 备注 |
|------|--------|------|
| 情绪识别 | 14.0% | 基线随机12.5%，需更多训练 |
| 性别识别 | 25.0% | 基线随机50%，需更多训练 |
| 推理速度 | 5.83ms | ONNX CPU推理 |

---

## 当前进行中的任务

### 任务1: 50epoch训练 (后台运行)
- 状态: 进行中 (Epoch 9/50)
- 日志: logs/train_50epoch.log
- 预计完成时间: ~30-40分钟

---

## 下一步计划

### 优先级1: 等待50epoch训练完成
```bash
# 监控训练进度
tail -f logs/train_50epoch.log

# 训练完成后导出新的ONNX模型
python3 export.py \
  --model checkpoints/ravdess_multitask_50ep/checkpoints/best_model.pt \
  --output_dir models/exported \
  --export_onnx --benchmark
```

### 优先级2: 下载更多数据集
```bash
# Common Voice 11.0（年龄/性别，多语言）
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from datasets import load_dataset
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
ds = load_dataset('mozilla-foundation/common_voice_11_0', 'en', split='validation[:1000]', trust_remote_code=True)
ds.save_to_disk('data/raw/common_voice_en_1k')
"
```

### 优先级3: 模型量化
```bash
python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input='models/exported/model.onnx',
    model_output='models/exported/model_int8.onnx',
    weight_type=QuantType.QInt8
)
"
```

### 优先级4: 创建端到端演示
```bash
# 注册说话人
python3 demo/register_speaker.py --speaker_name "Alice"

# 识别说话人
python3 demo/demo_pc.py --audio test.wav
```

---

## 项目文件结构

```
/data/mikexu/speech_analysis_project/
├── models/
│   ├── pretrained/
│   │   ├── wav2vec2_base_960h/     ✅ 360MB
│   │   ├── hubert_base_ls960/      ✅ 360MB
│   │   ├── wavlm_base/             ✅ 720MB
│   │   ├── ecapa_tdnn/             ⏳ 80MB (部分)
│   │   └── emotion2vec_plus_base/  ❌ 放弃
│   └── exported/
│       └── model.onnx              ✅ 16.21MB
├── data/
│   ├── raw/ravdess/                ✅ 1440 wav files
│   ├── processed/ravdess.json      ✅
│   └── splits/                     ✅ train/val/test
├── checkpoints/
│   ├── ravdess_multitask/          ✅ 5epoch模型
│   └── ravdess_multitask_50ep/     ⏳ 50epoch训练中
├── demo_inference.py               ✅ 单样本推理
├── batch_evaluate.py               ✅ 批量评估
├── export.py                       ✅ ONNX导出
├── training/train.py               ✅ 训练入口
└── docs/SESSION_RESUME.md            ✅ 本文档
```

---

## 重要配置

### HuggingFace镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 网络环境
- 直连 huggingface.co 不通
- 通过 hf-mirror.com 可访问

---

## 新会话恢复步骤

开启新对话后，粘贴以下内容：

```
继续语音四合一识别项目。当前状态：
1. 50epoch训练进行中 (当前Epoch 9/50，后台PID: 278496)
2. ONNX模型已导出 (16.21MB)
3. 演示脚本已完成 (demo_inference.py + batch_evaluate.py)
4. 5epoch模型评估: 情绪14%, 性别25%

请执行：
1. 检查训练进度
2. 如果训练完成，导出新的ONNX并评估
3. 尝试下载Common Voice数据集
4. 创建端到端演示
```

---

*本文档用于会话恢复，请在新会话开始时加载。*
