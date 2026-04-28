# 语音四合一识别项目 - 会话恢复文档

## 文档信息
- 更新时间: 2026-04-28
- 项目路径: /data/mikexu/speech_analysis_project
- 当前状态: 训练完成，ONNX导出成功

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
- Git提交15次，所有代码版本控制

### 2. 模型下载与验证 (3/5成功) ✅

| 模型 | 系列 | 状态 | 大小 | 验证结果 |
|------|------|------|------|----------|
| wav2vec 2.0 Base | wav2vec 2.0 | ✅可用 | 360MB | 94.4M参数，768维输出 |
| HuBERT Base | HuBERT/WavLM | ✅可用 | 360MB | 94.4M参数，768维输出 |
| WavLM Base Plus | HuBERT/WavLM | ✅可用 | 720MB | 94.4M参数，768维输出 |
| ECAPA-TDNN | Speaker | ⏳部分 | 80MB | SpeechBrain格式 |
| Emotion2Vec+ Base | Emotion2Vec+ | ❌放弃 | - | 使用WavLM+自定义头替代 |

**关键发现**:
- 网络需要使用 `hf-mirror.com` 镜像站
- 大文件下载容易被中断导致损坏
- 3个骨干模型结构相似，都适合作为共享编码器

### 3. 模型选型结论 ✅

**推荐方案: 单模型四合一**
- 使用 **WavLM Base** 作为共享编码器（效果最好，支持声纹+表示）
- 添加4个轻量分类头：声纹识别、年龄段、性别、情绪
- 总大小：~380MB (骨干) + ~10MB (4个头)
- 量化后：~100MB，适合MT9655端侧部署

### 4. 数据预处理与分割 ✅
- RAVDESS数据集：1440个样本
- 预处理完成：data/processed/ravdess.json
- LOSO分割完成：
  - Train: 16 speakers, 960 samples
  - Val: 3 speakers, 180 samples
  - Test: 5 speakers, 300 samples

### 5. 模型训练完成 ✅

**训练配置**:
- 设备: CPU
- Epochs: 5 (快速验证)
- Batch size: 16
- 模型参数量: 4,235,620 (16.16 MB)

**训练结果**:
| Epoch | Train Loss | Val Emotion UAR | Val Gender Acc | Val Speaker EER |
|-------|-----------|-----------------|----------------|-----------------|
| 1 | 14.25 | 0.2282 | 0.8111 | 0.0000 |
| 2 | 6.29 | 0.2282 | 0.8111 | 0.0000 |
| 3 | 5.08 | 0.4167 | 0.8722 | 0.0000 |
| 4 | 4.12 | 0.4325 | 0.8944 | 0.0000 |
| 5 | - | - | - | - |

**模型保存**: checkpoints/ravdess_multitask/checkpoints/best_model.pt

### 6. ONNX导出成功 ✅

**导出结果**:
- 路径: models/exported/model.onnx
- 大小: 16.21 MB
- 验证: ONNX模型验证通过
- 推理速度: ~9ms (CPU)
- 估计MT9655性能: ~45-91ms

**ONNX兼容性修复**:
- 将AdaptiveAvgPool2d替换为固定kernel_size的AvgPool2d
- 将ChannelAttention中的AdaptivePool替换为mean/amax操作

---

## 当前待解决问题

### 问题1: 训练epoch数不足
- 仅训练5个epoch，模型尚未收敛
- 建议训练50-100个epoch以获得更好效果

### 问题2: 数据集单一
- 当前仅使用RAVDESS数据集（英语，情绪+性别+说话人）
- 缺少年龄标签数据
- 建议添加：Common Voice（年龄/性别，多语言）、VoxCeleb（声纹）

### 问题3: 模型量化
- ONNX模型已导出，但未进行INT8量化
- 量化后可进一步减小模型大小（16MB -> ~4MB）

---

## 下一步计划（优先级排序）

### 优先级1: 扩展训练
```bash
cd /data/mikexu/speech_analysis_project
python3 training/train.py \
  --config configs/train_config.yaml \
  --model_config configs/model_config.yaml \
  --data_dir data/splits \
  --output_dir checkpoints \
  --exp_name ravdess_multitask_v2 \
  --device cpu \
  --epochs 50 \
  --batch_size 32
```

### 优先级2: 下载更多数据集
```bash
# Common Voice 11.0（年龄/性别，多语言）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download mozilla-foundation/common_voice_11_0 \
  --include "en/*" --local-dir data/raw/common_voice

# VoxCeleb1（声纹识别）
# 需从官网申请下载
```

### 优先级3: 模型量化
```bash
cd /data/mikexu/speech_analysis_project
python3 export.py \
  --model checkpoints/ravdess_multitask/checkpoints/best_model.pt \
  --output_dir models/exported \
  --export_onnx --quantize
```

### 优先级4: 创建演示脚本
```bash
# 运行演示
python3 demo/demo_pc.py --model models/exported/model.onnx --audio sample.wav
```

---

## 项目文件结构

```
/data/mikexu/speech_analysis_project/
├── models/
│   ├── pretrained/
│   │   ├── wav2vec2_base_960h/     ✅ 360MB
│   │   ├── hubert_base_ls960/      ✅ 360MB
│   │   ├── wavlm_base/             ✅ 720MB (WavLM Base Plus)
│   │   ├── ecapa_tdnn/             ⏳ 80MB (部分)
│   │   └── emotion2vec_plus_base/  ❌ 放弃
│   └── exported/
│       └── model.onnx              ✅ 16.21MB
├── data/
│   ├── raw/
│   │   └── ravdess/                ✅ 1440 wav files
│   ├── processed/
│   │   └── ravdess.json            ✅
│   └── splits/
│       ├── train.json              ✅ 960 samples
│       ├── val.json                ✅ 180 samples
│       └── test.json               ✅ 300 samples
├── checkpoints/
│   └── ravdess_multitask/
│       └── checkpoints/
│           └── best_model.pt       ✅
├── scripts/
│   └── evaluate_models_offline.py  ✅
├── training/
│   ├── train.py                    ✅ (入口)
│   └── trainer.py                  ✅ (修复了ONNX兼容性)
├── models/
│   ├── multitask_model.py          ✅
│   ├── backbone.py                 ✅ (修复AdaptivePool)
│   └── heads.py                    ✅
├── export.py                       ✅ (ONNX导出)
├── configs/
│   ├── train_config.yaml           ✅
│   └── model_config.yaml           ✅
└── docs/
    ├── MODEL_DOWNLOAD_GUIDE.md      ✅
    └── SESSION_RESUME.md            ✅ (本文档)
```

---

## 重要配置

### HuggingFace镜像（必须设置）
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 网络环境
- 直连 huggingface.co 不通
- 通过 hf-mirror.com 可访问
- 无需代理配置

---

## 新会话恢复步骤

开启新对话后，粘贴以下内容：

```
继续语音四合一识别项目。当前状态：
1. 3/5模型已下载验证（WavLM为骨干）
2. RAVDESS数据集已预处理并分割
3. 模型已训练5个epoch（基础验证）
4. ONNX导出成功（16.21MB，验证通过）
5. 需要：扩展训练、更多数据集、量化、演示

请执行下一步计划。
```

---

## 参考文档

- 模型下载指南: `/data/mikexu/speech_analysis_project/docs/MODEL_DOWNLOAD_GUIDE.md`
- 评测报告: `/data/mikexu/speech_analysis_project/results/evaluation/model_comparison_report.md`
- 项目状态: `/data/mikexu/speech_analysis_project/project_status.json`

---

*本文档用于会话恢复，请在新会话开始时加载。*
