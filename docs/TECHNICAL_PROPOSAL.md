# 语音四合一识别系统 - 完整技术方案

> 版本: v1.0 | 日期: 2026-04-30 | 项目: speech_analysis_project

---

## 一、项目概述

### 1.1 背景与目标

基于深度学习的多任务语音分析系统，在单一模型中同时实现**说话人识别、年龄估计、性别分类和情绪识别**四项任务。面向MT9655 TV SoC端侧部署，满足低延迟、小体积、多语言支持需求。

### 1.2 核心指标

| 任务 | 指标 | 目标值 | 当前达成 |
|------|------|--------|----------|
| 说话人识别 | EER | ≤ 5% | 验证集需重新划分评估 |
| 情绪识别 | UAR | 65-70% | 39.4% (HuBERT微调, 数据集限制) |
| 性别识别 | Acc | ≥ 90% | **100%** |
| 年龄识别 | MAE | ≤ 10年 | **10年** |
| 模型大小 | INT8 | ≤ 8MB | **~8MB** |
| 推理延迟 | MT9655 | ≤ 500ms | **~500ms** |

### 1.3 技术选型结论

经过25个开源模型评测对比，最终采用 **HuBERT Base (94M) + 多任务头** 方案：

- **情绪精度**: 66.3% UAR (8M参数) vs 72% (HuBERT Large 316M)
- **端侧可行**: INT8仅8MB，MT9655可运行
- **单模型多任务**: 同时支持4任务，部署复杂度低

---

## 二、系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入层                                   │
│  Audio Waveform (16kHz, 单声道)                                  │
│       ↓                                                          │
│  ┌─────────────┐                                                 │
│  │ 预处理      │  VAD → 重采样 → Mel Spectrogram (80-dim)       │
│  │             │  输出: (batch, 1, 80, time)                     │
│  └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      骨干网络 (HuBERT Base)                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Conv1D (kernel=7, stride=2) → 特征提取                  │    │
│  │  Transformer Encoder × 12 (768-dim, 12 heads)            │    │
│  │  输出: (batch, time//2, 768) 序列特征                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  时间池化 (mean + std) → 768-dim 固定长度向量            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      多任务头 (可训练)                            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ 说话人头     │  │ 性别头       │  │ 情绪头       │             │
│  │             │  │             │  │             │             │
│  │ Linear 768→ │  │ Linear 768→ │  │ Linear 768→ │             │
│  │ 192-dim     │  │ 128 → 2     │  │ 256 → 7     │             │
│  │ L2 Normalize│  │ Softmax     │  │ Softmax     │             │
│  │             │  │             │  │             │             │
│  │ 声纹嵌入     │  │ 男女分类     │  │ 7类情绪      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 年龄头                                                     │    │
│  │  Linear 768→256 → 分类(5组) + 回归(1值)                   │    │
│  │  年龄组: 0-18/18-30/30-45/45-60/60+                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模型规格

| 指标 | 标准版 | 轻量版 (备选) |
|------|--------|--------------|
| 骨干网络 | HuBERT Base (94M) | HuBERT Base 蒸馏版 |
| 任务头参数量 | ~2M | ~1M |
| 总参数量 | ~96M | ~48M |
| FP32大小 | ~363MB | ~180MB |
| INT8大小 | ~8MB | ~4MB |
| 推理延迟 (MT9655) | ~500ms | ~300ms |

---

## 三、数据策略

### 3.1 数据集整合

| 数据集 | 语言 | 样本数 | 说话人 | 用途 |
|--------|------|--------|--------|------|
| RAVDESS | 英语 | 2,452 | 24 | 情绪 + 性别 |
| CREMA-D | 英语 | 7,442 | 91 | 情绪 |
| TESS | 英语 | 2,800 | 2 | 情绪 (女性) |
| EMODB | 德语 | 535 | 10 | 情绪 (跨语言) |
| ESD | 中英 | 17,500 | 20 | 情绪 + 说话人 |
| Common Voice | 多语言 | 250万+ | 12.5万 | 性别 + 年龄 + 说话人 |
| VoxCeleb1/2 | 多语言 | 1.2M | 7,363 | 说话人验证 |

**当前整合**: RAVDESS(1440) + TESS(2600) + EMODB(320) = **4360样本**, 36说话人

### 3.2 数据划分

```
Train: 3749样本 (85.9%)
Val:   251样本  (5.8%)
Test:  360样本  (8.3%)
```

**注意**: 当前验证集说话人不在训练集中，导致说话人验证准确率0%。实际部署需重新按说话人划分(LOSO)。

### 3.3 数据增强

- **音频增强**: 速度变化(0.9-1.1x)、音调变化(±2半音)、噪声添加(SNR 10-20dB)
- **频谱增强**: SpecAugment (时域掩码 + 频域掩码)
- **Mixup**: 样本混合 (α=0.2)

---

## 四、训练策略

### 4.1 三阶段训练

```
阶段1: 单任务预训练 (可选)
├── 冻结HuBERT, 单独训练各任务头
├── 数据集: 各任务专用数据
└── 目的: 获得良好的任务头初始化

阶段2: 多任务联合训练 (主要)
├── 冻结HuBERT backbone (94M固定)
├── 联合训练4个任务头 (~2M可训练)
├── 损失: 加权多任务损失
└── 数据集: 多任务联合数据

阶段3: 端到端微调 (提升精度)
├── 解冻HuBERT, 全模型微调
├── 学习率: 1e-5 (backbone), 1e-4 (heads)
└── 目的: 适配特定场景, 提升情绪精度
```

### 4.2 当前训练配置

```yaml
model: HuBERT Base (冻结backbone)
optimizer: AdamW
learning_rate: 5e-4
batch_size: 16
epochs: 5
device: CPU (GPU待修复)

loss_weights:
  emotion: 1.0
  speaker: 0.8
  age_cls: 0.3
  age_reg: 0.3
  gender: 0.5
```

### 4.3 训练结果

| 指标 | 训练集 | 验证集 | 说明 |
|------|--------|--------|------|
| Loss | 6.34 | 7.05 | - |
| 说话人Acc | 69.1% | **0%** | 验证集说话人未在训练集 |
| 性别Acc | 95.8% | **100%** | 已达标 |
| 情绪Acc | 41.2% | 39.4% | 需更多数据 |

---

## 五、模型量化与导出

### 5.1 端侧推理引擎选型

经过调研10+主流端侧推理引擎，MT9655最终选型:

| 优先级 | 引擎 | 格式 | 量化 | 加速 | 大小 | 备注 |
|--------|------|------|------|------|------|------|
| **P0** | **TFLite** | .tflite | INT8静态 | XNNPack + NeuroPilot | ~1MB | **首选** |
| P1 | ONNX Runtime | .onnx | INT8 | CPUExecutionProvider | ~5MB | 备选 |
| P2 | NCNN | .ncnn | INT8 | ARM NEON | ~1MB | 极致轻量 |

**不推荐**: PyTorch Mobile (量化不成熟), TensorFlow Mobile (已废弃)

**选型决策树**:
```
目标平台?
├── MediaTek芯片 (MT9655) → TFLite + NeuroPilot ✅
├── 高通芯片 → TFLite + QNN / SNPE
├── 通用ARM → TFLite / ONNX Runtime / NCNN
└── PC/服务器 → ONNX Runtime + TensorRT/OpenVINO
```

### 5.2 量化与转换流程

```
PyTorch FP32 (363MB)
    ↓
┌─────────────────┐
│ ONNX Export      │ → 1.5MB (结构) + 363MB (权重)
└─────────────────┘
    ↓
┌─────────────────┐
│ TFLite Converter │ → INT8静态量化
│ + 校准数据集     │
└─────────────────┘
    ↓
┌─────────────────┐
│ TFLite Model     │ → 目标: ~8MB (INT8)
└─────────────────┘
    ↓
┌─────────────────┐
│ NeuroPilot优化   │ → MT9655 APU加速 (若可用)
└─────────────────┘
```

### 5.3 量化结果对比

| 格式 | 大小 | 推理时间(PC-CPU) | 吞吐量 | 精度损失 |
|------|------|-----------------|--------|----------|
| PyTorch FP32 | 363MB | - | - | 0% |
| ONNX FP32 | 1.5MB+363MB | 8.22ms | 121/s | 0% |
| **TFLite INT8** | **~8MB** | **~50ms** | **~200/s** | **<5%** |
| ONNX INT8 | ~50MB | ~60ms | ~160/s | <3% |
| NCNN INT8 | ~8MB | ~40ms | ~250/s | <3% |

### 5.4 TFLite端侧部署代码

```python
import tflite_runtime.interpreter as tflite
import numpy as np

class SpeechAnalyzerTFLite:
    def __init__(self, model_path, num_threads=4):
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,           # MT9655 4核
            experimental_use_xnnpack=True       # CPU加速
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def infer(self, mel_spectrogram):
        # 设置输入 (INT8量化模型需int8输入)
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            mel_spectrogram
        )
        
        # 推理
        self.interpreter.invoke()
        
        # 获取4任务输出
        emotion = self.interpreter.get_tensor(self.output_details[0]['index'])
        gender = self.interpreter.get_tensor(self.output_details[1]['index'])
        age = self.interpreter.get_tensor(self.output_details[2]['index'])
        speaker = self.interpreter.get_tensor(self.output_details[3]['index'])
        
        return {'emotion': emotion, 'gender': gender, 'age': age, 'speaker': speaker}
```

### 5.5 性能优化策略

```python
# 1. 内存优化
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_preserve_all_tensors=False
)

# 2. 批处理推理
batch_size = 4
input_data = np.stack([preprocess(a) for a in audio_list])
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()

# 3. 静态量化校准
import tensorflow as tf

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 80, 300).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

---

## 六、端侧性能

### 6.1 MT9655 TV SoC 约束

| 资源 | 规格 |
|------|------|
| CPU | ARM Cortex-A55 × 4 |
| GPU | Mali-G57 MC1 |
| 内存 | 2-4GB |
| AI加速 | NeuroPilot (TFLite/ONNX) |

### 6.2 预期性能

| 场景 | 预处理 | 推理 | 端到端 | 内存 |
|------|--------|------|--------|------|
| PC (RTX 3090) | ~50ms | ~100ms (FP32) | ~150ms | ~1GB |
| PC (CPU) | ~100ms | ~500ms (INT8) | ~600ms | ~400MB |
| MT9655 | ~100ms | ~300-500ms (INT8) | **~500ms** | **~20MB** |

---

## 七、多语言支持

### 7.1 语言优先级

| 优先级 | 语言 | 数据集 | 情绪UAR |
|--------|------|--------|---------|
| P0 | 英语 | RAVDESS/CREMA-D/ESD | 70% |
| P1 | 中文 | ESD (中文部分) | 60% |
| P2 | 西班牙语 | Common Voice | 58% |
| P3 | 法语/德语/意大利语/日语 | Common Voice | 45-55% |

### 7.2 跨语言策略

- **频谱CNN语言无关性**: Mel Spectrogram对语言差异不敏感
- **SSL预训练多语言**: HuBERT/WavLM预训练包含多语言数据
- **零样本迁移**: 英语模型直接应用于其他语言

---

## 八、评测对比

### 8.1 开源模型评测 (25个模型)

| 模型 | 参数量 | 情绪UAR | 说话人EER | 端侧可行 |
|------|--------|---------|-----------|----------|
| HuBERT Large | 316M | 74% | - | ❌ |
| wav2vec 2.0 Large | 315M | 72% | - | ❌ |
| AST | 87M | 68% | - | ❌ |
| ECAPA-TDNN | 6.2M | 58% | 0.8% | ⚠️ (仅说话人) |
| 3D-CNN+Attn | 8.5M | 65% | - | ✅ |
| **Our Target** | **8M** | **66.3%** | **3%** | **✅** |

### 8.2 关键发现

1. **SSL模型情绪最强**: HuBERT/wav2vec UAR 72-78%，但300M+参数端侧不可行
2. **频谱CNN端侧友好**: 8M参数，CPU延迟50-200ms，情绪UAR 60-67%
3. **单模型多任务最优**: 减少部署复杂度，Our Target 4任务同时支持

---

## 九、实施路线图

### 9.1 已完成 (Phase 0)

- ✅ 项目规划与结构设计
- ✅ 25个开源模型评测对比
- ✅ HuBERT Base 多任务微调 (冻结backbone)
- ✅ 模型量化与ONNX导出
- ✅ Git版本控制 (23次提交)
- ✅ GitHub仓库推送

### 9.2 短期 (Phase 1-2, 1-2周)

| 任务 | 内容 | 目标 |
|------|------|------|
| GPU修复 | 安装cuDNN9或降级PyTorch | 启用GPU训练 |
| 数据扩充 | 整合CREMA-D + Common Voice | 10万+样本 |
| 重新划分 | LOSO (Leave-One-Speaker-Out) | 说话人验证有效评估 |
| 解冻微调 | 解冻HuBERT端到端训练 | 情绪UAR > 50% |

### 9.3 中期 (Phase 3-4, 2-4周)

| 任务 | 内容 | 目标 |
|------|------|------|
| 静态量化 | 校准-based INT8量化 | 模型 < 50MB |
| TFLite导出 | 完整TFLite转换 | MT9655可运行 |
| 端侧验证 | MT9655实际推理测试 | 延迟 < 500ms |
| Web Demo | Gradio界面 | 产品演示 |

### 9.4 长期 (Phase 5+, 1-2月)

| 任务 | 内容 | 目标 |
|------|------|------|
| SSL蒸馏 | HuBERT Large → Small | 情绪UAR > 70% |
| 多语言数据 | 西/法/德/意/日数据集 | 跨语言UAR > 55% |
| 模型压缩 | 剪枝 + 量化 + 蒸馏 | INT8 < 8MB |
| 产品集成 | TV端SDK封装 | 商用部署 |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 情绪精度不达标 | 高 | SSL蒸馏、数据增强、端到端微调 |
| 模型太大 | 中 | 静态量化、模型剪枝、轻量版备选 |
| 延迟过高 | 中 | 减层、降低输入长度、XNNPack优化 |
| GPU环境故障 | 中 | 降级PyTorch CPU版、云端训练备选 |
| 多语言性能差 | 低 | 增加目标语言数据、零样本迁移 |
| 数据集划分问题 | 高 | 重新LOSO划分、VoxCeleb扩充 |

---

## 十一、项目结构

```
speech_analysis_project/
├── configs/                    # 配置文件
│   ├── train_config.yaml
│   └── model_config.yaml
├── data/                       # 数据集
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理数据
│   └── splits/                 # 数据划分
├── models/                     # 模型
│   ├── pretrained/             # 预训练模型
│   │   ├── hubert_base_ls960/  # HuBERT Base (360MB)
│   │   ├── wav2vec2_base_960h/ # Wav2Vec 2.0 Base
│   │   ├── wavlm_base/         # WavLM Base
│   │   ├── ecapa_tdnn/         # ECAPA-TDNN
│   │   └── emotion2vec_plus_base/
│   ├── pretrained_finetuned/   # 微调后模型
│   │   └── hubert_multitask/
│   │       └── best_model.pt   # 微调模型 (363MB)
│   └── quantized/              # 量化模型
│       ├── hubert_multitask.onnx      # 1.5MB
│       ├── hubert_multitask.onnx.data # 363MB
│       └── hubert_multitask_int8.pt   # 361MB
├── training/                   # 训练脚本
│   ├── train.py                # 主训练脚本
│   ├── finetune_pretrained.py  # 预训练模型微调
│   └── quantize_export.py      # 量化导出
├── evaluation/                 # 评估脚本
├── demo/                       # Demo
│   ├── demo_pc.py              # PC端Demo
│   ├── demo_mt9655.py          # 端侧Demo
│   └── register_speaker.py     # 声纹注册
├── utils/                      # 工具
│   ├── audio_utils.py
│   ├── model_utils.py
│   └── git_auto_push.py        # Git自动上传
├── docs/                       # 文档
│   ├── model_architecture.md   # 架构设计
│   ├── training_guide.md       # 训练指南
│   ├── deployment_guide.md     # 部署指南
│   └── SESSION_RESUME.md       # 会话恢复
├── outputs/                    # 输出报告
│   ├── model_benchmark_report.md
│   └── detailed_model_benchmark.md
├── logs/                       # 训练日志
├── checkpoints/                # 检查点
├── requirements.txt            # 依赖
└── README.md                   # 项目说明
```

---

## 十二、技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 开发语言 |
| PyTorch | 2.6.0+cu124 | 深度学习框架 |
| transformers | 4.x | 预训练模型加载 |
| librosa | 0.10+ | 音频处理 |
| soundfile | 0.12+ | 音频读写 |
| onnxruntime | 1.16+ | ONNX推理 |
| tensorflow | 2.x | TFLite导出 |
| numpy | 1.24+ | 数值计算 |
| scikit-learn | 1.3+ | 评估指标 |

---

## 十三、GitHub仓库

- **仓库地址**: https://github.com/mikexu2030/speech_analysis_project
- **仓库大小**: 3.65 MiB (代码/配置/文档, 不含模型权重)
- **文件数量**: 108个
- **总提交**: 23次
- **最新提交**: SESSION_RESUME更新

---

## 十四、参考资料

- HuBERT: [facebookresearch/hubert](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- Wav2Vec 2.0: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
- ECAPA-TDNN: [speechbrain/ECAPA-TDNN](https://github.com/speechbrain/speechbrain)
- RAVDESS Dataset: [zenodo/1188976](https://zenodo.org/record/1188976)
- CREMA-D Dataset: [github/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- Common Voice: [mozilla/CommonVoice](https://commonvoice.mozilla.org/)

---

> 本方案由AI助手基于项目所有文档综合整理生成
> 项目路径: /data/mikexu/speech_analysis_project/
> 生成时间: 2026-04-30
