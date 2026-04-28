# 语音四合一识别项目 - 会话恢复文档

## 文档信息
- 创建时间: 2026-04-28
- 项目路径: /data/mikexu/speech_analysis_project
- 当前会话状态: 工具调用次数已达上限，需要开启新会话继续

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

### 2. 模型下载与验证 (4/5成功) ✅

| 模型 | 系列 | 状态 | 大小 | 验证结果 |
|------|------|------|------|----------|
| wav2vec 2.0 Base | wav2vec 2.0 | ✅可用 | 360MB | 94.4M参数，768维输出 |
| HuBERT Base | HuBERT/WavLM | ✅可用 | 360MB | 94.4M参数，768维输出 |
| WavLM Base Plus | HuBERT/WavLM | ✅可用 | 720MB | 94.4M参数，768维输出 |
| ECAPA-TDNN | Speaker | ✅可用 | 80MB | SpeechBrain格式 |
| Emotion2Vec+ Base | Emotion2Vec+ | ❌损坏 | 227MB | model.pt反复下载损坏 |

**关键发现**:
- 网络需要使用 `hf-mirror.com` 镜像站（不是 huggingface.co）
- 大文件下载容易被中断导致损坏（360MB+ 文件）
- 3个骨干模型结构相似，都适合作为共享编码器

### 3. 模型选型结论 ✅

**推荐方案: 单模型四合一**
- 使用 **WavLM Base** 作为共享编码器（效果最好，支持声纹+表示）
- 添加4个轻量分类头：声纹识别、年龄段、性别、情绪
- 总大小：~380MB (骨干) + ~10MB (4个头)
- 量化后：~100MB，适合MT9655端侧部署

### 4. 关键脚本已创建 ✅

- `download_all_models.py` - 模型下载脚本
- `check_model_status.py` - 状态检查脚本
- `download_models.sh` - 多种下载方式脚本
- `scripts/evaluate_models_offline.py` - 离线评测脚本
- `auto_download_on_network_recovery.py` - 自动恢复下载
- `docs/MODEL_DOWNLOAD_GUIDE.md` - 详细下载指南

---

## 当前待解决问题

### 问题1: Emotion2Vec+ model.pt 反复损坏
- 已尝试下载10+次，每次文件大小不同（120M-451M），均损坏
- 原因：curl下载大文件时网络不稳定导致文件不完整
- **解决方案**: 使用wget或aria2c替代curl，或后台nohup下载

### 问题2: 数据集尚未下载
- RAVDESS（情绪识别，英语）
- Common Voice（年龄/性别，多语言）
- VoxCeleb（声纹识别）

### 问题3: 多任务模型尚未训练
- 训练脚本框架已设计，但未实际运行

---

## 下一步计划（优先级排序）

### 优先级1: 修复Emotion2Vec+下载（或放弃）
```bash
# 方案A: 使用wget替代curl
cd /data/mikexu/speech_analysis_project
rm -f models/pretrained/emotion2vec_plus_base/model.pt
wget -c -O models/pretrained/emotion2vec_plus_base/model.pt \
  "https://hf-mirror.com/emotion2vec/emotion2vec_plus_base/resolve/main/model.pt"

# 方案B: 后台下载
nohup wget -c -O models/pretrained/emotion2vec_plus_base/model.pt \
  "https://hf-mirror.com/emotion2vec/emotion2vec_plus_base/resolve/main/model.pt" \
  > logs/emotion2vec_download.log 2>&1 &

# 方案C: 放弃Emotion2Vec+，使用WavLM+自定义情绪头（推荐）
# 已在模型选型中确定此方案
```

### 优先级2: 下载数据集
```bash
# 数据集目录
mkdir -p data/raw/{ravdess,common_voice,voxceleb,iemocap}

# RAVDESS（情绪识别，英语，~1GB）
wget -r -np -nH --cut-dirs=3 -P data/raw/ravdess \
  "https://zenodo.org/record/1188976/files/"

# Common Voice 11.0（年龄/性别，多语言）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download mozilla-foundation/common_voice_11_0 \
  --include "en/*" --local-dir data/raw/common_voice

# VoxCeleb1（声纹识别）
# 需从官网申请下载
```

### 优先级3: 创建并运行训练脚本
```bash
# 训练脚本路径: scripts/train_multitask.py
# 基于WavLM骨干 + 4个分类头
# 见下方"关键代码模板"
```

### 优先级4: 模型导出与量化
```bash
# ONNX导出
python3 scripts/export_onnx.py

# INT8量化
python3 scripts/quantize_model.py
```

---

## 关键代码模板

### 多任务模型定义
```python
import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

class MultiTaskSpeechModel(nn.Module):
    def __init__(self, num_speakers=100, num_ages=7, num_genders=2, num_emotions=8):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(
            "/data/mikexu/speech_analysis_project/models/pretrained/wavlm_base"
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "/data/mikexu/speech_analysis_project/models/pretrained/wavlm_base"
        )
        
        # 冻结骨干网络（可选，先冻结预训练权重）
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 4个任务头
        self.speaker_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, num_speakers)
        )
        self.age_head = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, num_ages)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(768, 64), nn.ReLU(), nn.Linear(64, num_genders)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, num_emotions)
        )
        
    def forward(self, waveform):
        inputs = self.feature_extractor(
            waveform.squeeze().cpu().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)  # [B, 768]
        
        return {
            'speaker': self.speaker_head(features),
            'age': self.age_head(features),
            'gender': self.gender_head(features),
            'emotion': self.emotion_head(features)
        }
```

### ONNX导出脚本
```python
import torch
from scripts.train_multitask import MultiTaskSpeechModel

model = MultiTaskSpeechModel()
model.eval()
dummy_input = torch.randn(1, 16000)

torch.onnx.export(
    model,
    dummy_input,
    "models/exported/multitask_speech.onnx",
    input_names=["waveform"],
    output_names=["speaker", "age", "gender", "emotion"],
    dynamic_axes={"waveform": {0: "batch_size", 1: "sequence_length"}},
    opset_version=11
)
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
│   │   ├── ecapa_tdnn/             ✅ 80MB
│   │   └── emotion2vec_plus_base/  ❌ model.pt损坏
│   └── exported/                    ⏳ 待创建
├── data/
│   └── raw/                         ⏳ 待下载数据集
│       ├── ravdess/                 ⏳
│       ├── common_voice/            ⏳
│       └── voxceleb/                ⏳
├── scripts/
│   ├── evaluate_models_offline.py   ✅
│   ├── train_multitask.py           ⏳ 待创建/运行
│   └── export_onnx.py               ⏳ 待创建/运行
├── results/
│   └── evaluation/
│       ├── all_evaluations.json      ✅
│       └── model_comparison_report.md ✅
├── docs/
│   ├── MODEL_DOWNLOAD_GUIDE.md      ✅
│   └── SESSION_RESUME.md            ✅ (本文档)
├── logs/                            ✅
├── download_all_models.py           ✅
├── check_model_status.py            ✅
└── auto_download_on_network_recovery.py ✅
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
1. 4/5模型已下载验证（WavLM为骨干）
2. Emotion2Vec+ model.pt反复损坏
3. 数据集未下载
4. 训练脚本未运行

请执行：
1. 检查当前模型状态
2. 尝试修复Emotion2Vec+或确认放弃
3. 下载数据集（后台方式）
4. 创建并运行训练脚本
5. 导出ONNX模型
```

---

## 参考文档

- 模型下载指南: `/data/mikexu/speech_analysis_project/docs/MODEL_DOWNLOAD_GUIDE.md`
- 评测报告: `/data/mikexu/speech_analysis_project/results/evaluation/model_comparison_report.md`
- 项目状态: `/data/mikexu/speech_analysis_project/project_status.json`

---

*本文档用于会话恢复，请在新会话开始时加载。*
