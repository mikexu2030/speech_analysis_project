# 模型与数据下载路径汇总

## 一、Top3 推荐模型下载路径

### 1. Emotion2Vec+ Large (情绪识别首选)

| 属性 | 详情 |
|------|------|
| **来源** | 阿里巴巴达摩院 (Alibaba DAMO Academy) |
| **论文** | emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation (AAAI 2024) |
| **模型ID** | `iic/emotion2vec_plus_large` |
| **下载路径** | |
| ModelScope | https://modelscope.cn/models/iic/emotion2vec_plus_large |
| HuggingFace | https://huggingface.co/emotion2vec/emotion2vec_plus_large |
| **参数量** | 316M |
| **INT8大小** | ~11MB |
| **情绪UAR** | 76% (跨数据集平均) |
| **支持语言** | 英语、中文、德语、法语、意大利语、日语、韩语、西班牙语 |
| **训练数据** | RAVDESS + CREMA-D + IEMOCAP + ESD + EmoDB + eNTERFACE + AESDD + URDU + SUBESCO |
| **适用场景** | 高精度情绪识别、多语言支持 |
| **端侧可行** | ⚠️ 需INT8量化，推理延迟较高 |

**安装与加载**:
```python
# 方式1: ModelScope (国内推荐)
pip install modelscope funasr
from funasr import AutoModel
model = AutoModel(model="iic/emotion2vec_plus_large")

# 方式2: HuggingFace
pip install transformers
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
model = AutoModelForAudioClassification.from_pretrained("emotion2vec/emotion2vec_plus_large")
```

---

### 2. Emotion2Vec+ Base (轻量版)

| 属性 | 详情 |
|------|------|
| **来源** | 阿里巴巴达摩院 |
| **模型ID** | `emotion2vec/emotion2vec_plus_base` |
| **下载路径** | |
| HuggingFace | https://huggingface.co/emotion2vec/emotion2vec_plus_base |
| **参数量** | 95M |
| **INT8大小** | ~8MB |
| **情绪UAR** | 70-72% |
| **支持语言** | 同Large版本 |
| **端侧可行** | ✅ 推荐 |

---

### 3. wav2vec 2.0 Base + 情绪微调 (基线对比)

| 属性 | 详情 |
|------|------|
| **来源** | Meta (Facebook) |
| **论文** | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (NeurIPS 2020) |
| **模型ID** | `facebook/wav2vec2-base-960h` |
| **下载路径** | |
| HuggingFace | https://huggingface.co/facebook/wav2vec2-base-960h |
| **参数量** | 95M |
| **INT8大小** | ~8MB |
| **情绪UAR** | 68-72% (需微调) |
| **支持语言** | 英语为主 |
| **端侧可行** | ⚠️ 需微调 |

**微调脚本**:
```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=8,  # 情绪类别数
    ignore_mismatched_sizes=True
)
```

---

## 二、备选模型下载路径

### 4. HuBERT Large (高精度但大)

| 属性 | 详情 |
|------|------|
| **来源** | Meta |
| **论文** | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (ICASSP 2021) |
| **模型ID** | `facebook/hubert-large-ls960-ft` |
| **下载路径** | https://huggingface.co/facebook/hubert-large-ls960-ft |
| **参数量** | 316M |
| **情绪UAR** | 74% |
| **端侧可行** | ❌ 太大 |

### 5. WavLM Large (多语言好)

| 属性 | 详情 |
|------|------|
| **来源** | 微软 |
| **论文** | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing (ICASSP 2022) |
| **模型ID** | `microsoft/wavlm-large` |
| **下载路径** | https://huggingface.co/microsoft/wavlm-large |
| **参数量** | 316M |
| **情绪UAR** | 73% |
| **多语言** | 优于wav2vec2 |

### 6. ECAPA-TDNN (说话人SOTA)

| 属性 | 详情 |
|------|------|
| **来源** | SpeechBrain |
| **论文** | ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification (Interspeech 2020) |
| **模型ID** | `speechbrain/spkrec-ecapa-voxceleb` |
| **下载路径** | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| **参数量** | 6.2M |
| **说话人EER** | 0.8% |
| **端侧可行** | ✅ |

### 7. SpeechBrain Emotion CNN (轻量情绪)

| 属性 | 详情 |
|------|------|
| **来源** | SpeechBrain |
| **模型ID** | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` |
| **下载路径** | https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP |
| **参数量** | 14M |
| **情绪UAR** | 62% |
| **端侧可行** | ✅ |

---

## 三、数据集下载路径

### 已下载

| 数据集 | 状态 | 路径 | 样本数 | 语言 |
|--------|------|------|--------|------|
| **RAVDESS** | ✅ 已下载 | `data/raw/ravdess/` | 1,440 (Speech) | 英语 |
| | | | 1,012 (Song) | |

### 推荐下载 (优先级排序)

| 优先级 | 数据集 | 下载路径 | 大小 | 语言 | 情绪类别 | 样本数 |
|--------|--------|---------|------|------|---------|--------|
| **1** | **CREMA-D** | https://www.kaggle.com/datasets/ejlok1/cremad | ~1.2GB | 英语 | 6类 | 7,442 |
| | | https://github.com/CheyneyComputerScience/CREMA-D | | | | |
| **2** | **ESD** | https://github.com/HLTSingapore/Emotional-Speech-Data | ~2.9GB | 英语+中文 | 5类 | 17,500 |
| | | https://www.kaggle.com/datasets/jacksonchou/esd | | | | |
| **3** | **EmoDB** | http://emodb.bilderbar.info/download/ | ~30MB | 德语 | 7类 | 535 |
| | | https://github.com/berlin-with0ut-emotion/emodb | | | | |
| **4** | **SAVEE** | http://kahlan.eps.surrey.ac.uk/savee/ | ~100MB | 英语 | 7类 | 480 |
| | | https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee | | | | |
| **5** | **TESS** | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess | ~100MB | 英语 | 7类 | 2,800 |
| **6** | **JL-Corpus** | https://www.kaggle.com/datasets/tli7544/jl-corpus | ~200MB | 英语 | 5类 | 2,400 |
| | | https://github.com/jacklanda/JL-Corpus | | | | |
| **7** | **IEMOCAP** | https://sail.usc.edu/iemocap/iemocap_release.htm | ~12GB | 英语 | 10类 | 10,039 |
| | | (需注册申请) | | | | |

### 其他数据集参考

| 数据集 | 下载路径 | 语言 | 情绪类别 | 样本数 |
|--------|---------|------|---------|--------|
| **eNTERFACE** | http://www.enterface.net/ | 英语 | 6类 | 1,296 |
| **AESDD** | https://github.com/kyrgyzov/hesitation-aesdd | 希腊语 | 5类 | 500 |
| **SUBESCO** | https://github.com/shantoroy/SUBESCO | 孟加拉语 | 7类 | 3,600 |
| **URDU** | https://github.com/siddiquelatif/URDU-Dataset | 乌尔都语 | 4类 | 400 |
| **RAVDESS Song** | https://zenodo.org/record/1188976 | 英语 | 8类 | 1,012 |
| **MSP-Improv** | https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html | 英语 | 多类 | 7,500 |
| **MELD** | https://affective-meld.github.io/ | 英语 | 7类 | 13,000 |
| **CMU-MOSEI** | http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/ | 英语 | 多类 | 23,500 |

---

## 四、下载脚本汇总

### 项目内脚本

| 脚本 | 功能 | 路径 |
|------|------|------|
| `download_ravdess.sh` | 下载RAVDESS | 项目根目录 |
| `download_emodb.sh` | 下载EmoDB | 项目根目录 |
| `download_datasets.sh` | 批量检查/下载数据集 | 项目根目录 |
| `download_models.sh` | 下载预训练模型 | 项目根目录 |
| `data/download_emotion_datasets.py` | Python数据集下载 | `data/` |
| `data/download_speaker_datasets.py` | Python说话人数据下载 | `data/` |

### 快速下载命令

```bash
# RAVDESS (已下载)
cd /data/mikexu/speech_analysis_project
cat data/raw/ravdess/audio_speech/ | wc -l  # 1440 files

# EmoDB (德语，小数据集)
wget -O data/raw/emodb.tar.gz http://emodb.bilderbar.info/download/emodb.tar.gz

# CREMA-D (需要Kaggle账号)
pip install kaggle
kaggle datasets download -d ejlok1/cremad -p data/raw/
unzip data/raw/cremad.zip -d data/raw/cremad/

# ESD (GitHub)
git clone https://github.com/HLTSingapore/Emotional-Speech-Data.git data/raw/esd/

# 模型下载 (HuggingFace)
pip install huggingface-hub
huggingface-cli download emotion2vec/emotion2vec_plus_base --local-dir models/pretrained/emotion2vec_plus_base
huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/pretrained/wav2vec2_base_960h
```

---

## 五、数据与模型匹配建议

### 训练数据组合

| 目标 | 推荐数据集组合 | 总样本数 | 语言覆盖 |
|------|---------------|---------|---------|
| **英语情绪** | RAVDESS + CREMA-D + TESS + SAVEE | ~13,000 | 英语 |
| **多语言情绪** | RAVDESS + EmoDB + ESD + AESDD | ~20,000 | 英/德/中/希 |
| **说话人识别** | VoxCeleb1 + VoxCeleb2 | 1M+ | 多语言 |
| **年龄/性别** | Common Voice + Mozilla | 10,000+ | 多语言 |
| **完整多任务** | 上述全部 | 1M+ | 多语言 |

### 模型选择策略

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **Demo演示** | Emotion2Vec+ Base | 精度高、大小适中 |
| **端侧部署** | 自定义轻量CNN | 3MB、推理快 |
| **高精度研究** | Emotion2Vec+ Large | SOTA精度 |
| **多语言支持** | WavLM Base | 多语言预训练 |
| **说话人优先** | ECAPA-TDNN | SOTA声纹 |
| **快速原型** | SpeechBrain Emotion CNN | 即拿即用 |

---

## 六、数据来源标注

| 信息 | 原始来源 | 项目内文档 |
|------|---------|-----------|
| RAVDESS数据集 | Zenodo: https://zenodo.org/record/1188976 | `docs/emotion_label_survey.md` |
| CREMA-D数据集 | https://github.com/CheyneyComputerScience/CREMA-D | `docs/emotion_label_survey.md` |
| ESD数据集 | https://github.com/HLTSingapore/Emotional-Speech-Data | `docs/emotion_label_survey.md` |
| EmoDB数据集 | http://emodb.bilderbar.info/ | `docs/emotion_label_survey.md` |
| Emotion2Vec+ | Ma et al., AAAI 2024 | `docs/emotion_label_survey.md` |
| wav2vec 2.0 | Baevski et al., NeurIPS 2020 | `evaluation/detailed_benchmark.py` |
| HuBERT | Hsu et al., ICASSP 2021 | `evaluation/detailed_benchmark.py` |
| WavLM | Chen et al., ICASSP 2022 | `evaluation/detailed_benchmark.py` |
| ECAPA-TDNN | Desplanques et al., Interspeech 2020 | `evaluation/detailed_benchmark.py` |
| 情绪识别率 | 多篇论文综合分析 | `docs/emotion_label_survey.md` |
| VAD模型 | Russell, JPSP 1980 | `docs/emotion_label_survey.md` |
| 混淆矩阵 | SER论文综合分析 | `docs/emotion_label_survey.md` |

---

*文档生成时间: 2025-04-28*
*维护: 语音四合一识别项目*
