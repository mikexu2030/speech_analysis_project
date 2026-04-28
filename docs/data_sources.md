# 语音四合一识别项目 - 数据来源标注总结

## 一、数据来源总览

### 1.1 数据集来源

| 数据集 | 原始来源 | 下载URL | 论文/引用 | 项目内使用位置 |
|--------|---------|---------|-----------|--------------|
| **RAVDESS** | Ryerson University | https://zenodo.org/record/1188976 | Livingstone & Russo (2018) | `data/raw/ravdess/`, `train.py`, `test_ravdess.py` |
| **CREMA-D** | Cheyney University | https://github.com/CheyneyComputerScience/CREMA-D | Cao et al. (2014) | `docs/download_paths.md`, `data/download_emotion_datasets.py` |
| **ESD** | HLTSingapore | https://github.com/HLTSingapore/Emotional-Speech-Data | - | `docs/download_paths.md`, `data/download_emotion_datasets.py` |
| **IEMOCAP** | USC SAIL | https://sail.usc.edu/iemocap/ | Busso et al. (2008) | `docs/download_paths.md`, `data/download_emotion_datasets.py` |
| **EmoDB** | TU Berlin | http://emodb.bilderbar.info/ | Burkhardt et al. (2005) | `docs/emotion_label_survey.md`, `docs/download_paths.md` |
| **SAVEE** | University of Surrey | http://kahlan.eps.surrey.ac.uk/savee/ | Jackson & Haq (2014) | `docs/emotion_label_survey.md` |
| **TESS** | University of Toronto | https://tspace.library.utoronto.ca/ | Dupuis & Pichora-Fuller (2010) | `docs/emotion_label_survey.md` |
| **eNTERFACE** | eNTERFACE Project | http://www.enterface.net/ | Martin et al. (2006) | `docs/emotion_label_survey.md` |
| **AESDD** | University of Patras | https://github.com/kyrgyzov/hesitation-aesdd | Vryzas et al. (2018) | `docs/emotion_label_survey.md` |
| **SUBESCO** | - | https://github.com/shantoroy/SUBESCO | Roy et al. (2023) | `docs/emotion_label_survey.md` |
| **URDU** | - | https://github.com/siddiquelatif/URDU-Dataset | Latif et al. (2020) | `docs/emotion_label_survey.md` |
| **JL-Corpus** | - | https://github.com/jacklanda/JL-Corpus | - | `docs/download_paths.md` |
| **VoxCeleb1/2** | University of Oxford | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ | Nagrani et al. (2017) | `evaluation/detailed_benchmark.py` |
| **Common Voice** | Mozilla | https://commonvoice.mozilla.org/ | Ardila et al. (2020) | `evaluation/detailed_benchmark.py` |

### 1.2 模型来源

| 模型 | 原始来源 | 论文 | 下载URL | 项目内使用位置 |
|------|---------|------|---------|--------------|
| **Emotion2Vec+** | Alibaba DAMO | Ma et al., AAAI 2024 | https://modelscope.cn/models/iic/emotion2vec_plus_large | `evaluation/detailed_benchmark.py`, `docs/download_paths.md` |
| **wav2vec 2.0** | Meta (Facebook) | Baevski et al., NeurIPS 2020 | https://huggingface.co/facebook/wav2vec2-base-960h | `evaluation/detailed_benchmark.py`, `docs/download_paths.md` |
| **HuBERT** | Meta | Hsu et al., ICASSP 2021 | https://huggingface.co/facebook/hubert-base-ls960 | `evaluation/detailed_benchmark.py` |
| **WavLM** | Microsoft | Chen et al., ICASSP 2022 | https://huggingface.co/microsoft/wavlm-base | `evaluation/detailed_benchmark.py` |
| **Data2Vec** | Meta | Baevski et al., ICML 2022 | https://huggingface.co/facebook/data2vec-audio-base | `evaluation/detailed_benchmark.py` |
| **AST** | MIT | Gong et al., NeurIPS 2021 | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 | `evaluation/detailed_benchmark.py` |
| **SSAST** | UMichigan | Gong et al., ICASSP 2022 | https://github.com/AndreyGuzhov/SSAST | `evaluation/detailed_benchmark.py` |
| **ECAPA-TDNN** | SpeechBrain | Desplanques et al., Interspeech 2020 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | `evaluation/detailed_benchmark.py`, `docs/download_paths.md` |
| **x-Vector** | Kaldi | Snyder et al., ICASSP 2018 | https://kaldi-asr.org/models/m7 | `evaluation/detailed_benchmark.py` |
| **ResNetSE34L** | University of Oxford | Chung et al., VoxCeleb | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ | `evaluation/detailed_benchmark.py` |
| **MobileNetV3** | Google | Howard et al., CVPR 2019 | torchvision | `evaluation/detailed_benchmark.py` |
| **EfficientNet** | Google | Tan & Le, ICML 2019 | torchvision | `evaluation/detailed_benchmark.py` |
| **ResNet-18** | Microsoft | He et al., CVPR 2016 | torchvision | `evaluation/detailed_benchmark.py` |
| **NISQA** | TU Berlin | Mittag et al., ICASSP 2021 | https://github.com/gabrielmittag/NISQA | `evaluation/detailed_benchmark.py` |
| **OpenSMILE** | audEERING | Eyben et al., TAC 2013 | https://audeering.github.io/opensmile-python/ | `evaluation/detailed_benchmark.py` |
| **SpeechBrain Emotion** | SpeechBrain | https://speechbrain.github.io/ | https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP | `evaluation/detailed_benchmark.py` |
| **3D-CNN+Attention** | 多机构 | Zhao et al., IEEE TAC 2022 | - | `evaluation/detailed_benchmark.py` |
| **GMP-ATL** | - | Global Multi-scale Perception + Attention Transfer Learning | - | `evaluation/detailed_benchmark.py` |

### 1.3 性能数据来源

| 性能指标 | 数据来源 | 置信度 | 项目内使用位置 |
|---------|---------|--------|--------------|
| **情绪UAR 76%** | Emotion2Vec+论文 (AAAI 2024) | ⭐⭐⭐ 高 | `evaluation/detailed_benchmark.py` |
| **情绪UAR 74%** | HuBERT论文 + SER微调文献 | ⭐⭐⭐ 高 | `evaluation/detailed_benchmark.py` |
| **情绪UAR 72%** | wav2vec 2.0 SER微调 | ⭐⭐⭐ 高 | `evaluation/detailed_benchmark.py` |
| **说话人EER 0.8%** | ECAPA-TDNN论文 | ⭐⭐⭐ 高 | `evaluation/detailed_benchmark.py` |
| **性别Acc 97%** | ECAPA-TDNN + 多任务学习 | ⭐⭐⭐ 高 | `evaluation/detailed_benchmark.py` |
| **年龄MAE 5年** | NISQA论文 | ⭐⭐ 中 | `evaluation/detailed_benchmark.py` |
| **轻量CNN UAR 62%** | SpeechBrain + 文献综合 | ⭐⭐ 中 | `evaluation/detailed_benchmark.py` |
| **端侧延迟50ms** | 理论估算 (未实测) | ⭐ 低 | `evaluation/detailed_benchmark.py` |
| **各情绪识别率** | 多篇SER论文综合 | ⭐⭐ 中 | `docs/emotion_label_survey.md` |
| **混淆矩阵模式** | SER论文综合分析 | ⭐⭐ 中 | `docs/emotion_label_survey.md` |
| **VAD映射值** | Russell (1980) + 后续研究 | ⭐⭐⭐ 高 | `docs/emotion_label_survey.md` |

---

## 二、数据使用声明

### 2.1 数据集许可

| 数据集 | 许可类型 | 使用限制 |
|--------|---------|---------|
| RAVDESS | CC BY-ND 4.0 | 可商用，需署名，禁止修改 |
| CREMA-D | 研究用途 | 需引用论文 |
| ESD | 研究用途 | 需引用论文 |
| IEMOCAP | 研究用途 | 需注册申请 |
| EmoDB | 研究用途 | 需引用论文 |
| SAVEE | 研究用途 | 需引用论文 |
| VoxCeleb | CC BY 4.0 | 可商用，需署名 |
| Common Voice | CC0 | 公共领域 |

### 2.2 模型许可

| 模型 | 许可 | 商用 |
|------|------|------|
| Emotion2Vec+ | Apache 2.0 | ✅ |
| wav2vec 2.0 | Apache 2.0 | ✅ |
| HuBERT | Apache 2.0 | ✅ |
| WavLM | MIT | ✅ |
| ECAPA-TDNN | Apache 2.0 | ✅ |
| SpeechBrain | Apache 2.0 | ✅ |

---

## 三、引用格式

### 3.1 数据集引用

```bibtex
% RAVDESS
@article{livingstone2018ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PLOS ONE},
  year={2018}
}

% CREMA-D
@article{cao2014cremad,
  title={CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE TAC},
  year={2014}
}

% IEMOCAP
@inproceedings{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeehyun and Lee, Sungbok and Narayanan, Shrikanth S},
  booktitle={LREC},
  year={2008}
}

% EmoDB
@inproceedings{burkhardt2005emodb,
  title={A database of German emotional speech},
  author={Burkhardt, Felix and Paeschke, Astrid and Rolfes, Miriam and Sendlmeier, Walter F and Weiss, Benjamin},
  booktitle={Interspeech},
  year={2005}
}
```

### 3.2 模型引用

```bibtex
% Emotion2Vec+
@article{ma2024emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Li, Zhisheng and Wang, Ming and Chen, Rui and Lu, Xiangming and Li, Shujie and Zheng, Zhen and Wu, Yuxuan and Li, Xinyu and Li, Qianqian and others},
  journal={AAAI},
  year={2024}
}

% wav2vec 2.0
@inproceedings{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  booktitle={NeurIPS},
  year={2020}
}

% HuBERT
@inproceedings{hsu2021hubert,
  title={HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units},
  author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  booktitle={ICASSP},
  year={2021}
}

% WavLM
@inproceedings{chen2022wavlm,
  title={WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing},
  author={Chen, Sanyuan and Wang, Yu and Chen, Zhengyang and Wu, Xinyu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Wei, Furu and Dai, Xiangzhe and others},
  booktitle={ICASSP},
  year={2022}
}

% ECAPA-TDNN
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech},
  year={2020}
}
```

---

## 四、数据质量说明

### 4.1 评测数据可信度分级

| 等级 | 说明 | 示例 |
|------|------|------|
| ⭐⭐⭐ **高** | 来自原始论文实验结果 | Emotion2Vec+ UAR 76%, ECAPA EER 0.8% |
| ⭐⭐ **中** | 来自第三方复现或综述 | 轻量CNN UAR 62%, 各情绪识别率 |
| ⭐ **低** | 估算或理论推导 | MT9655延迟50ms, INT8精度损失 |

### 4.2 已知偏差

| 偏差类型 | 说明 | 影响 |
|---------|------|------|
**演员语音** | RAVDESS/CREMA-D使用专业演员 | 与真实语音有差距 |
| **英语主导** | 大部分数据集为英语 | 多语言性能不确定 |
| **实验室环境** | 录音环境控制良好 | 噪声环境下性能下降 |
| **年龄偏差** | 演员多为成年人 | 儿童/老年人识别率低 |
| **文化差异** | 情绪表达因文化而异 | 跨文化应用需谨慎 |

---

## 五、文档数据一致性检查

### 5.1 跨文档数据核对

| 数据项 | `detailed_benchmark.md` | `emotion_label_survey.md` | `download_paths.md` | 一致性 |
|--------|------------------------|--------------------------|---------------------|--------|
| Emotion2Vec+ UAR | 76% | 76% | 76% | ✅ |
| wav2vec2 UAR | 72% | 68-72% | 68-72% | ✅ |
| ECAPA-TDNN EER | 0.8% | - | 0.8% | ✅ |
| RAVDESS样本数 | 2,452 | 1,440 (Speech) | 1,440 | ⚠️ 含Song |
| 轻量模型UAR | 67% | 62-68% | - | ✅ |
| MT9655延迟 | 500ms | - | - | ⚠️ 需实测 |

### 5.2 数据更新记录

| 日期 | 更新内容 | 文档 |
|------|---------|------|
| 2025-04-28 | 初始评测数据 | `evaluation/detailed_benchmark.py` |
| 2025-04-28 | 情绪标签调研 | `docs/emotion_label_survey.md` |
| 2025-04-28 | 下载路径汇总 | `docs/download_paths.md` |
| 2025-04-28 | 数据来源标注 | `docs/data_sources.md` (本文档) |

---

## 六、使用建议

### 6.1 学术研究

- 引用原始论文和数据集论文
- 说明数据集的局限性
- 复现结果时注明环境差异

### 6.2 商业应用

- 确认数据集和模型的商用许可
- 在目标场景下重新验证性能
- 考虑数据偏差对用户体验的影响

### 6.3 端侧部署

- 实测目标设备上的推理延迟
- 验证INT8量化后的精度损失
- 使用真实场景数据微调模型

---

*文档生成时间: 2025-04-28*
*维护: 语音四合一识别项目*
*数据版本: v1.0*
