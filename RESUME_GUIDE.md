# 语音四合一识别项目 - 断网续执行完整指南

## 项目概述

**目标**: 在MT9655端侧实现单模型多任务语音分析
**任务**: 声纹识别 + 年龄段识别 + 性别识别 + 情绪识别
**语言**: 优先英语，其次西语，再次法德意日
**平台**: MT9655 (INT8量化, ~8MB模型)

## 当前状态 (2025-04-28)

### 已完成
- [x] 项目结构搭建 (16目录, 41文件)
- [x] 详细模型评测报告 (25个模型对比)
- [x] 添加 Emotion2Vec+ 和 GMP-ATL 模型到评测
- [x] 执行计划脚本 (execution_plan.py)
- [x] 评测脚本框架 (run_real_benchmark.py)
- [x] RAVDESS数据集下载中 (后台运行)

### 进行中
- [ ] RAVDESS数据集下载 (预计2-3小时)
- [ ] 模型依赖安装 (torch, transformers, funasr等)

### 待执行
- [ ] 下载其他数据集 (CREMA-D, ESD, EmoDB)
- [ ] 实际模型评测 (Emotion2Vec+, wav2vec2, HuBERT等)
- [ ] 模型训练
- [ ] 量化与导出
- [ ] 端侧验证

---

## 断网续执行步骤

### 步骤1: 检查当前状态
```bash
cd /data/mikexu/speech_analysis_project
python3 execution_plan.py
```

### 步骤2: 继续下载数据集

#### RAVDESS (优先级1, 208MB)
```bash
# 检查下载进度
tail -f logs/ravdess_download.log

# 如果下载中断，重新启动
cd /data/mikexu/speech_analysis_project
python3 data/download_emotion_datasets.py --dataset ravdess --output_dir data/raw
```

#### EmoDB (优先级5, 30MB, 德语)
```bash
# 直接下载
cd /data/mikexu/speech_analysis_project/data/raw
wget http://emodb.bilderbar.info/download/emodb.tar.gz
# 或
curl -L -o emodb.tar.gz http://emodb.bilderbar.info/download/emodb.tar.gz
```

#### CREMA-D (优先级2, 1.2GB)
```bash
# 需要手动从Kaggle下载
# 访问: https://www.kaggle.com/datasets/ejlok1/cremad
# 下载后解压到: data/raw/cremad/
```

#### ESD (优先级3, 2.9GB)
```bash
# 需要手动从GitHub下载
# 访问: https://github.com/HLTSingapore/Emotional-Speech-Data
# 下载后解压到: data/raw/esd/
```

### 步骤3: 安装模型依赖

```bash
# 基础依赖
pip install torch torchaudio transformers

# Emotion2Vec+ 需要
pip install modelscope funasr

# SpeechBrain (ECAPA-TDNN)
pip install speechbrain

# 其他工具
pip install librosa soundfile numpy scipy scikit-learn
```

### 步骤4: 运行模型评测

#### 评测 Emotion2Vec+ Large
```bash
cd /data/mikexu/speech_analysis_project
python3 evaluation/run_real_benchmark.py \
    --model emotion2vec_large \
    --dataset ravdess \
    --output outputs/emotion2vec_large_ravdess.json
```

#### 评测 Emotion2Vec+ Base
```bash
python3 evaluation/run_real_benchmark.py \
    --model emotion2vec_base \
    --dataset ravdess \
    --output outputs/emotion2vec_base_ravdess.json
```

#### 评测 wav2vec 2.0
```bash
python3 evaluation/run_real_benchmark.py \
    --model wav2vec2 \
    --dataset ravdess \
    --output outputs/wav2vec2_ravdess.json
```

### 步骤5: 分析评测结果

```bash
# 查看结果
ls -lh outputs/*_results.json

# 对比不同模型
python3 -c "
import json
for model in ['emotion2vec_large', 'emotion2vec_base', 'wav2vec2']:
    try:
        with open(f'outputs/{model}_ravdess.json') as f:
            data = json.load(f)
        print(f'{model}: {len(data)} samples processed')
    except:
        print(f'{model}: not yet evaluated')
"
```

---

## 关键文件位置

| 文件 | 路径 |
|------|------|
| 评测报告 | /data/mikexu/speech_analysis_project/outputs/detailed_model_benchmark.md |
| 执行计划 | /data/mikexu/speech_analysis_project/execution_plan.py |
| 评测脚本 | /data/mikexu/speech_analysis_project/evaluation/run_real_benchmark.py |
| 状态文件 | /data/mikexu/speech_analysis_project/project_status.json |
| 数据集 | /data/mikexu/speech_analysis_project/data/raw/ |
| 项目计划 | /data/mikexu/speech_analysis_project/PROJECT_PLAN.md |

---

## 模型评测优先级

### 情绪识别 (重点)
1. **Emotion2Vec+ Large** - 当前SOTA，必须评测
2. **Emotion2Vec+ Base** - 轻量版本，端侧候选
3. **wav2vec 2.0 Large** - 基线对比
4. **HuBERT Large** - 基线对比
5. **GMP-ATL** - 用户关注模型

### 说话人识别
1. **ECAPA-TDNN** - SOTA说话人模型
2. **WavLM Base SV** - 微软模型

### 年龄/性别
1. **NISQA + Age/Gender** - 多任务模型
2. **Age/Gender CNN** - 轻量模型

---

## 常见问题

### Q: 下载速度太慢怎么办?
A: 
- RAVDESS可以从镜像下载
- 使用aria2c多线程下载: `aria2c -x 16 -s 16 <url>`
- 或者先下载小数据集(EmoDB, 30MB)进行初步评测

### Q: 模型太大无法加载?
A:
- Emotion2Vec+ Large需要16GB+内存
- 先评测Base版本(95MB)
- 使用CPU推理而非GPU

### Q: 断网后如何继续?
A:
1. 检查 `project_status.json` 了解当前状态
2. 检查数据集目录 `data/raw/` 看哪些已下载
3. 运行 `python3 execution_plan.py` 查看状态
4. 从断点继续执行相应步骤

### Q: 评测需要多长时间?
A:
- RAVDESS完整评测(2452样本):
  - Emotion2Vec+ Large: ~2小时(CPU), ~30分钟(GPU)
  - Emotion2Vec+ Base: ~30分钟(CPU), ~10分钟(GPU)
  - 轻量模型: ~10分钟(CPU)

---

## 下一步行动建议

### 如果网络不稳定:
1. 优先下载 **EmoDB** (30MB, 德语情绪数据)
2. 安装依赖并评测 **Emotion2Vec+ Base**
3. 记录结果并保存状态

### 如果网络良好:
1. 等待RAVDESS下载完成
2. 安装所有依赖
3. 按优先级运行模型评测
4. 更新评测报告

### 如果完全断网:
1. 使用已下载的数据(如有)
2. 完善代码和文档
3. 准备训练脚本
4. 网络恢复后继续下载和评测

---

## 联系信息

项目目录: `/data/mikexu/speech_analysis_project/`
评测报告: `outputs/detailed_model_benchmark.md`
状态文件: `project_status.json`

---

*最后更新: 2025-04-28*
*状态: RAVDESS下载中, 等待网络稳定*
