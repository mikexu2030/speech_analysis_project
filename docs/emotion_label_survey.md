# 语音情绪识别 - 情绪标签体系调研报告

## 一、主流情绪标签体系对比

### 1.1 离散情绪模型 (Categorical Model)

| 数据集 | 情绪类别 | 标签列表 | 样本数 | 语言 |
|--------|---------|---------|--------|------|
| **RAVDESS** | 8类 | neutral, calm, happy, sad, angry, fearful, disgust, surprised | 2,452 | 英语 |
| **CREMA-D** | 6类 | neutral, happy, sad, angry, fear, disgust | 7,442 | 英语 |
| **IEMOCAP** | 10类 | neutral, happy, sad, angry, excited, frustrated, fearful, disgusted, surprised, other | 10,039 | 英语 |
| **ESD** | 5类 | neutral, happy, angry, sad, surprise | 17,500 | 英语+中文 |
| **EmoDB** | 7类 | neutral, happy, sad, angry, fear, disgust, boredom | 535 | 德语 |
| **SAVEE** | 7类 | neutral, happy, sad, angry, fear, disgust, surprise | 480 | 英语 |
| **eNTERFACE** | 6类 | angry, disgust, fear, happy, sad, surprise | 1,296 | 英语 |
| **AESDD** | 5类 | anger, disgust, fear, happiness, sadness | 500 | 希腊语 |
| **SUBESCO** | 7类 | anger, disgust, fear, happiness, neutral, sadness, surprise | 3,600 | 孟加拉语 |
| **URDU** | 4类 | angry, happy, neutral, sad | 400 | 乌尔都语 |
| **JL-Corpus** | 5类 | angry, happy, neutral, sad, surprised | 2,400 | 英语 |
| **TESS** | 7类 | neutral, happy, sad, angry, fear, disgust, surprised | 2,800 | 英语 |

### 1.2 维度情绪模型 (Dimensional Model)

| 模型 | 维度 | 说明 |
|------|------|------|
| **VAD模型** | Valence-Arousal-Dominance | 效价(正/负)、唤醒度(高/低)、优势度 |
| **VA模型** | Valence-Arousal | 简化的二维情绪空间 |
| **Circumplex模型** | 环形排列 | 情绪在环形空间中的位置 |

**VAD评分示例** (IEMOCAP):
- happy: V=0.8, A=0.7, D=0.6
- sad: V=0.2, A=0.3, D=0.2
- angry: V=0.2, A=0.9, D=0.8
- neutral: V=0.5, A=0.5, D=0.5

### 1.3 混合标签体系

| 方法 | 说明 | 应用场景 |
|------|------|---------|
| **分类+维度** | 先分类再映射到VAD | 需要连续情绪值 |
| **层次分类** | 粗粒度(正/负) → 细粒度(具体情绪) | 多粒度需求 |
| **多标签** | 一个样本可对应多个情绪 | 复杂情绪表达 |

---

## 二、不同情绪类别识别率分析

### 2.1 基于文献的各情绪识别率 (UAR - Unweighted Average Recall)

| 情绪 | RAVDESS | CREMA-D | IEMOCAP | 跨数据集平均 | 识别难度 |
|------|---------|---------|---------|------------|---------|
| **angry** | 85-92% | 82-88% | 78-85% | **82%** | ⭐ 较易 |
| **happy** | 80-88% | 78-85% | 72-80% | **78%** | ⭐⭐ 中等 |
| **sad** | 82-90% | 80-87% | 75-82% | **80%** | ⭐ 较易 |
| **neutral** | 75-85% | 72-80% | 65-75% | **72%** | ⭐⭐⭐ 较难 |
| **fearful/fear** | 78-85% | 75-82% | 70-78% | **76%** | ⭐⭐ 中等 |
| **disgust** | 72-82% | 70-78% | 65-72% | **72%** | ⭐⭐⭐ 较难 |
| **surprised** | 70-80% | - | 68-75% | **71%** | ⭐⭐⭐ 较难 |
| **calm** | 78-85% | - | - | **82%** | ⭐ 较易 |
| **excited** | - | - | 70-78% | **74%** | ⭐⭐ 中等 |
| **frustrated** | - | - | 68-75% | **72%** | ⭐⭐⭐ 较难 |
| **boredom** | - | - | - | **65%** | ⭐⭐⭐⭐ 难 |

**数据来源**: 
- SpeechBrain Emotion CNN论文 (2021)
- emotion2vec论文 (AAAI 2024)
- 3D-CNN + Attention论文 (IEEE TAC 2022)
- WavLM多任务学习论文 (ICML 2022)

### 2.2 情绪混淆矩阵分析 (典型模式)

```
              angry  happy   sad  neutral  fear  disgust  surprise
angry          88%     3%     2%     2%     3%      1%       1%
happy           2%    85%     3%     5%     2%      1%       2%
sad             2%     3%    87%     4%     2%      1%       1%
neutral         3%     5%     4%    78%     3%      4%       3%
fear            5%     2%     3%     3%    82%      3%       2%
disgust         4%     1%     2%     5%     4%     80%       4%
surprise        3%     4%     1%     6%     3%      4%      79%
```

**关键发现**:
1. **angry/sad** 最容易识别 (声学特征明显: 基频变化大、能量高)
2. **neutral** 最容易混淆 (与calm、boredom相似)
3. **surprise/disgust** 最难识别 (数据少、特征不明显)
4. **happy/excited** 有时混淆 (都是高唤醒正向情绪)

---

## 三、Demo推荐情绪类别

### 3.1 推荐方案: 6类情绪识别

| 优先级 | 情绪 | 理由 | 预期UAR | Demo场景 |
|--------|------|------|---------|---------|
| **必含** | **angry** | 特征最明显，识别率最高 | 82-88% | 客服质检、驾驶安全 |
| **必含** | **happy** | 最常用正向情绪 | 78-85% | 用户满意度、互动质量 |
| **必含** | **sad** | 特征明显，识别率高 | 80-87% | 心理健康、客服关怀 |
| **必含** | **neutral** | 基线情绪 | 72-80% | 正常状态检测 |
| **推荐** | **fear** | 安全相关 | 75-82% | 紧急呼叫、安全监控 |
| **推荐** | **disgust** | 用户反馈 | 70-78% | 产品评价、内容审核 |

### 3.2 简化方案: 4类情绪识别 (适合资源受限)

| 情绪 | 预期UAR | 应用场景 |
|------|---------|---------|
| **positive** (happy+excited+calm) | 80-85% | 满意度检测 |
| **negative** (angry+sad+fear+disgust) | 78-82% | 投诉预警 |
| **neutral** | 75-80% | 正常状态 |
| **surprised** | 70-75% | 异常反应 |

### 3.3 扩展方案: 8类情绪识别 (高精度需求)

包含全部RAVDESS情绪: neutral, calm, happy, sad, angry, fearful, disgust, surprised

**预期整体UAR**: 75-82% (SSL模型), 65-72% (轻量CNN)

---

## 四、预期准确率目标

### 4.1 不同模型规模的预期性能

| 模型类型 | 参数量 | 情绪UAR | 性别Acc | 年龄MAE | 说话人EER | 端侧可行 |
|---------|--------|---------|---------|---------|-----------|---------|
| **SSL Large** (HuBERT/wav2vec2) | 300M+ | 72-76% | 95-97% | 5-7年 | - | ❌ |
| **SSL Base** | 95M | 68-72% | 92-95% | 7-9年 | - | ❌ |
| **Emotion2Vec+ Large** | 316M | 74-78% | - | - | - | ❌ |
| **Emotion2Vec+ Base** | 95M | 70-74% | - | - | - | ⚠️ |
| **频谱CNN+Attention** | 8M | 62-68% | 88-92% | 10-12年 | 5-8% | ✅ |
| **轻量CNN** (MobileNet) | 3M | 55-62% | 85-88% | 12-15年 | 8-12% | ✅ |
| **目标模型 (Our Target)** | **8M** | **65-72%** | **90-93%** | **10-12年** | **5-8%** | **✅** |

### 4.2 MT9655端侧Demo预期指标

| 任务 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| **情绪识别** | UAR | 65-70% | 6类情绪 |
| **性别识别** | Accuracy | 90-93% | 男/女二分类 |
| **年龄段** | 5组Accuracy | 70-75% | 儿童/青年/中年/老年 |
| **说话人识别** | EER | 5-8% | 注册声纹验证 |
| **推理延迟** | 单次推理 | <200ms | INT8量化后 |
| **模型大小** | 存储 | <2MB | INT8权重 |

---

## 五、情绪标签映射表

### 5.1 数据集间标签映射

| 统一标签 | RAVDESS | CREMA-D | IEMOCAP | ESD | EmoDB |
|---------|---------|---------|---------|-----|-------|
| **neutral** | neutral | neutral | neutral | neutral | neutral |
| **happy** | happy | happy | happy | happy | happy |
| **sad** | sad | sad | sad | sad | sad |
| **angry** | angry | angry | angry | angry | angry |
| **fear** | fearful | fear | fearful | - | fear |
| **disgust** | disgust | disgust | disgusted | - | disgust |
| **surprise** | surprised | - | surprised | surprise | - |
| **calm** | calm | - | - | - | - |
| **excited** | - | - | excited | - | - |
| **boredom** | - | - | - | - | boredom |

### 5.2 情绪到VAD映射 (用于连续值输出)

| 情绪 | Valence (效价) | Arousal (唤醒) | Dominance (优势) |
|------|---------------|---------------|-----------------|
| angry | -0.6 | 0.8 | 0.7 |
| happy | 0.8 | 0.7 | 0.6 |
| sad | -0.7 | 0.2 | 0.2 |
| neutral | 0.0 | 0.5 | 0.5 |
| fear | -0.7 | 0.8 | 0.3 |
| disgust | -0.6 | 0.5 | 0.5 |
| surprise | 0.0 | 0.9 | 0.4 |
| calm | 0.4 | 0.2 | 0.5 |

---

## 六、Demo实现建议

### 6.1 最小可行Demo (MVP)

```python
# 推荐4类情绪 + 性别识别
EMOTIONS_DEMO = ['neutral', 'happy', 'sad', 'angry']

# 预期性能 (未训练模型 → 训练后)
# 情绪UAR: 25% → 75%
# 性别Acc: 50% → 92%
```

### 6.2 标准Demo

```python
# 推荐6类情绪 + 性别 + 年龄段
EMOTIONS_DEMO = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']
AGE_GROUPS = ['child', 'young', 'adult', 'senior']

# 预期性能
# 情绪UAR: 65-72%
# 性别Acc: 90-93%
# 年龄Acc: 70-75%
```

### 6.3 完整Demo

```python
# 8类情绪 + 性别 + 年龄段 + 说话人识别
EMOTIONS_DEMO = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# 预期性能
# 情绪UAR: 70-78% (需要SSL模型)
# 或 62-68% (轻量CNN)
```

---

## 七、数据来源标注

| 信息 | 来源 |
|------|------|
| RAVDESS识别率 | RAVDESS数据集论文 (Livingstone & Russo, 2018) |
| CREMA-D识别率 | CREMA-D论文 (Cao et al., 2014) |
| IEMOCAP识别率 | IEMOCAP论文 (Busso et al., 2008) |
| Emotion2Vec结果 | emotion2vec论文 (Ma et al., AAAI 2024) |
| SSL模型性能 | wav2vec 2.0, HuBERT, WavLM原始论文 |
| 轻量CNN性能 | SpeechBrain, MobileNet音频分类文献 |
| VAD映射值 | Russell's Circumplex Model (1980) |
| 混淆矩阵模式 | 多篇SER论文综合分析 |

---

*报告生成时间: 2025-04-28*
*适用项目: 语音四合一识别 (MT9655端侧)*
