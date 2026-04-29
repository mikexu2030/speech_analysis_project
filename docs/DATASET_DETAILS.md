# 数据集详细文档

## 已整合数据集

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **样本数**: 1440
- **说话人**: 24 (12男12女)
- **情绪**: 8类 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **语言**: 英语
- **场景**: 实验室表演
- **获取**: Zenodo/Kaggle免费下载 (CC BY-ND 4.0)
- **路径**: data/raw/ravdess/

### TESS (Toronto Emotional Speech Set)
- **样本数**: 2600 (已下载)
- **说话人**: 2 (女性)
- **情绪**: 7类
- **语言**: 英语
- **场景**: 实验室表演
- **获取**: Kaggle免费下载 (CC BY-NC-ND 4.0)

### EMODB (Berlin Database of Emotional Speech)
- **样本数**: 320 (已下载)
- **说话人**: 10 (5男5女)
- **情绪**: 7类
- **语言**: 德语
- **场景**: 实验室表演
- **获取**: 官网免费下载

## 数据划分 (按说话人隔离)

| 数据集 | 样本数 | 说话人数 | 来源 |
|--------|--------|---------|------|
| Train | 3749 | 25 | RAVDESS 900 + TESS 2600 + EMODB 249 |
| Val | 251 | 5 | RAVDESS 180 + EMODB 71 |
| Test | 360 | 6 | RAVDESS 360 |

## 待下载数据集

| 数据集 | 规模 | 语言 | 场景 | 获取难度 |
|--------|------|------|------|---------|
| MSP-Podcast | ~5046小时 | 英语 | 真实播客 | 需学术协议 |
| IEMOCAP | ~12小时 | 英语 | 半自然对话 | 需机构签名 |
| CREMA-D | 7442片段 | 英语 | 实验室表演 | 免费 |
| MELD | 13708片段 | 英语 | 电视剧对话 | 免费 |
| Common Voice | 大规模 | 多语言 | 自然 | 免费 |

## 核心问题

1. **表演数据 vs 真实数据**: 当前数据集均为表演数据, 与真实场景有差距
2. **语言单一**: 主要为英语, 中文情绪数据缺乏
3. **规模不足**: 4360样本对于深度学习仍偏少
