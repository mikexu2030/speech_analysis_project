# 语音四合一识别项目 - 进展报告

**日期**: 2026-04-28
**项目路径**: /data/mikexu/speech_analysis_project

---

## 本次会话完成的工作

### 1. 50epoch训练完成并评估
- **训练配置**: 50 epochs, batch_size=32, CPU训练
- **训练时间**: ~40分钟
- **早停**: Epoch 15（15个epoch无改善）
- **最佳检查点**: checkpoint_epoch_15.pt

### 2. 关键Bug修复

#### Bug 1: 数据预处理不一致
- **问题**: demo_inference使用n_fft=2048/hop=512，训练使用n_fft=1024/hop=256
- **影响**: 评估结果完全错误（情绪14%→32%，性别0%→96%）
- **修复**: 统一使用训练时的预处理参数
- **文件**: demo_inference.py, batch_evaluate.py

#### Bug 2: 模型输出类别数不匹配
- **问题**: 模型输出7类，但评估期望8类（包含surprised）
- **影响**: surprised样本无法正确评估
- **修复**: 将surprised（标签7）映射到disgust（标签6）
- **文件**: batch_evaluate.py

#### Bug 3: best_model保存逻辑
- **问题**: 基于val_loss保存，但val_loss第1个epoch后未改善
- **影响**: best_model.pt实际上是随机初始化的epoch 0
- **修复**: 使用epoch 15的检查点作为最佳模型
- **文件**: export.py（已修正benchmark函数参数）

### 3. 模型INT8量化
- **原始模型**: 16.21 MB (FP32 ONNX)
- **量化模型**: 4.23 MB (INT8 ONNX)
- **压缩率**: 73.9%
- **精度损失**: 极小（情绪32%→33%，性别96.33%不变）

### 4. 端到端演示脚本
- **文件**: demo_end2end.py
- **功能**: 音频文件 → 预处理 → 推理 → 结果展示
- **性能**: 预处理7ms + 推理37ms = 总时间44ms（CPU）
- **预估MT9655**: 总时间200-400ms

---

## 当前模型性能

### FP32模型 (model.onnx, 16.21MB)
| 任务 | 准确率 | 备注 |
|------|--------|------|
| 情绪识别 | 32.00% | 7类分类，随机基线14% |
| 性别识别 | 96.33% | 2类分类，非常好 |
| 推理速度 | 8.22 ms | CPU单线程 |

### INT8模型 (model_int8.onnx, 4.23MB)
| 任务 | 准确率 | 备注 |
|------|--------|------|
| 情绪识别 | 33.00% | 与FP32几乎相同 |
| 性别识别 | 96.33% | 与FP32相同 |
| 推理速度 | 36.85 ms | CPU单线程（INT8优化未生效） |

### 情绪混淆矩阵 (INT8, 300样本)
```
              neutral  calm  happy   sad  angry fearful  disgust+surprised
neutral            24     0     30     0      5       1                  0
calm                4     3     11     0     16       0                  6
happy               5     2     24     0      9       0                  1
sad                 0     3      7     1     23       1                  5
angry               0     1     18     0     20       0                  1
fearful             2     2     12     0     14      11                  2
disgust+surprised   0     0      6     0     16       1                 16
```

---

## 项目文件清单

### 核心代码
| 文件 | 功能 |
|------|------|
| models/multitask_model.py | 多任务模型定义 |
| training/trainer.py | 训练器 |
| training/train.py | 训练脚本 |
| export.py | 模型导出和量化 |
| demo_inference.py | 推理演示 |
| demo_end2end.py | 端到端演示 |
| batch_evaluate.py | 批量评估 |
| utils/data_loader.py | 数据加载器 |
| utils/audio_utils.py | 音频处理工具 |

### 模型文件
| 文件 | 大小 | 说明 |
|------|------|------|
| models/exported/model.onnx | 16.21 MB | FP32 ONNX模型 |
| models/exported/model_int8.onnx | 4.23 MB | INT8量化模型 |
| checkpoints/.../checkpoint_epoch_15.pt | ~16 MB | 最佳训练检查点 |

### 数据文件
| 文件 | 样本数 | 说明 |
|------|--------|------|
| data/splits/train.json | 960 | 训练集 |
| data/splits/val.json | 180 | 验证集 |
| data/splits/test.json | 300 | 测试集 |

---

## 已知问题与限制

1. **情绪识别准确率偏低**: 32%对于7类分类有提升空间（目标>50%）
   - 原因: RAVDESS数据集小（960训练样本），演员朗读式语音
   - 解决: 需要更多真实场景数据集（Common Voice, IEMOCAP等）

2. **surprised情绪缺失**: RAVDESS的surprised样本被合并到disgust
   - 原因: 原始模型设计为7类
   - 解决: 重新训练8类模型

3. **年龄估计未验证**: 当前数据集没有年龄标签
   - 解决: 需要Common Voice等带年龄标签的数据集

4. **声纹识别未验证**: 当前模型输出嵌入向量，但未测试说话人识别准确率
   - 解决: 需要设计声纹注册和比对流程

5. **网络不可用**: 无法下载外部数据集
   - 原因: 当前环境网络限制
   - 解决: 需要手动下载或使用代理

---

## 下一步建议

### 短期（1-2天）
1. **修复训练保存逻辑**: 改为基于val_emotion_uar保存best_model
2. **声纹识别验证**: 测试说话人识别准确率
3. **多语言支持**: 准备西班牙语、法语等数据集

### 中期（1周）
1. **下载更多数据集**: Common Voice（多语言，带年龄/性别标签）
2. **数据增强**: 添加SpecAugment、时间拉伸等
3. **模型优化**: 尝试更轻量的骨干网络

### 长期（2-4周）
1. **端到端优化**: 减少预处理时间，优化ONNXRuntime配置
2. **MT9655部署**: 测试INT8模型在目标设备上的性能
3. **持续学习**: 设计在线学习机制，支持新说话人注册

---

## 快速开始命令

```bash
# 1. 进入项目目录
cd /data/mikexu/speech_analysis_project

# 2. 运行端到端演示
python3 demo_end2end.py --audio data/raw/ravdess/audio_speech/Actor_11/03-01-05-01-02-01-11.wav

# 3. 批量评估
python3 batch_evaluate.py --model models/exported/model_int8.onnx --test_json data/splits/test.json

# 4. 导出ONNX并量化
python3 export.py --model checkpoints/ravdess_multitask_50ep/checkpoints/checkpoint_epoch_15.pt \
  --output_dir models/exported --export_onnx --quantize --benchmark

# 5. 训练（如需重新训练）
python3 training/train.py --config configs/train_config.yaml \
  --model_config configs/model_config.yaml --data_dir data/splits \
  --output_dir checkpoints --exp_name ravdess_multitask --device cpu
```

---

*报告生成时间: 2026-04-28*
*模型版本: checkpoint_epoch_15 (50epoch训练, 早停于epoch 15)*
