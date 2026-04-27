# 部署指南

## 概述

将训练好的多任务语音模型部署到MT9655 TV SoC端侧。

## 部署流程

```
PyTorch FP32 Model
      │
      ▼
┌─────────────┐
│  PTQ/QAT    │  量化到INT8
│  Quantization│
└─────────────┘
      │
      ▼
┌─────────────┐
│  ONNX Export │  中间格式
└─────────────┘
      │
      ▼
┌─────────────┐
│ TFLite Export│  端侧格式
│   INT8       │
└─────────────┘
      │
      ▼
┌─────────────┐
│ MT9655 Demo  │  端侧推理
│  (TFLite)    │
└─────────────┘
```

## 1. 模型量化

### 1.1 训练后量化 (PTQ)

```bash
python quantization/ptq.py \
    --model checkpoints/best_model.pt \
    --data data/splits/train.json \
    --output checkpoints/model_int8_ptq.pt \
    --n_samples 500
```

### 1.2 量化感知训练 (QAT)

```bash
python quantization/qat.py \
    --model checkpoints/best_model.pt \
    --data_dir data/splits \
    --output checkpoints/model_int8_qat.pt \
    --qat_epochs 10 \
    --lr 1e-4
```

**推荐**: QAT精度更高，但需要额外训练时间。

## 2. 模型导出

### 2.1 导出ONNX

```bash
python quantization/export_onnx.py \
    --model checkpoints/best_model.pt \
    --output checkpoints/model.onnx \
    --simplify
```

### 2.2 导出TFLite

```bash
# 方法1: 直接导出 (推荐)
python quantization/export_tflite.py \
    --model checkpoints/best_model.pt \
    --output checkpoints/model.tflite \
    --method direct

# 方法2: 通过ONNX导出 (备用)
python quantization/export_tflite.py \
    --model checkpoints/best_model.pt \
    --output checkpoints/model_int8.tflite \
    --method onnx \
    --quantize \
    --data data/splits/train.json \
    --n_calib 100
```

## 3. MT9655端侧部署

### 3.1 环境准备

```bash
# 安装TFLite Runtime
pip install tflite-runtime

# 或安装完整TensorFlow
pip install tensorflow
```

### 3.2 运行Demo

```bash
python demo/demo_mt9655.py \
    --model checkpoints/model_int8.tflite \
    --audio test.wav \
    --threads 4 \
    --benchmark
```

### 3.3 集成到MT9655应用

```python
# 在MT9655应用中集成
import tflite_runtime.interpreter as tflite

class SpeechAnalyzer:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=4  # 使用4核
        )
        self.interpreter.allocate_tensors()
    
    def analyze(self, audio_data):
        # 预处理
        input_data = self.preprocess(audio_data)
        
        # 推理
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        
        # 获取结果
        emotion = self.interpreter.get_tensor(self.emotion_output_index)
        gender = self.interpreter.get_tensor(self.gender_output_index)
        age = self.interpreter.get_tensor(self.age_output_index)
        
        return {
            'emotion': emotion,
            'gender': gender,
            'age': age
        }
```

## 4. 性能优化

### 4.1 内存优化

```python
# 使用内存映射加载模型
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_preserve_all_tensors=False
)
```

### 4.2 推理优化

```python
# 启用XNNPACK delegate
interpreter = tflite.Interpreter(
    model_path=model_path,
    num_threads=4,
    experimental_use_xnnpack=True
)
```

### 4.3 批处理推理

```python
# 批量处理多个音频
batch_size = 4
input_data = np.stack([preprocess(audio) for audio in audio_list])
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
```

## 5. 声纹注册

### 5.1 注册说话人

```bash
python demo/register_speaker.py \
    --model checkpoints/best_model.pt \
    --action register \
    --name "张三" \
    --audio "samples/zhangsan_*.wav" \
    --registry speaker_registry.pkl
```

### 5.2 验证说话人

```bash
python demo/register_speaker.py \
    --model checkpoints/best_model.pt \
    --action verify \
    --audio test.wav \
    --registry speaker_registry.pkl \
    --threshold 0.5
```

## 6. 多语言支持

### 6.1 语言优先级

1. **英语** (优先)
   - RAVDESS, CREMA-D, ESD (英文部分)
   - Common Voice English

2. **西班牙语**
   - Common Voice Spanish
   - 需要额外数据

3. **法语/德语/意大利语/日语**
   - Common Voice 各语言版本
   - 零样本/少样本迁移

### 6.2 多语言部署

```python
# 根据语言选择模型或调整阈值
language_configs = {
    'en': {'emotion_threshold': 0.5, 'speaker_threshold': 0.5},
    'es': {'emotion_threshold': 0.45, 'speaker_threshold': 0.5},
    'fr': {'emotion_threshold': 0.45, 'speaker_threshold': 0.5}
}
```

## 7. 预期性能

### 7.1 PC端 (RTX 3090)

| 指标 | 数值 |
|------|------|
| 预处理 | ~50ms |
| 推理 (FP32) | ~100ms |
| 推理 (INT8) | ~50ms |
| 端到端 | ~150ms |

### 7.2 MT9655端

| 指标 | 数值 |
|------|------|
| 预处理 | ~100ms |
| 推理 (INT8) | ~300-500ms |
| 端到端 | ~500ms |
| 内存占用 | ~20MB |

## 8. 故障排查

### 8.1 TFLite转换失败

**问题**: `ValueError: Cannot convert model to TFLite`

**解决方案:**
1. 检查ONNX opset版本 (推荐13)
2. 使用 `--method onnx` 备用方案
3. 安装 ai-edge-torch: `pip install ai-edge-torch`

### 8.2 INT8精度下降

**问题**: 量化后精度明显下降

**解决方案:**
1. 使用QAT代替PTQ
2. 增加校准样本数
3. 检查输入量化范围
4. 使用混合精度 (仅权重INT8)

### 8.3 推理延迟过高

**问题**: MT9655上推理超过1秒

**解决方案:**
1. 使用轻量模型 `--lightweight`
2. 减少输入长度 (target_length=200)
3. 启用XNNPACK
4. 使用单线程 (有时多线程有开销)

## 9. 文件清单

| 文件 | 说明 |
|------|------|
| `checkpoints/model_int8.tflite` | 端侧模型 |
| `speaker_registry.pkl` | 声纹库 |
| `demo/demo_mt9655.py` | 端侧Demo |
| `demo/register_speaker.py` | 声纹注册工具 |

## 10. 参考资料

- [TFLite Guide](https://www.tensorflow.org/lite/guide)
- [MTK NeuroPilot](https://neuropilot.mediatek.com/)
- [ONNX to TFLite](https://github.com/onnx/onnx-tensorflow)
