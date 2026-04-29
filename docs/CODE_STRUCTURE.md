# 代码结构详细文档

## 目录结构

```
speech_analysis_project/
├── configs/              # 配置文件
│   ├── train_config.yaml
│   └── model_config.yaml
├── data/                 # 数据集
│   ├── raw/              # 原始数据
│   ├── processed/        # 预处理数据
│   ├── splits/           # 划分数据
│   ├── expand_datasets.py
│   ├── preprocess_multidata.py
│   └── download_datasets_simple.py
├── docs/                 # 文档
│   ├── TECH_INSIGHT_SUMMARY.md
│   ├── DATASET_DETAILS.md
│   ├── MODEL_DETAILS.md
│   ├── DEPLOYMENT_DETAILS.md
│   ├── CODE_STRUCTURE.md
│   └── SESSION_RESUME.md
├── models/               # 模型定义
│   ├── multitask_model.py
│   ├── backbone.py
│   └── heads.py
├── training/             # 训练
│   ├── trainer.py
│   ├── train.py
│   ├── finetune_pretrained.py
│   └── quantize_export.py
├── utils/                # 工具
│   ├── data_loader.py
│   ├── audio_utils.py
│   ├── data_augmentation.py
│   └── git_auto_push.py
├── tests/                # 测试
│   └── test_runner.py
├── mcp-servers/          # MCP服务
│   └── docx_reader.py
├── checkpoints/          # 检查点
├── results/              # 结果
│   └── evaluation/
├── logs/                 # 日志
├── export.py             # 模型导出
├── demo_inference.py     # 推理演示
├── demo_end2end.py       # 端到端演示
├── batch_evaluate.py     # 批量评估
├── speech_analyzer.py    # 主分析器
├── requirements.txt      # 依赖
└── README.md             # 项目说明
```

## 核心文件说明

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| models/multitask_model.py | 多任务模型 | MultiTaskSpeechModel |
| models/backbone.py | 骨干网络 | SpectralBackbone, LightweightBackbone |
| models/heads.py | 任务头 | SpeakerHead, AgeHead, GenderHead, EmotionHead |
| training/trainer.py | 训练器 | MultiTaskTrainer |
| training/train.py | 训练脚本 | main() |
| training/finetune_pretrained.py | 预训练微调 | finetune_hubert() |
| utils/data_loader.py | 数据加载 | SpeechDataset, collate_fn |
| utils/audio_utils.py | 音频处理 | load_audio, mel_spectrogram, mfcc |
| export.py | 模型导出 | export_onnx, quantize_int8 |
| demo_end2end.py | 端到端演示 | run_inference |
| speech_analyzer.py | 主分析器 | SpeechAnalyzer |

## 技术栈

- **框架**: PyTorch 2.x, transformers 4.x
- **音频**: torchaudio, librosa
- **推理**: ONNX Runtime
- **量化**: PyTorch quantization
- **数据**: numpy, pandas
- **日志**: logging
