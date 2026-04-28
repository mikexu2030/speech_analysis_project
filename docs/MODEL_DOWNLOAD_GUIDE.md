# Top3 模型系列下载指南

## 网络恢复后执行步骤

### 1. 检查网络状态
```bash
cd /data/mikexu/speech_analysis_project
python3 check_model_status.py
```

### 2. 下载所有缺失模型（自动断点续传）
```bash
cd /data/mikexu/speech_analysis_project
python3 download_all_models.py
```

### 3. 下载指定模型
```bash
cd /data/mikexu/speech_analysis_project
python3 check_model_status.py --download emotion2vec_plus_base
python3 check_model_status.py --download wav2vec2_base_960h
python3 check_model_status.py --download hubert_base_ls960
python3 check_model_status.py --download wavlm_base
python3 check_model_status.py --download ecapa_tdnn
```

---

## 模型清单

### Series 1: Emotion2Vec+
| 模型 | 参数量 | 用途 | HuggingFace链接 |
|------|--------|------|----------------|
| emotion2vec_plus_base | 95M | 情绪识别 | https://huggingface.co/emotion2vec/emotion2vec_plus_base |

### Series 2: wav2vec 2.0
| 模型 | 参数量 | 用途 | HuggingFace链接 |
|------|--------|------|----------------|
| wav2vec2_base_960h | 95M | 自监督表示 | https://huggingface.co/facebook/wav2vec2-base-960h |

### Series 3: HuBERT / WavLM
| 模型 | 参数量 | 用途 | HuggingFace链接 |
|------|--------|------|----------------|
| hubert_base_ls960 | 95M | 自监督表示 | https://huggingface.co/facebook/hubert-base-ls960 |
| wavlm_base | 95M | 自监督表示 | https://huggingface.co/microsoft/wavlm-base |

### Bonus: Speaker Recognition
| 模型 | 参数量 | 用途 | HuggingFace链接 |
|------|--------|------|----------------|
| ecapa_tdnn | 6.2M | 声纹识别 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |

---

## 备用下载方法

### 方法1: 使用 git-lfs
```bash
# 安装 git-lfs
sudo apt-get install git-lfs
git lfs install

# 克隆模型仓库
cd /data/mikexu/speech_analysis_project/models/pretrained
git clone https://huggingface.co/emotion2vec/emotion2vec_plus_base
git clone https://huggingface.co/facebook/wav2vec2-base-960h
git clone https://huggingface.co/facebook/hubert-base-ls960
git clone https://huggingface.co/microsoft/wavlm-base
git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
```

### 方法2: 使用 wget 下载关键文件
```bash
cd /data/mikexu/speech_analysis_project/models/pretrained

# Emotion2Vec+ Base
mkdir -p emotion2vec_plus_base
cd emotion2vec_plus_base
wget https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json
wget https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin
wget https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/preprocessor_config.json
cd ..

# wav2vec 2.0 Base
mkdir -p wav2vec2_base_960h
cd wav2vec2_base_960h
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json
cd ..

# HuBERT Base
mkdir -p hubert_base_ls960
cd hubert_base_ls960
wget https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json
wget https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/hubert-base-ls960/resolve/main/preprocessor_config.json
cd ..

# WavLM Base
mkdir -p wavlm_base
cd wavlm_base
wget https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json
wget https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/microsoft/wavlm-base/resolve/main/preprocessor_config.json
cd ..

# ECAPA-TDNN
mkdir -p ecapa_tdnn
cd ecapa_tdnn
wget https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.yaml
wget https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt
cd ..
```

### 方法3: 使用 aria2c 多线程下载
```bash
# 安装 aria2
sudo apt-get install aria2

# 创建下载列表
cat > model_download_list.txt << 'EOF'
https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json
https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin
https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json
https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin
https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json
https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin
EOF

# 多线程下载
aria2c -i model_download_list.txt -j 4 -x 4 -c
```

### 方法4: 使用 curl
```bash
# Emotion2Vec+ Base
mkdir -p emotion2vec_plus_base
curl -L -o emotion2vec_plus_base/config.json https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json
curl -L -o emotion2vec_plus_base/pytorch_model.bin https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin

# 其他模型类似...
```

---

## 下载后验证

```bash
cd /data/mikexu/speech_analysis_project
python3 check_model_status.py
```

预期输出：
```
Progress: 5/5 models complete
```

---

## 模型文件结构验证

下载完成后，模型目录应包含以下文件：

### Emotion2Vec+ Base
```
emotion2vec_plus_base/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重 (~380MB)
└── preprocessor_config.json  # 预处理配置
```

### wav2vec 2.0 Base
```
wav2vec2_base_960h/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重 (~380MB)
├── preprocessor_config.json # 预处理配置
└── vocab.json               # CTC词汇表
```

### HuBERT Base
```
hubert_base_ls960/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重 (~380MB)
└── preprocessor_config.json  # 预处理配置
```

### WavLM Base
```
wavlm_base/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重 (~380MB)
└── preprocessor_config.json  # 预处理配置
```

### ECAPA-TDNN
```
ecapa_tdnn/
├── hyperparams.yaml         # 超参数配置
├── embedding_model.ckpt     # 模型权重 (~25MB)
└── label_encoder.txt        # 标签编码器
```

---

## 常见问题

### Q: 下载中断怎么办？
A: 脚本支持断点续传，重新运行 `python3 download_all_models.py` 即可。

### Q: 网络不稳定？
A: 使用 `aria2c` 或 `wget -c` 支持断点续传。

### Q: 磁盘空间不足？
A: 确保至少有 2GB 可用空间（5个模型约 1.5GB）。

### Q: 下载速度慢？
A: 使用 `aria2c -x 4 -j 4` 多线程下载，或设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 download_all_models.py
```

---

## 下一步

所有模型下载完成后，执行：
```bash
# 1. 模型评测对比
python3 scripts/evaluate_models.py

# 2. 数据集准备
python3 scripts/prepare_datasets.py

# 3. 模型训练
python3 scripts/train_model.py

# 4. 模型导出
python3 scripts/export_model.py
```
