#!/bin/bash
# 模型下载脚本 - 使用多种方法尝试下载
# 用法: bash download_models.sh [method]
# method: python|git|wget|aria2|curl

set -e

BASE_DIR="/data/mikexu/speech_analysis_project"
MODEL_DIR="$BASE_DIR/models/pretrained"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# 模型列表
MODELS=(
    "emotion2vec_plus_base:emotion2vec/emotion2vec_plus_base"
    "wav2vec2_base_960h:facebook/wav2vec2-base-960h"
    "hubert_base_ls960:facebook/hubert-base-ls960"
    "wavlm_base:microsoft/wavlm-base"
    "ecapa_tdnn:speechbrain/spkrec-ecapa-voxceleb"
)

METHOD=${1:-python}

echo "========================================"
echo "Model Download Script"
echo "Method: $METHOD"
echo "Target: $MODEL_DIR"
echo "========================================"

case $METHOD in
    python)
        echo "Using Python huggingface_hub..."
        cd "$BASE_DIR"
        python3 download_all_models.py 2>&1 | tee "$LOG_DIR/model_download_$(date +%Y%m%d_%H%M%S).log"
        ;;
    
    git)
        echo "Using git clone..."
        cd "$MODEL_DIR"
        for model in "${MODELS[@]}"; do
            name=$(echo $model | cut -d: -f1)
            repo=$(echo $model | cut -d: -f2)
            
            if [ -d "$name" ]; then
                echo "[$name] Already exists, skipping"
                continue
            fi
            
            echo "[$name] Cloning $repo..."
            git clone "https://huggingface.co/$repo" "$name" 2>&1 | tee "$LOG_DIR/${name}_git.log" || {
                echo "[$name] Git clone failed, trying next..."
                continue
            }
        done
        ;;
    
    wget)
        echo "Using wget..."
        cd "$MODEL_DIR"
        
        # Emotion2Vec+
        mkdir -p emotion2vec_plus_base
        wget -c -O emotion2vec_plus_base/config.json "https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json" || true
        wget -c -O emotion2vec_plus_base/pytorch_model.bin "https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin" || true
        
        # wav2vec 2.0
        mkdir -p wav2vec2_base_960h
        wget -c -O wav2vec2_base_960h/config.json "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json" || true
        wget -c -O wav2vec2_base_960h/pytorch_model.bin "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin" || true
        
        # HuBERT
        mkdir -p hubert_base_ls960
        wget -c -O hubert_base_ls960/config.json "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json" || true
        wget -c -O hubert_base_ls960/pytorch_model.bin "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin" || true
        
        # WavLM
        mkdir -p wavlm_base
        wget -c -O wavlm_base/config.json "https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json" || true
        wget -c -O wavlm_base/pytorch_model.bin "https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin" || true
        
        # ECAPA-TDNN
        mkdir -p ecapa_tdnn
        wget -c -O ecapa_tdnn/hyperparams.yaml "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.yaml" || true
        wget -c -O ecapa_tdnn/embedding_model.ckpt "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt" || true
        ;;
    
    aria2)
        echo "Using aria2c..."
        if ! command -v aria2c &> /dev/null; then
            echo "aria2c not found, installing..."
            sudo apt-get update && sudo apt-get install -y aria2
        fi
        
        cd "$MODEL_DIR"
        
        # 创建下载列表
        cat > aria2_download_list.txt << 'EOF'
https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json
    dir=emotion2vec_plus_base
https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin
    dir=emotion2vec_plus_base
https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
    dir=wav2vec2_base_960h
https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
    dir=wav2vec2_base_960h
https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json
    dir=hubert_base_ls960
https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin
    dir=hubert_base_ls960
https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json
    dir=wavlm_base
https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin
    dir=wavlm_base
EOF
        
        aria2c -i aria2_download_list.txt -j 4 -x 4 -c --auto-file-renaming=false 2>&1 | tee "$LOG_DIR/aria2_download.log"
        ;;
    
    curl)
        echo "Using curl..."
        cd "$MODEL_DIR"
        
        mkdir -p emotion2vec_plus_base
        curl -L -C - -o emotion2vec_plus_base/config.json "https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/config.json" || true
        curl -L -C - -o emotion2vec_plus_base/pytorch_model.bin "https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/pytorch_model.bin" || true
        
        mkdir -p wav2vec2_base_960h
        curl -L -C - -o wav2vec2_base_960h/config.json "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json" || true
        curl -L -C - -o wav2vec2_base_960h/pytorch_model.bin "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin" || true
        
        mkdir -p hubert_base_ls960
        curl -L -C - -o hubert_base_ls960/config.json "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json" || true
        curl -L -C - -o hubert_base_ls960/pytorch_model.bin "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin" || true
        
        mkdir -p wavlm_base
        curl -L -C - -o wavlm_base/config.json "https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json" || true
        curl -L -C - -o wavlm_base/pytorch_model.bin "https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin" || true
        ;;
    
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: bash download_models.sh [python|git|wget|aria2|curl]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Download attempt complete"
echo "========================================"
echo ""
echo "Checking status..."
cd "$BASE_DIR"
python3 check_model_status.py
