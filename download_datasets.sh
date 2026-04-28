#!/bin/bash
# 下载相关数据集脚本

set -e
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Downloading Datasets"
echo "=========================================="

# 1. RAVDESS (已下载，验证完整性)
echo "[1/5] Checking RAVDESS..."
if [ -f "$DATA_DIR/ravdess/Audio_Speech_Actors_01-24.zip" ]; then
    echo "  RAVDESS Speech zip found"
    unzip -t "$DATA_DIR/ravdess/Audio_Speech_Actors_01-24.zip" > /dev/null 2>&1 && echo "  Valid zip" || echo "  Corrupted!"
fi
if [ -f "$DATA_DIR/ravdess/Audio_Song_Actors_01-24.zip" ]; then
    echo "  RAVDESS Song zip found"
    unzip -t "$DATA_DIR/ravdess/Audio_Song_Actors_01-24.zip" > /dev/null 2>&1 && echo "  Valid zip" || echo "  Corrupted!"
fi

# 2. EmoDB (德语，小数据集，30MB)
echo ""
echo "[2/5] Downloading EmoDB (German, 30MB)..."
EMODB_DIR="$DATA_DIR/emodb"
mkdir -p "$EMODB_DIR"
cd "$EMODB_DIR"

if [ ! -f "emodb.tar.gz" ]; then
    # 尝试多个镜像
    wget -c http://emodb.bilderbar.info/download/emodb.tar.gz 2>/dev/null || \
    wget -c https://github.com/berlin-with0ut-emotion/emodb/archive/refs/heads/main.zip 2>/dev/null || \
    echo "  Manual download required from: http://emodb.bilderbar.info/download/"
fi

if [ -f "emodb.tar.gz" ]; then
    tar -xzf emodb.tar.gz 2>/dev/null || echo "  Extract failed"
fi
cd - > /dev/null

# 3. SAVEE (英语，小数据集，~100MB)
echo ""
echo "[3/5] Downloading SAVEE (English, ~100MB)..."
SAVEE_DIR="$DATA_DIR/savee"
mkdir -p "$SAVEE_DIR"
cd "$SAVEE_DIR"

if [ ! -f "savee.zip" ]; then
    # SAVEE需要从Kaggle或官网下载
    echo "  SAVEE requires manual download from:"
    echo "    https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee"
    echo "    or http://kahlan.eps.surrey.ac.uk/savee/"
fi
cd - > /dev/null

# 4. TESS (英语，多伦多情绪语音，~100MB)
echo ""
echo "[4/5] Downloading TESS (English, ~100MB)..."
TESS_DIR="$DATA_DIR/tess"
mkdir -p "$TESS_DIR"
cd "$TESS_DIR"

if [ ! -f "tess.zip" ]; then
    echo "  TESS requires manual download from:"
    echo "    https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"
fi
cd - > /dev/null

# 5. JL-Corpus (英语，~200MB)
echo ""
echo "[5/5] Downloading JL-Corpus (English, ~200MB)..."
JL_DIR="$DATA_DIR/jl-corpus"
mkdir -p "$JL_DIR"
cd "$JL_DIR"

if [ ! -d "jl-corpus" ]; then
    echo "  JL-Corpus requires manual download from:"
    echo "    https://www.kaggle.com/datasets/tli7544/jl-corpus"
    echo "    or https://github.com/jacklanda/JL-Corpus"
fi
cd - > /dev/null

echo ""
echo "=========================================="
echo "Dataset Download Summary"
echo "=========================================="
echo "Downloaded/Verified:"
for d in ravdess emodb savee tess jl-corpus cremad esd iemocap; do
    dir="$DATA_DIR/$d"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        count=$(find "$dir" -name "*.wav" 2>/dev/null | wc -l)
        echo "  ✅ $d: $count wav files"
    else
        echo "  ❌ $d: not downloaded"
    fi
done

echo ""
echo "Manual download required:"
echo "  - CREMA-D: https://www.kaggle.com/datasets/ejlok1/cremad"
echo "  - ESD: https://github.com/HLTSingapore/Emotional-Speech-Data"
echo "  - IEMOCAP: https://sail.usc.edu/iemocap/ (requires registration)"
echo "  - SAVEE: http://kahlan.eps.surrey.ac.uk/savee/"
echo "  - TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"
echo "  - JL-Corpus: https://www.kaggle.com/datasets/tli7544/jl-corpus"
