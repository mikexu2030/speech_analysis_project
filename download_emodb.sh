#!/bin/bash
# 下载EmoDB数据集 (德语，小数据集，适合快速测试)

set -e
DATA_DIR="data/raw/emodb"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading EmoDB (Berlin Database of Emotional Speech)..."
echo "Size: ~30MB"
echo "Language: German"
echo "Emotions: 7 (neutral, happy, sad, angry, fear, disgust, boredom)"
echo "Samples: 535"
echo ""

# 尝试多个下载源
if [ ! -f "emodb.tar.gz" ]; then
    echo "Trying primary source..."
    wget -c --timeout=30 http://emodb.bilderbar.info/download/emodb.tar.gz 2>/dev/null && echo "  Success!" || echo "  Failed"
fi

if [ ! -f "emodb.tar.gz" ]; then
    echo "Trying GitHub mirror..."
    wget -c --timeout=30 https://github.com/berlin-with0ut-emotion/emodb/archive/refs/heads/main.zip 2>/dev/null && echo "  Success!" || echo "  Failed"
fi

if [ -f "emodb.tar.gz" ]; then
    echo "Extracting..."
    tar -xzf emodb.tar.gz && echo "  Extracted!" || echo "  Extract failed"
    
    # 统计
    count=$(find . -name "*.wav" 2>/dev/null | wc -l)
    echo "WAV files: $count"
else
    echo "Download failed. Manual download required:"
    echo "  http://emodb.bilderbar.info/download/"
fi
