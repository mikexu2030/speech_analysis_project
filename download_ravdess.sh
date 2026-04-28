#!/bin/bash
set -e
OUTDIR="data/raw/ravdess"
mkdir -p "$OUTDIR"

# Download with resume support
echo "[$(date)] Starting RAVDESS download..."

# Audio Speech Actors
echo "Downloading Audio_Speech_Actors_01-24.zip..."
curl -L -C - -o "$OUTDIR/Audio_Speech_Actors_01-24.zip" \
  "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip" \
  || wget -c -O "$OUTDIR/Audio_Speech_Actors_01-24.zip" \
  "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

# Audio Song Actors  
echo "Downloading Audio_Song_Actors_01-24.zip..."
curl -L -C - -o "$OUTDIR/Audio_Song_Actors_01-24.zip" \
  "https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip" \
  || wget -c -O "$OUTDIR/Audio_Song_Actors_01-24.zip" \
  "https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip"

echo "[$(date)] Download complete!"
echo "Verifying files..."
unzip -t "$OUTDIR/Audio_Speech_Actors_01-24.zip" >/dev/null && echo "Speech zip OK" || echo "Speech zip FAILED"
unzip -t "$OUTDIR/Audio_Song_Actors_01-24.zip" >/dev/null && echo "Song zip OK" || echo "Song zip FAILED"
