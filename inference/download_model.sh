#!/usr/bin/env bash
# Download DeepSeek-R1 model using HuggingFace CLI

set -e

source "$(dirname "$0")/benchmark.env"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Please set it in benchmark.env"
    exit 1
fi

echo "=== Downloading DeepSeek-R1 Model ==="
echo "Model: $MODEL"
echo "Cache: $HF_HUB_CACHE"
echo "Size: ~600GB (this will take time)"
echo ""

mkdir -p "$HF_HUB_CACHE"

echo "Starting download..."
huggingface-cli download "$MODEL" \
    --cache-dir "$HF_HUB_CACHE" \
    --resume-download \
    --token "$HF_TOKEN"

echo ""
echo "✓ Model download complete!"
echo "Location: $HF_HUB_CACHE/models--deepseek-ai--DeepSeek-R1-0528"
