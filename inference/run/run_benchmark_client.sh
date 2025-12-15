#!/usr/bin/env bash

# SGLang benchmark client runner
# This script runs the benchmark client against an SGLang server
#
# Usage: ./run_benchmark_client.sh [OPTIONS]
#   Options:
#     --host HOST            Server hostname/IP (default: bmk-server for single-node, localhost for multi-node)
#     --port PORT            Server port (default: 8888)
#     --network-mode MODE    Docker network mode: bridge or host (default: bridge)
#     --conc CONC            Concurrency (default: 8)
#     --num-prompts NUM      Number of prompts (default: 48)
#     --isl ISL              Input sequence length (default: 512)
#     --osl OSL              Output sequence length (default: 512)
#     --random-range-ratio RATIO  Random range ratio (default: 0.8)
#     --output-file FILE     Output file path (default: auto-generated)
#     --test-name NAME       Test name for output file (default: benchmark)

set -e

# ============================================
# CONFIGURATION
# ============================================
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
export IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"

# Default values
HOST=""
PORT=8888
NETWORK_MODE="bridge"
CONC=8
NUM_PROMPTS=48
ISL=512
OSL=512
RANDOM_RANGE_RATIO=0.8
OUTPUT_FILE=""
TEST_NAME="benchmark"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --network-mode)
            NETWORK_MODE="$2"
            shift 2
            ;;
        --conc)
            CONC="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --isl)
            ISL="$2"
            shift 2
            ;;
        --osl)
            OSL="$2"
            shift 2
            ;;
        --random-range-ratio)
            RANDOM_RANGE_RATIO="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --test-name)
            TEST_NAME="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option '$1'"
            echo "Usage: $0 [--host HOST] [--port PORT] [--network-mode MODE] [--conc CONC] [--num-prompts NUM] [--isl ISL] [--osl OSL] [--random-range-ratio RATIO] [--output-file FILE] [--test-name NAME]"
            exit 1
            ;;
    esac
done

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to get to inference/ directory
WORKSPACE="$(cd "$WORKSPACE/.." && pwd)"

# Set default host based on network mode
if [ -z "$HOST" ]; then
    if [ "$NETWORK_MODE" == "host" ]; then
        HOST="localhost"
    else
        HOST="bmk-server"
    fi
fi

# Generate output file name if not provided
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="/workspace/outputs/${TEST_NAME}_${NUM_PROMPTS}req_custom_rccl.json"
fi

client_name="bmk-client"

echo "=== Running benchmark client ==="
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Network mode: $NETWORK_MODE"
echo "  Concurrency: $CONC"
echo "  Num prompts: $NUM_PROMPTS"
echo "  Input length: $ISL"
echo "  Output length: $OSL"
echo "  Output file: $OUTPUT_FILE"
echo ""

# Build docker network argument
DOCKER_NETWORK_ARG=""
if [ "$NETWORK_MODE" == "host" ]; then
    DOCKER_NETWORK_ARG="--network=host"
else
    DOCKER_NETWORK_ARG="--network=$NETWORK_MODE"
fi

docker run --rm $DOCKER_NETWORK_ARG --name=$client_name \
-v $HF_HUB_CACHE:$HF_HUB_CACHE:ro \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN \
--entrypoint=python3 \
$IMAGE \
-m sglang.bench_serving \
--host $HOST --port $PORT --model $MODEL \
--dataset-name random --backend sglang \
--dataset-path $HF_HUB_CACHE/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
--profile \
--max-concurrency $CONC \
--random-input-len $ISL \
--random-output-len $OSL \
--random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $NUM_PROMPTS \
--output-file $OUTPUT_FILE

echo "=== Benchmark client completed ==="

