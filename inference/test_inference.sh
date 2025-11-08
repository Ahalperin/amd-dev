#!/usr/bin/env bash

# InferenceMAX Benchmark Test Script
# Usage: ./test_inference.sh [NUM_PROMPTS]
# Example: ./test_inference.sh 100  (runs 100 requests)
# Default: 50 requests if not specified

set -e

# Parse command-line arguments
NUM_PROMPTS=${1:-50}  # Use first argument or default to 50

# Validate NUM_PROMPTS is a positive integer
if ! [[ "$NUM_PROMPTS" =~ ^[0-9]+$ ]] || [ "$NUM_PROMPTS" -lt 1 ]; then
    echo "ERROR: NUM_PROMPTS must be a positive integer"
    echo "Usage: $0 [NUM_PROMPTS]"
    echo "Example: $0 100"
    exit 1
fi

echo "=== Benchmark Configuration ==="
echo "Number of prompts: $NUM_PROMPTS"
echo ""

# HF_TOKEN removed for security

export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL_PATH="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"  # Local path for server
# export HF_HUB_CACHE="/mnt/data/data/archive/huggingface/hub/"
# export MODEL_PATH="/mnt/data/data/archive/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"

export MODEL="deepseek-ai/DeepSeek-R1-0528"  # Repo ID for client tokenizer
export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
export TP=8  # Tensor parallel size
export PORT=8888
export CONC=8  # Concurrency
export ISL=512  # Input sequence length
export OSL=512  # Output sequence length
export RANDOM_RANGE_RATIO=0.8
export NUM_PROMPTS

WORKSPACE="/home/dn/amd-dev/inference"

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

echo "=== Checking GPU availability ==="
rocm-smi --showproductname | grep -E "GPU\[|Card series" || echo "Warning: Could not detect GPUs with rocm-smi"
echo ""

echo "=== Cleaning up any existing containers/networks ==="
docker stop $server_name 2>/dev/null || true
docker rm -f $server_name 2>/dev/null || true
docker stop $client_name 2>/dev/null || true
docker rm $client_name 2>/dev/null || true
docker network rm $network_name 2>/dev/null || true

echo "=== Creating Docker network ==="
docker network create $network_name

echo "=== Starting SGLang server container with TP=$TP (RCCL should activate) ==="
docker run -d --ipc=host --shm-size=16g --network=$network_name --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE:$HF_HUB_CACHE \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL_PATH -e TP -e PORT \
-e SGLANG_USE_AITER=1 \
-e NCCL_DEBUG=INFO \
-e NCCL_DEBUG_FILE=/workspace/inferenceMAX-benchmark/rccl_debug.log \
-e NCCL_TOPO_DUMP_FILE=/workspace/inferenceMAX-benchmark/rccl_topo.xml \
-e NCCL_GRAPH_DUMP_FILE=/workspace/inferenceMAX-benchmark/rccl_graph.xml \
-e NCCL_SOCKET_IFNAME=^lo,docker0 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
--entrypoint=/bin/bash \
$IMAGE \
-c "python3 -m sglang.launch_server \
    --model-path \$MODEL_PATH \
    --host=0.0.0.0 \
    --port \$PORT \
    --tensor-parallel-size \$TP \
    --trust-remote-code \
    --mem-fraction-static 0.85"

echo "=== Waiting for server to be ready ==="
timeout=300
elapsed=0
while ! docker logs $server_name 2>&1 | grep -q "Application startup complete"; do
    # Check if container is still running
    if ! docker ps -q -f name=$server_name | grep -q .; then
        echo ""
        echo "ERROR: Container exited! Showing logs:"
        echo "========================================"
        docker logs $server_name 2>&1 | tail -100
        echo "========================================"
        docker network rm $network_name 2>/dev/null
        exit 1
    fi
    
    sleep 5
    elapsed=$((elapsed + 5))
    
    # Show progress with log snippet every 30 seconds
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Still waiting (${elapsed}s)... Last log lines:"
        docker logs $server_name 2>&1 | tail -3
    else
        echo "Waiting... (${elapsed}s / ${timeout}s)"
    fi
    
    if [ $elapsed -ge $timeout ]; then
        echo ""
        echo "ERROR: Timeout after $timeout seconds. Last 50 log lines:"
        echo "========================================"
        docker logs $server_name 2>&1 | tail -50
        echo "========================================"
        docker stop $server_name
        docker network rm $network_name
        exit 1
    fi
done

echo "=== Server is ready! ==="

echo "=== Running benchmark client with ${NUM_PROMPTS} requests ==="
docker run --rm --network=$network_name --name=$client_name \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN \
--entrypoint=python3 \
$IMAGE \
bench_serving/benchmark_serving.py \
--model=$MODEL --backend=vllm --base-url="http://$server_name:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$NUM_PROMPTS \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=/workspace/inferenceMAX-benchmark --result-filename=test_${NUM_PROMPTS}req.json

echo "=== Cleaning up ==="
docker stop $server_name 2>/dev/null
docker rm -f $server_name 2>/dev/null
docker network rm $network_name

echo "=== Test complete! Results saved to test_${NUM_PROMPTS}req.json ==="
echo ""
if [ -f "inferenceMAX-benchmark/rccl_debug.log" ]; then
    echo "✓ RCCL debug log created: rccl_debug.log"
    echo "  RCCL WAS ACTIVE with TP=$TP"
    echo "  Log size: $(du -h inferenceMAX-benchmark/rccl_debug.log | cut -f1)"
    echo "  Preview:"
    head -20 inferenceMAX-benchmark/rccl_debug.log | grep -E "NCCL|init|rank" || head -5 inferenceMAX-benchmark/rccl_debug.log
else
    echo "✗ No RCCL debug log found - RCCL was NOT used!"
    echo "  This is unexpected with TP=$TP (should use RCCL for multi-GPU)"
    echo "  Check server logs for issues:"
    docker logs $server_name 2>&1 | tail -50
fi



