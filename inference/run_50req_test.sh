#!/usr/bin/env bash

# Simple 50-request test for InferenceMAX
# [Test script: sets up Docker network, launches SGLang server, runs benchmark client]

set -e

# HF_TOKEN removed for security
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"  # Full 70B model - local path with snapshot
export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
export TP=4  # Tensor parallel size - start with 2 GPUs to verify RCCL works
export PORT=8888
export CONC=8  # Concurrency
export ISL=512  # Input sequence length
export OSL=512  # Output sequence length
export RANDOM_RANGE_RATIO=0.8
export NUM_PROMPTS=50

WORKSPACE="/home/dn/dev/reps/inferenceMAX_benchmark"

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

echo "=== Checking GPU availability ==="
rocm-smi --showproductname | grep -E "GPU\[|Card series" || echo "Warning: Could not detect GPUs with rocm-smi"
echo ""

echo "=== Cleaning up any existing containers/networks ==="
docker stop $server_name 2>/dev/null || true
docker rm $server_name 2>/dev/null || true
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
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e PORT \
-e SGLANG_USE_AITER=1 \
-e NCCL_DEBUG=INFO \
-e NCCL_DEBUG_FILE=rccl.debug.log \
-e NCCL_TOPO_DUMP_FILE=rccl.topo.log \
-e NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
-e NCCL_SOCKET_IFNAME=^lo,docker0 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
--entrypoint=/bin/bash \
$IMAGE \
-c "python3 -m sglang.launch_server \
    --model-path \$MODEL \
    --host=0.0.0.0 \
    --port \$PORT \
    --tensor-parallel-size \$TP \
    --trust-remote-code \
    --mem-fraction-static 0.3"

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

echo "=== Running benchmark client with 50 requests ==="
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
--result-dir=/workspace/ --result-filename=test_50req.json

echo "=== Cleaning up ==="
docker stop $server_name 2>/dev/null
docker rm $server_name 2>/dev/null
docker network rm $network_name

echo "=== Test complete! Results saved to test_50req.json ==="
echo ""
if [ -f "rccl.debug.log" ]; then
    echo "✓ RCCL debug log created: rccl.debug.log"
    echo "  RCCL WAS ACTIVE with TP=$TP"
    echo "  Log size: $(du -h rccl.debug.log | cut -f1)"
    echo "  Preview:"
    head -20 rccl.debug.log | grep -E "NCCL|init|rank" || head -5 rccl.debug.log
else
    echo "✗ No RCCL debug log found - RCCL was NOT used!"
    echo "  This is unexpected with TP=$TP (should use RCCL for multi-GPU)"
    echo "  Check server logs for issues:"
    docker logs $server_name 2>&1 | tail -50
fi



