#!/usr/bin/env bash

set -e


# ============================================
# BENCHMARK CONFIGURATION
# ============================================
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
export TP=8  # Tensor parallel size
export EP=1  # Expert parallel size
export PORT=8888
export CONC=8  # Concurrency
export ISL=512  # Input sequence length
export OSL=512  # Output sequence length
export RANDOM_RANGE_RATIO=0.8
export NUM_PROMPTS=48

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

# Track if we started the server (for cleanup)
SERVER_STARTED_BY_SCRIPT=false

if docker ps -q -f name=$server_name | grep -q .; then
    echo "=== Server container '$server_name' is already running. Using the existing server. ==="
else
    echo "=== Cleaning up any existing containers/networks ==="
    docker stop $server_name 2>/dev/null || true
    docker rm $server_name 2>/dev/null || true
    docker network rm $network_name 2>/dev/null || true

    echo "=== Starting SGLang server using start_server_container.sh ==="
    # Call start_server_container.sh to start the server in detached mode
    "$WORKSPACE/start_server_container.sh" server --detached --tp $TP --ep $EP --port $PORT
    SERVER_STARTED_BY_SCRIPT=true
fi

echo "=== Waiting for server to be ready ==="
timeout=900
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
echo ""
echo "=== Checking which RCCL library is actually loaded ==="
docker exec $server_name bash -c "cat /proc/\$(pgrep -f sglang.launch_server | head -1)/maps | grep librccl" || echo "Could not verify loaded RCCL"
echo ""

echo "=== Running benchmark client with 50 requests ==="
docker run --rm --network=$network_name --name=$client_name \
-v $HF_HUB_CACHE:$HF_HUB_CACHE:ro \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN \
--entrypoint=python3 \
$IMAGE \
-m sglang.bench_serving \
--host $server_name --port $PORT --model $MODEL \
--dataset-name random --backend sglang \
--max-concurrency $CONC \
--random-input-len $ISL \
--random-output-len $OSL \
--random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $NUM_PROMPTS \
--output-file /workspace/outputs/test_50req_custom_rccl.json

mkdir -p $WORKSPACE/outputs/logs
docker logs $server_name > $WORKSPACE/outputs/logs/server.log 2> $WORKSPACE/outputs/logs/server.error.log

if [ "$SERVER_STARTED_BY_SCRIPT" == "true" ]; then
    echo "=== Cleaning up ==="
    docker stop $server_name 2>/dev/null
    docker rm $server_name 2>/dev/null
    docker network rm $network_name
else
    echo "=== Skipping cleanup (server was already running) ==="
fi

echo "=== Test complete! ==="



