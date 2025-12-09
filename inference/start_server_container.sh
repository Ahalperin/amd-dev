#!/usr/bin/env bash

# Simple 50-request test for InferenceMAX with CUSTOM RCCL
# Uses custom-built RCCL libraries from host instead of container's libraries
#
# Usage: ./start_server_container.sh [benchmark|server] [OPTIONS]
#   benchmark: Run offline benchmark
#   server:    Launch SGLang server
#   Options:
#     --detached, -d          Run container in detached mode (for server mode only)
#     --tp TP                 Set tensor parallel size (default: 8)
#     --ep EP                 Set expert parallel size (default: 1)
#     --port PORT             Set server port (default: 8888)
#     --num-prompts NUM       Set number of prompts for benchmark (default: 50)

set -e

# ============================================
# USAGE FUNCTION
# ============================================
print_usage() {
    echo "Usage: $0 [benchmark|server] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  benchmark: Run offline benchmark"
    echo "  server:    Launch SGLang server"
    echo ""
    echo "Options:"
    echo "  --detached, -d          Run container in detached mode (for server mode only)"
    echo "  --tp TP                 Set tensor parallel size (default: 8)"
    echo "  --ep EP                 Set expert parallel size (default: 1)"
    echo "  --port PORT             Set server port (default: 8888)"
    echo "  --num-prompts NUM       Set number of prompts for benchmark (default: 50)"
}

# ============================================
# PARAMETER PARSING
# ============================================
MODE=""
DETACHED=false
TP_OVERRIDE=""
EP_OVERRIDE=""
PORT_OVERRIDE=""
NUM_PROMPTS_OVERRIDE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        benchmark|server)
            MODE="$1"
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --tp)
            TP_OVERRIDE="$2"
            shift 2
            ;;
        --ep)
            EP_OVERRIDE="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option '$1'"
            echo ""
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: Missing mode parameter"
    echo ""
    print_usage
    exit 1
fi

if [ "$MODE" != "benchmark" ] && [ "$MODE" != "server" ]; then
    echo "ERROR: Invalid mode '$MODE'"
    echo ""
    print_usage
    exit 1
fi

if [ "$DETACHED" == "true" ] && [ "$MODE" == "benchmark" ]; then
    echo "WARNING: --detached flag is ignored for benchmark mode"
    DETACHED=false
fi

# Validate numeric parameters
if [ -n "$TP_OVERRIDE" ]; then
    if ! [[ "$TP_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$TP_OVERRIDE" -le 0 ]; then
        echo "ERROR: --tp must be a positive integer, got: $TP_OVERRIDE"
        exit 1
    fi
fi

if [ -n "$EP_OVERRIDE" ]; then
    if ! [[ "$EP_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$EP_OVERRIDE" -le 0 ]; then
        echo "ERROR: --ep must be a positive integer, got: $EP_OVERRIDE"
        exit 1
    fi
fi

if [ -n "$PORT_OVERRIDE" ]; then
    if ! [[ "$PORT_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$PORT_OVERRIDE" -le 0 ] || [ "$PORT_OVERRIDE" -gt 65535 ]; then
        echo "ERROR: --port must be a positive integer between 1 and 65535, got: $PORT_OVERRIDE"
        exit 1
    fi
fi

if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
    if ! [[ "$NUM_PROMPTS_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NUM_PROMPTS_OVERRIDE" -le 0 ]; then
        echo "ERROR: --num-prompts must be a positive integer, got: $NUM_PROMPTS_OVERRIDE"
        exit 1
    fi
fi

# ============================================
# CUSTOM RCCL CONFIGURATION
# ============================================
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT=$(git -C "$WORKSPACE" rev-parse --show-toplevel)

CUSTOM_ROCM_PATH="/opt/rocm-clean/"
CUSTOM_RCCL_PATH="${GIT_ROOT}/dn/rccl/build/release"
CUSTOM_RCCL_NET_PATH="${GIT_ROOT}/dn/amd-anp/build"
EXTRA_LIBS_PATH="/lib/x86_64-linux-gnu"

if [ ! -f "$CUSTOM_RCCL_PATH/librccl.so" ]; then
    echo "ERROR: Custom RCCL not found at $CUSTOM_RCCL_PATH/librccl.so"
    echo "Please set CUSTOM_RCCL_PATH to your RCCL build directory"
    exit 1
fi

if [ ! -f "$CUSTOM_RCCL_NET_PATH/librccl-net.so" ]; then
    echo "ERROR: librccl-net.so not found at $CUSTOM_RCCL_NET_PATH/librccl-net.so"
    echo "Please set CUSTOM_RCCL_NET_PATH to your RCCL NET build directory"
    exit 1
fi


# ============================================
# BENCHMARK CONFIGURATION
# ============================================
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
# export IMAGE="rocm/sgl-dev:v0.5.5.post3-rocm700-mi35x-20251202"
export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
export TP="${TP_OVERRIDE:-8}"  # Tensor parallel size (can be overridden with --tp)
export EP="${EP_OVERRIDE:-1}"  # Expert parallel size (can be overridden with --ep)
export PORT="${PORT_OVERRIDE:-8888}"  # Server port (can be overridden with --port)
export NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE:-50}"  # Number of prompts (can be overridden with --num-prompts)
#export NPKIT_FLAGS=0xFFFFFFFFFFFFFFFF
export NPKIT_FLAGS=0x0000000000000000

network_name="bmk-net"
server_name="bmk-server"

echo "=== Creating Docker network ==="
docker network inspect $network_name >/dev/null 2>&1 || docker network create $network_name

echo "=== Starting SGLang with CUSTOM RCCL (TP=$TP, EP=$EP, PORT=$PORT, MODE=$MODE) ==="
if [ "$MODE" == "benchmark" ]; then
    echo "    NUM_PROMPTS=$NUM_PROMPTS"
fi
echo "    Custom RCCL will be mounted from: $CUSTOM_RCCL_PATH"
echo ""

# Prepare the command to run inside the container
if [ "$MODE" == "benchmark" ]; then
    echo "=== Running offline benchmark ==="
    DOCKER_CMD="set -e && \
        echo '=== Patching SGLang code ===' && \
        patch -i /workspace/sglang_patch/comm_shutdown.patch -p 1 -d /sgl-workspace/sglang/ && \
        echo '=== Starting offline benchmark ===' && \
        python -m sglang.bench_offline_throughput --model-path \$MODEL --tensor-parallel-size \$TP --dataset-name random --num-prompts \$NUM_PROMPTS --disable-custom-all-reduce --cuda-graph-bs \$NUM_PROMPTS --max-running-requests \$NUM_PROMPTS --skip-warmup --profile --mem-fraction-static 0.6"
else
    echo "=== Launching SGLang server ==="
    DOCKER_CMD="set -e && \
        echo '=== Patching SGLang code ===' && \
        patch -i /workspace/sglang_patch/comm_shutdown.patch -p 1 -d /sgl-workspace/sglang/ && \
        echo '=== Starting SGLang server ===' && \
        python3 -m sglang.launch_server --model-path \$MODEL --host=0.0.0.0 --port \$PORT --tensor-parallel-size \$TP --expert-parallel-size \$EP --trust-remote-code --mem-fraction-static 0.6"
fi


# Determine docker run flags based on mode
if [ "$DETACHED" == "true" ]; then
    DOCKER_RUN_FLAGS="-d"
else
    DOCKER_RUN_FLAGS="--rm -it"
fi

mkdir -p $WORKSPACE/outputs/logs

docker run $DOCKER_RUN_FLAGS --ipc=host --shm-size=16g --network=$network_name --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE:$HF_HUB_CACHE \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-v $CUSTOM_ROCM_PATH:/opt/rocm-custom:ro \
-v $CUSTOM_RCCL_PATH:/opt/rccl/lib:ro \
-v $CUSTOM_RCCL_NET_PATH:/opt/rccl/lib-net:ro \
-v $EXTRA_LIBS_PATH:/opt/lib-extra:ro \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e PORT -e EP -e NUM_PROMPTS \
-e SGLANG_USE_AITER=1 \
-e SGLANG_TORCH_PROFILER_DIR=/workspace/outputs/sglang_profile \
-e SGLANG_NCCL_SO_PATH=/opt/rccl/lib/librccl.so \
-e LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rccl/lib:/opt/rccl/lib-net:/opt/lib-extra \
-e LD_PRELOAD=/opt/rccl/lib/librccl.so:/opt/rccl/lib-net/librccl-net.so \
-e NCCL_DEBUG=INFO \
-e NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,COLL,P2P,NET,CALL,PROFILE \
-e NCCL_DEBUG_TIMESTAMP_LEVELS=INFO \
-e NPKIT_DUMP_DIR=/workspace/outputs/npkit \
-e NPKIT_FLAGS \
-e NCCL_DEBUG_FILE=/workspace/outputs/logs/rccl.debug.%h.%p.log \
-e NCCL_TOPO_DUMP_FILE=/workspace/outputs/logs/rccl.topo.log \
-e NCCL_GRAPH_DUMP_FILE=/workspace/outputs/logs/rccl.graph.log \
-e NCCL_SOCKET_IFNAME=^lo,docker0 \
-e NCCL_P2P_LEVEL=SYS \
-e NCCL_IB_DISABLE=0 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
--entrypoint=/bin/bash \
$IMAGE -c "$DOCKER_CMD"

