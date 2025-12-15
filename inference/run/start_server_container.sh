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
#   Multi-node Options (for server mode):
#     --dist-init-addr ADDR   Master node address for multi-node (e.g., 172.30.160.145:20000)
#     --nnodes N              Total number of nodes (default: 1, single-node)
#     --node-rank RANK        Rank of this node (0, 1, 2, ...) (default: 0)

set -e

# ============================================
# USAGE FUNCTION
# ============================================
print_usage() {
    echo "Usage: $0 [benchmark|server|interactive] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  benchmark:   Run offline benchmark"
    echo "  server:      Launch SGLang server"
    echo "  interactive: Run container in interactive mode"
    echo ""
    echo "Options:"
    echo "  --detached, -d          Run container in detached mode (for server mode only)"
    echo "  --tp TP                 Set tensor parallel size (default: 8)"
    echo "  --ep EP                 Set expert parallel size (default: 1)"
    echo "  --port PORT             Set server port (default: 8888)"
    echo "  --apply-sync-patch      Apply sync_on_batch.patch for profiling (default: off)"
    echo "Benchmark Options:"
    echo "  --num-prompts NUM       Set number of prompts for benchmark (default: 32)"
    echo "  --concurrency CONC      Set concurrency for benchmark (default: 32)"
    echo "Multi-node Options (server mode only):"
    echo "  --dist-init-addr ADDR   Master node address (IP:PORT) for multi-node setup"
    echo "                          Example: 172.30.160.145:20000"
    echo "  --nnodes N              Total number of nodes (default: 1)"
    echo "  --node-rank RANK        Rank of this node (0, 1, 2, ...) (default: 0)"
    echo ""
    echo "Additional Arguments:"
    echo "  --                      Pass all remaining arguments to the benchmark/server command"
    echo "                         Example: ./start_server_container.sh server -- --mem-fraction-static 0.8"
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
CONCURRENCY_OVERRIDE=""
APPLY_SYNC_PATCH=false
DIST_INIT_ADDR=""
NNODES_OVERRIDE=""
NODE_RANK_OVERRIDE=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            # Everything after -- goes to EXTRA_ARGS
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_ARGS+=("$1")
                shift
            done
            break
            ;;
        benchmark|server|interactive)
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
        --apply-sync-patch)
            APPLY_SYNC_PATCH=true
            shift
            ;;
        --concurrency)
            CONCURRENCY_OVERRIDE="$2"
            shift 2
            ;;
        --dist-init-addr)
            DIST_INIT_ADDR="$2"
            shift 2
            ;;
        --nnodes)
            NNODES_OVERRIDE="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK_OVERRIDE="$2"
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

if [ "$MODE" != "benchmark" ] && [ "$MODE" != "server" ] && [ "$MODE" != "interactive" ]; then
    echo "ERROR: Invalid mode '$MODE'"
    echo ""
    print_usage
    exit 1
fi

if [ "$DETACHED" == "true" ] && [ "$MODE" == "benchmark" ] || [ "$MODE" == "interactive" ]; then
    echo "WARNING: --detached flag is ignored for benchmark and interactive mode"
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

if [ -n "$CONCURRENCY_OVERRIDE" ]; then
    if ! [[ "$CONCURRENCY_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$CONCURRENCY_OVERRIDE" -le 0 ]; then
        echo "ERROR: --concurrency must be a positive integer, got: $CONCURRENCY_OVERRIDE"
        exit 1
    fi
fi

# Validate multi-node parameters
if [ -n "$NNODES_OVERRIDE" ]; then
    if ! [[ "$NNODES_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NNODES_OVERRIDE" -le 0 ]; then
        echo "ERROR: --nnodes must be a positive integer, got: $NNODES_OVERRIDE"
        exit 1
    fi
    if [ "$MODE" == "benchmark" ]; then
        echo "WARNING: Multi-node options are only valid for server mode, ignoring --nnodes"
        NNODES_OVERRIDE=""
    fi
fi

if [ -n "$NODE_RANK_OVERRIDE" ]; then
    if ! [[ "$NODE_RANK_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$NODE_RANK_OVERRIDE" -lt 0 ]; then
        echo "ERROR: --node-rank must be a non-negative integer, got: $NODE_RANK_OVERRIDE"
        exit 1
    fi
    if [ "$MODE" == "benchmark" ]; then
        echo "WARNING: Multi-node options are only valid for server mode, ignoring --node-rank"
        NODE_RANK_OVERRIDE=""
    fi
fi

if [ -n "$DIST_INIT_ADDR" ]; then
    if [ "$MODE" == "benchmark" ]; then
        echo "WARNING: Multi-node options are only valid for server mode, ignoring --dist-init-addr"
        DIST_INIT_ADDR=""
    elif [ -z "$NNODES_OVERRIDE" ] || [ "$NNODES_OVERRIDE" == "1" ]; then
        echo "WARNING: --dist-init-addr specified but --nnodes is 1 or not set. Multi-node requires --nnodes > 1"
    fi
fi

# ============================================
# CUSTOM RCCL CONFIGURATION
# ============================================
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to get to inference/ directory
WORKSPACE="$(cd "$WORKSPACE/.." && pwd)"
GIT_ROOT=$(git -C "$WORKSPACE" rev-parse --show-toplevel)

CUSTOM_ROCM_PATH="/opt/rocm-clean/"
CUSTOM_RCCL_PATH="${GIT_ROOT}/dn/rccl/build/release"
CUSTOM_RCCL_NET_PATH="${GIT_ROOT}/dn/amd-anp/build"
EXTRA_LIBS="/lib/x86_64-linux-gnu/libionic.so.1"
EXTRA_LIBS_MOUNTS=""
for lib in $EXTRA_LIBS; do
    EXTRA_LIBS_MOUNTS="$EXTRA_LIBS_MOUNTS -v $lib:$lib:ro"
done

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
export IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
# export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
export TP="${TP_OVERRIDE:-4}"  # Tensor parallel size (can be overridden with --tp)
export EP="${EP_OVERRIDE:-1}"  # Expert parallel size (can be overridden with --ep)
export PORT="${PORT_OVERRIDE:-8888}"  # Server port (can be overridden with --port)
export NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE:-50}"  # Number of prompts (can be overridden with --num-prompts)
export CONCURRENCY="${CONCURRENCY_OVERRIDE:-32}"  # Concurrency (can be overridden with --concurrency)
#export NPKIT_FLAGS=0xFFFFFFFFFFFFFFFF
export NPKIT_FLAGS=0x0000000000000000

# Multi-node configuration
export NNODES="${NNODES_OVERRIDE:-1}"
export NODE_RANK="${NODE_RANK_OVERRIDE:-0}"

# Determine if we're in multi-node mode
MULTI_NODE=false
if [ "$MODE" == "server" ] && [ "$NNODES" -gt 1 ]; then
    MULTI_NODE=true
    if [ -z "$DIST_INIT_ADDR" ]; then
        echo "ERROR: Multi-node mode requires --dist-init-addr to be specified"
        exit 1
    fi
    export DIST_INIT_ADDR="$DIST_INIT_ADDR"
fi

network_name="bmk-net"
server_name="bmk-server"

# For multi-node, use host networking; otherwise use bridge network
if [ "$MULTI_NODE" == "true" ]; then
    echo "=== Multi-node mode detected (nnodes=$NNODES, node-rank=$NODE_RANK) ==="
    echo "=== Using host networking for inter-node communication ==="
    NETWORK_MODE="host"
    export NCCL_SOCKET_IFNAME=enp81s0f1np1
else
    echo "=== Creating Docker network ==="
    docker network inspect $network_name >/dev/null 2>&1 || docker network create $network_name
    NETWORK_MODE="$network_name"
    export NCCL_SOCKET_IFNAME=^lo,docker0
fi

echo "=== Starting SGLang with CUSTOM RCCL (TP=$TP, EP=$EP, PORT=$PORT, MODE=$MODE) ==="
if [ "$MULTI_NODE" == "true" ]; then
    echo "    Multi-node: nnodes=$NNODES, node-rank=$NODE_RANK, dist-init-addr=$DIST_INIT_ADDR"
fi
if [ "$MODE" == "benchmark" ]; then
    echo "    NUM_PROMPTS=$NUM_PROMPTS"
fi
echo "    Custom RCCL will be mounted from: $CUSTOM_RCCL_PATH"
echo ""

# Prepare the command to run inside the container
# Build patch commands
PATCH_CMDS="echo '=== Patching aiter code ===' && \
    patch -i /workspace/run/aiter_patch/fix_arg_parsing.patch -p 1 -d /sgl-workspace/aiter/"

if [ "$APPLY_SYNC_PATCH" == "true" ]; then
    PATCH_CMDS="$PATCH_CMDS && \
    echo '=== Patching SGLang code ===' && \
    patch -i /workspace/run/sglang_patch/comm_shutdown.patch -p 1 -d /sgl-workspace/sglang/"
    PATCH_CMDS="$PATCH_CMDS && \
    patch -i /workspace/run/sglang_patch/sync_on_batch.patch -p 1 -d /sgl-workspace/sglang/"
fi

# Build extra arguments string if any were provided
EXTRA_ARGS_STR=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    # Properly escape each argument for use in the shell command
    for arg in "${EXTRA_ARGS[@]}"; do
        # Use printf %q to properly escape for shell, then wrap in single quotes for the inner command
        escaped_arg=$(printf %q "$arg")
        EXTRA_ARGS_STR="$EXTRA_ARGS_STR $escaped_arg"
    done
    echo "    Additional arguments: ${EXTRA_ARGS[*]}"
fi

if [ "$MODE" == "benchmark" ]; then
    echo "=== Running offline benchmark ==="
    DOCKER_CMD="set -e && \
        $PATCH_CMDS && \
        echo '=== Starting offline benchmark ===' && \
        python -m sglang.bench_offline_throughput --model-path \$MODEL --tensor-parallel-size \$TP --dataset-name random --num-prompts \$NUM_PROMPTS --disable-custom-all-reduce --cuda-graph-bs \$CONCURRENCY --max-running-requests \$CONCURRENCY --skip-warmup --profile --mem-fraction-static 0.6$EXTRA_ARGS_STR"
elif [ "$MODE" == "interactive" ]; then
    echo "=== Running interactive mode ==="
    echo "python3 -m sglang.launch_server --model-path \$MODEL --host=0.0.0.0 --port \$PORT --tensor-parallel-size \$TP --expert-parallel-size \$EP --trust-remote-code --mem-fraction-static 0.6 --cuda-graph-max-bs 64"
    echo "python -m sglang.bench_offline_throughput --model-path \$MODEL --tensor-parallel-size \$TP --dataset-name random --num-prompts \$NUM_PROMPTS --disable-custom-all-reduce --cuda-graph-bs \$CONCURRENCY --max-running-requests \$CONCURRENCY --skip-warmup --profile --mem-fraction-static 0.6$EXTRA_ARGS_STR"
    DOCKER_CMD=""
else
    echo "=== Launching SGLang server ==="
    # Build the launch_server command
    SERVER_CMD="python3 -m sglang.launch_server --model-path \$MODEL --host=0.0.0.0 --port \$PORT --tensor-parallel-size \$TP --expert-parallel-size \$EP --trust-remote-code --mem-fraction-static 0.6 --cuda-graph-max-bs 64"
    
    # Add multi-node parameters if in multi-node mode
    if [ "$MULTI_NODE" == "true" ]; then
        SERVER_CMD="$SERVER_CMD --dist-init-addr \$DIST_INIT_ADDR --nnodes \$NNODES --node-rank \$NODE_RANK"
    fi
    
    SERVER_CMD="$SERVER_CMD$EXTRA_ARGS_STR"
    
    DOCKER_CMD="set -e && \
        $PATCH_CMDS && \
        echo '=== Starting SGLang server ===' && \
        $SERVER_CMD"
fi


# Determine docker run flags based on mode
if [ "$DETACHED" == "true" ]; then
    DOCKER_RUN_FLAGS="-d"
else
    DOCKER_RUN_FLAGS="--rm -it"
fi

mkdir -p $WORKSPACE/outputs/logs

# -v $CUSTOM_ROCM_PATH:/opt/rocm-custom:ro \

# -e SGLANG_NCCL_SO_PATH=/opt/rccl/lib/librccl.so \
# -e LD_LIBRARY_PATH=/opt/rccl/lib:/opt/rccl/lib-net \
# -e LD_PRELOAD=/opt/rccl/lib/librccl.so:/opt/rccl/lib-net/librccl-net.so \

docker run $DOCKER_RUN_FLAGS --ipc=host --shm-size=16g --network=$NETWORK_MODE --name=$server_name \
--privileged \
--ulimit memlock=-1 \
--cap-add=CAP_SYS_ADMIN \
--cap-add=IPC_LOCK \
--device=/dev/kfd \
--device=/dev/dri \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v $HF_HUB_CACHE:$HF_HUB_CACHE \
-v $WORKSPACE:/workspace/ -w /workspace/ \
-v $CUSTOM_RCCL_PATH:/opt/rccl/lib:ro \
-v $CUSTOM_RCCL_NET_PATH:/opt/rccl/lib-net:ro \
$EXTRA_LIBS_MOUNTS \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e PORT -e EP -e NUM_PROMPTS -e CONCURRENCY \
-e NNODES -e NODE_RANK -e DIST_INIT_ADDR \
-e AITER_JIT_DIR=/workspace/aiter_jit_cache \
-e SGLANG_USE_AITER=1 \
-e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
-e SGLANG_TORCH_PROFILER_DIR=/workspace/outputs/sglang_profile \
-e IONIC_LOCKFREE=all \
-e IONIC_PRIVATE_SERVICE_FORCE=1 \
-e IONIC_RCQ_NUM_PATHS=128 \
-e IONIC_RCQ_SIGN_BIT=15 \
-e NCCL_DEBUG_FILE=/workspace/outputs/logs/rccl.debug.%h.%p.log \
-e NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,COLL,P2P,NET,PROXY,CALL,PROFILE,REG \
-e NCCL_DEBUG_TIMESTAMP_LEVELS=INFO \
-e NCCL_DEBUG=TRACE \
-e NCCL_GDR_FLUSH_DISABLE=1 \
-e NCCL_GDRCOPY_ENABLE=0 \
-e NCCL_IB_DISABLE=0 \
-e NCCL_IB_FIFO_TC=192 \
-e NCCL_IB_GID_INDEX=1 \
-e NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
-e NCCL_IB_QPS_PER_CONNECTION=1 \
-e NCCL_IB_TC=104 \
-e NCCL_IB_TIMEOUT=5 \
-e NCCL_IB_USE_INLINE=1 \
-e NCCL_IGNORE_CPU_AFFINITY=1 \
-e NCCL_P2P_LEVEL=SYS \
-e NCCL_PXN_DISABLE=0 \
-e NCCL_SOCKET_IFNAME \
-e NET_OPTIONAL_RECV_COMPLETION=1 \
-e RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1 \
-e NPKIT_DUMP_DIR=/workspace/outputs/npkit \
-e NPKIT_FLAGS \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
--entrypoint=/bin/bash \
$IMAGE ${DOCKER_CMD:+-c "$DOCKER_CMD"}

