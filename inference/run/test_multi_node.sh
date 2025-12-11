#!/usr/bin/env bash

# Multi-node SGLang inference benchmark test
# This script starts SGLang servers on multiple nodes via SSH and runs a benchmark client
#
# Usage: ./test_multi_node.sh [OPTIONS]
#   Options:
#     --node1-ip IP          IP address of first node (default: 172.30.160.145)
#     --node2-ip IP          IP address of second node (default: 172.30.160.150)
#     --node1-user USER      SSH user for first node (default: current user)
#     --node2-user USER      SSH user for second node (default: current user)
#     --tp TP                Tensor parallel size per node (default: 8)
#     --ep EP                Expert parallel size (default: 1)
#     --port PORT            Server port (default: 8888)
#     --dist-port PORT       Distributed init port (default: 20000)
#     --conc CONC            Concurrency (default: 8)
#     --num-prompts NUM      Number of prompts (default: 48)

set -e

# ============================================
# CONFIGURATION
# ============================================
export HF_HUB_CACHE="/mnt/hf_hub_cache"
export MODEL="/mnt/hf_hub_cache/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
export IMAGE="rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"

# Default values
NODE1_IP="${NODE1_IP:-172.30.160.145}"
NODE2_IP="${NODE2_IP:-172.30.160.150}"
NODE1_USER="${NODE1_USER:-$(whoami)}"
NODE2_USER="${NODE2_USER:-$(whoami)}"
TP=16
EP=1
PORT=8888
DIST_PORT=20000
CONC=8
ISL=512
OSL=512
RANDOM_RANGE_RATIO=0.8
NUM_PROMPTS=48

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --node1-ip)
            NODE1_IP="$2"
            shift 2
            ;;
        --node2-ip)
            NODE2_IP="$2"
            shift 2
            ;;
        --node1-user)
            NODE1_USER="$2"
            shift 2
            ;;
        --node2-user)
            NODE2_USER="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --ep)
            EP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dist-port)
            DIST_PORT="$2"
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
        *)
            echo "ERROR: Unknown option '$1'"
            echo "Usage: $0 [--node1-ip IP] [--node2-ip IP] [--node1-user USER] [--node2-user USER] [--tp TP] [--ep EP] [--port PORT] [--dist-port PORT] [--conc CONC] [--num-prompts NUM]"
            exit 1
            ;;
    esac
done

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to get to inference/ directory
WORKSPACE="$(cd "$WORKSPACE/.." && pwd)"

# Function to check if an IP is the local node
is_local_node() {
    local ip="$1"
    # Check if IP matches localhost variants
    if [ "$ip" == "localhost" ] || [ "$ip" == "127.0.0.1" ]; then
        return 0
    fi
    # Check if IP matches any of the local IP addresses
    local local_ips
    local_ips=$(hostname -I 2>/dev/null | tr ' ' '\n' || echo "")
    if echo "$local_ips" | grep -q "^${ip}$"; then
        return 0
    fi
    # Also check hostname
    local hostname_ip
    hostname_ip=$(hostname -i 2>/dev/null || echo "")
    if [ "$ip" == "$hostname_ip" ]; then
        return 0
    fi
    return 1
}

# Determine if nodes are local
NODE1_IS_LOCAL=false
NODE2_IS_LOCAL=false

if is_local_node "$NODE1_IP"; then
    NODE1_IS_LOCAL=true
    echo "=== Node 1 ($NODE1_IP) is detected as local node ==="
fi

if is_local_node "$NODE2_IP"; then
    NODE2_IS_LOCAL=true
    echo "=== Node 2 ($NODE2_IP) is detected as local node ==="
fi

# Function to run command on Node 1 (local or remote)
run_on_node1() {
    if [ "$NODE1_IS_LOCAL" == "true" ]; then
        echo "Running: $*"
        bash -c "$*"
    else
        ssh ${NODE1_USER}@${NODE1_IP} "$*"
    fi
}

# Function to run command on Node 2 (local or remote)
run_on_node2() {
    if [ "$NODE2_IS_LOCAL" == "true" ]; then
        bash -c "$*"
    else
        ssh ${NODE2_USER}@${NODE2_IP} "$*"
    fi
}

# Function to rsync scripts to a remote node
sync_scripts_to_node() {
    local node_ip="$1"
    local node_user="$2"
    local is_local="$3"
    
    if [ "$is_local" == "true" ]; then
        echo "=== Skipping rsync for local node ($node_ip) ==="
        return 0
    fi
    
    echo "=== Syncing scripts to ${node_user}@${node_ip} ==="
    
    # Sync only the run/ directory which contains all scripts and patches
    # Ensure the run directory exists on remote host first
    ssh ${node_user}@${node_ip} "mkdir -p ${WORKSPACE}/run" 2>/dev/null || true
    
    rsync -avz --delete \
        "$WORKSPACE/run/" \
        ${node_user}@${node_ip}:${WORKSPACE}/run/ || {
        echo "WARNING: Failed to rsync scripts to ${node_user}@${node_ip}"
        echo "Continuing anyway, but remote node may have outdated scripts"
        return 1
    }
    
    echo "=== Successfully synced scripts to ${node_user}@${node_ip} ==="
}

DIST_INIT_ADDR="${NODE1_IP}:${DIST_PORT}"
NNODES=2

server_name="bmk-server"

# Track if we started servers (for cleanup)
SERVERS_STARTED_BY_SCRIPT=false

echo "=== Multi-node SGLang Benchmark Configuration ==="
echo "  Node 1: ${NODE1_USER}@${NODE1_IP} (rank 0)"
echo "  Node 2: ${NODE2_USER}@${NODE2_IP} (rank 1)"
echo "  TP per node: $TP"
echo "  EP: $EP"
echo "  Server port: $PORT"
echo "  Dist init address: $DIST_INIT_ADDR"
echo "  Concurrency: $CONC"
echo "  Num prompts: $NUM_PROMPTS"
echo ""

# Check if servers are already running
NODE1_RUNNING=false
NODE2_RUNNING=false

if run_on_node1 "docker ps -q -f name=$server_name | grep -q ." 2>/dev/null; then
    echo "=== Server on Node 1 is already running ==="
    NODE1_RUNNING=true
fi

if run_on_node2 "docker ps -q -f name=$server_name | grep -q ." 2>/dev/null; then
    echo "=== Server on Node 2 is already running ==="
    NODE2_RUNNING=true
fi

# Start servers if not running
if [ "$NODE1_RUNNING" == "false" ] || [ "$NODE2_RUNNING" == "false" ]; then
    SERVERS_STARTED_BY_SCRIPT=true
    
    # Sync scripts to remote nodes before starting servers
    echo "=== Syncing scripts to remote nodes ==="
    sync_scripts_to_node "$NODE1_IP" "$NODE1_USER" "$NODE1_IS_LOCAL"
    sync_scripts_to_node "$NODE2_IP" "$NODE2_USER" "$NODE2_IS_LOCAL"
    echo ""
    
    echo "=== Cleaning up any existing containers on both nodes ==="
    run_on_node1 "docker stop $server_name 2>/dev/null || true; docker rm $server_name 2>/dev/null || true" || true
    run_on_node2 "docker stop $server_name 2>/dev/null || true; docker rm $server_name 2>/dev/null || true" || true
    
    echo "=== Starting SGLang server on Node 1 (rank 0) ==="
    run_on_node1 "cd $WORKSPACE && ./run/start_server_container.sh server --detached --tp $TP --ep $EP --port $PORT --dist-init-addr $DIST_INIT_ADDR --nnodes $NNODES --node-rank 0" &
    NODE1_PID=$!
    
    echo "=== Starting SGLang server on Node 2 (rank 1) ==="
    run_on_node2 "cd $WORKSPACE && ./run/start_server_container.sh server --detached --tp $TP --ep $EP --port $PORT --dist-init-addr $DIST_INIT_ADDR --nnodes $NNODES --node-rank 1" &
    NODE2_PID=$!
    
    # Wait for both SSH commands to complete
    wait $NODE1_PID
    wait $NODE2_PID
    
    echo "=== Servers started, waiting for them to be ready ==="
fi

# Wait for both servers to be ready
echo "=== Waiting for Node 1 server to be ready ==="
timeout=900
elapsed=0
while ! run_on_node1 "docker logs $server_name 2>&1 | grep -q 'Application startup complete'" 2>/dev/null; do
    if ! run_on_node1 "docker ps -q -f name=$server_name | grep -q ." 2>/dev/null; then
        echo ""
        echo "ERROR: Container on Node 1 exited! Showing logs:"
        echo "========================================"
        run_on_node1 "docker logs $server_name 2>&1 | tail -100"
        echo "========================================"
        exit 1
    fi
    
    sleep 5
    elapsed=$((elapsed + 5))
    
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Still waiting for Node 1 (${elapsed}s)..."
        run_on_node1 "docker logs $server_name 2>&1 | tail -3"
    fi
    
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Timeout waiting for Node 1 server"
        run_on_node1 "docker logs $server_name 2>&1 | tail -50"
        exit 1
    fi
done

echo "=== Node 1 server is ready! ==="

echo "=== Waiting for Node 2 server to be ready ==="
elapsed=0
while ! run_on_node2 "docker logs $server_name 2>&1 | grep -q 'Application startup complete'" 2>/dev/null; do
    if ! run_on_node2 "docker ps -q -f name=$server_name | grep -q ." 2>/dev/null; then
        echo ""
        echo "ERROR: Container on Node 2 exited! Showing logs:"
        echo "========================================"
        run_on_node2 "docker logs $server_name 2>&1 | tail -100"
        echo "========================================"
        exit 1
    fi
    
    sleep 5
    elapsed=$((elapsed + 5))
    
    if [ $((elapsed % 30)) -eq 0 ]; then
        echo "Still waiting for Node 2 (${elapsed}s)..."
        run_on_node2 "docker logs $server_name 2>&1 | tail -3"
    fi
    
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Timeout waiting for Node 2 server"
        run_on_node2 "docker logs $server_name 2>&1 | tail -50"
        exit 1
    fi
done

echo "=== Node 2 server is ready! ==="
echo ""

echo "=== Checking which RCCL library is loaded on Node 1 ==="
run_on_node1 "docker exec $server_name bash -c \"cat /proc/\\\$(pgrep -f sglang.launch_server | head -1)/maps | grep librccl\"" || echo "Could not verify loaded RCCL"
echo ""

# Determine which node to use for the client (use Node 1)
echo "=== Running benchmark client from Node 1 ==="
# Since we're using host networking in multi-node mode, we connect to localhost
run_on_node1 "cd $WORKSPACE && ./run/run_benchmark_client.sh \
    --host localhost \
    --port $PORT \
    --network-mode host \
    --conc $CONC \
    --num-prompts $NUM_PROMPTS \
    --isl $ISL \
    --osl $OSL \
    --random-range-ratio $RANDOM_RANGE_RATIO \
    --test-name 'test_multi_node'"

echo "=== Collecting logs ==="
mkdir -p $WORKSPACE/outputs/logs
run_on_node1 "docker logs $server_name > $WORKSPACE/outputs/logs/node1_server.log 2> $WORKSPACE/outputs/logs/node1_server.error.log" || true
run_on_node2 "docker logs $server_name > $WORKSPACE/outputs/logs/node2_server.log 2> $WORKSPACE/outputs/logs/node2_server.error.log" || true

# Copy logs from remote nodes if needed
if [ "$NODE1_IS_LOCAL" == "false" ]; then
    echo "=== Copying logs from Node 1 ==="
    run_on_node1 "cat $WORKSPACE/outputs/logs/node1_server.log" > $WORKSPACE/outputs/logs/node1_server.log 2>/dev/null || true
    run_on_node1 "cat $WORKSPACE/outputs/logs/node1_server.error.log" > $WORKSPACE/outputs/logs/node1_server.error.log 2>/dev/null || true
fi

if [ "$NODE2_IS_LOCAL" == "false" ]; then
    echo "=== Copying logs from Node 2 ==="
    run_on_node2 "cat $WORKSPACE/outputs/logs/node2_server.log" > $WORKSPACE/outputs/logs/node2_server.log 2>/dev/null || true
    run_on_node2 "cat $WORKSPACE/outputs/logs/node2_server.error.log" > $WORKSPACE/outputs/logs/node2_server.error.log 2>/dev/null || true
fi

if [ "$SERVERS_STARTED_BY_SCRIPT" == "true" ]; then
    echo "=== Cleaning up ==="
    run_on_node1 "docker stop $server_name 2>/dev/null || true; docker rm $server_name 2>/dev/null || true" || true
    run_on_node2 "docker stop $server_name 2>/dev/null || true; docker rm $server_name 2>/dev/null || true" || true
else
    echo "=== Skipping cleanup (servers were already running) ==="
fi

echo "=== Test complete! ==="

