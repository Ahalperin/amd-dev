#!/bin/bash

##############################################################################
# RCCL Test Runner in Ephemeral Container
#
# This script runs RCCL tests in a temporary container that is automatically
# cleaned up after test completion.
#
# Usage:
#   ./run_rccl_test_container.sh <test_name> [test_args...]
#
# Examples:
#   ./run_rccl_test_container.sh all_reduce_perf -b 8 -e 128M -f 2 -g 1
#   ./run_rccl_test_container.sh all_gather_perf -t 1 -g 1 -b 4 -e 8G -f 2
#   ./run_rccl_test_container.sh broadcast_perf -b 1M -e 1G -f 2
#
# Environment Variables:
#   RCCL_CONTAINER_NAME  - Custom container name (default: rccl-test-runner-<timestamp>)
#   RCCL_OUTPUT_DIR      - Output directory for logs (default: ./rccl_test_results)
#   NCCL_DEBUG           - NCCL debug level (default: INFO)
#   NCCL_DEBUG_SUBSYS    - NCCL debug subsystems (default: not set)
#   USE_PRIVILEGED       - Use privileged mode (default: 0, set to 1 for full access)
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_DEV_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONTAINER_NAME="${RCCL_CONTAINER_NAME:-rccl-test-runner-${TIMESTAMP}}"
OUTPUT_DIR="${RCCL_OUTPUT_DIR:-${AMD_DEV_DIR}/rccl_test_results}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG:-INFO}"
USE_PRIVILEGED="${USE_PRIVILEGED:-0}"

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 <test_name> [test_args...]

Available RCCL tests:
  - all_reduce_perf
  - all_gather_perf
  - reduce_scatter_perf
  - broadcast_perf
  - reduce_perf
  - alltoall_perf
  - sendrecv_perf

Common test arguments:
  -b, --minbytes <size>       Minimum size in bytes
  -e, --maxbytes <size>       Maximum size in bytes
  -f, --stepfactor <factor>   Increment factor
  -g, --ngpus <count>         GPUs per thread
  -t, --nthreads <count>      Number of threads
  -n, --iters <count>         Iteration count
  -w, --warmup_iters <count>  Warmup iteration count

Examples:
  $0 all_reduce_perf -b 8 -e 128M -f 2 -g 1
  $0 all_gather_perf -t 1 -g 1 -b 4 -e 8G -f 2
  
Environment Variables:
  RCCL_CONTAINER_NAME  - Custom container name
  RCCL_OUTPUT_DIR      - Output directory for logs
  NCCL_DEBUG           - NCCL debug level (default: INFO)
  NCCL_DEBUG_SUBSYS    - NCCL debug subsystems
  USE_PRIVILEGED       - Use privileged mode (0/1)

EOF
    exit 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [ -n "$CONTAINER_NAME" ]; then
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_info "Cleaning up container: ${CONTAINER_NAME}"
            docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
            docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
            print_success "Container cleaned up"
        fi
    fi
    
    if [ $exit_code -ne 0 ]; then
        print_error "Script exited with error code: $exit_code"
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Check arguments
if [ $# -lt 1 ]; then
    print_error "Missing test name"
    usage
fi

TEST_NAME="$1"
shift
TEST_ARGS="$@"

# Validate test name
VALID_TESTS=("all_reduce_perf" "all_gather_perf" "reduce_scatter_perf" "broadcast_perf" "reduce_perf" "alltoall_perf" "sendrecv_perf")
if [[ ! " ${VALID_TESTS[@]} " =~ " ${TEST_NAME} " ]]; then
    print_warning "Test '${TEST_NAME}' is not in the standard list, but will try to run it anyway"
fi

# Check prerequisites
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker images | grep -q "rccl-builder"; then
    print_error "rccl-builder image not found. Please build it first."
    print_info "Build command: cd ${AMD_DEV_DIR}/rccl-builder && ./build_rccl_builder"
    exit 1
fi

# Check if amd directory exists
if [ ! -d "${AMD_DEV_DIR}/amd" ]; then
    print_error "AMD source directory not found: ${AMD_DEV_DIR}/amd"
    print_info "Please clone RCCL sources first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${TEST_NAME}_${TIMESTAMP}.log"

print_info "============================================"
print_info "RCCL Test Runner Configuration"
print_info "============================================"
print_info "Test Name:      ${TEST_NAME}"
print_info "Test Args:      ${TEST_ARGS:-<none>}"
print_info "Container:      ${CONTAINER_NAME}"
print_info "AMD Dev Dir:    ${AMD_DEV_DIR}"
print_info "Output Dir:     ${OUTPUT_DIR}"
print_info "Log File:       ${LOG_FILE}"
print_info "NCCL Debug:     ${NCCL_DEBUG_LEVEL}"
if [ -n "$NCCL_DEBUG_SUBSYS" ]; then
    print_info "Debug Subsys:   ${NCCL_DEBUG_SUBSYS}"
fi
print_info "============================================"
echo

# Build docker run command based on privileged mode
if [ "$USE_PRIVILEGED" = "1" ]; then
    print_info "Using PRIVILEGED mode for full GPU access"
    DOCKER_RUN_CMD="docker run --rm \
        --name ${CONTAINER_NAME} \
        --workdir /workspace \
        -v ${AMD_DEV_DIR}/amd:/workspace \
        -v ${AMD_DEV_DIR}/tools:/tools \
        -v ${OUTPUT_DIR}:/output \
        -e NCCL_DEBUG=${NCCL_DEBUG_LEVEL} \
        ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}} \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt seccomp=unconfined \
        --security-opt apparmor=unconfined \
        --shm-size=512m \
        --privileged \
        --network=host \
        --pid=host \
        --ipc=host \
        -v /dev:/dev \
        -v /sys:/sys:ro \
        -v /proc:/proc \
        rccl-builder:latest"
else
    print_info "Using STANDARD mode for GPU access"
    DOCKER_RUN_CMD="docker run --rm \
        --name ${CONTAINER_NAME} \
        --workdir /workspace \
        -v ${AMD_DEV_DIR}/amd:/workspace \
        -v ${AMD_DEV_DIR}/tools:/tools \
        -v ${OUTPUT_DIR}:/output \
        -e NCCL_DEBUG=${NCCL_DEBUG_LEVEL} \
        ${NCCL_DEBUG_SUBSYS:+-e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}} \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt seccomp=unconfined \
        --shm-size=512m \
        rccl-builder:latest"
fi

# Construct the test command
TEST_EXEC="/workspace/rccl-tests/build/${TEST_NAME}"
TEST_CMD="${TEST_EXEC} ${TEST_ARGS}"

print_info "Starting container and running test..."
print_info "Command: ${TEST_CMD}"
echo

# Run the test
START_TIME=$(date +%s)

if $DOCKER_RUN_CMD bash -c "$TEST_CMD" 2>&1 | tee "$LOG_FILE"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo
    print_success "Test completed successfully!"
    print_success "Duration: ${DURATION} seconds"
    print_success "Log saved to: ${LOG_FILE}"
    
    # Check if test produced any output files
    OUTPUT_FILES=$(find "$OUTPUT_DIR" -type f -name "*${TEST_NAME}*${TIMESTAMP}*" 2>/dev/null | wc -l)
    if [ "$OUTPUT_FILES" -gt 1 ]; then
        print_success "Found ${OUTPUT_FILES} output files in ${OUTPUT_DIR}"
    fi
    
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo
    print_error "Test failed after ${DURATION} seconds"
    print_error "Check log file: ${LOG_FILE}"
    exit 1
fi

