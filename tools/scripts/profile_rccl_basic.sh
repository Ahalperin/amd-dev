#!/bin/bash

# Basic RCCL Profiling Script
# Usage: ./profile_rccl_basic.sh <test_name> [additional_args]

set -e

# Default parameters
TEST_NAME=${1:-all_reduce_perf}
RCCL_TESTS_DIR="/workspace/rccl-tests"
OUTPUT_DIR="/workspace/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variables for better profiling
export NCCL_DEBUG=INFO
export HSA_NO_SCRATCH_RECLAIM=1

echo "=== Basic RCCL Profiling ==="
echo "Test: ${TEST_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"

# Check if test executable exists
TEST_EXEC="${RCCL_TESTS_DIR}/build/${TEST_NAME}"
if [[ ! -f "${TEST_EXEC}" ]]; then
    echo "Error: Test executable not found: ${TEST_EXEC}"
    echo "Available tests:"
    ls -1 "${RCCL_TESTS_DIR}/build/" | grep "_perf$" | head -10
    exit 1
fi

echo "=== GPU Status Check ==="
rocm-smi --showtemp --showpower --showuse

echo "=== Running Basic Profiling ==="

# 1. Basic kernel and API tracing
echo "1. Basic kernel and HIP API tracing..."
rocprof --hip-trace --hsa-trace \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_basic_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

# 2. Performance statistics
echo "2. Performance statistics..."
rocprof --hip-trace --stats \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_stats_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

# 3. Memory-focused profiling
echo "3. Memory-focused profiling..."
rocprof --hip-trace \
        -m TCC_HIT,TCC_MISS,TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_memory_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

echo "=== Profiling Complete ==="
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}/"*"${TIMESTAMP}"*

echo ""
echo "=== Quick Analysis ==="
if [[ -f "${OUTPUT_DIR}/${TEST_NAME}_stats_${TIMESTAMP}.csv" ]]; then
    echo "Top 10 longest running kernels:"
    head -1 "${OUTPUT_DIR}/${TEST_NAME}_stats_${TIMESTAMP}.csv"
    tail -n +2 "${OUTPUT_DIR}/${TEST_NAME}_stats_${TIMESTAMP}.csv" | sort -t, -k7 -nr | head -10
fi