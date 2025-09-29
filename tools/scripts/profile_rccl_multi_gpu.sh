#!/bin/bash

# Multi-GPU RCCL Profiling Script
# Usage: ./profile_rccl_multi_gpu.sh <test_name> <num_gpus> [additional_args]

set -e

# Default parameters
TEST_NAME=${1:-all_reduce_perf}
NUM_GPUS=${2:-2}
RCCL_TESTS_DIR="/workspace/rccl-tests"
OUTPUT_DIR="/workspace/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variables
export NCCL_DEBUG=INFO
export HSA_NO_SCRATCH_RECLAIM=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "=== Multi-GPU RCCL Profiling ==="
echo "Test: ${TEST_NAME}"
echo "Number of GPUs: ${NUM_GPUS}"
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
rocm-smi --showid --showtemp --showuse

# Check available GPUs
AVAILABLE_GPUS=$(rocm-smi --showid | grep -c "GPU" || echo "0")
echo "Available GPUs: ${AVAILABLE_GPUS}"

if [[ ${NUM_GPUS} -gt ${AVAILABLE_GPUS} ]]; then
    echo "Warning: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available"
    echo "Adjusting to use ${AVAILABLE_GPUS} GPUs"
    NUM_GPUS=${AVAILABLE_GPUS}
fi

echo "=== Running Multi-GPU Profiling ==="

# 1. Single process multi-GPU profiling
echo "1. Single process multi-GPU profiling..."
rocprof --hip-trace --hsa-trace \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_single_proc_${NUM_GPUS}gpu_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g ${NUM_GPUS} ${@:3}

# 2. Multi-process single GPU per process (recommended for performance)
echo "2. Multi-process profiling (1 GPU per process)..."
mpirun --allow-run-as-root -np ${NUM_GPUS} --bind-to numa \
       sh -c "rocprof --hip-trace --output-file ${OUTPUT_DIR}/${TEST_NAME}_multi_proc_rank_\${OMPI_COMM_WORLD_RANK}_${TIMESTAMP} \
              ${TEST_EXEC} --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:3}"

# 3. Advanced profiling with rocprofv3
echo "3. Advanced multi-GPU profiling with rocprofv3..."
rocprofv3 --plugin csv \
          --kernel-trace \
          --hip-trace \
          --output-file "${OUTPUT_DIR}/${TEST_NAME}_advanced_${NUM_GPUS}gpu_${TIMESTAMP}.csv" \
          -- "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g ${NUM_GPUS} ${@:3}

# 4. Memory bandwidth focused profiling
echo "4. Memory bandwidth profiling..."
rocprof --hip-trace \
        -m TCC_HIT,TCC_MISS,TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum,TCC_EA_RDREQ,TCC_EA_WRREQ \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_bandwidth_${NUM_GPUS}gpu_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g ${NUM_GPUS} ${@:3}

echo "=== Profiling Complete ==="
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}/"*"${TIMESTAMP}"*

echo ""
echo "=== Multi-GPU Analysis ==="
echo "Per-rank profiling files:"
ls -la "${OUTPUT_DIR}/"*"rank_"*"${TIMESTAMP}"* 2>/dev/null || echo "No per-rank files found"

echo ""
echo "=== Performance Summary ==="
if [[ -f "${OUTPUT_DIR}/${TEST_NAME}_advanced_${NUM_GPUS}gpu_${TIMESTAMP}.csv" ]]; then
    echo "Kernel execution summary from advanced profiling:"
    head -1 "${OUTPUT_DIR}/${TEST_NAME}_advanced_${NUM_GPUS}gpu_${TIMESTAMP}.csv"
    tail -n +2 "${OUTPUT_DIR}/${TEST_NAME}_advanced_${NUM_GPUS}gpu_${TIMESTAMP}.csv" | \
        awk -F, '{sum+=$7; count++} END {if(count>0) printf "Total kernels: %d, Average duration: %.3f ms\n", count, sum/count}'
fi

echo ""
echo "=== GPU Utilization Check ==="
echo "Final GPU status:"
rocm-smi --showtemp --showuse