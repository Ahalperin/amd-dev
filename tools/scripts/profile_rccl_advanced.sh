#!/bin/bash

# Advanced RCCL Profiling Script with rocprofv3
# Usage: ./profile_rccl_advanced.sh <test_name> [additional_args]

set -e

# Default parameters
TEST_NAME=${1:-all_reduce_perf}
RCCL_TESTS_DIR="/workspace/rccl-tests"
OUTPUT_DIR="/workspace/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variables for enhanced profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export HSA_NO_SCRATCH_RECLAIM=1
export ROCP_TOOL_LIB=/opt/rocm/lib/librocprofiler64.so

echo "=== Advanced RCCL Profiling with rocprofv3 ==="
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

echo "=== System Information ==="
echo "ROCm Version:"
cat /opt/rocm/.info/version || echo "Version file not found"
echo ""
echo "Available GPUs:"
rocm-smi --showid --showtemp --showuse

echo "=== Running Advanced Profiling ==="

# 1. Comprehensive profiling with Perfetto output (Chrome tracing)
echo "1. Comprehensive profiling with Perfetto tracing..."
rocprofv3 --plugin perfetto \
          --kernel-trace \
          --hip-trace \
          --hsa-trace \
          --output-file "${OUTPUT_DIR}/${TEST_NAME}_perfetto_${TIMESTAMP}.pftrace" \
          -- "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

# 2. Detailed CSV output with performance counters
echo "2. Detailed CSV profiling..."
rocprofv3 --plugin csv \
          --kernel-trace \
          --hip-trace \
          --output-file "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" \
          -- "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

# 3. System-level profiling (if available)
echo "3. System-level profiling..."
if command -v rocprof-sys-run &> /dev/null; then
    rocprof-sys-run --trace \
                    --sample-freq=1000 \
                    --output="${OUTPUT_DIR}/${TEST_NAME}_system_${TIMESTAMP}" \
                    -- "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}
else
    echo "rocprof-sys-run not available, skipping system profiling"
fi

# 4. Performance counter profiling with specific metrics
echo "4. Performance counter profiling..."
rocprof --hip-trace \
        -m SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR \
        --output-file "${OUTPUT_DIR}/${TEST_NAME}_counters_${TIMESTAMP}" \
        "${TEST_EXEC}" --allow-run-as-root -b 8 -e 128M -f 2 -g 1 ${@:2}

echo "=== Profiling Complete ==="
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}/"*"${TIMESTAMP}"*

echo ""
echo "=== Analysis Instructions ==="
echo "1. Perfetto trace can be viewed at: https://ui.perfetto.dev/"
echo "   Upload: ${OUTPUT_DIR}/${TEST_NAME}_perfetto_${TIMESTAMP}.pftrace"
echo ""
echo "2. CSV data can be analyzed with:"
echo "   - Spreadsheet applications (Excel, LibreOffice Calc)"
echo "   - Python pandas: pd.read_csv('${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv')"
echo ""
echo "3. Quick kernel analysis:"
if [[ -f "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" ]]; then
    echo "   Total kernels executed:"
    tail -n +2 "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" | wc -l
    echo "   Top 5 longest kernels:"
    head -1 "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv"
    tail -n +2 "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" | sort -t, -k7 -nr | head -5
fi