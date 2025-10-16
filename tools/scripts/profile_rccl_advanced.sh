#!/bin/bash

# Advanced RCCL Profiling Script with rocprofv3
# Usage: ./profile_rccl_advanced.sh <test_name> [additional_args]

set -e

# Default parameters
TEST_NAME=${1:-all_reduce_perf}
RCCL_TESTS_DIR="/workspace/rccl-tests/"
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
# Note: For multi-GPU runs, we use only kernel-trace to avoid data corruption from millions of HIP API calls
# --hip-trace and --hsa-trace generate excessive data (~3M+ calls) that causes profiler corruption
rocprofv3 --output-format pftrace \
          --memory-copy-trace \
          --memory-allocation-trace \
          --rccl-trace \
          --marker-trace \
          --kernel-trace \
          -o "${OUTPUT_DIR}/${TEST_NAME}_perfetto_${TIMESTAMP}" \
          -- "${TEST_EXEC}" ${@:2}

# # 2. Detailed CSV profiling...
# echo "2. Detailed CSV profiling..."
# rocprofv3 --output-format csv \
#           --kernel-trace \
#           --hip-trace \
#           -o "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}" \
#           -- "${TEST_EXEC}" ${@:2}

# # 3. JSON format for analysis tools
# echo "3. JSON profiling for analysis tools..."
# rocprofv3 --output-format json \
#           --kernel-trace \
#           --hip-trace \
#           --stats \
#           -o "${OUTPUT_DIR}/${TEST_NAME}_json_${TIMESTAMP}" \
#           -- "${TEST_EXEC}" ${@:2}

# 4. System-level profiling (if available)
# echo "4. System-level profiling..."
# if command -v rocprof-sys-run &> /dev/null; then
#     rocprof-sys-run --trace \
#                     --sample-freq=1000 \
#                     --output="${OUTPUT_DIR}/${TEST_NAME}_system_${TIMESTAMP}" \
#                     -- "${TEST_EXEC}" ${@:2}
# else
#     echo "rocprof-sys-run not available, skipping system profiling"
# fi

# 5. Performance counter profiling with specific metrics
# echo "5. Performance counter profiling..."
# rocprof --hip-trace \
#         -m SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR \
#         -o "${OUTPUT_DIR}/${TEST_NAME}_counters_${TIMESTAMP}.csv" \
#         "${TEST_EXEC}" ${@:2}

echo "=== Profiling Complete ==="
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}/"*"${TIMESTAMP}"*

echo ""
echo "=== Graphical Visualization Instructions ==="
echo ""
echo "🎨 GRAPHICAL VIEWING OPTIONS:"
echo ""
echo "1. 🌐 Perfetto Trace Viewer (RECOMMENDED):"
echo "   • Open: https://ui.perfetto.dev/"
echo "   • Upload: ${OUTPUT_DIR}/${TEST_NAME}_perfetto_${TIMESTAMP}.pftrace"
echo "   • Features: Timeline view, kernel details, API calls, memory operations"
echo ""
echo "2. 🔥 Chrome Tracing:"
echo "   • Open Chrome browser"
echo "   • Navigate to: chrome://tracing/"
echo "   • Load file: ${OUTPUT_DIR}/${TEST_NAME}_json_${TIMESTAMP}.json"
echo ""
echo "3. 📊 CSV Analysis Tools:"
echo "   • Spreadsheet: Excel, LibreOffice Calc, Google Sheets"
echo "   • Python: pandas, matplotlib, seaborn"
echo "   • File: ${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv"
echo ""
echo "4. 🔧 Analysis Scripts:"
echo "   • Basic analysis: python3 /tools/scripts/analyze_rccl_profile.py ${OUTPUT_DIR}/"
echo ""
echo "=== Quick Analysis ==="
if [[ -f "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" ]]; then
    echo "   Total kernels executed:"
    tail -n +2 "${OUTPUT_DIR}/${TEST_NAME}_detailed_${TIMESTAMP}.csv" | wc -l 2>/dev/null || echo "   CSV file not found"
fi