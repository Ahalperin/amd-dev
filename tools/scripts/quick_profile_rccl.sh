#!/bin/bash

# Quick RCCL Profiling Script - Get started fast!
# Usage: ./quick_profile_rccl.sh [test_name]

set -e

# Configuration
TEST_NAME=${1:-all_reduce_perf}
RCCL_TESTS_DIR="/workspace/rccl-tests"
OUTPUT_DIR="/workspace/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Quick RCCL Profiling Setup ===${NC}"
echo "Test: ${TEST_NAME}"
echo "Timestamp: ${TIMESTAMP}"

# Create directories
mkdir -p "${OUTPUT_DIR}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

# Check if RCCL tests are built
TEST_EXEC="${RCCL_TESTS_DIR}/build/${TEST_NAME}"
if [[ ! -f "${TEST_EXEC}" ]]; then
    echo -e "${RED}Error: RCCL test not found: ${TEST_EXEC}${NC}"
    echo "Available tests:"
    ls -1 "${RCCL_TESTS_DIR}/build/" | grep "_perf$" | head -10
    echo -e "\n${YELLOW}To build RCCL tests:${NC}"
    echo "cd ${RCCL_TESTS_DIR}"
    echo "mkdir -p build && cd build"
    echo "cmake -DCMAKE_BUILD_TYPE=Release .."
    echo "make -j\$(nproc)"
    exit 1
fi

# Check ROCm tools
if ! command -v rocprof &> /dev/null; then
    echo -e "${RED}Error: rocprof not found. Please install ROCm.${NC}"
    exit 1
fi

# Check GPUs
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "0")
if [[ ${GPU_COUNT} -eq 0 ]]; then
    echo -e "${RED}Error: No GPUs detected.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"
echo -e "${GREEN}✓ RCCL tests available${NC}"
echo -e "${GREEN}✓ ROCm profiling tools ready${NC}"

# Set environment variables
export NCCL_DEBUG=INFO
export HSA_NO_SCRATCH_RECLAIM=1

echo -e "\n${BLUE}=== Running Quick Profile ===${NC}"
echo "This will run a basic profiling session with essential metrics..."

# Run basic profiling with essential metrics
rocprof --hip-trace --hsa-trace \
        -m GRBM_GUI_ACTIVE,TCC_HIT,TCC_MISS,SQ_WAVES,TCC_EA_RDREQ,TCC_EA_WRREQ \
        -o "${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}.csv" \
        "${TEST_EXEC}" -b 8 -e 128M -f 2 -g 1

echo -e "\n${GREEN}=== Profiling Complete! ===${NC}"
echo "Results saved to: ${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}"

# Quick analysis
if [[ -f "${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}.csv" ]]; then
    echo -e "\n${BLUE}=== Quick Analysis ===${NC}"
    CSV_FILE="${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}.csv"
    
    # Count kernels
    KERNEL_COUNT=$(tail -n +2 "${CSV_FILE}" | wc -l)
    echo "Total kernels executed: ${KERNEL_COUNT}"
    
    # Show top 3 kernels by duration
    echo -e "\nTop 3 longest kernels:"
    echo "Kernel Name | Duration (ms)"
    echo "------------|-------------"
    tail -n +2 "${CSV_FILE}" | sort -t, -k7 -nr | head -3 | \
        awk -F, '{printf "%-50s | %.3f\n", $13, $7/1000000}'
fi

echo -e "\n${YELLOW}=== Next Steps ===${NC}"
echo "1. View detailed results:"
echo "   python3 /tools/scripts/analyze_rccl_profile.py ${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}.csv"
echo ""
echo "2. Run more comprehensive profiling:"
echo "   /tools/scripts/profile_rccl_advanced.sh ${TEST_NAME}"
echo ""
echo "3. Multi-GPU profiling:"
echo "   /tools/scripts/profile_rccl_multi_gpu.sh ${TEST_NAME} ${GPU_COUNT}"
echo ""
echo "4. View Chrome trace (if .json file exists):"
echo "   Open chrome://tracing and load ${OUTPUT_DIR}/${TEST_NAME}_quick_${TIMESTAMP}.json"

echo -e "\n${GREEN}Quick profiling session complete!${NC}"