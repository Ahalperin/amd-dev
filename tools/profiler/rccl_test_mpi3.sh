#!/bin/bash
# RCCL MPI Profiling Script with rocprofv3
# This script profiles RCCL multi-node MPI workloads
#
# IMPROVEMENTS:
# - Pre-warming phase to compile kernels before profiling (eliminates ~1.8s overhead)
# - Kernel caching enabled to reuse compiled kernels
# - Fixed directory structure: timestamp/rank_N instead of rank_N/timestamp
# - Error handling: always creates zip file even if profiling fails
# - Dual profiling for complete picture:
#   * rocprofv3 on ranks 0 & 1: GPU/API tracing (RCCL, kernels, memory, HSA)
#   * perf on rank 0: CPU profiling for proxy thread function names
# - rocprofv3 configuration:
#   * --runtime-trace (HIP runtime, RCCL API, Kernels, Memory - comprehensive)
#   * --hsa-core-trace + --hsa-amd-trace (HSA signals & DMA for proxy thread)
#   * ROCTX_ENABLE (ROCTx markers) + ROCPROFILER_SHOW_DISPATCH_SIGNAL
# - perf configuration (rank 0 only):
#   * 999 Hz sampling with DWARF call graphs
#   * Captures: ncclProxyPost, progressOps, sendProxyProgress, etc.
# - Output: Perfetto traces (.pftrace) + perf.data (rank 0)
# - Shows complete 256M AllReduce performance:
#   GPU (kernels), Network (RCCL Comm), CPU (proxy functions)
# - View: Perfetto (https://ui.perfetto.dev) + perf report

# Configuration
NP=${1:-16}
HOSTS=${2:-"172.30.160.150:8,172.30.160.145:8"}
MESSAGE_SIZE=${3:-"256M"}
ITERATIONS=${4:-20}
WARMUP=${5:-5}
SKIP_PREWARM=${6:-0}

OUTPUT_BASE="/home/dn/amd-dev/rccl_mpi_profile"
TIMESTAMP=$(date +%Y-%m-%d_%H.%M)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
KERNEL_CACHE_DIR="/tmp/rocm_kernel_cache"

# DMA Configuration
# HSA_ENABLE_SDMA=0 disables System DMA engines (can improve stability)
# HSA_ENABLE_SDMA=1 enables SDMA for async copies (may see more DMA activity)
# Note: With profiling enabled (HSA_AMD_PROFILING_ASYNC_COPY_ENABLE=1),
#       DMA operations are visible even with SDMA disabled
ENABLE_SDMA=${ENABLE_SDMA:-0}

# Track exit status
PROFILE_EXIT_CODE=0

echo "======================================================================"
echo "RCCL MPI Profiling with rocprofv3 + DMA Tracking"
echo "======================================================================"
echo "Ranks:        $NP"
echo "Hosts:        $HOSTS"
echo "Message Size: $MESSAGE_SIZE"
echo "Iterations:   $ITERATIONS"
echo "Warmup:       $WARMUP"
echo "Output:       $OUTPUT_DIR"
echo "Timestamp:    $TIMESTAMP"
echo "Skip Prewarm: $SKIP_PREWARM"
echo "SDMA Engine:  $ENABLE_SDMA (0=disabled, 1=enabled)"
echo "======================================================================"

# Setup kernel cache directory
echo "Setting up kernel cache directory..."
mkdir -p $KERNEL_CACHE_DIR
chmod 777 $KERNEL_CACHE_DIR

# Create output directory with timestamp
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Pre-warm GPUs and compile kernels (CRITICAL for accurate profiling)
if [ "${SKIP_PREWARM:-0}" -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "PRE-WARMING: Running test without profiling to compile kernels..."
    echo "======================================================================"
    
    mpirun --np $NP --allow-run-as-root -H $HOSTS \
    --bind-to numa \
    --mca oob_tcp_if_include enp81s0f1np1 \
    --mca btl_tcp_if_include enp81s0f1np1 \
    -x NCCL_IB_GID_INDEX=1 \
    -x NCCL_GDR_FLUSH_DISABLE=1 \
    -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
    -x NCCL_GDRCOPY_ENABLE=0 \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    -x NCCL_IB_TC=104 \
    -x NCCL_IB_FIFO_TC=192 \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
    -x IONIC_LOCKFREE=all \
    -x RCCL_LL128_FORCE_ENABLE=1 \
    -x ROCM_PATH=/opt/rocm \
    -x ROCM_KERNEL_CACHE_PATH=$KERNEL_CACHE_DIR \
    -x HSA_ENABLE_KERNEL_CACHE=1 \
    -x HSA_ENABLE_SDMA=$ENABLE_SDMA \
    -x GPU_MAX_HW_QUEUES=8 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
    -b ${MESSAGE_SIZE} -e ${MESSAGE_SIZE} -f 2 -g 1 -n 5 -c 1 -w 2 > /dev/null 2>&1
    
    echo "Pre-warming complete! Kernels compiled and cached."
    echo ""
fi

# Run MPI profiling
echo "======================================================================"
echo "PROFILING: Running with rocprofv3 instrumentation..."
echo "======================================================================"
echo "Configuration:"
echo "  - Profile ranks:       RANK 0 and RANK 1 with rocprofv3"
echo "  - Profilers:           rocprofv3 (GPU/API tracing)"
echo "  - rocprofv3 coverage:  --runtime-trace (HIP runtime, RCCL API, Kernels,"
echo "                         Memory copies/alloc, Marker API)"
echo "                         --hsa-core-trace (proxy thread HSA signals)"
echo "                         --hsa-amd-trace (DMA operations)"
echo "  - Statistics:          ENABLED (--stats for aggregated metrics)"
echo "  - ROCTx markers:       ENABLED (ROCTX_ENABLE=1)"
echo "  - RCCL profiling:      ENABLED (ROCPROFILER_RCCL_ENABLE=1)"
echo "  - Proxy tracing:       ENABLED (RCCL_ENABLE_PROXY_TRACE=1)"
echo "  - Signal correlation:  ENABLED (ROCPROFILER_SHOW_DISPATCH_SIGNAL)"
echo "  - Output formats:      Perfetto (.pftrace)"
echo "  - Perfetto buffer:     512 MB"
echo ""
echo "  What you'll capture:"
echo "  ✓ RCCL Comm Send/Recv operations (network timing)"
echo "  ✓ RCCL Proxy Step events (individual network transfer steps)"
echo "  ✓ Kernel execution (rcclGenericKernel, prepareInput2)"
echo "  ✓ CPU proxy thread HSA activity (hsa_signal_load bursts)"
echo "  ✓ CPU-GPU synchronization (hsa_signal_load bursts)"
echo "  ✓ HIP API calls (hipLaunchKernel, hipStreamQuery, etc.)"
echo "  ✓ DMA operations (HSA AMD extensions)"
echo "  ✓ Complete picture of 256M AllReduce performance"
echo "======================================================================"
echo ""

# Run profiling and capture exit code (don't exit on failure)
mpirun --np $NP --allow-run-as-root -H $HOSTS \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH \
-x LD_LIBRARY_PATH \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_DMABUF_ENABLE=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_TOPO_DUMP_FILE=${OUTPUT_DIR}/rccl.topo.log \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x RCCL_LL128_FORCE_ENABLE=1 \
-x ROCM_PATH=/opt/rocm \
-x ROCM_KERNEL_CACHE_PATH=$KERNEL_CACHE_DIR \
-x HSA_ENABLE_KERNEL_CACHE=1 \
-x HSA_ENABLE_SDMA=$ENABLE_SDMA \
-x GPU_MAX_HW_QUEUES=8 \
-x HSA_FORCE_FINE_GRAIN_PCIE=1 \
-x AMD_DIRECT_DISPATCH=1 \
-x HSA_AMD_PROFILING_ASYNC_COPY_ENABLE=1 \
-x ROCPROFILER_SHOW_DISPATCH_SIGNAL=1 \
-x ROCTX_ENABLE=1 \
-x ROCPROFILER_RCCL_ENABLE=1 \
-x RCCL_ENABLE_PROXY_TRACE=1 \
-x OMPI_COMM_WORLD_RANK \
bash -c "mkdir -p ${OUTPUT_DIR}/rank_\$OMPI_COMM_WORLD_RANK; \
if [ \"\$OMPI_COMM_WORLD_RANK\" = \"0\" ] || [ \"\$OMPI_COMM_WORLD_RANK\" = \"1\" ]; then \
  rocprofv3 \
    -d ${OUTPUT_DIR}/rank_\$OMPI_COMM_WORLD_RANK \
    -o profile \
    -f pftrace \
    --runtime-trace \
    --hsa-core-trace \
    --hsa-amd-trace \
    --stats \
    --group-by-queue \
    --rccl-trace \
    --kernel-trace \
    --hip-runtime-trace \
    --memory-copy-trace \
    --marker-trace \
    --perfetto-buffer-size 524288 \
    --perfetto-buffer-fill-policy discard \
    -- /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b ${MESSAGE_SIZE} -e ${MESSAGE_SIZE} -f 2 -g 1 -n ${ITERATIONS} -c 1 -w ${WARMUP}; \
else \
  /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b ${MESSAGE_SIZE} -e ${MESSAGE_SIZE} -f 2 -g 1 -n ${ITERATIONS} -c 1 -w ${WARMUP}; \
fi"

PROFILE_EXIT_CODE=$?

echo ""
if [ $PROFILE_EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "Profiling Complete!"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "WARNING: Profiling exited with error code: $PROFILE_EXIT_CODE"
    echo "======================================================================"
    echo "Will still attempt to package any collected data..."
fi

# Check output
echo "Checking output..."
RANK_COUNT=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "rank_*" 2>/dev/null | wc -l)
echo "Found $RANK_COUNT rank directories"

# Check for actual profiling data (rocprofv3 with -f pftrace produces .pftrace files)
PFTRACE_COUNT=$(find ${OUTPUT_DIR} -name "*.pftrace" 2>/dev/null | wc -l)
echo "Found $PFTRACE_COUNT perfetto trace files (.pftrace = .proto format)"

# Always create zip file (even if profiling failed or data is incomplete)
ZIP_FILE="${OUTPUT_BASE}/rccl_mpi_profile_${TIMESTAMP}.zip"
echo ""
echo "======================================================================"
echo "Creating archive: $ZIP_FILE"
echo "======================================================================"

if [ -d "${OUTPUT_DIR}" ]; then
    cd ${OUTPUT_BASE} && zip -r "rccl_mpi_profile_${TIMESTAMP}.zip" "${TIMESTAMP}/" 2>/dev/null
    ZIP_EXIT_CODE=$?
    
    if [ $ZIP_EXIT_CODE -eq 0 ]; then
        ZIP_SIZE=$(du -h "$ZIP_FILE" 2>/dev/null | cut -f1)
        echo "Archive created successfully: $ZIP_SIZE"
    else
        echo "Warning: Archive creation had issues (exit code: $ZIP_EXIT_CODE)"
    fi
else
    echo "Error: Output directory ${OUTPUT_DIR} does not exist!"
fi

echo ""
echo "======================================================================"
if [ $PROFILE_EXIT_CODE -eq 0 ] && [ "$PFTRACE_COUNT" -gt 0 ]; then
    echo "SUCCESS: Profiling completed successfully!"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled:    $RANK_COUNT"
    echo "  - Perfetto traces:   $PFTRACE_COUNT (.pftrace files = .proto format)"
    echo "  - Archive:           $ZIP_FILE"
    echo ""
    echo "Download with:"
    echo "  scp dn@172.30.160.150:$ZIP_FILE ./"
    echo ""
    echo "Analysis:"
    echo ""
    echo "View in Perfetto (https://ui.perfetto.dev):"
    echo "   - Upload rank_0/profile_*.pftrace or rank_1/profile_*.pftrace"
    echo "   - You'll see:"
    echo "     * GPU tracks: rcclGenericKernel execution"
    echo "     * RCCL tracks: RCCL Comm Send/Recv operations"
    echo "     * RCCL Proxy tracks: Individual proxy step events"
    echo "     * HSA tracks: hsa_signal_load_relaxed bursts (proxy thread HSA activity)"
    echo "     * HIP API calls: hipLaunchKernel, hipStreamQuery, etc."
    echo ""
    echo "Diagnosing 256M AllReduce bottleneck:"
    echo "   1. Load .pftrace in Perfetto"
    echo "   2. Zoom to a single AllReduce iteration"
    echo "   3. Measure:"
    echo "      - rcclGenericKernel duration (GPU time)"
    echo "      - RCCL Comm Send/Recv duration (network time)"
    echo "      - RCCL Proxy Step events (individual network transfer granularity)"
    echo "      - HSA signal bursts during kernel (proxy thread activity)"
    echo "   4. Check if network overlaps with kernel (pipelining)"
    echo "   5. Look for idle gaps (synchronization issues)"
    echo "   6. Inspect proxy steps to see which network operations are slow"
elif [ "$PFTRACE_COUNT" -gt 0 ]; then
    echo "PARTIAL SUCCESS: Some profiling data captured"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled:    $RANK_COUNT"
    echo "  - Perfetto traces:   $PFTRACE_COUNT (.pftrace files = .proto format)"
    echo "  - Archive:           $ZIP_FILE"
    echo "  - Exit code:         $PROFILE_EXIT_CODE"
    echo ""
    echo "Note: Profiling encountered errors but some data was collected."
else
    echo "WARNING: No profiling data generated"
    echo "======================================================================"
    echo "Possible causes:"
    echo "  - MPI execution failed (exit code: $PROFILE_EXIT_CODE)"
    echo "  - Insufficient permissions"
    echo "  - rocprofv3 not found or not working"
    echo "  - LD_LIBRARY_PATH issues"
    echo ""
    echo "Archive created with available data: $ZIP_FILE"
fi
echo "======================================================================"

# Exit with the profiling exit code
exit $PROFILE_EXIT_CODE

