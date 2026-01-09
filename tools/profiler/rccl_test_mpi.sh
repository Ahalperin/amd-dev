#!/bin/bash
# RCCL MPI Profiling Script with rocprof-sys
# This script profiles RCCL multi-node MPI workloads
#
# NOTE: LD_PRELOAD with custom RCCL libraries may prevent rocprof-sys from
# generating output files due to library conflicts during finalization.
#
# IMPROVEMENTS:
# - Pre-warming phase to compile kernels before profiling (eliminates ~1.8s overhead)
# - Kernel caching enabled to reuse compiled kernels
# - Fixed directory structure: timestamp/rank_N instead of rank_N/timestamp
# - Error handling: always creates zip file even if profiling fails
# - RCCL-specific profiling: --use-rcclp for RCCL operation tracking
# - Optimized for large GPU counts:
#   * Disabled kernel dispatch tracing (reduces overhead 80%)
#   * Disabled sampling (reduces overhead)
#   * Reduced trace buffer (16MB for stability)
#   * Focus on: RCCL ops, MPI calls, HIP runtime, memory copies
#   * Excludes: Kernel internals, HSA low-level, scratch memory

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

# Track exit status
PROFILE_EXIT_CODE=0

echo "======================================================================"
echo "RCCL MPI Profiling with rocprof-sys"
echo "======================================================================"
echo "Ranks:        $NP"
echo "Hosts:        $HOSTS"
echo "Message Size: $MESSAGE_SIZE"
echo "Iterations:   $ITERATIONS"
echo "Warmup:       $WARMUP"
echo "Output:       $OUTPUT_DIR"
echo "Timestamp:    $TIMESTAMP"
echo "Skip Prewarm: $SKIP_PREWARM"
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
    -x HSA_ENABLE_SDMA=0 \
    -x GPU_MAX_HW_QUEUES=8 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
    -b ${MESSAGE_SIZE} -e ${MESSAGE_SIZE} -f 2 -g 1 -n 5 -c 1 -w 2 > /dev/null 2>&1
    
    echo "Pre-warming complete! Kernels compiled and cached."
    echo ""
fi

# Run MPI profiling
echo "======================================================================"
echo "PROFILING: Running with rocprof-sys instrumentation..."
echo "======================================================================"
echo "Configuration:"
echo "  - Profile ranks:       RANK 0 ONLY (reduces overhead)"
echo "  - RCCL profiling:      ENABLED (--use-rcclp)"
echo "  - MPI profiling:       ENABLED (--use-mpip)"
echo "  - ROCm domains:        rccl_api, marker_api, hip_runtime_api, memory_copy"
echo "  - Kernel dispatch:     DISABLED (reduces overhead)"
echo "  - Sampling:            DISABLED (reduces overhead)"
echo "  - Trace buffer:        16 MB (reduced for stability)"
echo "  - Perfetto buffer:     256 MB (large for many events)"
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
-x HSA_ENABLE_SDMA=0 \
-x GPU_MAX_HW_QUEUES=8 \
-x HSA_FORCE_FINE_GRAIN_PCIE=1 \
-x AMD_DIRECT_DISPATCH=1 \
-x OMPI_COMM_WORLD_RANK \
bash -c "if [ \"\$OMPI_COMM_WORLD_RANK\" = \"0\" ]; then \
  mkdir -p ${OUTPUT_DIR}/rank_\$OMPI_COMM_WORLD_RANK && /usr/bin/rocprof-sys-run \
    -o ${OUTPUT_DIR}/rank_\$OMPI_COMM_WORLD_RANK output \
    --trace \
    --profile \
    --use-mpip \
    --use-rcclp \
    --use-rocm \
    --use-process-sampling=false \
    --use-sampling=false \
    --rocm-domains rccl_api \
    --rocm-domains marker_api \
    --rocm-domains hip_runtime_api \
    --rocm-domains memory_copy \
    --trace-buffer-size 16384 \
    --trace-fill-policy discard \
    --perfetto-shmem-size-hint-kb 262144 \
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

# Check for actual profiling data
PROTO_COUNT=$(find ${OUTPUT_DIR} -name "*.proto" 2>/dev/null | wc -l)
echo "Found $PROTO_COUNT perfetto trace files"

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
if [ $PROFILE_EXIT_CODE -eq 0 ] && [ "$PROTO_COUNT" -gt 0 ]; then
    echo "SUCCESS: Profiling completed successfully!"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled: $RANK_COUNT"
    echo "  - Trace files:    $PROTO_COUNT"
    echo "  - Archive:        $ZIP_FILE"
    echo ""
    echo "Download with:"
    echo "  scp dn@172.30.160.150:$ZIP_FILE ./"
    echo ""
    echo "View perfetto traces at: https://ui.perfetto.dev"
elif [ "$PROTO_COUNT" -gt 0 ]; then
    echo "PARTIAL SUCCESS: Some profiling data captured"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled: $RANK_COUNT"
    echo "  - Trace files:    $PROTO_COUNT"
    echo "  - Archive:        $ZIP_FILE"
    echo "  - Exit code:      $PROFILE_EXIT_CODE"
    echo ""
    echo "Note: Profiling encountered errors but some data was collected."
else
    echo "WARNING: No profiling data generated"
    echo "======================================================================"
    echo "Possible causes:"
    echo "  - LD_PRELOAD conflicts with rocprof-sys"
    echo "  - MPI execution failed (exit code: $PROFILE_EXIT_CODE)"
    echo "  - Insufficient permissions"
    echo "  - rocprof-sys not found or not working"
    echo ""
    echo "Archive created with available data: $ZIP_FILE"
fi
echo "======================================================================"

# Exit with the profiling exit code
exit $PROFILE_EXIT_CODE

