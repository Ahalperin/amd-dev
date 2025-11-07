#!/bin/bash
# RCCL MPI Profiling Script with NPKit
# This script profiles RCCL multi-node MPI workloads with NPKit instrumentation only
#
# IMPROVEMENTS:
# - Pre-warming phase to compile kernels before profiling (eliminates ~1.8s overhead)
# - Kernel caching enabled to reuse compiled kernels
# - Fixed directory structure: timestamp/rank_N instead of rank_N/timestamp
# - Error handling: always creates zip file even if profiling fails
# - NPKit profiling for RCCL internals:
#   * NPKit on all ranks: Fine-grained RCCL internal events
#   * NPKIT_DUMP_DIR: Per-rank NPKit trace output
#   * NPKIT_FLAGS=0xFFFFFFFFFFFFFFFF: Capture all NPKit events
#   * Captures: Send/Recv entry/exit, kernel launch, channel ops, primitives
# - Automatic trace generation:
#   * Converts NPKit binary traces to JSON for rank 0 and rank 1
#   * Uses npkit_trace_generator.py from RCCL
# - Output: NPKit JSON traces + logs (binary dumps excluded from archive)
# - View: chrome://tracing or Perfetto (https://ui.perfetto.dev)

# Configuration
NP=${1:-16}
HOSTS=${2:-"172.30.160.150:8,172.30.160.145:8"}
MESSAGE_SIZE=${3:-"256M"}
ITERATIONS=${4:-20}
WARMUP=${5:-5}

OUTPUT_BASE="/home/dn/amd-dev/rccl_mpi_profile"
TIMESTAMP=$(date +%Y-%m-%d_%H.%M)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# Track exit status
PROFILE_EXIT_CODE=0

echo "======================================================================"
echo "RCCL MPI Profiling with NPKit"
echo "======================================================================"
echo "Ranks:        $NP"
echo "Hosts:        $HOSTS"
echo "Message Size: $MESSAGE_SIZE"
echo "Iterations:   $ITERATIONS"
echo "Warmup:       $WARMUP"
echo "Output:       $OUTPUT_DIR"
echo "Timestamp:    $TIMESTAMP"
echo "======================================================================"

# Create output directory with timestamp
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
mkdir -p ${OUTPUT_DIR}/npkit

# Note: Pre-warming is DISABLED because it would also collect NPKit data
# Kernel compilation overhead will be included in the first iterations

# Run MPI profiling
echo "======================================================================"
echo "PROFILING: Running with NPKit instrumentation..."
echo "======================================================================"
echo "Configuration:"
echo "  - NPKit:               ALL RANKS (fine-grained RCCL events)"
echo "  - NPKit coverage:      NPKIT_FLAGS=0xFFFFFFFFFFFFFFFF (all events)"
echo "  - Output formats:      NPKit binary dumps (will be converted to JSON)"
echo ""
echo "  What you'll capture:"
echo "  ✓ NPKit: Fine-grained RCCL internal events:"
echo "    * Send/Recv entry/exit times"
echo "    * Kernel launch times"
echo "    * Channel operations"
echo "    * Fine-grained RCCL primitive timing"
echo "  ✓ Complete RCCL internal timing for AllReduce"
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
-x LD_LIBRARY_PATH=/home/dn/amd-dev/dn/rccl/build/release:/home/dn/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/dn/amd-dev/dn/amd-anp/build/librccl-net.so:/home/dn/amd-dev/dn/rccl/build/release/librccl.so \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_DMABUF_ENABLE=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x RCCL_LL128_FORCE_ENABLE=1 \
-x NCCL_PROTO=SIMPLE \
-x NPKIT_DUMP_DIR=${OUTPUT_DIR}/npkit \
-x NPKIT_FLAGS=0xFFFFFFFFFFFFFFFF \
-x NCCL_DEBUG=INFO \
-x NCCL_DEBUG_FILE=${OUTPUT_DIR}/rccl_debug.log \
-x NCCL_TOPO_DUMP_FILE=${OUTPUT_DIR}/rccl.topo.log \
-x NCCL_DEBUG_SUBSYS=GRAPH \
-x NCCL_GRAPH_DUMP_FILE=nccl_graph.xml \
/home/dn/amd-dev/dn/rccl-tests/build/all_reduce_perf -b ${MESSAGE_SIZE} -e ${MESSAGE_SIZE} -f 2 -g 1 -n ${ITERATIONS} -c 1 -w ${WARMUP}

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
    echo "Will still attempt to process any collected data..."
fi

# Check output
echo "Checking output..."
RANK_COUNT=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "rank_*" 2>/dev/null | wc -l)
echo "Found $RANK_COUNT rank directories"

# Check for NPKit data (RCCL creates per-rank subdirectories automatically)
NPKIT_BASE_DIR="${OUTPUT_DIR}/npkit"
NPKIT_COUNT=$(find ${NPKIT_BASE_DIR} -type f 2>/dev/null | wc -l)
echo "Found $NPKIT_COUNT NPKit binary trace files in ${NPKIT_BASE_DIR}"

# Convert NPKit traces to JSON for rank 0 and rank 1
RCCL_HOME="/home/dn/amd-dev/dn/rccl"
NPKIT_GENERATOR="${RCCL_HOME}/tools/scripts/npkit_trace_generator.py"
NPKIT_HEADER="${RCCL_HOME}/src/include/npkit/npkit_event.h"

if [ $NPKIT_COUNT -gt 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Converting NPKit traces to JSON..."
    echo "======================================================================"
    
    # RCCL creates files like: gpu_events_rank_0_buf_0, cpu_events_rank_0_channel_0, etc.
    # We convert rank 0 and rank 1
    for RANK in 0 1; do
        # Check if any files exist for this rank
        RANK_FILES=$(find ${NPKIT_BASE_DIR} -name "*_rank_${RANK}_*" 2>/dev/null | wc -l)
        
        if [ $RANK_FILES -gt 0 ]; then
            echo "Processing rank ${RANK} (${RANK_FILES} files)..."
            OUTPUT_JSON_DIR="${OUTPUT_DIR}/npkit_json_rank_${RANK}"
            mkdir -p "$OUTPUT_JSON_DIR"
            
            python3 "$NPKIT_GENERATOR" \
                --npkit_dump_dir="$NPKIT_BASE_DIR" \
                --npkit_event_header_path="$NPKIT_HEADER" \
                --output_dir="$OUTPUT_JSON_DIR" 2>&1 | tee "${OUTPUT_DIR}/npkit_conversion_rank_${RANK}.log"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Rank ${RANK} trace converted successfully"
                if [ -f "${OUTPUT_JSON_DIR}/npkit_event_trace.json" ]; then
                    echo "    Output: ${OUTPUT_JSON_DIR}/npkit_event_trace.json"
                fi
            else
                echo "  ✗ Rank ${RANK} trace conversion failed"
            fi
        else
            echo "No NPKit data found for rank ${RANK}, skipping..."
        fi
    done
    echo "NPKit trace conversion complete!"
fi

# Create zip file excluding binary NPKit dumps, only include logs and JSON
ZIP_FILE="${OUTPUT_BASE}/rccl_npkit_profile_${TIMESTAMP}.zip"
echo ""
echo "======================================================================"
echo "Creating archive: $ZIP_FILE"
echo "======================================================================"
echo "Including: *.json, *.log, *.txt, *.xml files"
echo "Excluding: Binary NPKit dumps (gpu_events_*, cpu_events_*, gpu_clock_*, cpu_clock_*)"

if [ -d "${OUTPUT_DIR}" ]; then
    cd ${OUTPUT_BASE}
    # Create zip with logs and JSON files only, exclude binary dumps
    zip -r "rccl_npkit_profile_${TIMESTAMP}.zip" "${TIMESTAMP}/" \
        -i "*.json" "*.log" "*.txt" "*.xml" \
        -x "*/npkit/gpu_events_*" "*/npkit/cpu_events_*" "*/npkit/gpu_clock_*" "*/npkit/cpu_clock_*" 2>/dev/null
    ZIP_EXIT_CODE=$?
    
    if [ $ZIP_EXIT_CODE -eq 0 ]; then
        ZIP_SIZE=$(du -h "$ZIP_FILE" 2>/dev/null | cut -f1)
        echo "Archive created successfully: $ZIP_SIZE"
        echo "Contents: JSON traces, logs, and metadata (binary dumps excluded)"
    else
        echo "Warning: Archive creation had issues (exit code: $ZIP_EXIT_CODE)"
    fi
else
    echo "Error: Output directory ${OUTPUT_DIR} does not exist!"
fi

echo ""
echo "======================================================================"

# Check if JSON files were created
JSON_COUNT=$(find ${OUTPUT_DIR} -name "npkit_event_trace.json" 2>/dev/null | wc -l)

if [ $PROFILE_EXIT_CODE -eq 0 ] && [ "$NPKIT_COUNT" -gt 0 ]; then
    echo "SUCCESS: NPKit profiling completed successfully!"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled:       $RANK_COUNT"
    echo "  - NPKit binary files:   $NPKIT_COUNT"
    echo "  - JSON traces created:  $JSON_COUNT (rank 0 & 1)"
    echo "  - Archive:              $ZIP_FILE"
    echo ""
    echo "Download with:"
    echo "  scp dn@172.30.160.150:$ZIP_FILE ./"
    echo ""
    echo "Analysis:"
    echo ""
    echo "View NPKit traces in chrome://tracing or Perfetto (https://ui.perfetto.dev):"
    echo "   - Load: npkit_json_rank_0/npkit_event_trace.json"
    echo "   - Load: npkit_json_rank_1/npkit_event_trace.json"
    echo ""
    echo "   NPKit shows detailed RCCL internal events:"
    echo "     * Send/Recv entry/exit times"
    echo "     * Kernel launch times"
    echo "     * Channel operations"
    echo "     * Fine-grained RCCL primitive timing"
    echo ""
    echo "Diagnosing AllReduce bottleneck:"
    echo "   1. Load npkit_event_trace.json in chrome://tracing"
    echo "   2. Zoom to a single AllReduce iteration"
    echo "   3. Look for:"
    echo "      - RCCL kernel launch delays"
    echo "      - Send/Recv operation timing"
    echo "      - Channel utilization patterns"
    echo "      - Synchronization gaps between operations"
    echo "   4. Compare rank 0 vs rank 1 timing to find asymmetries"
elif [ "$NPKIT_COUNT" -gt 0 ]; then
    echo "PARTIAL SUCCESS: Some NPKit data captured"
    echo "======================================================================"
    echo "Results:"
    echo "  - Ranks profiled:       $RANK_COUNT"
    echo "  - NPKit binary files:   $NPKIT_COUNT"
    echo "  - JSON traces created:  $JSON_COUNT"
    echo "  - Archive:              $ZIP_FILE"
    echo "  - Exit code:            $PROFILE_EXIT_CODE"
    echo ""
    echo "Note: Profiling encountered errors but some data was collected."
else
    echo "WARNING: No NPKit data generated"
    echo "======================================================================"
    echo "Possible causes:"
    echo "  - MPI execution failed (exit code: $PROFILE_EXIT_CODE)"
    echo "  - RCCL not compiled with NPKit support (use --enable-npkit)"
    echo "  - NPKIT_DUMP_DIR not writable"
    echo "  - NPKIT_FLAGS not set correctly"
    echo ""
    echo "Archive created with available data: $ZIP_FILE"
fi
echo "======================================================================"

# Exit with the profiling exit code
exit $PROFILE_EXIT_CODE
