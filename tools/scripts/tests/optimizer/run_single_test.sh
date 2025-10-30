#!/bin/bash
# Quick script to run a single RCCL test manually for debugging
# This simulates what the optimizer does

set -x  # Print commands as they execute

# Load environment from config
export PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib:
export LD_PRELOAD=/home/dn/amd-dev/amd/amd-anp/build/librccl-net.so:/home/dn/amd-dev/amd/rccl/build/release/librccl.so

# Fixed NCCL settings
export NCCL_SOCKET_IFNAME=enp81s0f1np1
export NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1
export NCCL_IB_GID_INDEX=1
export NCCL_DEBUG=VERSION
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_GDRCOPY_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export IONIC_LOCKFREE=all

# Test parameters
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_TC=104
export RCCL_LL128_FORCE_ENABLE=1

echo "================================================================================"
echo "Running single RCCL test with timeout..."
echo "This should complete in ~30-60 seconds"
echo "================================================================================"

# Run with timeout
timeout 120s mpirun \
  --np 16 \
  --allow-run-as-root \
  -H 172.30.160.145:8,172.30.160.150:8 \
  --bind-to numa \
  --mca oob_tcp_if_include enp81s0f1np1 \
  --mca btl_tcp_if_include enp81s0f1np1 \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x LD_PRELOAD \
  -x NCCL_IB_GID_INDEX \
  -x NCCL_GDR_FLUSH_DISABLE=1 \
  -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
  -x NCCL_GDRCOPY_ENABLE \
  -x NCCL_IB_HCA \
  -x NCCL_DMABUF_ENABLE \
  -x NCCL_IB_QPS_PER_CONNECTION \
  -x HSA_NO_SCRATCH_RECLAIM \
  -x NCCL_IB_TC \
  -x NCCL_IB_FIFO_TC=192 \
  -x NCCL_IGNORE_CPU_AFFINITY \
  -x NCCL_DEBUG \
  -x NET_OPTIONAL_RECV_COMPLETION=1 \
  -x NCCL_IB_USE_INLINE=1 \
  -x NCCL_SOCKET_IFNAME \
  -x IONIC_LOCKFREE \
  -x NCCL_PXN_DISABLE=0 \
  -x RCCL_LL128_FORCE_ENABLE \
  /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf \
  -b 1M -e 1M -f 2 -g 1 -n 5 -c 1 -w 1

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Test completed successfully!"
elif [ $EXIT_CODE -eq 124 ]; then
  echo "✗ Test TIMED OUT (>120s)"
else
  echo "✗ Test failed with exit code: $EXIT_CODE"
fi
echo "================================================================================"

exit $EXIT_CODE


