################################################################################
# all_reduce_perf test command lines
################################################################################
# Amir develop
mpirun --np 16 --allow-run-as-root \
-H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/home/amir/amd-dev/dn/rccl/build/release:/home/amir/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/amir/amd-dev/dn/amd-anp/build/librccl-net.so:/home/amir/amd-dev/dn/rccl/build/release/librccl.so \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_MIN_NCHANNELS=64 \
-x NCCL_MAX_NCHANNELS=64 \
-x NCCL_GFX9_CHEAP_FENCE_OFF=0 \
-x NCCL_DEBUG=VERSION \
-x NCCL_DEBUG_FILE=nccl_debug.log \
/home/amir/amd-dev/dn/rccl-tests/build/all_gather_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

# DN TAG 
mpirun --np 16 --allow-run-as-root \
-H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/home/dn/amd-dev/dn/rccl/build/release:/home/dn/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/dn/amd-dev/dn/amd-anp/build/librccl-net.so:/home/dn/amd-dev/dn/rccl/build/release/librccl.so \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_MIN_NCHANNELS=64 \
-x NCCL_MAX_NCHANNELS=64 \
-x NCCL_DEBUG=VERSION \
-x NCCL_DEBUG_FILE=nccl_debug.log \
/home/dn/amd-dev/dn/rccl-tests/build/all_gather_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5
