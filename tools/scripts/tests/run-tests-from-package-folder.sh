# export MY_PATH=/home/dn/rccl-bins/drop_2025-08
# export MY_PATH=/home/dn/rccl-bins/develop
export MY_PATH=/home/dn/rccl-bins/2025-06-J13A-1

# 172.30.160.131:8 is under use of GPU mem

/opt/ompi-4.1.6/bin/mpirun --np 8 --allow-run-as-root \
-H 172.30.160.127:8 \
--bind-to numa \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x IONIC_PRIVATE_SERVICE_FORCE=1 \
-x IONIC_RCQ_NUM_PATHS=255 \
-x IONIC_RCQ_SIGN_BIT=15 \
-x NCCL_IB_TIMEOUT=5 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
--mca btl '^vader,openib' \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=VERSION \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x NCCL_GRAPH_DUMP_FILE=/tmp/dump_graph.txt \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=${MY_PATH}:/usr/local/lib: \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

/opt/ompi-4.1.6/bin/mpirun --np 64 --allow-run-as-root \
-H 172.30.160.127:8,172.30.160.126:8,172.30.160.165:8,172.30.160.201:8,172.30.160.111:8,172.30.160.119:8,172.30.160.204:8,172.30.160.193:8 \
--bind-to numa \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x IONIC_PRIVATE_SERVICE_FORCE=1 \
-x IONIC_RCQ_NUM_PATHS=255 \
-x IONIC_RCQ_SIGN_BIT=15 \
-x NCCL_IB_TIMEOUT=5 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
--mca btl '^vader,openib' \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=VERSION \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x NCCL_GRAPH_DUMP_FILE=/tmp/dump_graph.txt \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=${MY_PATH}:/usr/local/lib: \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

/opt/ompi-4.1.6/bin/mpirun --np 72 --allow-run-as-root \
-H 172.30.160.127:8,172.30.160.126:8,172.30.160.165:8,172.30.160.201:8,172.30.160.131:8,172.30.160.111:8,172.30.160.119:8,172.30.160.204:8,172.30.160.193:8 \
--bind-to numa \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x IONIC_RCQ_NUM_PATHS=255 \
-x IONIC_PRIVATE_SERVICE_FORCE=1 \
-x IONIC_RCQ_SIGN_BIT=15 \
-x NCCL_IB_TIMEOUT=5 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=1 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
--mca btl '^vader,openib' \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=VERSION \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x NCCL_GRAPH_DUMP_FILE=/tmp/dump_graph.txt \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=${MY_PATH}:/usr/local/lib: \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

mpirun --np 16 --allow-run-as-root \
-H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=${MY_PATH}:/usr/local/lib: \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
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
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5