export MY_PATH=/home/amir/rccl-bins/drop_2025-08
export MY_PATH=/home/amir/rccl-bins/develop
export MY_PATH=/home/amir/rccl-bins/2025-06-J13A-1
export MY_PATH=/home/dn/rccl-bins/show_alltoall_algo_proto_and_channels
export MY_PATH=/home/dn/rccl-bins/rocm-7.1.1_show_alltoall_algo_proto_and_channels
export MY_PATH=/home/dn/rccl-bins/rocm-7.2.0_show_alltoall_algo_proto_and_channels
export MY_PATH=/home/dn/rccl-bins/develop_dn_tuner

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
-x NCCL_IB_QPS_PER_CONNECTION=2 \
-x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

/opt/ompi-4.1.6/bin/mpirun --np 72 --allow-run-as-root \
-H 172.30.160.127:8,172.30.160.126:8,172.30.160.165:8,172.30.160.201:8,172.30.160.131:8,172.30.160.111:8,172.30.160.119:8,172.30.160.204:8,172.30.160.193:8 \
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
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_GFX9_CHEAP_FENCE_OFF=0 \
-x NCCL_DEBUG=VERSION \
-x NCCL_DEBUG_SUBSYS=TUNING \
-x NCCL_DEBUG_FILE=nccl_debug.log \
-x NCCL_IB_QPS_PER_CONNECTION=2 \
-x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5 -M 1

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
-x NCCL_MIN_NCHANNELS=64 \
-x NCCL_MAX_NCHANNELS=64 \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5

mpirun --np 16 --allow-run-as-root \
-H 172.30.160.127:8,172.30.160.126:8 \
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
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_GFX9_CHEAP_FENCE_OFF=0 \
-x NCCL_DEBUG=VERION \
-x NCCL_DEBUG_SUBSYS=TUNING \
-x NCCL_DEBUG_FILE=nccl_debug.log \
-x NCCL_IB_QPS_PER_CONNECTION=2 \
-x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
-x NCCL_LOCAL_REGISTER=1 \
-x NCCL_GRAPH_REGISTER=1 \
-x NCCL_LEGACY_CUDA_REGISTER=1 \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 5 -c 1 -w 5 -R 1 -M 1

-x NCCL_MIN_NCHANNELS=1 \
-x NCCL_MAX_NCHANNELS=64 \
-x NCCL_ALGO=Tree \
-x NCCL_PROTO=SIMPLE \
-x NCCL_MIN_NCHANNELS=64 \
-x NCCL_MAX_NCHANNELS=64 \


mpirun --np 8 --allow-run-as-root \
-H 172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
--mca btl '^vader,openib' \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/usr/local/lib:${MY_PATH}/:/opt/rocm/bin \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
-x IONIC_RCQ_NUM_PATHS=255 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x IONIC_PRIVATE_SERVICE_FORCE=1 \
-x IONIC_RCQ_SIGN_BIT=15 \
-x IONIC_LOCKFREE=all \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x NCCL_IB_TIMEOUT=5 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
-x NCCL_IB_QPS_PER_CONNECTION=2 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_DEBUG=VERSION \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x NCCL_GRAPH_DUMP_FILE=/tmp/dump_graph.txt \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5 -M 1


#
# latest setup for XAI cluster 6/1/2026
#
/opt/ompi-4.1.6/bin/mpirun --np 72 --allow-run-as-root \
-H 172.30.160.127:8,172.30.160.126:8,172.30.160.165:8,172.30.160.201:8,172.30.160.131:8,172.30.160.111:8,172.30.160.119:8,172.30.160.193:8,172.30.160.204:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
--mca btl '^vader,openib' \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/usr/local/lib:${MY_PATH}/:/opt/rocm/bin \
-x LD_PRELOAD=${MY_PATH}/librccl-net.so:${MY_PATH}/librccl.so \
-x IONIC_RCQ_NUM_PATHS=255 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x IONIC_PRIVATE_SERVICE_FORCE=1 \
-x IONIC_RCQ_SIGN_BIT=15 \
-x IONIC_LOCKFREE=all \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x NCCL_IB_TIMEOUT=5 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
-x NCCL_IB_QPS_PER_CONNECTION=2 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_PXN_DISABLE=0 \
-x NCCL_DEBUG=TRACE \
-x NCCL_DEBUG_FILE=/tmp/nccl_debug.log \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x NCCL_GRAPH_DUMP_FILE=/tmp/dump_graph.txt \
-x NCCL_TUNER_PLUGIN=${MY_PATH}/librccl-tunerv4-dn.so \
-x NCCL_TUNER_CONFIG_FILE=${MY_PATH}/dn-tuner-conf/xai_gfx950_tuner_512M.conf \
${MY_PATH}/all_reduce_perf -b 1M -e 16G -f 2 -g 1 -n 20 -c 1 -w 5 -M 1

-x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
-x RCCL_LL128_FORCE_ENABLE=1 \
-x NCCL_MIN_NCHANNELS=64 \
-x NCCL_MAX_NCHANNELS=64 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_TUNER_PLUGIN=${MY_PATH}/librccl-tunerv4-dn.so
-x NCCL_TUNER_CONFIG_FILE=${MY_PATH}/dn-tuner-conf/xai_gfx950_tuner_512M.conf \