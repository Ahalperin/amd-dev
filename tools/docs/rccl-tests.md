# RCCL Tests Usage Guide

RCCL tests executables are located under `/workspace/rccl-tests/build/` in the container.

## Container Structure

```
/ (container root)
├── workspace/
│   ├── rccl/              # RCCL source and build
│   └── rccl-tests/        # RCCL tests source and build
│       └── build/         # Test executables location
└── tools/
    ├── docs/              # Documentation
    └── scripts/           # Profiling scripts
```

## Example All-Gather Test Command and Options

```shell
/workspace/rccl-tests/build/all_gather_perf --help
USAGE: all_gather_perf 
        [-t,--nthreads <num threads>] 
        [-g,--ngpus <gpus per thread>] 
        [-b,--minbytes <min size in bytes>] 
        [-e,--maxbytes <max size in bytes>] 
        [-i,--stepbytes <increment size>] 
        [-f,--stepfactor <increment factor>] 
        [-n,--iters <iteration count>] 
        [-m,--agg_iters <aggregated iteration count>] 
        [-w,--warmup_iters <warmup iteration count>] 
        [-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] 
        [-p,--parallel_init <0/1>] 
        [-c,--check <check iteration count>] 
        [-o,--op <sum/prod/min/max/avg/mulsum/all>] 
        [-d,--datatype <nccltype/all>] 
        [-r,--root <root/all>] 
        [-z,--blocking <0/1>] 
        [-y,--stream_null <0/1>] 
        [-T,--timeout <time in seconds>] 
        [-G,--cudagraph <num graph launches>] 
        [-C,--report_cputime <0/1>] 
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
        [-R,--local_register <0/1/2> enable local (1) or symmetric (2) buffer registration on send/recv buffers (default: disable (0))] 
        [-Y,--memory_type <coarse/fine/host/managed>] 
        [-u,--cumask <d0,d1,d2,d3>] 
        [-O,--out_of_place <0/1>] 
        [-q,--delay <delay between out-of-place and in-place in microseconds>] 
        [-F,--cache_flush <number of iterations between instruction cache flush>] 
        [-E,--rotating_tensor <0/1>] 
        [-x,--output_file <output file name>] 
        [-Z,--output_format <output format <csv|json>] 
        [-h,--help]
```

mpirun -H 172.30.160.145:8 -np 8 --bind-to numa --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 \
bash -c " \
NCCL_MIN_NCHANNELS=64 \
NCCL_MAX_NCHANNELS=64 \
RCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_PCI_RELAXED_ORDERING=0 \
NCCL_IB_QPS_PER_CONNECTION=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
NCCL_GDRCOPY_ENABLE=0 \
NCCL_IB_TC=104 \
NCCL_IB_FIFO_TC=192 \
NCCL_IGNORE_CPU_AFFINITY=1 \
RCCL_LL128_FORCE_ENABLE=1 \
NCCL_PXN_DISABLE=0 \
NET_OPTIONAL_RECV_COMPLETION=1 \
NCCL_IB_USE_INLINE=1 \
NCCL_GDR_FLUSH_DISABLE=1 \
RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
NCCL_DEBUG_SUBSYS=GRAPH,ALL \
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE=rccl.debug.log \
NCCL_TOPO_DUMP_FILE=rccl.topo.log \
NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=enp81s0f1np1 \
UCX_NET_DEVICES=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1"

mpirun -H 172.30.160.145:8,172.30.160.150:8 -np 16 --bind-to numa --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 \
bash -c " \
NCCL_MIN_NCHANNELS=64 \
NCCL_MAX_NCHANNELS=64 \
RCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_PCI_RELAXED_ORDERING=0 \
NCCL_IB_QPS_PER_CONNECTION=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
NCCL_GDRCOPY_ENABLE=0 \
NCCL_IB_TC=104 \
NCCL_IB_FIFO_TC=192 \
NCCL_IGNORE_CPU_AFFINITY=1 \
RCCL_LL128_FORCE_ENABLE=1 \
NCCL_PXN_DISABLE=0 \
NET_OPTIONAL_RECV_COMPLETION=1 \
NCCL_IB_USE_INLINE=1 \
NCCL_GDR_FLUSH_DISABLE=1 \
RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
NCCL_DEBUG_SUBSYS=GRAPH,ALL \
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE=rccl.debug.log \
NCCL_TOPO_DUMP_FILE=rccl.topo.log \
NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=enp81s0f1np1 \
UCX_NET_DEVICES=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1"

mpirun -H 172.30.160.145:8,172.30.160.150:8 -np 16 --bind-to numa --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 \
bash -c " \
NCCL_MIN_NCHANNELS=64 \
NCCL_MAX_NCHANNELS=64 \
RCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_PCI_RELAXED_ORDERING=0 \
NCCL_IB_QPS_PER_CONNECTION=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
NCCL_GDRCOPY_ENABLE=0 \
NCCL_IB_TC=104 \
NCCL_IB_FIFO_TC=192 \
NCCL_IGNORE_CPU_AFFINITY=1 \
RCCL_LL128_FORCE_ENABLE=1 \
NCCL_PXN_DISABLE=0 \
NET_OPTIONAL_RECV_COMPLETION=1 \
NCCL_IB_USE_INLINE=1 \
NCCL_GDR_FLUSH_DISABLE=1 \
RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
NCCL_DEBUG_SUBSYS=GRAPH,ALL \
NCCL_DEBUG=TRACE \
NCCL_DEBUG_FILE=rccl.debug.log \
NCCL_TOPO_DUMP_FILE=rccl.topo.log \
NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=enp81s0f1np1 \
UCX_NET_DEVICES=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1"

mpirun -H 172.30.160.145:8,172.30.160.150:8 -np 16 --bind-to numa --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 \
bash -c " \
NCCL_MIN_NCHANNELS=64 \
NCCL_MAX_NCHANNELS=64 \
RCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_PCI_RELAXED_ORDERING=0 \
NCCL_IB_QPS_PER_CONNECTION=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
NCCL_GDRCOPY_ENABLE=0 \
NCCL_IB_TC=104 \
NCCL_IB_FIFO_TC=192 \
NCCL_IGNORE_CPU_AFFINITY=1 \
RCCL_LL128_FORCE_ENABLE=1 \
NCCL_PXN_DISABLE=0 \
NET_OPTIONAL_RECV_COMPLETION=1 \
NCCL_IB_USE_INLINE=1 \
NCCL_GDR_FLUSH_DISABLE=1 \
RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
NCCL_DEBUG_SUBSYS=GRAPH,ALL \
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE=rccl.debug.log \
NCCL_TOPO_DUMP_FILE=rccl.topo.log \
NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=enp81s0f1np1 \
UCX_NET_DEVICES=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
LD_LIBRARY_PATH=/usr/local/lib: \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1"

#
# latest Arik runs 28/10
#
mpirun --np 16 --allow-run-as-root -H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/dn/amd-dev/amd/amd-anp/build/librccl-net.so:/home/dn/amd-dev/amd/rccl/build/release/librccl.so \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=INFO \
-x NCCL_DEBUG_FILE=rccl.debug.log \
-x NCCL_TOPO_DUMP_FILE=rccl.topo.log \
-x NCCL_GRAPH_DUMP_FILE=rccl.graph.log \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_LL128_FORCE_ENABLE=1  \
-x NCCL_ALGO=RING \
-x NCCL_BUFFSIZE=1194304 \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 256M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5

#
# latest Arik runs 28/10
#
mpirun --np 16 --allow-run-as-root -H 172.30.160.145:8,172.30.160.150:8 --bind-to numa \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=VERSION \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_LL128_FORCE_ENABLE=1  \
-x NCCL_ALGO=RING \
-x LD_PRELOAD=/home/dn/amd-dev/amd/amd-anp/build/librccl-net.so:/home/dn/amd-dev/amd/rccl/build/release/librccl.so \
-x NCCL_BUFFSIZE=1194304 \
-x NCCL_PROXY_APPEND_BATCH_SIZE= \
-x NCCL_PROGRESS_APPENDOP_FREQ= \
/home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5

/usr/bin/mpirun --np 16 --allow-run-as-root -H 172.30.160.145:8,172.30.160.150:8 --bind-to numa -x NCCL_IB_GID_INDEX=1 -x NCCL_GDR_FLUSH_DISABLE=1 -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 -x NCCL_GDRCOPY_ENABLE=0 -x LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/usr/local/lib: -x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 -x NCCL_DMABUF_ENABLE=0 --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 -x NCCL_IB_QPS_PER_CONNECTION=1 -x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_IB_TC=104 -x NCCL_IB_FIFO_TC=192 -x NCCL_IGNORE_CPU_AFFINITY=1 -x NCCL_DEBUG=VERSION -x NET_OPTIONAL_RECV_COMPLETION=1 -x NCCL_IB_USE_INLINE=1 -x NCCL_SOCKET_IFNAME=enp81s0f1np1 -x IONIC_LOCKFREE=all -x NCCL_PXN_DISABLE=0 -x RCCL_LL128_FORCE_ENABLE=1 -x RCCL_DISABLE_RAIL_TREES=1 -x NCCL_WORK_FIFO_BYTES=17179869184  /home/dn/amd-dev/amd/rccl-tests/build/alltoall_perf -b 256M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5

# last best Arik's run 
mpirun --np 16 --allow-run-as-root -H 172.30.160.145:8,172.30.160.150:8 --bind-to numa -x NCCL_IB_GID_INDEX=1 -x NCCL_GDR_FLUSH_DISABLE=1 -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 -x NCCL_GDRCOPY_ENABLE=0 -x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin -x LD_LIBRARY_PATH=/home/dn/amd-dev/amd/rccl/build/release:/home/dn/amd-dev/amd/amd-anp/build:/usr/local/lib: -x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 -x NCCL_DMABUF_ENABLE=0 --mca oob_tcp_if_include enp81s0f1np1 --mca btl_tcp_if_include enp81s0f1np1 -x NCCL_IB_QPS_PER_CONNECTION=1 -x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_IB_TC=104 -x NCCL_IB_FIFO_TC=192 -x NCCL_IGNORE_CPU_AFFINITY=1 -x NCCL_DEBUG=VERSION -x NET_OPTIONAL_RECV_COMPLETION=1 -x NCCL_IB_USE_INLINE=1 -x NCCL_SOCKET_IFNAME=enp81s0f1np1 -x IONIC_LOCKFREE=all -x NCCL_PXN_DISABLE=0 -x RCCL_LL128_FORCE_ENABLE=1 -x LD_PRELOAD=/home/dn/amd-dev/amd/amd-anp/build/librccl-net.so:/home/dn/amd-dev/amd/rccl/build/release/librccl.so /home/dn/amd-dev/amd/rccl-tests/build/all_reduce_perf -b 256M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5



# perfetto profiling output
TEST_NAME="all_reduce_perf" \
RCCL_TESTS_DIR="/workspace/rccl-tests/" \
OUTPUT_DIR="/workspace/profiling_results" \
TIMESTAMP=$(date +%Y%m%d_%H%M%S) \
TEST_EXEC="${RCCL_TESTS_DIR}/build/${TEST_NAME}" \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=ALL \
HSA_NO_SCRATCH_RECLAIM=1 \
OUT_FORMAT="pftrace" \
BASE_OUTFILE_NAME="${OUTPUT_DIR}/${TEST_NAME}_${TIMESTAMP}" \
&& rocprofv3 -f ${OUT_FORMAT} --kernel-trace --hip-trace --hsa-trace --stats --rccl-trace \
          -o "${BASE_OUTFILE_NAME}" \
          -- "${TEST_EXEC}" -b 8 -e 128M -f 2 -g 1 \
          -x "${BASE_OUTFILE_NAME}.rccl_test" | tee "${BASE_OUTFILE_NAME}.rccl_test.log"

# csv profiling output
TEST_NAME="all_reduce_perf" \
RCCL_TESTS_DIR="/workspace/rccl-tests/" \
OUTPUT_DIR="/workspace/profiling_results" \
TIMESTAMP=$(date +%Y%m%d_%H%M%S) \
TEST_EXEC="${RCCL_TESTS_DIR}/build/${TEST_NAME}" \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=ALL \
HSA_NO_SCRATCH_RECLAIM=1 \
OUT_FORMAT="csv" \
BASE_OUTFILE_NAME="${OUTPUT_DIR}/${TEST_NAME}_${TIMESTAMP}" \
&& rocprofv3 -f ${OUT_FORMAT} --kernel-trace --hip-trace --hsa-trace --stats --rccl-trace \
          -o "${BASE_OUTFILE_NAME}" \
          -- "${TEST_EXEC}" -b 8 -e 128M -f 2 -g 1 \
          -x "${BASE_OUTFILE_NAME}.rccl_test" | tee "${BASE_OUTFILE_NAME}.rccl_test.log"

```

## Output and Results

- Test results and profiling data are saved to `/workspace/profiling_results/`
- Use the analysis script to examine results: `/tools/scripts/analyze_rccl_profile.py`
- Use chrome://tracing/ to examin /workspace/profiling_results/


## Useful env
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=nccl_debug.log
export NCCL_TOPO_DUMP_FILE=topo.xml
export NCCL_GRAPH_DUMP_FILE=graph.xml
export NCCL_DEBUG_SUBSYS=GRAPH,COLL
env | grep NCCL

#
# DN
#
mpirun --np 16 --allow-run-as-root -H 172.30.160.145:8,172.30.160.150:8 --bind-to numa \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x LD_LIBRARY_PATH=/home/dn/amd-dev/dn/rccl/build/release:/home/dn/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt \
-x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NCCL_DEBUG=VERSION \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_LL128_FORCE_ENABLE=1  \
-x NCCL_ALGO=RING \
-x LD_PRELOAD=/home/dn/amd-dev/dn/amd-anp/build/librccl-net.so:/home/dn/amd-dev/dn/rccl/build/release/librccl.so \
-x NCCL_BUFFSIZE=1194304 \
/home/dn/amd-dev/dn/rccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5

mpirun --np 16 --allow-run-as-root \
-H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_DMABUF_ENABLE=0 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_LL128_FORCE_ENABLE=1 \
-x LD_LIBRARY_PATH=/home/dn/amd-dev/dn/rccl/build/release:/home/dn/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/dn/amd-dev/dn/amd-anp/build/librccl-net.so:/home/dn/amd-dev/dn/rccl/build/release/librccl.so \
/home/dn/amd-dev/dn/rccl-tests/build/all_reduce_perf -b 256M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5

-x NCCL_DEBUG=INFO \
-x NCCL_DEBUG_FILE=nccl_debug.log \
-x NCCL_TOPO_DUMP_FILE=nccl_topo.xml \
-x NCCL_GRAPH_DUMP_FILE=nccl_graph.xml \
-x NCCL_DEBUG_SUBSYS=GRAPH,COLL \

mpirun --np 16 --allow-run-as-root \
-H 172.30.160.145:8,172.30.160.150:8 \
--bind-to numa \
--mca oob_tcp_if_include enp81s0f1np1 \
--mca btl_tcp_if_include enp81s0f1np1 \
-x NCCL_IB_GID_INDEX=1 \
-x NCCL_GDR_FLUSH_DISABLE=1 \
-x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
-x NCCL_GDRCOPY_ENABLE=0 \
-x PATH=/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
-x NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1 \
-x NCCL_DMABUF_ENABLE=0 \
-x NCCL_IB_QPS_PER_CONNECTION=1 \
-x HSA_NO_SCRATCH_RECLAIM=1 \
-x NCCL_IB_TC=104 \
-x NCCL_IB_FIFO_TC=192 \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
-x NET_OPTIONAL_RECV_COMPLETION=1 \
-x NCCL_IB_USE_INLINE=1 \
-x NCCL_SOCKET_IFNAME=enp81s0f1np1 \
-x IONIC_LOCKFREE=all \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_LL128_FORCE_ENABLE=1 \
-x LD_LIBRARY_PATH=/home/amir/amd-dev/dn/rccl/build/release:/home/amir/amd-dev/dn/amd-anp/build:/usr/local/lib: \
-x LD_PRELOAD=/home/amir/amd-dev/dn/amd-anp/build/librccl-anp.so:/home/amir/amd-dev/dn/rccl/build/release/librccl.so \
/home/amir/amd-dev/dn/rccl-tests/build/all_reduce_perf -b 256M -e 256M -f 2 -g 1 -n 20 -c 1 -w 5
