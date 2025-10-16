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

## Running Tests in Container Context

### To allow running in context of root use --allow-run-as-root

### Execution for example on a single node single GPU

```shell
# /workspace/rccl-tests/build/all_gather_perf --allow-run-as-root -t 1 -g 1 -b 4 -e 8G -f 2 -N 8
/workspace/rccl-tests/build/all_gather_perf -t 1 -g 1 -b 4 -e 8G -f 2 -N 8
NCCL_DEBUG=INFO /workspace/rccl-tests/build/all_gather_perf -t 1 -g 8 -b 4 -e 8G -f 2 -N 1 | tee ah_all_gather_perf_8gpus_rccl.log
```

### Execution for example on 1 node single GPU via mpirun

```shell
mpirun --allow-run-as-root -np 1 --bind-to numa /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Execution for example on 2 nodes single GPU each

```shell
mpirun --allow-run-as-root -np 2 --bind-to numa /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Available Test Executables

Common RCCL test executables available in `/workspace/rccl-tests/build/`:

- **all_reduce_perf**: All-reduce collective operation
- **all_gather_perf**: All-gather collective operation  
- **reduce_scatter_perf**: Reduce-scatter collective operation
- **broadcast_perf**: Broadcast collective operation
- **reduce_perf**: Reduce collective operation
- **alltoall_perf**: All-to-all collective operation
- **sendrecv_perf**: Send/receive point-to-point operations

## Integration with Profiling Scripts

The profiling scripts in `/tools/scripts/` automatically use the correct container paths:

```bash
# Quick profiling
/tools/scripts/quick_profile_rccl.sh all_reduce_perf

# Basic profiling with custom arguments
/tools/scripts/profile_rccl_basic.sh all_gather_perf -b 1M -e 1G -f 2

# Advanced profiling
/tools/scripts/profile_rccl_advanced.sh broadcast_perf

# Multi-GPU profiling
/tools/scripts/profile_rccl_multi_gpu.sh all_reduce_perf 4
```

## Profiling rccl-tests using rocprofv3 directly without helper script

```bash

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