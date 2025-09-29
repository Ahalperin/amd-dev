# RCCL Tests Profiling Guide

This guide covers how to profile RCCL tests execution flow on both CPU and GPU using ROCm profiling tools.

## Available ROCm Profiling Tools

Your environment has several profiling tools available:

### 1. rocprof (Classic ROCProfiler)
- **Purpose**: GPU kernel profiling, memory operations, API tracing
- **Best for**: Basic GPU performance analysis, kernel timing
- **Command**: `rocprof`

### 2. rocprofv2 (ROCProfiler v2)
- **Purpose**: Enhanced profiling with better performance counter support
- **Best for**: Detailed performance counter analysis
- **Command**: `rocprofv2`

### 3. rocprofv3 (ROCProfiler v3 - Latest)
- **Purpose**: Most advanced profiling with comprehensive metrics
- **Best for**: Advanced performance analysis, latest GPU architectures
- **Command**: `rocprofv3`

### 4. rocprof-sys-* Tools (System-level Profiling)
- **rocprof-sys-run**: Run applications with system-level profiling
- **rocprof-sys-sample**: CPU sampling profiler
- **rocprof-sys-instrument**: Binary instrumentation
- **rocprof-sys-causal**: Causal profiling

## Basic GPU Profiling with rocprof

### Simple Kernel Profiling
```bash
# Profile all GPU kernels and HIP API calls
rocprof --hip-trace --hsa-trace ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Profile with specific metrics
rocprof --hip-trace --stats ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Detailed Performance Counter Profiling
```bash
# Profile with performance counters (example metrics)
rocprof --hip-trace \
        -m SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE \
        ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Profile memory bandwidth and cache metrics
rocprof --hip-trace \
        -m TCC_HIT,TCC_MISS,TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum \
        ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Advanced Profiling with rocprofv3

### Comprehensive Analysis
```bash
# Use rocprofv3 for latest GPU architectures (gfx942, gfx90a, etc.)
rocprofv3 --plugin perfetto \
          --kernel-trace \
          --hip-trace \
          --hsa-trace \
          -- ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Generate detailed reports
rocprofv3 --plugin csv \
          --output-file rccl_profile.csv \
          --kernel-trace \
          -- ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## System-Level CPU/GPU Profiling

### Using rocprof-sys-run for Comprehensive Profiling
```bash
# Profile both CPU and GPU with system-level insights
rocprof-sys-run --trace --sample-freq=1000 \
                --output=rccl_sys_profile \
                -- ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# CPU sampling with GPU correlation
rocprof-sys-sample --freq=1000 --duration=30 \
                   --output=rccl_cpu_sample \
                   -- ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Multi-GPU and Multi-Node Profiling

### Single Node Multi-GPU
```bash
# Profile multi-GPU RCCL operations
rocprof --hip-trace --hsa-trace \
        --output-file multi_gpu_profile \
        -- mpirun --allow-run-as-root -np 8 --bind-to numa \
           ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### MPI-based Multi-Node Profiling
```bash
# Each MPI rank can be profiled separately
mpirun --allow-run-as-root -np 2 --bind-to numa \
       sh -c 'rocprof --hip-trace --output-file profile_rank_${OMPI_COMM_WORLD_RANK} \
              ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1'
```

## Profiling Different RCCL Operations

### All-Reduce Profiling
```bash
rocprof --hip-trace --stats \
        --output-file allreduce_profile \
        ./rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### All-Gather Profiling
```bash
rocprof --hip-trace --stats \
        --output-file allgather_profile \
        ./rccl-tests/build/all_gather_perf -b 8 -e 128M -f 2 -g 1
```

### Broadcast Profiling
```bash
rocprof --hip-trace --stats \
        --output-file broadcast_profile \
        ./rccl-tests/build/broadcast_perf -b 8 -e 128M -f 2 -g 1
```

## Output Analysis

### Generated Files
- **results.csv**: Kernel timing and performance counters
- **results.json**: Detailed JSON format results
- **results.db**: SQLite database for advanced queries
- **trace.json**: Chrome tracing format (viewable in chrome://tracing)

### Key Metrics to Analyze
1. **Kernel Duration**: Time spent in GPU kernels
2. **Memory Bandwidth**: Data transfer rates
3. **GPU Utilization**: How efficiently GPU resources are used
4. **API Overhead**: Time spent in HIP/HSA API calls
5. **Memory Operations**: Copy operations, allocations

## Environment Variables for Enhanced Profiling

```bash
# Enable additional RCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# ROCm profiling environment
export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so
export ROCP_TOOL_LIB=/opt/rocm/lib/librocprofiler64.so

# For better performance on MI300 series
export HSA_NO_SCRATCH_RECLAIM=1
```

## Performance Counter Reference

### Common GPU Metrics
- **SQ_WAVES**: Number of wavefronts
- **SQ_INSTS_VALU**: Vector ALU instructions
- **GRBM_GUI_ACTIVE**: GPU busy percentage
- **TCC_HIT/TCC_MISS**: L2 cache hit/miss rates
- **TCP_TCC_READ/WRITE_REQ**: Memory read/write requests

### Memory Metrics
- **TCP_TOTAL_CACHE_ACCESSES**: Total cache accesses
- **TCC_EA_RDREQ**: External memory read requests
- **TCC_EA_WRREQ**: External memory write requests

## Troubleshooting

### Common Issues
1. **Permission Errors**: Use `--allow-run-as-root` for MPI
2. **Missing Libraries**: Ensure `LD_LIBRARY_PATH` includes ROCm libraries
3. **GPU Access**: Check GPU visibility with `rocm-smi`

### Debug Commands
```bash
# Check GPU status
rocm-smi

# Verify RCCL installation
ldd ./rccl-tests/build/all_reduce_perf

# Test basic functionality
./rccl-tests/build/all_reduce_perf --help
```

## Quick Start

For immediate profiling results, use the quick start script:
```bash
/root/amd-dev/scripts/quick_profile_rccl.sh all_reduce_perf
```

## Available Scripts

### 1. Quick Profiling
- **Script**: `/root/amd-dev/scripts/quick_profile_rccl.sh`
- **Purpose**: Fast profiling with essential metrics
- **Usage**: `./quick_profile_rccl.sh [test_name]`

### 2. Basic Profiling
- **Script**: `/root/amd-dev/scripts/profile_rccl_basic.sh`
- **Purpose**: Comprehensive basic profiling with multiple metric sets
- **Usage**: `./profile_rccl_basic.sh [test_name] [additional_args]`

### 3. Advanced Profiling
- **Script**: `/root/amd-dev/scripts/profile_rccl_advanced.sh`
- **Purpose**: Advanced profiling with rocprofv3 and Perfetto traces
- **Usage**: `./profile_rccl_advanced.sh [test_name] [additional_args]`

### 4. Multi-GPU Profiling
- **Script**: `/root/amd-dev/scripts/profile_rccl_multi_gpu.sh`
- **Purpose**: Profile multi-GPU RCCL operations
- **Usage**: `./profile_rccl_multi_gpu.sh [test_name] [num_gpus] [additional_args]`

### 5. Analysis Script
- **Script**: `/root/amd-dev/scripts/analyze_rccl_profile.py`
- **Purpose**: Analyze profiling results from CSV/JSON files
- **Usage**: `python3 ./analyze_rccl_profile.py [file_or_directory]`

## Additional Documentation

- **Performance Counters Reference**: `/root/amd-dev/docs/rocm-performance-counters.md`
- **RCCL Tests Usage**: `/root/amd-dev/docs/rccl-tests.md`

## Next Steps

1. Start with the quick profiling script to get immediate results
2. Use advanced profiling for detailed performance analysis on modern GPUs
3. Employ multi-GPU profiling for distributed RCCL operations
4. Analyze results using the provided analysis tools to identify bottlenecks and optimization opportunities
