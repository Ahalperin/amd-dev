# ROCm Performance Counters Reference

This document provides a comprehensive reference for performance counters available in ROCm profiling tools for RCCL profiling inside the container environment.

## Container Structure

```
/ (container root)
├── workspace/
│   ├── rccl/              # RCCL source and build
│   └── rccl-tests/        # RCCL tests source and build
└── tools/
    ├── docs/              # Documentation
    └── scripts/           # Profiling scripts
```

## Key Performance Counter Categories

### 1. Shader Processor (SQ) Counters
These counters provide insight into GPU compute unit utilization and instruction execution.

#### Wave and Thread Metrics
- **SQ_WAVES**: Total number of wavefronts processed
- **SQ_WAVES_EQ_64**: Wavefronts with exactly 64 active threads
- **SQ_WAVES_LT_64**: Wavefronts with <64 active threads
- **SQ_WAVE_CYCLES**: Wave-cycles spent by waves in CUs
- **SQ_BUSY_CYCLES**: Clock cycles while SQ is busy
- **SQ_BUSY_CU_CYCLES**: Count quad-cycles each CU is busy

#### Instruction Execution
- **SQ_INSTS**: Total number of instructions issued
- **SQ_INSTS_VALU**: Vector ALU instructions
- **SQ_INSTS_VMEM_RD**: Vector memory read instructions
- **SQ_INSTS_VMEM_WR**: Vector memory write instructions
- **SQ_INSTS_SALU**: Scalar ALU instructions
- **SQ_INSTS_SMEM**: Scalar memory instructions
- **SQ_INSTS_LDS**: Local Data Share instructions

#### Matrix Operations (MFMA)
- **SQ_INSTS_VALU_MFMA_F16**: Matrix FMA operations (FP16)
- **SQ_INSTS_VALU_MFMA_BF16**: Matrix FMA operations (BF16)
- **SQ_INSTS_VALU_MFMA_F32**: Matrix FMA operations (FP32)
- **SQ_VALU_MFMA_BUSY_CYCLES**: Cycles MFMA ALU is busy

#### Wait States and Stalls
- **SQ_WAIT_ANY**: Wave-cycles waiting for anything
- **SQ_WAIT_INST_ANY**: Wave-cycles waiting for instruction issue
- **SQ_WAIT_INST_LDS**: Wave-cycles waiting for LDS instruction

### 2. Texture Cache Processor (TCP) Counters
These counters monitor L1 cache performance and memory access patterns.

#### Cache Performance
- **TCP_TOTAL_CACHE_ACCESSES**: Total cache line accesses (hits + misses)
- **TCP_TOTAL_READ**: Total read operations from TA
- **TCP_TOTAL_WRITE**: Total write operations from TA
- **TCP_TOTAL_ATOMIC_WITH_RET**: Atomic operations with return value
- **TCP_TOTAL_ATOMIC_WITHOUT_RET**: Atomic operations without return

#### L2 Communication
- **TCP_TCC_READ_REQ**: Read requests to L2 cache (TCC)
- **TCP_TCC_WRITE_REQ**: Write requests to L2 cache
- **TCP_TCC_ATOMIC_WITH_RET_REQ**: Atomic requests with return to L2
- **TCP_TCC_ATOMIC_WITHOUT_RET_REQ**: Atomic requests without return to L2

#### Memory Types
- **TCP_TCC_NC_READ_REQ**: Non-coherent read requests
- **TCP_TCC_UC_READ_REQ**: Uncached read requests
- **TCP_TCC_CC_READ_REQ**: Coherently cached read requests

### 3. Texture Cache Coherent (TCC) - L2 Cache Counters
Monitor L2 cache behavior and memory bandwidth.

#### Basic Cache Metrics
- **TCC_HIT**: L2 cache hits
- **TCC_MISS**: L2 cache misses
- **TCC_REQ**: Total requests to L2
- **TCC_READ**: Read requests to L2
- **TCC_WRITE**: Write requests to L2
- **TCC_ATOMIC**: Atomic requests to L2

#### Memory Bandwidth
- **TCC_EA_RDREQ**: External memory read requests
- **TCC_EA_WRREQ**: External memory write requests
- **TCC_EA_RDREQ_32B**: 32-byte external read requests
- **TCC_EA_WRREQ_64B**: 64-byte external write requests

#### Cache Behavior
- **TCC_WRITEBACK**: Lines written back to main memory
- **TCC_NORMAL_EVICT**: Cache evictions due to capacity
- **TCC_PROBE_EVICT**: Cache evictions due to coherency probes

### 4. Graphics Resource Block Manager (GRBM) Counters
High-level GPU utilization metrics.

- **GRBM_GUI_ACTIVE**: GPU is active (overall utilization)
- **GRBM_CP_BUSY**: Command processor busy
- **GRBM_SPI_BUSY**: Shader pipe interpolators busy
- **GRBM_TC_BUSY**: Texture cache blocks busy

## Useful Counter Combinations for RCCL Profiling

### Memory Bandwidth Analysis
```bash
rocprof -m TCC_HIT,TCC_MISS,TCC_EA_RDREQ,TCC_EA_WRREQ,TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum \
        /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Compute Utilization
```bash
rocprof -m SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE,SQ_BUSY_CYCLES,SQ_WAVE_CYCLES \
        /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Matrix Operations (for AI workloads)
```bash
rocprof -m SQ_INSTS_VALU_MFMA_F16,SQ_INSTS_VALU_MFMA_BF16,SQ_INSTS_VALU_MFMA_F32,SQ_VALU_MFMA_BUSY_CYCLES \
        /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Memory Access Patterns
```bash
rocprof -m TCP_TOTAL_READ,TCP_TOTAL_WRITE,TCP_TOTAL_ATOMIC_WITH_RET,TCC_READ,TCC_WRITE,TCC_ATOMIC \
        /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Cache Efficiency
```bash
rocprof -m TCC_HIT,TCC_MISS,TCP_TOTAL_CACHE_ACCESSES,SQC_ICACHE_HITS,SQC_ICACHE_MISSES \
        /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Counter Interpretation Guidelines

### Memory Bandwidth Utilization
```
Effective Bandwidth = (TCC_EA_RDREQ * 32 + TCC_EA_WRREQ * 32) / execution_time_seconds
Hit Rate = TCC_HIT / (TCC_HIT + TCC_MISS)
```

### GPU Utilization
```
Wave Occupancy = SQ_LEVEL_WAVES / max_waves_per_CU
Instruction Throughput = SQ_INSTS / execution_time_seconds
VALU Utilization = SQ_INSTS_VALU / SQ_INSTS
```

### RCCL-Specific Metrics
- **High TCP_TCC_READ_REQ**: Indicates significant GPU-to-GPU data movement
- **High TCC_EA_WRREQ**: Shows memory writes (results being stored)
- **High SQ_INSTS_VALU_MFMA**: Matrix operations (reduction kernels)
- **GRBM_GUI_ACTIVE**: Overall GPU utilization during collective operations

## Performance Counter Limitations

1. **Nondeterministic Counters**: Some counters (marked as nondeterministic) may vary between runs
2. **Per-SIMD Counters**: Values are per-SIMD unit, multiply by number of SIMDs for total
3. **Windowed vs Non-Windowed**: Some counters support windowing, others don't
4. **Counter Conflicts**: Some performance blocks have limited counters, requiring multiple runs

## Example Analysis Workflow

1. **Start with High-Level Metrics**:
   ```bash
   rocprof -m GRBM_GUI_ACTIVE,TCC_HIT,TCC_MISS,SQ_WAVES \
           /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
   ```

2. **Drill Down to Memory**:
   ```bash
   rocprof -m TCC_EA_RDREQ,TCC_EA_WRREQ,TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum \
           /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
   ```

3. **Analyze Compute Patterns**:
   ```bash
   rocprof -m SQ_INSTS_VALU,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,SQ_BUSY_CYCLES \
           /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
   ```

4. **Check for Bottlenecks**:
   ```bash
   rocprof -m SQ_WAIT_ANY,TCP_PENDING_STALL_CYCLES,TCC_TAG_STALL \
           /workspace/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
   ```

## Container-Specific Usage

### Available Scripts for Counter Analysis
- **Quick profiling**: `/tools/scripts/quick_profile_rccl.sh`
- **Basic profiling**: `/tools/scripts/profile_rccl_basic.sh`
- **Advanced profiling**: `/tools/scripts/profile_rccl_advanced.sh`
- **Multi-GPU profiling**: `/tools/scripts/profile_rccl_multi_gpu.sh`

### Results Location
All profiling results are saved to `/workspace/profiling_results/` for easy access and analysis.

This systematic approach helps identify performance bottlenecks in RCCL collective operations within the container environment.
