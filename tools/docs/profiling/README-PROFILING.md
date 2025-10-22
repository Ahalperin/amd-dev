# RCCL Profiling with ROCm Tools - Container Setup

This directory contains a complete setup for profiling RCCL tests execution flow on both CPU and GPU using ROCm profiling tools inside the rccl-builder container.

## 📁 Container Structure

```
/ (container root)
├── workspace/
│   ├── rccl/                           # RCCL source and build
│   ├── rccl-tests/                     # RCCL tests source and build
│   │   └── build/                      # Test executables
│   └── profiling_results/              # Auto-created profiling output directory
│       ├── *_quick_*.csv               # Quick profiling results
│       ├── *_basic_*.csv               # Basic profiling results
│       ├── *_advanced_*.csv            # Advanced profiling results
│       ├── *_perfetto_*.pftrace        # Perfetto trace files
│       └── *_multi_proc_rank_*.csv     # Multi-GPU per-rank results
└── tools/
    ├── docs/
    │   ├── rccl-profiling-guide.md     # Comprehensive profiling guide
    │   ├── rocm-performance-counters.md # Performance counters reference
    │   └── rccl-tests.md               # RCCL tests usage guide
    └── scripts/
        ├── quick_profile_rccl.sh       # Quick start profiling
        ├── profile_rccl_basic.sh       # Basic profiling scenarios
        ├── profile_rccl_advanced.sh    # Advanced profiling with rocprofv3
        ├── profile_rccl_multi_gpu.sh   # Multi-GPU profiling
        └── analyze_rccl_profile.py     # Results analysis tool
```

## 🚀 Quick Start (30 seconds)

1. **Run quick profiling** to get started immediately:
   ```bash
   /tools/scripts/quick_profile_rccl.sh all_reduce_perf
   ```

2. **View results**:
   ```bash
   python3 /tools/scripts/analyze_rccl_profile.py /workspace/profiling_results/
   ```

## 🛠 Available ROCm Profiling Tools

Your container has these profiling tools available:

- **rocprof** - Classic profiler for basic GPU analysis
- **rocprofv2** - Enhanced profiler with better performance counters
- **rocprofv3** - Latest profiler with comprehensive metrics
- **rocprof-sys-*** - System-level profiling tools

## 📊 Profiling Scenarios

### 1. Basic Single-GPU Profiling
```bash
/tools/scripts/profile_rccl_basic.sh all_reduce_perf
```
- Profiles kernel execution, memory operations, and API calls
- Generates CSV reports with timing data
- Good for understanding execution flow

### 2. Advanced Profiling with Perfetto
```bash
/tools/scripts/profile_rccl_advanced.sh all_reduce_perf
```
- Uses rocprofv3 for detailed analysis
- Generates Perfetto traces viewable in Chrome
- Includes system-level profiling

### 3. Multi-GPU Profiling
```bash
/tools/scripts/profile_rccl_multi_gpu.sh all_reduce_perf 2
```
- Profiles distributed RCCL operations
- Supports both single-process and multi-process modes
- Analyzes inter-GPU communication patterns

## 📈 Key Metrics to Monitor

### GPU Utilization
- **GRBM_GUI_ACTIVE**: Overall GPU busy percentage
- **SQ_WAVES**: Number of wavefronts executed
- **SQ_BUSY_CYCLES**: GPU compute unit utilization

### Memory Bandwidth
- **TCC_EA_RDREQ**: External memory read requests
- **TCC_EA_WRREQ**: External memory write requests
- **TCC_HIT/TCC_MISS**: L2 cache hit/miss rates

### RCCL-Specific Patterns
- **TCP_TCC_READ_REQ**: GPU-to-GPU data transfers
- **SQ_INSTS_VALU_MFMA**: Matrix operations (reductions)
- **TCP_TOTAL_ATOMIC**: Synchronization operations

## 🔍 Analysis Workflow

1. **Start with Quick Profiling**: Get immediate insights
2. **Identify Bottlenecks**: Look for high wait times or low utilization
3. **Deep Dive**: Use advanced profiling for detailed analysis
4. **Multi-GPU Analysis**: Profile distributed operations
5. **Optimize**: Apply insights to improve RCCL performance

## 📋 Example Commands

### Profile All-Reduce Operation
```bash
# Quick profile
/tools/scripts/quick_profile_rccl.sh all_reduce_perf

# Detailed analysis
/tools/scripts/profile_rccl_advanced.sh all_reduce_perf

# Multi-GPU (4 GPUs)
/tools/scripts/profile_rccl_multi_gpu.sh all_reduce_perf 4
```

### Profile All-Gather Operation
```bash
/tools/scripts/profile_rccl_basic.sh all_gather_perf -b 1M -e 1G -f 2
```

### Analyze Results
```bash
# Analyze specific file
python3 /tools/scripts/analyze_rccl_profile.py results.csv

# Analyze all results in directory
python3 /tools/scripts/analyze_rccl_profile.py /workspace/profiling_results/
```

## 🎯 Container-Specific Notes

### File Locations
- **RCCL Tests**: `/workspace/rccl-tests/build/`
- **Profiling Results**: `/workspace/profiling_results/`
- **Scripts**: `/tools/scripts/`
- **Documentation**: `/tools/docs/`

### Environment Variables
The scripts automatically set these for optimal profiling:
- `NCCL_DEBUG=INFO`: Enable RCCL debugging
- `HSA_NO_SCRATCH_RECLAIM=1`: Better performance on MI300 series
- `OMPI_ALLOW_RUN_AS_ROOT=1`: Allow MPI as root

### Available RCCL Tests
- `all_reduce_perf` - All-reduce operations
- `all_gather_perf` - All-gather operations
- `broadcast_perf` - Broadcast operations
- `reduce_scatter_perf` - Reduce-scatter operations
- `sendrecv_perf` - Point-to-point operations

## 🎯 Troubleshooting

### Common Issues
1. **"Test executable not found"**: Build RCCL tests first
   ```bash
   cd /workspace/rccl-tests
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j$(nproc)
   ```

2. **"No GPUs detected"**: Check GPU status
   ```bash
   rocm-smi
   ```

3. **Permission errors with MPI**: Scripts automatically use `--allow-run-as-root`

## 📚 Documentation

- **[Complete Profiling Guide](tools/docs/rccl-profiling-guide.md)**: Detailed profiling instructions
- **[Graphical Visualization Guide](tools/docs/rccl-visualization-guide.md)**: How to view results graphically
- **[ROCm Profiler Libraries Guide](tools/docs/rocm-profiler-libraries-guide.md)**: Advanced profiling with shared libraries
- **[Performance Counters](tools/docs/rocm-performance-counters.md)**: Counter reference and interpretation
- **[RCCL Tests Guide](tools/docs/rccl-tests.md)**: RCCL tests usage and options

## 🎉 Ready to Profile!

You now have a complete profiling setup for RCCL tests within the container environment. Start with the quick profiling script and work your way up to advanced analysis as needed.

For questions or issues, refer to the detailed documentation in the `/tools/docs/` directory.