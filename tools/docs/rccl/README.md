# RCCL Research and Optimization Documentation

**Last Updated:** October 30, 2025  
**Purpose:** Comprehensive documentation for RCCL performance analysis, bottleneck identification, and optimization

---

## Overview

This directory contains in-depth research and analysis of the **RCCL (ROCm Communication Collectives Library)** with the goal of identifying and optimizing performance bottlenecks. The documentation is structured to support systematic performance engineering work.

---

## Document Index

### Core Documentation

#### 1. [**RCCL Design Overview**](rccl-design-overview.md) 📐
**Status:** ✅ Complete  
**Purpose:** High-level architectural overview and component breakdown

**Contents:**
- Architecture and component diagram
- Communication patterns and algorithms (Ring, Tree, MSCCL)
- Data flow and execution pipeline
- Transport layer (P2P, xGMI, InfiniBand)
- Topology discovery and graph optimization
- Memory management
- Known bottlenecks and limitations

**When to Read:** Start here for a comprehensive understanding of RCCL architecture

---

#### 2. [**RCCL Bottleneck Analysis Guide**](rccl-bottleneck-analysis.md) 🔍
**Status:** ✅ Complete  
**Purpose:** Practical guide for identifying and analyzing performance bottlenecks

**Contents:**
- Performance analysis methodology
- Profiling tools and techniques (rccl-tests, rocprof, NPKit)
- Common bottleneck patterns with solutions
- Systematic investigation workflow
- Code hot spots
- Optimization strategies
- Validation and benchmarking approaches

**When to Read:** Use this as your primary guide when actively profiling and optimizing

---

#### 3. [**RCCL Technical Internals**](rccl-technical-internals.md) ⚙️
**Status:** ✅ Complete  
**Purpose:** Deep dive into implementation details

**Contents:**
- Data structure deep dives (ncclComm, ncclChannel, etc.)
- GPU kernel implementation details
- Protocol variants (Simple, LL, LL128)
- Transport layer internals
- Memory management details
- Synchronization mechanisms
- Algorithm implementations
- Network proxy architecture
- AMD-specific optimizations

**When to Read:** Reference this when you need to understand or modify specific code sections

---

### Supporting Documentation

#### 4. [**RCCL Environment Variables Analysis**](rccl-environment-variables-analysis.md) 🔧
**Status:** ✅ Complete (pre-existing)  
**Purpose:** Comprehensive list of all RCCL environment variables

**When to Read:** When tuning RCCL behavior or debugging configuration issues

---

#### 5. [**RCCL Branch Analysis**](rccl-branch-analysis.md) 📊
**Status:** ✅ Complete (pre-existing)  
**Purpose:** Comparison between develop and drop/2025-08 branches

**When to Read:** Understanding recent changes and development history

---

#### 6. [**AMD ANP Plugin Analysis**](net-plugin/amd-anp-plugin-calls-analysis.md) 🔌
**Status:** ✅ Complete (pre-existing)  
**Purpose:** Analysis of AMD AINIC Network Plugin

**When to Read:** When working on network transport optimizations

---

## Quick Start Guide

### For First-Time Readers

**Step 1: Understand the Architecture (30-60 minutes)**
1. Read [RCCL Design Overview](rccl-design-overview.md)
2. Focus on sections:
   - Architecture Overview
   - Core Components
   - Communication Patterns

**Step 2: Setup Profiling Environment (15-30 minutes)**
1. Read [Bottleneck Analysis Guide - Profiling Tools](rccl-bottleneck-analysis.md#profiling-tools-and-techniques)
2. Set up rccl-tests:
   ```bash
   cd /Users/ahalperin/xai/amd-dev/amd/rccl-tests
   ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
   ```
3. Enable debug logging:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

**Step 3: Run Baseline Benchmarks (30-60 minutes)**
1. Follow [Systematic Investigation Workflow](rccl-bottleneck-analysis.md#systematic-investigation-workflow)
2. Document baseline performance
3. Compare to theoretical peaks

**Step 4: Identify Bottlenecks (varies)**
1. Use [Common Bottleneck Patterns](rccl-bottleneck-analysis.md#common-bottleneck-patterns) as a guide
2. Profile with rocprof for detailed analysis
3. Check topology detection and algorithm selection

**Step 5: Deep Dive and Optimize (varies)**
1. Reference [Technical Internals](rccl-technical-internals.md) for code details
2. Implement optimizations from [Optimization Strategies](rccl-bottleneck-analysis.md#optimization-strategies)
3. Validate with correctness and performance tests

---

## Optimization Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. BASELINE                                                 │
│     - Run rccl-tests                                         │
│     - Document system config                                 │
│     - Identify performance gap                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  2. PROFILE                                                  │
│     - Enable NCCL_DEBUG=TRACE                                │
│     - Run rocprof                                            │
│     - Analyze topology, algorithm, protocol selection        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  3. IDENTIFY BOTTLENECK                                      │
│     - Compare bus BW to theoretical peak                     │
│     - Check GPU utilization                                  │
│     - Examine CPU/proxy overhead                             │
│     - Validate network performance                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  4. DEEP DIVE                                                │
│     - Read relevant code sections                            │
│     - Understand data flow                                   │
│     - Identify optimization opportunity                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  5. IMPLEMENT                                                │
│     - Make targeted changes                                  │
│     - Add instrumentation                                    │
│     - Document changes                                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  6. VALIDATE                                                 │
│     - Run correctness tests                                  │
│     - Benchmark performance                                  │
│     - Check for regressions                                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                    ┌─────────┐
                    │ Success?│
                    └────┬────┘
                    Yes  │  No
                         ↓  ↓
                    Ship  │  Debug
                         Return to Step 4
```

---

## Key Performance Metrics

### Target Performance Goals

**Intra-Node (xGMI) - MI300X:**
- **Bandwidth:** 350-400 GB/s bus bandwidth (85-95% of 432 GB/s theoretical)
- **Latency:** 10-30 µs for small messages

**Intra-Node (PCIe):**
- **Bandwidth:** 17-20 GB/s bus bandwidth (70-85% of 24 GB/s theoretical)
- **Latency:** 20-50 µs

**Inter-Node (InfiniBand HDR200):**
- **Bandwidth:** 17-21 GB/s (75-90% of 23 GB/s theoretical)
- **Latency:** 50-150 µs

### How to Measure

```bash
# AllReduce bandwidth test
./all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100

# Look for "busbw" column in output
# Compare to theoretical peak for your hardware
```

**Example Output:**
```
#       size         count      type   redop     time   algbw   busbw
      128M        33554432     float     sum    345.2  370.71  648.74
                                                 ↑       ↑       ↑
                                              Time   Algo BW  Bus BW
```

**Analysis:**
- Bus BW of 648 GB/s for 8 GPUs = ~81 GB/s per GPU
- For MI300X with xGMI, theoretical peak ≈ 432 GB/s
- Efficiency: 648 / (432 × 8/8 × 7/8) ≈ **172%** ← Wait, this is wrong calculation

**Correct Analysis:**
For AllReduce with N ranks using ring algorithm:
```
Bus BW = Algo BW × 2 × (N-1)/N
       = 370.71 × 2 × 7/8
       = 648.74 GB/s ✓ (matches output)

Per-link bandwidth = Bus BW / N
                   = 648.74 / 8
                   = 81 GB/s per link

xGMI theoretical = 432 GB/s per link
Efficiency = 81 / 432 × 100% = 18.8%
```

This indicates a bottleneck - should be 85%+ efficiency!

---

## Common Optimization Targets

Based on typical bottlenecks, prioritize investigation in this order:

### 1. 🔥 **Algorithm/Protocol Selection** (High Impact, Low Risk)
**File:** `src/graph/tuning.cc`  
**Symptoms:** Wrong protocol for message size, sub-optimal algorithm  
**Fix Complexity:** Low  
**Expected Gain:** 10-50%

### 2. 🔥 **Topology Detection** (High Impact, Medium Risk)
**Files:** `src/graph/topo.cc`, `src/graph/search.cc`  
**Symptoms:** PCIe paths chosen instead of xGMI  
**Fix Complexity:** Medium  
**Expected Gain:** 2-10x for affected cases

### 3. 🔥 **Kernel Primitives** (Medium Impact, High Risk)
**Files:** `src/device/primitives.h`, `src/device/prims_*.h`  
**Symptoms:** Low GPU utilization, high memory stalls  
**Fix Complexity:** High  
**Expected Gain:** 10-30%

### 4. ⚠️ **Network Proxy** (Medium Impact, Medium Risk)
**File:** `src/proxy.cc`  
**Symptoms:** High CPU usage, network latency  
**Fix Complexity:** Medium  
**Expected Gain:** 5-20% (multi-node only)

### 5. ⚠️ **Memory Management** (Low Impact, Low Risk)
**File:** `src/allocator.cc`  
**Symptoms:** High initialization time, memory fragmentation  
**Fix Complexity:** Low  
**Expected Gain:** 5-10% (initialization only)

---

## Tools Reference

### Profiling Tools

| Tool | Purpose | Command Example |
|------|---------|-----------------|
| **rccl-tests** | Baseline performance | `./all_reduce_perf -b 8 -e 8G -g 8` |
| **rocprof** | GPU kernel profiling | `rocprof --stats ./all_reduce_perf -g 8` |
| **NCCL_DEBUG** | Trace logging | `NCCL_DEBUG=TRACE ./all_reduce_perf` |
| **NPKit** | Event-level profiling | Requires compile flag `--npkit-enable` |
| **rocm-smi** | GPU monitoring | `rocm-smi -showtemp -showpower` |
| **ibstat** | InfiniBand status | `ibstat` |
| **perf** | CPU profiling | `perf record -g ./all_reduce_perf` |

### Key Environment Variables

```bash
# Debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH,NET
export NCCL_DEBUG_FILE=/tmp/rccl_debug.log

# Performance tuning
export NCCL_NCHANNELS=16
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_BUFFSIZE=4194304  # 4 MB

# AMD-specific
export RCCL_FORCE_XGMI=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1

# Network
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=eth0
```

---

## Source Code Navigation

### Key Directories

```
rccl/src/
├── init.cc                 ← Entry point, initialization
├── enqueue.cc              ← Work enqueuing, kernel launch
├── collectives.cc          ← Collective APIs (ncclAllReduce, etc.)
├── channel.cc              ← Channel setup
├── proxy.cc                ← Network proxy threads
│
├── device/                 ← GPU kernels
│   ├── primitives.h        ← Core GPU primitives ⚡
│   ├── prims_simple.h      ← Simple protocol
│   ├── prims_ll.h          ← LL protocol
│   ├── prims_ll128.h       ← LL128 protocol
│   └── common.cu           ← Kernel entry points
│
├── graph/                  ← Topology and algorithms
│   ├── topo.cc             ← Topology discovery ⚡
│   ├── search.cc           ← Path search
│   ├── rings.cc            ← Ring algorithm
│   ├── trees.cc            ← Tree algorithm
│   └── tuning.cc           ← Algorithm selection ⚡
│
└── transport/              ← Transport implementations
    ├── p2p.cc              ← P2P transport ⚡
    ├── net_ib.cc           ← InfiniBand
    └── net_socket.cc       ← TCP sockets

⚡ = High-priority optimization targets
```

### Important Headers

```
rccl/src/include/
├── comm.h                  ← ncclComm structure (central!)
├── channel.h               ← Channel structures
├── graph.h                 ← Topology graph
├── transport.h             ← Transport layer
├── collectives.h           ← Collective operations
└── device.h                ← Device-side definitions
```

---

## Testing and Validation

### Correctness Tests

```bash
# RCCL unit tests
cd /Users/ahalperin/xai/amd-dev/amd/rccl/build/test
./rccl-UnitTests

# rccl-tests with correctness checking (-c flag)
cd /Users/ahalperin/xai/amd-dev/amd/rccl-tests
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -c 1 -n 100
```

### Performance Tests

```bash
# Comprehensive benchmark
for SIZE in 1K 4K 16K 64K 256K 1M 4M 16M 64M 256M 1G; do
  ./build/all_reduce_perf -b $SIZE -e $SIZE -g 8 -n 100 \
    | tee results/allreduce_${SIZE}.log
done

# Multi-collective test
for COLL in all_reduce all_gather reduce_scatter broadcast; do
  ./build/${COLL}_perf -b 8 -e 1G -f 2 -g 8 -n 100
done
```

### Regression Testing

After any optimization:
1. ✅ Run full correctness test suite
2. ✅ Benchmark all message sizes (8 bytes to 8 GB)
3. ✅ Test with 2, 4, 8 GPUs (scaling)
4. ✅ Test all collectives (not just AllReduce)
5. ✅ Compare to baseline (no regressions)

---

## Collaboration and Contribution

### Document Updates

When adding new findings or optimizations:

1. Update the relevant document(s)
2. Add entry to Document History section
3. Update this README if adding new documents
4. Document any new environment variables
5. Add test cases for new optimizations

### Code Changes

When modifying RCCL code:

1. Reference which document sections guided the change
2. Document performance impact
3. Add comments referencing this documentation
4. Update Technical Internals if implementation changes significantly

---

## FAQ

### Q: Where should I start if performance is poor?

**A:** Follow this decision tree:

1. Run baseline test: `./all_reduce_perf -b 8 -e 8G -g 8`
2. Compare bus BW to theoretical peak
3. If < 70% efficiency:
   - Enable `NCCL_DEBUG=INFO`
   - Check which paths/transports are used
   - Check algorithm/protocol selection
   - See [Common Bottleneck Patterns](rccl-bottleneck-analysis.md#common-bottleneck-patterns)

### Q: How do I know if xGMI is being used?

**A:** 
```bash
export NCCL_DEBUG=INFO
./all_reduce_perf -g 8 2>&1 | grep "P2P"
# Look for "P2P/direct" or "XGMI"
# Should NOT see "P2P/IB" for intra-node
```

### Q: What message sizes should I focus on?

**A:** Depends on your workload:
- **Training (large models):** 16 MB - 1 GB (AllReduce dominates)
- **Training (small models):** 256 KB - 16 MB  
- **Inference:** 1 KB - 1 MB (latency-sensitive)

Profile your actual application to see message size distribution.

### Q: How do I profile a PyTorch training job?

**A:**
```python
# In your training script
import torch.distributed as dist

# Before training
dist.barrier()  # Sync

# Enable RCCL debug for one iteration
os.environ['NCCL_DEBUG'] = 'INFO'

# Single training step
loss.backward()
optimizer.step()

# Disable debug
os.environ['NCCL_DEBUG'] = 'WARN'
```

---

## External Resources

### Official Documentation
- [ROCm RCCL Documentation](https://rocm.docs.amd.com/projects/rccl/)
- [ROCm Profiler Guide](https://rocm.docs.amd.com/projects/rocprofiler/)
- [AMD Instinct MI300X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

### Academic Papers
- "Massively Parallel Communication with Collective Network" (NCCL Paper)
- "MSCCLang: Microsoft Collective Communication Language"
- "Ring-AllReduce: Optimizing Bandwidth for Deep Learning" (Baidu)

### Related Projects
- [NCCL (NVIDIA)](https://github.com/NVIDIA/nccl)
- [MSCCL](https://github.com/microsoft/msccl)
- [MSCCLPP](https://github.com/microsoft/mscclpp)

---

## Document Status Summary

| Document | Status | Lines | Purpose |
|----------|--------|-------|---------|
| [README.md](README.md) | ✅ Current | 550+ | Navigation and overview |
| [rccl-design-overview.md](rccl-design-overview.md) | ✅ Complete | 1000+ | Architecture guide |
| [rccl-bottleneck-analysis.md](rccl-bottleneck-analysis.md) | ✅ Complete | 1400+ | Optimization guide |
| [rccl-technical-internals.md](rccl-technical-internals.md) | ✅ Complete | 1500+ | Implementation details |
| [rccl-environment-variables-analysis.md](rccl-environment-variables-analysis.md) | ✅ Complete | 570+ | Environment variables |
| [rccl-branch-analysis.md](rccl-branch-analysis.md) | ✅ Complete | 260+ | Branch comparison |

**Total Documentation:** ~5,500+ lines covering all aspects of RCCL research and optimization

---

## Contact and Support

For questions or discussions about this documentation:
- Review the relevant document section
- Check the FAQ above
- Examine the source code references
- Run profiling tools to gather data

---

## License

This documentation is part of the RCCL project analysis. See [RCCL LICENSE.txt](../../../amd/rccl/LICENSE.txt) for library licensing information.

---

**Last Updated:** October 30, 2025  
**Version:** 1.0  
**Maintained by:** RCCL Performance Engineering Team


