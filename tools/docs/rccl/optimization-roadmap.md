# RCCL Optimization Roadmap

**Date:** October 30, 2025  
**Purpose:** Structured approach to systematically optimize RCCL performance  
**Related:** [Bottleneck Analysis](rccl-bottleneck-analysis.md), [Technical Internals](rccl-technical-internals.md)

---

## Overview

This roadmap provides a structured approach to identifying and implementing RCCL optimizations. Each phase builds on the previous one, starting with low-risk, high-impact optimizations and progressing to more complex kernel-level improvements.

---

## Phase 1: Profiling and Baseline (Week 1)

**Goal:** Establish baseline performance and identify low-hanging fruit

### Tasks

#### 1.1 System Characterization (Days 1-2)
- [ ] Document hardware configuration
  - GPU model, count, topology
  - xGMI connectivity
  - Network configuration (IB, NICs)
  - CPU/NUMA layout
- [ ] Verify system health
  - GPU clocks and power states
  - Thermal throttling check
  - Network link status
- [ ] Document software versions
  - ROCm version
  - RCCL version and branch
  - Kernel version
  - Network driver versions

**Commands:**
```bash
# GPU info
rocm-smi --showproductname --showbus
rocm-smi --showtopo

# Network info
ibstat
ibv_devinfo

# Software versions
cat /opt/rocm/.info/version
git -C rccl/ log -1 --oneline
uname -r
```

**Deliverable:** `system_config.txt` with complete system profile

#### 1.2 Baseline Performance (Days 2-3)
- [ ] Run comprehensive rccl-tests
  - All collectives (AllReduce, AllGather, etc.)
  - Full message size sweep (8B to 8GB)
  - Multiple GPU counts (2, 4, 8)
- [ ] Measure theoretical peaks
- [ ] Calculate efficiency gaps
- [ ] Identify performance cliffs

**Commands:**
```bash
#!/bin/bash
# baseline_benchmark.sh

COLLECTIVES="all_reduce all_gather reduce_scatter broadcast sendrecv"
GPU_COUNTS="2 4 8"

for COLL in $COLLECTIVES; do
  for NGPU in $GPU_COUNTS; do
    echo "Testing $COLL with $NGPU GPUs"
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1))) \
      ./build/${COLL}_perf -b 8 -e 8G -f 2 -g $NGPU -n 100 \
      | tee baseline_${COLL}_${NGPU}gpu.log
  done
done
```

**Deliverable:** Baseline performance report with efficiency analysis

#### 1.3 Initial Profiling (Days 3-5)
- [ ] Enable NCCL_DEBUG logging
- [ ] Analyze topology detection
- [ ] Verify algorithm/protocol selection
- [ ] Profile with rocprof for hot spots
- [ ] Identify top 3 bottlenecks

**Commands:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH,NET
export NCCL_DEBUG_FILE=/tmp/rccl_profile.log

# Run test
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100

# Analyze
grep -i "channel\|algo\|proto\|xgmi" /tmp/rccl_profile.log

# GPU profiling
rocprof --stats --timestamp on \
  ./build/all_reduce_perf -b 128M -e 128M -g 8 -n 100
```

**Deliverable:** Profiling report identifying top bottlenecks

---

## Phase 2: Environment Tuning (Week 2)

**Goal:** Optimize using environment variables (no code changes)

### Tasks

#### 2.1 Channel Count Optimization (Days 1-2)
Test different channel counts to find optimal parallelism.

- [ ] Test NCCL_NCHANNELS from 4 to 32
- [ ] Measure bandwidth vs. channel count
- [ ] Identify optimal configuration per message size

**Script:**
```bash
#!/bin/bash
for NCHANNELS in 4 8 12 16 20 24 28 32; do
  export NCCL_NCHANNELS=$NCHANNELS
  echo "Testing with $NCHANNELS channels"
  ./build/all_reduce_perf -b 128M -e 128M -g 8 -n 100 \
    | tee channels_${NCHANNELS}.log
done

# Analyze results
python3 analyze_channels.py channels_*.log
```

**Expected Gain:** 5-20% bandwidth improvement

#### 2.2 Protocol Threshold Tuning (Days 2-3)
Optimize protocol selection thresholds.

- [ ] Test different NCCL_PROTO settings
- [ ] Identify protocol transition points
- [ ] Compare to current thresholds

**Script:**
```bash
# Test each protocol
for PROTO in LL LL128 Simple; do
  export NCCL_PROTO=$PROTO
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100 \
    | tee proto_${PROTO}.log
done

# Find crossover points
python3 find_protocol_thresholds.py proto_*.log
```

**Expected Gain:** 10-30% for specific message sizes

#### 2.3 Buffer Size Tuning (Days 3-4)
Optimize buffer sizes for target workload.

- [ ] Test NCCL_BUFFSIZE from 1MB to 16MB
- [ ] Measure impact on large message bandwidth
- [ ] Check memory consumption

**Script:**
```bash
for BUFFSIZE in 1048576 2097152 4194304 8388608 16777216; do
  export NCCL_BUFFSIZE=$BUFFSIZE
  echo "Testing buffer size: $BUFFSIZE bytes"
  ./build/all_reduce_perf -b 64M -e 1G -f 2 -g 8 -n 100 \
    | tee buffsize_${BUFFSIZE}.log
done
```

**Expected Gain:** 5-15% for large messages

#### 2.4 System Configuration (Days 4-5)
Optimize system-level settings.

- [ ] CPU governor to performance mode
- [ ] Disable NUMA balancing
- [ ] GPU performance profiles
- [ ] Network tuning (MTU, buffer sizes)

**Script:**
```bash
# CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable NUMA balancing
sudo sysctl kernel.numa_balancing=0

# GPU performance
sudo rocm-smi --setperflevel high

# Network tuning (if applicable)
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.wmem_max=268435456
```

**Expected Gain:** 5-10% consistency improvement

**Deliverable:** Optimized environment configuration file

---

## Phase 3: Code Analysis and Planning (Week 3)

**Goal:** Deep dive into code to plan targeted optimizations

### Tasks

#### 3.1 Algorithm Selection Analysis (Days 1-2)
Review algorithm selection logic in `src/graph/tuning.cc`.

- [ ] Trace algorithm selection for different scenarios
- [ ] Identify sub-optimal decisions
- [ ] Design improved selection logic
- [ ] Plan MI300X-specific tuning

**Files to Review:**
- `src/graph/tuning.cc:~line 150-300`
- `src/graph/search.cc:~line 100-400`
- `src/graph/rings.cc`
- `src/graph/trees.cc`

**Analysis Questions:**
1. Are thresholds optimal for MI300X?
2. Does selection consider xGMI topology?
3. Can we use performance history?
4. Are there ML opportunities?

#### 3.2 Topology Detection Analysis (Days 2-3)
Review topology discovery in `src/graph/topo.cc`.

- [ ] Verify xGMI detection accuracy
- [ ] Check path bandwidth calculations
- [ ] Validate ring construction
- [ ] Identify optimization opportunities

**Files to Review:**
- `src/graph/topo.cc:~line 800-1200`
- `src/graph/xml.cc`
- `src/graph/paths.cc`

**Known Issues to Investigate:**
1. Are all xGMI links discovered?
2. Are link bandwidths accurate?
3. Is path search optimal?
4. Can we cache more aggressively?

#### 3.3 Kernel Primitive Analysis (Days 3-4)
Review GPU kernel implementations.

- [ ] Profile primitive operations
- [ ] Check memory access patterns
- [ ] Analyze synchronization overhead
- [ ] Identify vectorization opportunities

**Files to Review:**
- `src/device/primitives.h:~line 300-800`
- `src/device/prims_simple.h`
- `src/device/prims_ll.h`
- `src/device/prims_ll128.h`
- `src/device/common.cu`

**Performance Metrics:**
- Memory bandwidth utilization (target: 95%+)
- Wavefront occupancy
- Register usage
- Atomic operation overhead

#### 3.4 Transport Layer Analysis (Days 4-5)
Review P2P and network transport.

- [ ] Analyze P2P setup overhead
- [ ] Check peer access patterns
- [ ] Review proxy thread efficiency
- [ ] Identify memory copy overhead

**Files to Review:**
- `src/transport/p2p.cc`
- `src/transport/net_ib.cc`
- `src/proxy.cc:~line 500-800`

**Deliverable:** Detailed code analysis report with optimization opportunities

---

## Phase 4: Low-Risk Optimizations (Weeks 4-5)

**Goal:** Implement safe, high-impact optimizations

### Priority 1: Algorithm Selection Improvements

#### 4.1 Hardware-Specific Tuning Tables
**Risk:** Low  
**Impact:** Medium-High (10-30%)  
**Effort:** 2-3 days

**Implementation:**
```cpp
// src/graph/tuning.cc
struct ncclTuningTable {
  int arch;                    // GFX942, GFX90A, etc.
  int nRanks;
  struct {
    size_t threshold;
    ncclAlgo_t algo;
    ncclProto_t proto;
  } entries[16];
};

static struct ncclTuningTable mi300xTuning = {
  .arch = GFX942,
  .nRanks = -1,  // any
  .entries = {
    {4096,    NCCL_ALGO_TREE, NCCL_PROTO_LL},
    {65536,   NCCL_ALGO_RING, NCCL_PROTO_LL128},
    {524288,  NCCL_ALGO_RING, NCCL_PROTO_SIMPLE},
    {0, 0, 0}  // sentinel
  }
};

ncclResult_t selectAlgorithmFromTable(struct ncclInfo* info) {
  // Lookup based on GPU arch and message size
}
```

**Testing:**
- Validate all collectives
- Test all message sizes
- Compare to baseline
- No correctness issues

#### 4.2 Topology-Aware Ring Construction
**Risk:** Low-Medium  
**Impact:** High (20-50% for poorly configured systems)  
**Effort:** 3-4 days

**Implementation:**
```cpp
// src/graph/rings.cc
ncclResult_t ncclTopoComputeRingsXGMI(struct ncclTopoSystem* system) {
  // 1. Identify all xGMI links
  struct xgmiLink {
    int gpu1, gpu2;
    float bandwidth;
  };
  struct xgmiLink xgmiLinks[MAX_LINKS];
  int nXgmiLinks = discoverXGMILinks(system, xgmiLinks);
  
  // 2. Build rings that maximize xGMI usage
  for (int c = 0; c < nChannels; c++) {
    // Prefer paths using xGMI over PCIe
    constructRingWithPreferredLinks(system, xgmiLinks, nXgmiLinks, &rings[c]);
  }
  
  // 3. Validate bandwidth
  verifyRingBandwidth(system, rings, nChannels);
  
  return ncclSuccess;
}
```

**Testing:**
- Verify xGMI links used
- Check NCCL_DEBUG output
- Measure bandwidth improvement
- Test on multiple topologies

#### 4.3 Protocol Threshold Auto-Tuning
**Risk:** Low  
**Impact:** Medium (10-20%)  
**Effort:** 2-3 days

**Implementation:**
```cpp
// src/graph/tuning.cc
ncclResult_t autoTuneThresholds(struct ncclComm* comm) {
  // Detect GPU architecture
  if (comm->arch == GFX942) {  // MI300X
    // Run micro-benchmarks
    size_t llThreshold = measureLLThreshold(comm);
    size_t ll128Threshold = measureLL128Threshold(comm);
    
    // Update thresholds
    comm->llThreshold = llThreshold;
    comm->ll128Threshold = ll128Threshold;
    
    INFO(NCCL_TUNING, "Auto-tuned thresholds: LL=%zu LL128=%zu",
         llThreshold, ll128Threshold);
  }
  
  return ncclSuccess;
}
```

**Testing:**
- Run during first collective
- Cache results
- Compare to fixed thresholds
- Verify no overhead

**Deliverable:** Patches with improvements, performance reports

---

## Phase 5: Medium-Risk Optimizations (Weeks 6-8)

**Goal:** Kernel and transport optimizations

### Priority 2: GPU Kernel Improvements

#### 5.1 Vectorized Memory Operations
**Risk:** Medium  
**Impact:** High (15-30%)  
**Effort:** 1 week

**Implementation:**
```cpp
// src/device/primitives.h
template<typename T>
__device__ void copyVectorized(T* dst, const T* src, int nelem) {
  // Use 128-bit loads/stores
  using VecType = typename std::conditional<
    sizeof(T) == 4, uint4,
    typename std::conditional<sizeof(T) == 2, uint4, uint4>::type
  >::type;
  
  VecType* dst4 = reinterpret_cast<VecType*>(dst);
  const VecType* src4 = reinterpret_cast<const VecType*>(src);
  int nelem4 = nelem * sizeof(T) / sizeof(VecType);
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = idx; i < nelem4; i += stride) {
    dst4[i] = src4[i];
  }
}
```

**Testing:**
- Verify alignment requirements
- Check all data types
- Measure memory bandwidth
- Run correctness tests

#### 5.2 Reduce Synchronization Overhead
**Risk:** Medium-High  
**Impact:** Medium (10-20% for small messages)  
**Effort:** 1 week

**Implementation:**
- Replace some `__syncthreads()` with warp-level sync
- Use lock-free synchronization where possible
- Optimize atomic operations

**Testing:**
- Extensive correctness validation
- Test on all GPU architectures
- Check edge cases (odd message sizes, etc.)

#### 5.3 Optimized Reduction Kernels
**Risk:** Medium  
**Impact:** Medium (10-25%)  
**Effort:** 1 week

**Implementation:**
- Use AMD-specific instructions (e.g., `__builtin_amdgcn_*`)
- Optimize for MI300X cache hierarchy
- Improve wavefront utilization

---

## Phase 6: High-Risk Optimizations (Weeks 9-12)

**Goal:** Advanced optimizations requiring significant changes

### Priority 3: Architectural Improvements

#### 6.1 Persistent Kernels
**Risk:** High  
**Impact:** High (20-40% for small messages)  
**Effort:** 2-3 weeks

**Benefits:**
- Eliminate kernel launch overhead
- Better for high-frequency collectives
- Improved latency

**Challenges:**
- Complex programming model
- Resource management
- Backward compatibility

#### 6.2 GPU-Driven Network Operations
**Risk:** High  
**Impact:** High (30-50% for multi-node)  
**Effort:** 3-4 weeks

**Benefits:**
- Reduce proxy thread overhead
- Lower latency
- Better overlap

**Challenges:**
- Requires hardware support
- Complex error handling
- Network plugin changes

#### 6.3 Machine Learning-Based Tuning
**Risk:** Medium  
**Impact:** Medium-High (15-30%)  
**Effort:** 4-6 weeks

**Approach:**
1. Collect performance data
2. Train model for algorithm selection
3. Integrate into RCCL
4. Validate across workloads

---

## Success Metrics

### Phase 1-2 (Weeks 1-2)
- [ ] Baseline established
- [ ] 10-20% improvement from env tuning
- [ ] Top bottlenecks identified

### Phase 3-4 (Weeks 3-5)
- [ ] 20-40% cumulative improvement
- [ ] No correctness regressions
- [ ] Documented code changes

### Phase 5 (Weeks 6-8)
- [ ] 40-60% cumulative improvement
- [ ] Kernel bandwidth > 90% of peak
- [ ] All tests passing

### Phase 6 (Weeks 9-12)
- [ ] 60-100% improvement (2x target)
- [ ] Production-ready code
- [ ] Upstreamed patches

---

## Risk Mitigation

### For Each Optimization:
1. **Backup:** Keep original code
2. **Test:** Run full test suite
3. **Benchmark:** Compare to baseline
4. **Review:** Code review before merge
5. **Validate:** Test on multiple systems

### Rollback Plan:
- Git branches for each optimization
- Easy revert if issues found
- Maintain compatibility

---

## Documentation Requirements

### For Each Phase:
- Performance report (before/after)
- Code changes with rationale
- Test results (correctness + performance)
- Known issues and limitations

### Final Deliverable:
- Complete optimization guide
- Performance characterization
- Tuning recommendations
- Upstream patches (if applicable)

---

## Timeline Summary

| Phase | Weeks | Focus | Expected Gain |
|-------|-------|-------|---------------|
| 1 | 1 | Profiling & Baseline | 0% (baseline) |
| 2 | 1 | Env Tuning | +10-20% |
| 3 | 1 | Code Analysis | 0% (planning) |
| 4 | 2 | Low-Risk Optimizations | +20-40% |
| 5 | 3 | Medium-Risk (Kernels) | +40-60% |
| 6 | 4 | High-Risk (Advanced) | +60-100% |
| **Total** | **12** | | **2x target** |

---

## Next Steps

1. **Review this roadmap** with team
2. **Adjust timeline** based on priorities
3. **Start Phase 1** (profiling)
4. **Weekly check-ins** to track progress
5. **Iterate** based on findings

---

**Remember:** Optimization is iterative. Be prepared to adjust the roadmap based on what you discover in each phase.


