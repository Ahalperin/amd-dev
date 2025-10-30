# RCCL Bottleneck Identification and Optimization Guide

**Date:** October 30, 2025  
**Purpose:** Practical guide for identifying and optimizing RCCL performance bottlenecks  
**Related:** [RCCL Design Overview](rccl-design-overview.md)

---

## Table of Contents
1. [Performance Analysis Methodology](#performance-analysis-methodology)
2. [Profiling Tools and Techniques](#profiling-tools-and-techniques)
3. [Common Bottleneck Patterns](#common-bottleneck-patterns)
4. [Systematic Investigation Workflow](#systematic-investigation-workflow)
5. [Code Hot Spots](#code-hot-spots)
6. [Optimization Strategies](#optimization-strategies)
7. [Validation and Benchmarking](#validation-and-benchmarking)

---

## Performance Analysis Methodology

### Top-Down Approach

```
1. Measure End-to-End Performance
   ↓
2. Identify Performance Gap (vs. Theoretical Peak)
   ↓
3. Breakdown by Component (GPU, Network, CPU)
   ↓
4. Drill Down into Specific Component
   ↓
5. Profile Hot Spots
   ↓
6. Root Cause Analysis
   ↓
7. Implement & Validate Fix
```

### Key Performance Metrics

#### 1. **Bandwidth (GB/s)**
```
Bus Bandwidth = (Message Size × 2 × (N-1)) / (N × Time)

For AllReduce with N ranks:
- Factor of 2: data sent and received
- (N-1)/N: efficiency factor (ring algorithm)
```

**Theoretical Peaks:**
- xGMI (MI300X): ~432 GB/s (48 GT/s × 9 lanes)
- xGMI (MI250X): ~288 GB/s (36 GT/s × 8 lanes)
- PCIe Gen4 x16: ~24 GB/s (bidirectional)
- InfiniBand HDR200: ~23 GB/s (200 Gb/s)
- InfiniBand NDR400: ~46 GB/s (400 Gb/s)

**Target Efficiency:**
- Intra-node (xGMI): 85-95% of peak
- Intra-node (PCIe): 70-85% of peak
- Inter-node (IB): 75-90% of peak

#### 2. **Latency (microseconds)**
```
Latency = Time for smallest message
```

**Typical Latencies:**
- Intra-node (xGMI): 10-30 us
- Intra-node (PCIe): 20-50 us
- Inter-node (IB): 50-150 us
- Kernel launch overhead: 5-20 us

#### 3. **Algorithm Bandwidth (Algo BW)**
```
Algo BW = Message Size / Time
```
This is the effective bandwidth from the user's perspective.

#### 4. **Bus Bandwidth (Bus BW)**
```
Bus BW = (Actual bytes transferred on interconnect) / Time
```
This accounts for the actual data movement required by the algorithm.

---

## Profiling Tools and Techniques

### 1. RCCL-Tests: Baseline Performance

**Location:** `amd-dev/amd/rccl-tests/`

#### Basic Usage
```bash
# AllReduce performance test
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# Options:
# -b: minimum size (bytes)
# -e: maximum size (bytes)
# -f: size factor (2 = double each iteration)
# -g: number of GPUs
# -c: check correctness
# -n: number of iterations per size
# -w: warmup iterations
```

#### Advanced Options
```bash
# Test specific data types
./build/all_reduce_perf -b 1M -e 1G -g 8 -d float,half,bfloat16

# Test on specific GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf -g 4

# Multi-node test (MPI)
mpirun -np 16 -npernode 8 ./build/all_reduce_perf -b 1M -e 1G -g 8

# Enable NCCL debug output
NCCL_DEBUG=INFO ./build/all_reduce_perf -g 8
```

#### Interpreting Results
```
# Sample output:
#                                                       out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
      1048576        262144     float     sum    145.2   7.22   12.64  3e-07    144.8   7.24   12.67  3e-07
      2097152        524288     float     sum    197.3  10.63   18.60  3e-07    196.9  10.65   18.63  3e-07
      4194304       1048576     float     sum    301.4  13.91   24.35  2e-07    300.8  13.94   24.40  2e-07

Analysis:
- algbw: Algorithm bandwidth (user perspective) = Message Size / Time
- busbw: Bus bandwidth (actual data transferred) = algbw × 2 × (N-1)/N
- Compare busbw to theoretical peak to calculate efficiency
```

### 2. NCCL Debug Output

#### Environment Variables
```bash
# Basic info
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Detailed tracing
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,P2P,NET

# Save to file
export NCCL_DEBUG_FILE=/tmp/rccl_debug_%h_%p.log

# Timestamps
export NCCL_DEBUG_TIMESTAMP_LEVELS=1
```

#### Key Information from Debug Output

**Initialization:**
```
NCCL INFO Bootstrap : Using eth0:192.168.1.100<0>
NCCL INFO NET/Plugin : Plugin load returned 0 : librccl-net.so.
NCCL INFO Channel 00/08 : 0 1 2 3 4 5 6 7
```

**Topology:**
```
NCCL INFO Channel 00 : 0[0] -> 1[1] [send] via NET/IB/0
NCCL INFO Channel 00 : 1[1] -> 2[2] [send] via P2P/XGMI
```

**Algorithm Selection:**
```
NCCL INFO AllReduce: opCount 0 sendbuff 0x7f... recvbuff 0x7f... count 1048576 datatype 0 op 0 root 0
NCCL INFO Pattern Ring : 8 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
NCCL INFO Using 8 channels, 1 nodes, protocol Simple
```

### 3. Collective Trace

**Enable:**
```bash
export RCCL_COLL_TRACE_ENABLE=1
export RCCL_COLL_TRACE_FILE=/tmp/rccl_coll_trace.txt
```

**Output:**
```
Rank 0 AllReduce count=1048576 datatype=float op=sum time=145.2us bandwidth=7.22GB/s
  - Kernel launch: 12.3us
  - Kernel execution: 128.5us
  - Synchronization: 4.4us
```

### 4. ROCm Profiler (rocprof)

#### Profile RCCL Test
```bash
# Profile kernel execution
rocprof --hip-trace --stats ./all_reduce_perf -b 1M -e 1M -g 8 -n 100

# Generate timeline
rocprof --hip-trace --sys-trace ./all_reduce_perf -b 1M -e 1M -g 8 -n 100

# Output: results.csv, results.json, results.db
```

#### Key Metrics to Look For
- Kernel duration
- Memory copy duration
- Kernel launch overhead
- GPU utilization
- Memory bandwidth utilization

### 5. NPKit Profiling

**Compile with NPKit:**
```bash
cd rccl
./install.sh --npkit-enable
```

**Run with NPKit:**
```bash
export NPKIT_DUMP_DIR=/tmp/npkit_dump
./all_reduce_perf -b 1M -e 1M -g 8 -n 10
```

**Analyze:**
```python
# NPKit provides event-level profiling
# Events: GPU send/recv, CPU events, network events
```

### 6. Perfetto Visualization

**Convert NPKit to Perfetto:**
```bash
python3 rccl/tools/scripts/npkit_trace_generator.py \
    --npkit_dump_dir=/tmp/npkit_dump \
    --npkit_event_header_path=rccl/src/include/npkit/npkit_event.h \
    --output_dir=/tmp/perfetto_trace

# Open in Chrome: chrome://tracing
```

---

## Common Bottleneck Patterns

### Pattern 1: Low Bandwidth for Large Messages

**Symptom:**
```
# Expected: 400+ GB/s bus bandwidth (xGMI)
# Actual: 200 GB/s bus bandwidth

  size       time    algbw   busbw
  128MB     1200us  106GB/s  186GB/s   ← Should be 400+ GB/s
```

**Potential Causes:**
1. **Sub-optimal topology routing**
   - Check: `NCCL_DEBUG=INFO` → Look for "Channel" lines
   - Validate: All channels use xGMI paths, not PCIe
   - Fix: Set `RCCL_FORCE_XGMI=1` or improve topology XML

2. **PCIe bottleneck**
   - Check: `NCCL_DEBUG=TRACE` → Look for "P2P/IB" instead of "P2P/XGMI"
   - Validate: Ensure GPUDirect and xGMI are enabled
   - Fix: Check hardware configuration, kernel parameters

3. **Memory bandwidth limitation**
   - Check: Profile with rocprof → Memory bandwidth utilization
   - Validate: Should be near 100% for large messages
   - Fix: Optimize memory access patterns in kernels

4. **Insufficient channels**
   - Check: `NCCL_DEBUG=INFO` → "Using X channels"
   - Validate: Should use multiple channels (typically 8-16)
   - Fix: Set `NCCL_NCHANNELS=16` (or higher)

### Pattern 2: High Latency for Small Messages

**Symptom:**
```
# Expected: 10-30 us
# Actual: 100+ us

  size       time    algbw
  1KB       150us   6.7MB/s   ← Should be 30-50us
```

**Potential Causes:**
1. **Kernel launch overhead**
   - Check: rocprof → kernel launch time
   - Validate: Should be <20us
   - Fix: Use persistent kernels, batch operations

2. **Wrong protocol selection**
   - Check: `NCCL_DEBUG=TRACE` → "Using protocol Simple"
   - Validate: Should use "LL" for small messages
   - Fix: Adjust `NCCL_PROTO` or protocol thresholds

3. **CPU overhead**
   - Check: Profile proxy threads with perf
   - Validate: CPU usage should be minimal for intra-node
   - Fix: Optimize proxy thread code, reduce CPU involvement

4. **Synchronization overhead**
   - Check: Enable collective trace
   - Validate: Kernel execution vs. synchronization time
   - Fix: Reduce synchronization points

### Pattern 3: Poor Scaling with Rank Count

**Symptom:**
```
# 4 GPUs: 300 GB/s bus BW
# 8 GPUs: 180 GB/s bus BW  ← Should scale linearly or near-linear
```

**Potential Causes:**
1. **Contention on shared resources**
   - Check: Topology graph → identify shared links
   - Validate: Look for bottleneck links
   - Fix: Better load balancing across channels

2. **Algorithm selection**
   - Check: `NCCL_DEBUG=INFO` → Algorithm used
   - Validate: Ring vs. Tree tradeoffs
   - Fix: Force specific algorithm with `NCCL_ALGO`

3. **Network saturation (multi-node)**
   - Check: Network bandwidth utilization
   - Validate: Network BW vs. aggregate GPU BW
   - Fix: Add more NICs, use NCCL_NET_SHARED_BUFFERS

### Pattern 4: Inconsistent Performance

**Symptom:**
```
# Run 1: 400 GB/s
# Run 2: 200 GB/s
# Run 3: 380 GB/s
```

**Potential Causes:**
1. **CPU frequency scaling**
   - Check: `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
   - Fix: Set to "performance"
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **GPU power/clock throttling**
   - Check: `rocm-smi` → GPU clocks, power state
   - Fix: Set power profile to high performance
   ```bash
   sudo rocm-smi --setperflevel high
   ```

3. **NUMA balancing**
   - Check: `cat /proc/sys/kernel/numa_balancing`
   - Fix: Disable
   ```bash
   sudo sysctl kernel.numa_balancing=0
   ```

4. **Network interference**
   - Check: Other network traffic on the same NICs
   - Fix: Dedicate NICs for RCCL, use separate VLAN

### Pattern 5: Multi-Node Slower than Expected

**Symptom:**
```
# Single node (8 GPUs): 400 GB/s
# Two nodes (16 GPUs): 15 GB/s  ← Should be ~23 GB/s (IB limit)
```

**Potential Causes:**
1. **Network transport not optimal**
   - Check: `NCCL_DEBUG=INFO` → "NET/IB" vs "NET/Socket"
   - Validate: Should use InfiniBand, not sockets
   - Fix: Install RCCL net plugin, set `NCCL_NET="IB"`

2. **GPUDirect not working**
   - Check: `NCCL_DEBUG=TRACE` → Look for "GDR" or "GPU Direct"
   - Validate: Memory should be registered for RDMA
   - Fix: Ensure nv_peer_mem or amd_peer_mem kernel module loaded

3. **Proxy thread bottleneck**
   - Check: CPU profiling → Proxy thread utilization
   - Fix: Reduce proxy thread load, optimize network path

4. **Network configuration**
   - Check: `ibstat`, `ibv_devinfo`
   - Validate: Links are ACTIVE, correct speed
   - Fix: Check physical connections, subnet manager

---

## Systematic Investigation Workflow

### Step 1: Establish Baseline

```bash
# 1. Run standard rccl-tests
cd /Users/ahalperin/xai/amd-dev/amd/rccl-tests
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100 | tee baseline_allreduce.log

# 2. Run for each collective
for TEST in all_reduce all_gather reduce_scatter broadcast; do
    ./build/${TEST}_perf -b 8 -e 8G -f 2 -g 8 -n 100 | tee baseline_${TEST}.log
done

# 3. Save system info
rocm-smi > system_info.txt
ibstat >> system_info.txt
lspci | grep -i amd >> system_info.txt
```

### Step 2: Enable Detailed Logging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH,NET
export NCCL_DEBUG_FILE=/tmp/rccl_debug.log

./build/all_reduce_perf -b 1M -e 1M -g 8 -n 10
```

**Analyze log for:**
- Topology detected: Are xGMI links found?
- Algorithm selected: Ring or Tree?
- Protocol used: Simple, LL, or LL128?
- Channel count: How many parallel channels?
- Transport paths: P2P/XGMI, P2P/IB, NET/IB, etc.

### Step 3: Profile Hot Paths

```bash
# Profile GPU kernels
rocprof --stats --timestamp on \
    ./build/all_reduce_perf -b 1M -e 1M -g 8 -n 100

# Analyze results
cat results.stats.csv

# Look for:
# - Kernel execution time
# - Memory copy time
# - Kernel launch gaps (overhead)
```

### Step 4: Identify Bottleneck Layer

**Decision Tree:**
```
Is Bus BW < 70% of theoretical peak?
├─ YES → Likely bottleneck found
│  ├─ Check NCCL_DEBUG: Which transport used?
│  │  ├─ P2P/XGMI → GPU or memory bottleneck
│  │  ├─ P2P/IB → PCIe bottleneck
│  │  └─ NET/IB → Network bottleneck
│  │
│  └─ Profile with rocprof
│     ├─ Low GPU utilization → CPU overhead
│     ├─ High memory stalls → Memory bandwidth
│     └─ High kernel time → GPU kernel optimization
│
└─ NO → Performance is reasonable
   └─ Compare to other systems/baselines
```

### Step 5: Deep Dive into Code

Based on bottleneck layer, examine:

**GPU Kernel Bottleneck:**
- `src/device/primitives.h` → Core send/recv primitives
- `src/device/prims_simple.h` → Simple protocol implementation
- `src/device/reduce_kernel.h` → Reduction operations

**Transport Bottleneck:**
- `src/transport/p2p.cc` → P2P transport setup
- `src/transport/net_ib.cc` → InfiniBand transport

**Algorithm Bottleneck:**
- `src/graph/tuning.cc` → Algorithm selection logic
- `src/graph/search.cc` → Path search algorithm

**Proxy Bottleneck:**
- `src/proxy.cc` → Proxy thread implementation

---

## Code Hot Spots

### Critical Path: AllReduce

**User API → Kernel Launch:**
```
1. ncclAllReduce()                       [src/collectives.cc:~line 300]
   ├─ Input validation
   └─ Forward to ncclEnqueueCheck()

2. ncclEnqueueCheck()                    [src/enqueue.cc:~line 800]
   ├─ Validate inputs
   ├─ Select algorithm + protocol        [src/graph/tuning.cc:~line 200]
   ├─ Calculate chunk sizes
   └─ Forward to ncclEnqueueEvents()

3. ncclEnqueueEvents()                   [src/enqueue.cc:~line 500]
   ├─ Setup work elements
   ├─ Setup kernel arguments
   └─ Launch kernel (ncclLaunchKernel)

4. ncclLaunchKernel()                    [src/enqueue.cc:~line 200]
   └─ hipLaunchKernel()
```

**Potential Optimization Points:**
- **Line enqueue.cc:~800**: Reduce validation overhead for repeated calls
- **Line tuning.cc:~200**: Cache algorithm decisions
- **Line enqueue.cc:~500**: Reduce work setup overhead

### GPU Kernel Execution

**Kernel Entry:**
```cpp
// src/device/common.cu:~line 500
__global__ void ncclKernel(struct ncclDevWorkColl work) {
  // 1. Setup (line ~550)
  ncclPrimitives prims(work);
  
  // 2. Main loop (line ~600)
  for (int chunk = 0; chunk < nChunks; chunk++) {
    prims.send(src, chunkSize);      // Send to next rank
    prims.recv(dst, chunkSize);      // Receive from prev rank
    prims.reduce(dst, src, chunkSize); // Apply reduction
  }
  
  // 3. Finalize (line ~700)
  prims.wait();
}
```

**Primitive Implementation:**
```cpp
// src/device/primitives.h:~line 300
template<...>
class ncclPrimitives {
  __device__ void send(T* src, int nelem) {
    // Copy to send buffer
    // Signal peer via atomic
    // Wait for acknowledgment
  }
  
  __device__ void recv(T* dst, int nelem) {
    // Wait for peer signal
    // Copy from recv buffer
    // Send acknowledgment
  }
};
```

**Optimization Opportunities:**
1. **Reduce synchronization overhead** (atomic operations)
2. **Optimize memory coalescing** (access patterns)
3. **Better wavefront utilization** (work distribution)
4. **Reduce register pressure** (kernel complexity)

### Transport Setup (P2P)

```cpp
// src/transport/p2p.cc:~line 200
ncclResult_t p2pSetup(struct ncclComm* comm, ...) {
  // 1. Check if P2P is possible
  int p2p = 0;
  hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2);
  
  // 2. Enable peer access
  if (p2p) hipDeviceEnablePeerAccess(cudaDev2, 0);
  
  // 3. Map remote memory
  send->transportComm->buffer = remoteGpuMem;
  
  // 4. Setup signaling
  // ...
}
```

**Optimization Opportunities:**
1. **Caching peer accessibility checks**
2. **Batch peer access enabling**
3. **Reduce memory mapping overhead**

### Algorithm Selection

```cpp
// src/graph/tuning.cc:~line 150
static ncclResult_t selectAlgorithm(struct ncclInfo* info, ...) {
  // 1. Get thresholds
  ssize_t threshold = ncclParamAutoThreshold();
  
  // 2. Select based on size
  if (nBytes < threshold) {
    *algorithm = NCCL_ALGO_TREE;
    *protocol = NCCL_PROTO_LL;
  } else {
    *algorithm = NCCL_ALGO_RING;
    *protocol = NCCL_PROTO_SIMPLE;
  }
  
  // 3. Override if specified
  if (ncclParamAlgo() != -1) *algorithm = ncclParamAlgo();
}
```

**Optimization Opportunities:**
1. **Machine learning-based selection**
2. **Hardware-specific tuning tables**
3. **Runtime performance feedback**

---

## Optimization Strategies

### Strategy 1: Topology-Aware Optimization

**Problem:** Sub-optimal channel-to-GPU mapping

**Solution:**
```cpp
// Modify src/graph/rings.cc or src/graph/search.cc

// Current: Heuristic-based ring construction
// Improved: Explicit xGMI link prioritization

ncclResult_t ncclTopoComputeRingsXGMI(struct ncclTopoSystem* system, ...) {
  // 1. Identify all xGMI links
  // 2. Build rings that maximize xGMI usage
  // 3. Avoid PCIe links when possible
  // 4. Balance load across multiple xGMI links
}
```

**Validation:**
- Check `NCCL_DEBUG=INFO` → All channels should use xGMI
- Measure bus bandwidth → Should be near 400+ GB/s for MI300X

### Strategy 2: Protocol Threshold Tuning

**Problem:** Wrong protocol selected for given message size

**Solution:**
```cpp
// Modify src/graph/tuning.cc

// Current thresholds (approximate):
// LL:     0 - 8 KB
// LL128:  8 KB - 512 KB
// Simple: 512 KB+

// Tune for specific hardware:
static ncclResult_t tuneThresholds(struct ncclComm* comm) {
  // Detect GPU architecture
  if (isGfx942()) {  // MI300X
    comm->llThreshold = 4096;      // Prefer Simple earlier
    comm->ll128Threshold = 262144; // Adjust LL128 range
  } else if (isGfx90a()) {  // MI250X
    comm->llThreshold = 8192;
    comm->ll128Threshold = 524288;
  }
}
```

**Validation:**
- Run rccl-tests with various sizes
- Check protocol used in `NCCL_DEBUG=TRACE`
- Measure latency/bandwidth curves

### Strategy 3: Kernel Optimization

**Problem:** GPU kernel inefficiency

**Example: Uncoalesced Memory Access**
```cpp
// Before (src/device/primitives.h):
__device__ void copyData(T* dst, T* src, int nelem) {
  for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
    dst[i] = src[i];  // May not be coalesced
  }
}

// After:
__device__ void copyData(T* dst, T* src, int nelem) {
  // Ensure 128-byte alignment and coalescing
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  // Vectorized load/store
  T4* dst4 = (T4*)dst;
  T4* src4 = (T4*)src;
  int nelem4 = nelem / 4;
  
  for (int i = idx; i < nelem4; i += stride) {
    dst4[i] = src4[i];  // 4x elements per instruction
  }
}
```

**Validation:**
- Profile with rocprof → Check memory bandwidth utilization
- Should approach ~95% of theoretical peak memory bandwidth

### Strategy 4: Reduce CPU Overhead

**Problem:** High proxy thread CPU usage

**Solution:**
```cpp
// src/proxy.cc:~line 500

// Before: Busy-wait polling
while (!done) {
  checkNetwork();  // Constantly polling
}

// After: Event-driven with adaptive polling
int pollCount = 0;
while (!done) {
  if (checkNetwork()) {
    pollCount = 0;  // Reset on activity
  } else {
    pollCount++;
    if (pollCount > POLL_THRESHOLD) {
      usleep(1);  // Back off when idle
      pollCount = 0;
    }
  }
}
```

**Validation:**
- Profile CPU usage with `perf` or `top`
- Proxy thread CPU should drop significantly when idle

### Strategy 5: Persistent Kernels

**Problem:** Kernel launch overhead for small messages

**Solution:**
```cpp
// Implement persistent kernel approach

__global__ void ncclPersistentKernel(struct ncclWorkQueue* queue) {
  while (true) {
    // Wait for work
    struct ncclWork work = dequeueWork(queue);
    if (work.type == EXIT) break;
    
    // Execute work
    executeCollective(work);
    
    // Signal completion
    signalCompletion(work);
  }
}

// Launch once, feed work continuously
ncclLaunchPersistentKernel(comm);
```

**Benefits:**
- Eliminate per-operation kernel launch overhead
- Reduce latency for small messages
- Better for high-frequency collectives

**Challenges:**
- More complex programming model
- Resource management (kernel stays resident)

---

## Validation and Benchmarking

### Correctness Validation

**After any optimization, ALWAYS validate correctness:**

```bash
# Run correctness tests
cd rccl-tests
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -c 1

# Run all collective tests with checking
for TEST in all_reduce all_gather reduce_scatter broadcast sendrecv; do
    echo "Testing $TEST..."
    ./build/${TEST}_perf -b 8 -e 1G -f 2 -g 8 -c 1 -n 10
    if [ $? -ne 0 ]; then
        echo "FAILED: $TEST"
        exit 1
    fi
done

# Run RCCL unit tests
cd rccl/build/test
./rccl-UnitTests
```

### Performance Regression Testing

**Create benchmark suite:**

```bash
#!/bin/bash
# benchmark_suite.sh

SIZES="1K 4K 16K 64K 256K 1M 4M 16M 64M 256M 1G"
COLLECTIVES="all_reduce all_gather reduce_scatter"
GPUS="2 4 8"

for GPU in $GPUS; do
  for COLL in $COLLECTIVES; do
    for SIZE in $SIZES; do
      echo "Benchmarking $COLL with $GPU GPUs, size $SIZE"
      ./build/${COLL}_perf -b $SIZE -e $SIZE -g $GPU -n 100 \
        | tee results/${COLL}_${GPU}gpu_${SIZE}.log
    done
  done
done

# Generate comparison report
python3 analyze_results.py results/
```

### Continuous Benchmarking

**Track performance over time:**

```python
# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt

def parse_rccl_test_output(filename):
    # Parse rccl-tests output
    # Return: message_sizes, bandwidths, latencies
    pass

def compare_baseline(current, baseline):
    # Compare current results to baseline
    # Flag any regressions > 5%
    pass

def plot_performance_curve(results):
    # Generate bandwidth vs. message size plot
    plt.figure(figsize=(12, 6))
    plt.semilogx(sizes, bandwidths)
    plt.xlabel('Message Size (bytes)')
    plt.ylabel('Bus Bandwidth (GB/s)')
    plt.title('RCCL AllReduce Performance')
    plt.savefig('performance_curve.png')
```

### A/B Testing

**Compare optimization impact:**

```bash
# Baseline (before optimization)
./all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 1000 > baseline.log

# Apply optimization (recompile RCCL)
cd rccl
# ... make changes ...
./install.sh
sudo dpkg -i build/*.deb

# Test optimized version
cd rccl-tests
./all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 1000 > optimized.log

# Compare
python3 compare_logs.py baseline.log optimized.log
```

---

## Practical Checklist

### Before Starting Optimization

- [ ] Establish baseline performance with rccl-tests
- [ ] Document system configuration (GPUs, interconnects, network)
- [ ] Identify specific workload characteristics (message sizes, collectives)
- [ ] Set performance goals (target bandwidth, latency)

### During Investigation

- [ ] Enable NCCL_DEBUG logging
- [ ] Run with various message sizes to identify patterns
- [ ] Profile with rocprof to identify hot spots
- [ ] Check topology detection (xGMI links found?)
- [ ] Validate algorithm and protocol selection

### During Optimization

- [ ] Make small, incremental changes
- [ ] Keep backup of original code
- [ ] Add comments explaining changes
- [ ] Test correctness after each change
- [ ] Measure performance impact

### After Optimization

- [ ] Run full correctness test suite
- [ ] Benchmark all message sizes
- [ ] Test on multiple GPU counts
- [ ] Test multi-node if applicable
- [ ] Document performance improvement
- [ ] Check for any regressions

---

## Next Steps

1. **Run Baseline Benchmarks**
   - Execute rccl-tests on your system
   - Compare to expected performance
   - Identify specific areas of concern

2. **Profile Target Workload**
   - If you have a specific application (PyTorch, etc.), profile it
   - Identify most common collective operations
   - Focus optimization efforts on hot paths

3. **Systematic Code Review**
   - Start with algorithm selection (tuning.cc)
   - Review topology detection (topo.cc)
   - Examine kernel primitives (primitives.h)

4. **Iterative Optimization**
   - Pick one bottleneck at a time
   - Implement, test, measure
   - Repeat

---

## References

- [RCCL Design Overview](rccl-design-overview.md)
- [RCCL Environment Variables](rccl-environment-variables-analysis.md)
- rccl-tests: `/Users/ahalperin/xai/amd-dev/amd/rccl-tests/`
- rocprof documentation: https://rocm.docs.amd.com/projects/rocprofiler/

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-10-30 | AI Assistant | Initial bottleneck analysis guide |


