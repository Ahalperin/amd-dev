# RCCL Quick Reference Guide

**Last Updated:** October 30, 2025  
**Purpose:** Fast lookup for common RCCL operations, debugging, and optimization

---

## 🚀 Quick Start Commands

### Run Basic Performance Test
```bash
cd /Users/ahalperin/xai/amd-dev/amd/rccl-tests
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100
```

### Enable Debug Logging
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH,NET
./build/all_reduce_perf -g 8 2>&1 | tee rccl_debug.log
```

### Profile GPU Kernels
```bash
rocprof --stats --timestamp on ./build/all_reduce_perf -b 1M -e 1M -g 8 -n 100
cat results.stats.csv
```

---

## 🔍 Debugging Checklist

### ❓ Is xGMI being used?
```bash
export NCCL_DEBUG=INFO
./build/all_reduce_perf -g 8 2>&1 | grep -E "P2P|XGMI|Channel"
# Should see: "P2P/direct" or "XGMI"
# Should NOT see: "P2P/IB" for intra-node
```

### ❓ What topology is detected?
```bash
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH
./build/all_reduce_perf -g 8 2>&1 | grep -i "graph\|topo\|channel"
```

### ❓ What algorithm is selected?
```bash
export NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL
./build/all_reduce_perf -b 1M -e 1M -g 8 -n 10 2>&1 | grep -i "algo\|proto"
# Look for: "Algorithm Ring/Tree" and "Protocol Simple/LL/LL128"
```

### ❓ What's my actual bandwidth?
```bash
./build/all_reduce_perf -b 128M -e 128M -g 8 -n 100 | grep "128M"
# Compare "busbw" column to theoretical peak:
# - xGMI MI300X: 432 GB/s per link (target: 350-400 GB/s)
# - xGMI MI250X: 288 GB/s per link (target: 240-270 GB/s)
# - PCIe Gen4 x16: 24 GB/s (target: 17-20 GB/s)
```

### ❓ Is there a performance cliff?
```bash
# Sweep message sizes
./build/all_reduce_perf -b 1K -e 1G -f 2 -g 8 -n 100 > sweep.log

# Plot or analyze where bandwidth drops
python3 << 'EOF'
import re
with open('sweep.log') as f:
    for line in f:
        if re.match(r'^\s+\d+', line):
            parts = line.split()
            size, time, algbw, busbw = parts[0], parts[4], parts[5], parts[6]
            print(f"{size:>12s}: {busbw:>8s} GB/s")
EOF
```

---

## 🎯 Performance Targets

### Bandwidth Targets (Bus Bandwidth)

| Hardware | Theoretical | Target (85%+) | Good | Excellent |
|----------|-------------|---------------|------|-----------|
| MI300X xGMI | 432 GB/s | 367 GB/s | 350+ GB/s | 400+ GB/s |
| MI250X xGMI | 288 GB/s | 245 GB/s | 240+ GB/s | 270+ GB/s |
| PCIe Gen4 x16 | 24 GB/s | 20 GB/s | 17+ GB/s | 22+ GB/s |
| IB HDR200 | 23 GB/s | 19.5 GB/s | 17+ GB/s | 21+ GB/s |

### Latency Targets (Small Messages)

| Message Size | Intra-Node (xGMI) | Intra-Node (PCIe) | Inter-Node (IB) |
|--------------|-------------------|-------------------|-----------------|
| 1 KB | <15 µs | <30 µs | <100 µs |
| 4 KB | <20 µs | <40 µs | <120 µs |
| 16 KB | <30 µs | <50 µs | <150 µs |

---

## 🔧 Tuning Environment Variables

### For Large Messages (>1 MB)
```bash
export NCCL_NCHANNELS=16          # More parallelism
export NCCL_PROTO=Simple          # Lowest overhead
export NCCL_BUFFSIZE=8388608      # 8 MB buffers
export NCCL_ALGO=Ring             # Best bandwidth
```

### For Small Messages (<64 KB)
```bash
export NCCL_NCHANNELS=4           # Less overhead
export NCCL_PROTO=LL              # Low latency
export NCCL_ALGO=Tree             # Lower latency
```

### For xGMI Systems
```bash
export RCCL_FORCE_XGMI=1          # Force xGMI detection
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1
```

### For Multi-Node (InfiniBand)
```bash
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Specify IB devices
export NCCL_IB_GID_INDEX=3         # For RoCE
export NCCL_SOCKET_IFNAME=eth0     # Fallback interface
export NCCL_IB_TIMEOUT=22          # Increase if needed
```

### Disable for Testing
```bash
export NCCL_P2P_DISABLE=1         # Disable P2P (use SHM)
export NCCL_SHM_DISABLE=1         # Disable shared memory
export NCCL_NET_SHARED_COMMS=0    # Separate network connections
```

---

## 📊 Interpreting rccl-tests Output

### Sample Output
```
#       size         count      type   redop     time   algbw   busbw
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)
      1048576        262144     float     sum    145.2   7.22   12.64
     67108864      16777216     float     sum   2405.1  27.90   48.83
    134217728      33554432     float     sum   4567.3  29.39   51.43
```

### What Each Column Means
- **size**: Message size in bytes
- **count**: Number of elements (size / sizeof(type))
- **type**: Data type (float, half, int, etc.)
- **redop**: Reduction operation (sum, prod, min, max)
- **time**: Average time in microseconds
- **algbw**: Algorithm bandwidth = size / time (user perspective)
- **busbw**: Bus bandwidth = actual bytes on interconnect / time

### Calculate Efficiency
```python
# For AllReduce with N ranks, ring algorithm:
# busbw = algbw × 2 × (N-1)/N

N = 8  # number of GPUs
busbw = 51.43  # GB/s from output
algbw = 29.39  # GB/s from output

# Verify relationship
calculated_busbw = algbw * 2 * (N-1) / N
print(f"Calculated: {calculated_busbw:.2f}, Actual: {busbw:.2f}")

# Efficiency vs. theoretical peak
theoretical_peak = 432  # GB/s for MI300X xGMI
per_link_bw = busbw / N
efficiency = per_link_bw / theoretical_peak * 100
print(f"Per-link BW: {per_link_bw:.2f} GB/s")
print(f"Efficiency: {efficiency:.1f}%")
```

---

## 🐛 Common Issues and Solutions

### Issue: Low Bandwidth (<50% of theoretical)

**Check:**
```bash
# 1. Is xGMI detected?
export NCCL_DEBUG=INFO
./test 2>&1 | grep -i xgmi

# 2. What paths are used?
export NCCL_DEBUG_SUBSYS=GRAPH
./test 2>&1 | grep "Channel.*send"

# 3. GPU clocks throttled?
rocm-smi --showclocks
rocm-smi --showtemp

# 4. CPU governor?
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Fix:**
```bash
# Force performance mode
sudo rocm-smi --setperflevel high
sudo rocm-smi --setfan 200

# CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Issue: High Latency for Small Messages

**Check:**
```bash
# Protocol used?
export NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL
./build/all_reduce_perf -b 1K -e 1K -g 8 -n 100 2>&1 | grep Protocol
```

**Fix:**
```bash
# Force LL protocol for low latency
export NCCL_PROTO=LL

# Or adjust thresholds (requires rebuild)
# Edit src/graph/tuning.cc
```

### Issue: Inconsistent Performance

**Check:**
```bash
# NUMA balancing enabled?
cat /proc/sys/kernel/numa_balancing

# Interrupts on wrong CPUs?
cat /proc/interrupts | grep mlx5

# Other processes using GPUs?
rocm-smi --showpids
```

**Fix:**
```bash
# Disable NUMA balancing
sudo sysctl kernel.numa_balancing=0

# Set CPU affinity (example for GPU 0)
numactl --cpunodebind=0 --membind=0 ./test

# Clear GPU processes
sudo fuser -k /dev/dri/renderD*  # Use with caution!
```

### Issue: Multi-Node Slower Than Expected

**Check:**
```bash
# Network reachable?
ping <remote_host>

# InfiniBand active?
ibstat
ibv_devinfo

# RCCL using IB?
export NCCL_DEBUG=INFO
mpirun -np 16 ./test 2>&1 | grep "NET/IB"
```

**Fix:**
```bash
# Ensure IB plugin loaded
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0

# Check GPUDirect
lsmod | grep peer_mem
# Should see: nv_peer_mem or amd_peer_mem

# Test IB bandwidth
ib_write_bw -d mlx5_0
```

---

## 📁 Quick File Reference

### Source Files to Check First

**Optimization Priority Order:**

1. **`src/graph/tuning.cc`** → Algorithm/protocol selection logic
2. **`src/graph/topo.cc`** → Topology detection (xGMI, PCIe)
3. **`src/device/primitives.h`** → GPU kernel primitives
4. **`src/graph/search.cc`** → Path search and ring construction
5. **`src/transport/p2p.cc`** → P2P transport setup
6. **`src/proxy.cc`** → Network proxy (multi-node)
7. **`src/enqueue.cc`** → Work enqueuing and kernel launch

### Key Data Structures

```c
// Main communicator
struct ncclComm          // src/include/comm.h:~line 150

// Channel (parallel execution path)
struct ncclChannel       // src/include/channel.h:~line 20

// Topology graph
struct ncclTopoSystem    // src/graph/topo.h:~line 100

// Connection between peers
struct ncclChannelPeer   // src/include/channel.h:~line 50
```

---

## 🧪 Test Matrix

### Basic Functionality
```bash
#!/bin/bash
for COLL in all_reduce all_gather reduce_scatter broadcast; do
    echo "Testing $COLL..."
    ./build/${COLL}_perf -b 8 -e 1G -f 2 -g 8 -c 1 -n 10
    [ $? -eq 0 ] && echo "✓ PASS" || echo "✗ FAIL"
done
```

### Performance Sweep
```bash
#!/bin/bash
SIZES="1K 4K 16K 64K 256K 1M 4M 16M 64M 256M 1G"
for SIZE in $SIZES; do
    ./build/all_reduce_perf -b $SIZE -e $SIZE -g 8 -n 100
done
```

### Scaling Test
```bash
#!/bin/bash
for NGPU in 2 4 8; do
    echo "Testing with $NGPU GPUs..."
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1))) \
        ./build/all_reduce_perf -b 128M -e 128M -g $NGPU -n 100
done
```

---

## 📈 Performance Regression Test

```bash
#!/bin/bash
# Save baseline
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100 > baseline.log

# After optimization
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 100 > optimized.log

# Compare (simplified)
python3 << 'EOF'
import re

def parse_log(filename):
    results = {}
    with open(filename) as f:
        for line in f:
            m = re.match(r'\s+(\d+)\s+\d+\s+\w+\s+\w+\s+[\d.]+\s+[\d.]+\s+([\d.]+)', line)
            if m:
                size, busbw = int(m.group(1)), float(m.group(2))
                results[size] = busbw
    return results

baseline = parse_log('baseline.log')
optimized = parse_log('optimized.log')

print(f"{'Size':>12s}  {'Baseline':>10s}  {'Optimized':>10s}  {'Change':>10s}")
print("-" * 50)
for size in sorted(baseline.keys()):
    if size in optimized:
        change = (optimized[size] / baseline[size] - 1) * 100
        emoji = "🚀" if change > 5 else "✓" if change > -5 else "⚠️"
        print(f"{size:12d}  {baseline[size]:10.2f}  {optimized[size]:10.2f}  {change:+9.1f}% {emoji}")
EOF
```

---

## 🔑 Key Formulas

### AllReduce Bandwidth
```
Algorithm Bandwidth = Message Size / Time
Bus Bandwidth = Algorithm BW × 2 × (N-1)/N

Where:
- 2: data sent and received
- (N-1)/N: ring efficiency (N-1 steps, N ranks)
```

### Efficiency
```
Efficiency = Measured BW / Theoretical Peak × 100%

Example:
- Measured bus BW: 350 GB/s (from rccl-tests)
- Theoretical (MI300X): 432 GB/s per link
- Efficiency: 350 / 432 = 81%
```

### Expected Time
```
Expected Time = Message Size / (Bandwidth × Efficiency)

Example:
- Message: 128 MB
- Bandwidth: 432 GB/s (theoretical)
- Expected efficiency: 85%
- Time = 128 MB / (432 GB/s × 0.85) = 348 µs
```

---

## 🎓 Learning Path

### Beginner (1-2 days)
1. Read [RCCL Design Overview](rccl-design-overview.md)
2. Run basic rccl-tests
3. Understand output metrics (algbw vs busbw)
4. Enable NCCL_DEBUG and interpret logs

### Intermediate (3-5 days)
1. Read [Bottleneck Analysis Guide](rccl-bottleneck-analysis.md)
2. Profile with rocprof
3. Understand topology detection
4. Test different tuning parameters

### Advanced (1-2 weeks)
1. Read [Technical Internals](rccl-technical-internals.md)
2. Review source code (tuning.cc, topo.cc, primitives.h)
3. Implement optimization
4. Validate and benchmark

---

## 📞 When You're Stuck

### Step 1: Check Environment
```bash
# Print all NCCL variables
env | grep -E "NCCL|RCCL|HSA"

# GPU status
rocm-smi

# Topology
rocm-smi --showtopo
```

### Step 2: Enable Verbose Logging
```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=/tmp/rccl_full_trace.log

# Run test
./test

# Analyze log
grep -i "error\|warn\|fail" /tmp/rccl_full_trace.log
```

### Step 3: Isolate Problem
```bash
# Test with minimal configuration
NCCL_NCHANNELS=1 \
NCCL_PROTO=Simple \
    ./build/all_reduce_perf -b 1M -e 1M -g 2 -n 10

# Test different pairs
HIP_VISIBLE_DEVICES=0,1 ./test  # GPU 0-1
HIP_VISIBLE_DEVICES=0,2 ./test  # GPU 0-2
HIP_VISIBLE_DEVICES=0,3 ./test  # GPU 0-3
```

### Step 4: Compare to Known Good
```bash
# Test with NVIDIA NCCL (if available)
LD_LIBRARY_PATH=/opt/nccl/lib ./test

# Or test on different system
scp test remote_host:
ssh remote_host './test'
```

---

## 📚 Documentation Map

- **[README.md](README.md)** → Start here, navigation
- **[rccl-design-overview.md](rccl-design-overview.md)** → Architecture
- **[rccl-bottleneck-analysis.md](rccl-bottleneck-analysis.md)** → Optimization guide
- **[rccl-technical-internals.md](rccl-technical-internals.md)** → Code details
- **[rccl-environment-variables-analysis.md](rccl-environment-variables-analysis.md)** → All env vars
- **[quick-reference.md](quick-reference.md)** → This document

---

**Tip:** Bookmark this page for quick access during debugging and optimization work!


