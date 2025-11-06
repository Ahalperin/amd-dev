# RCCL Proxy Thread Optimization - Quick Start Guide

**Date:** November 4, 2025  
**Purpose:** Quick guide to apply and test proxy thread optimizations

---

## TL;DR - Quick Implementation

### Step 1: Apply the Lock-Free Queue Patch (Highest Priority)

```bash
cd /home/dn/amd-dev/amd/rccl
git apply /home/dn/amd-dev/tools/docs/rccl/proxy-lockfree-queue.patch
```

### Step 2: Rebuild RCCL

```bash
cd /home/dn/amd-dev/amd/rccl
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Step 3: Quick Test

```bash
# Enable the optimization
export NCCL_PROXY_LOCKFREE_QUEUE=1

# Test small message latency
./build/test/single/sendrecv_test -b 1 -e 4096 -f 2 -g 1 -n 1000

# Compare with baseline (disable optimization)
export NCCL_PROXY_LOCKFREE_QUEUE=0
./build/test/single/sendrecv_test -b 1 -e 4096 -f 2 -g 1 -n 1000
```

---

## Detailed Implementation Steps

### 1. Lock-Free Queue (Primary Optimization)

This is the highest-impact, lowest-risk optimization. Expected improvement: 15-25% latency reduction.

**Files Modified:**
- `src/include/proxy.h` - Add ring buffer structure
- `src/proxy.cc` - Implement lock-free queue functions

**Testing:**
```bash
# Compile with optimization
cd /home/dn/amd-dev/amd/rccl
git apply tools/docs/rccl/proxy-lockfree-queue.patch
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
make -j$(nproc)

# Run comprehensive tests
export NCCL_PROXY_LOCKFREE_QUEUE=1
export NCCL_DEBUG=INFO

# Small message test (latency sensitive)
./test/single/sendrecv_test -b 1 -e 8192 -f 2 -g 2 -n 10000

# Medium message test
./test/single/all_reduce_test -b 64K -e 1M -f 2 -g 2 -n 1000

# Large message test (bandwidth)
./test/single/all_reduce_test -b 1M -e 128M -f 2 -g 2 -n 100
```

**Expected Results:**
- Small messages (1B-8KB): 15-25% lower latency
- No regression on large messages
- Lower proxy thread CPU usage

### 2. Additional Optimizations (Optional)

#### 2A. Batched Network Polling

**Location:** `src/transport/net.cc`

Add batched test support:

```cpp
// Around line 1400 in sendProxyProgress()
#define MAX_BATCH_TEST 16

// Collect requests
void* batchRequests[MAX_BATCH_TEST];
int batchCount = 0;

for (int s = 0; s < args->nsubs && batchCount < MAX_BATCH_TEST; s++) {
    struct ncclProxySubArgs* sub = args->subs + s;
    if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base + sub->done) % NCCL_STEPS;
        batchRequests[batchCount++] = sub->requests[buffSlot];
    }
}

// Batch test (if network plugin supports it)
if (batchCount > 1 && proxyState->ncclNet->testBatch) {
    // Use batched API
    int doneFlags[MAX_BATCH_TEST];
    int sizes[MAX_BATCH_TEST];
    NCCLCHECK(proxyState->ncclNet->testBatch(batchCount, batchRequests, doneFlags, sizes));
    
    // Process results
    for (int i = 0; i < batchCount; i++) {
        if (doneFlags[i]) {
            // Handle completion
        }
    }
}
```

**Note:** Requires network plugin to implement `testBatch()` API.

#### 2B. Software Prefetching

Already included in the lock-free queue patch. Adds prefetch hints for linked list traversal.

**Verify it's working:**
```bash
# Check if prefetch instructions are generated
objdump -d build/librccl.so | grep -A 5 "progressOps" | grep prefetch
```

#### 2C. Adaptive Spin-Wait

Also included in the lock-free queue patch. Tunable via environment variables:

```bash
export NCCL_PROXY_SPIN_COUNT=100        # Tight spin iterations (default: 100)
export NCCL_PROXY_YIELD_THRESHOLD=1000  # Cycles before yielding CPU (default: 1000)
```

**Tuning Guidelines:**
- **High message rate workloads:** Increase `SPIN_COUNT` to 200-500
- **Low message rate workloads:** Decrease `SPIN_COUNT` to 50
- **Shared CPU systems:** Decrease both to yield faster
- **Dedicated CPU cores:** Increase both to minimize latency

---

## Benchmarking and Profiling

### Microbenchmarks

```bash
#!/bin/bash
# save as benchmark_proxy.sh

NCCL_BUILD=/home/dn/amd-dev/amd/rccl/build

echo "=== Small Message Latency Test ==="
for opt in 0 1; do
    export NCCL_PROXY_LOCKFREE_QUEUE=$opt
    echo "NCCL_PROXY_LOCKFREE_QUEUE=$opt"
    $NCCL_BUILD/test/single/sendrecv_test -b 1 -e 4096 -f 2 -g 2 -n 10000 \
        | grep "Avg latency"
done

echo ""
echo "=== Multi-Operation Throughput Test ==="
for opt in 0 1; do
    export NCCL_PROXY_LOCKFREE_QUEUE=$opt
    echo "NCCL_PROXY_LOCKFREE_QUEUE=$opt"
    $NCCL_BUILD/test/single/sendrecv_test -b 64K -e 64K -f 1 -g 2 -n 1000 -w 16 \
        | grep "Avg bus bandwidth"
done

echo ""
echo "=== Large Message Bandwidth Test ==="
for opt in 0 1; do
    export NCCL_PROXY_LOCKFREE_QUEUE=$opt
    echo "NCCL_PROXY_LOCKFREE_QUEUE=$opt"
    $NCCL_BUILD/test/single/all_reduce_test -b 1M -e 128M -f 2 -g 2 -n 100 \
        | grep "Avg bus bandwidth"
done
```

### Profiling with perf

```bash
# Profile proxy thread CPU usage
sudo perf record -g -p $(pgrep -f "NCCL Progress") sleep 10
sudo perf report

# Look for improvements in:
# 1. Reduced time in pthread_mutex_lock/unlock
# 2. Reduced time in sched_yield
# 3. Better instruction throughput (IPC)
```

### Profiling with rocprof (AMD-specific)

```bash
# Profile GPU-side effects
export ROCPROF_ENABLE=1
rocprof --stats ./your_benchmark

# Check for:
# 1. Reduced kernel launch latency
# 2. Improved GPU utilization
# 3. Fewer stalls waiting for host
```

---

## Debugging

### Enable Debug Output

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROXY,NET

# Should see log messages like:
# [Proxy Progress] Using lock-free queue optimization
# Proxy lock-free ring buffer initialized (size=1024)
```

### Common Issues

#### Issue 1: Build Errors

```bash
# Error: undefined reference to `proxyRingPush`
# Solution: Ensure patch applied cleanly
git status
git diff src/proxy.cc src/include/proxy.h

# If patch failed, try manual application
git apply --reject proxy-lockfree-queue.patch
# Then manually apply failed hunks from *.rej files
```

#### Issue 2: No Performance Improvement

```bash
# Check if optimization is actually enabled
export NCCL_DEBUG=INFO
export NCCL_PROXY_LOCKFREE_QUEUE=1
./your_test 2>&1 | grep "lock-free"

# Should see: "[Proxy Progress] Using lock-free queue optimization"

# If not enabled:
# 1. Check environment variable is set
# 2. Verify opsRing is initialized (check logs)
# 3. Ensure recompiled after patching
```

#### Issue 3: Crashes or Hangs

```bash
# Enable address sanitizer
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
make -j$(nproc)

# Run test
./test/single/sendrecv_test -b 1 -e 4096 -f 2 -g 2 -n 100

# Common causes:
# 1. Ring buffer overflow (increase PROXY_RING_SIZE)
# 2. Memory ordering issues (check atomic operations)
# 3. Race conditions (verify proper synchronization)
```

---

## Performance Validation Checklist

- [ ] Baseline measurements captured (NCCL_PROXY_LOCKFREE_QUEUE=0)
- [ ] Optimized measurements captured (NCCL_PROXY_LOCKFREE_QUEUE=1)
- [ ] Small message latency improved by ≥10%
- [ ] No regression on large message bandwidth
- [ ] Multi-operation throughput improved by ≥15%
- [ ] Proxy thread CPU usage decreased
- [ ] All existing tests pass (make test)
- [ ] No memory leaks (valgrind --leak-check=full)
- [ ] No data races (built with -fsanitize=thread)

---

## Integration Testing

### Test Suite

```bash
cd /home/dn/amd-dev/amd/rccl/build

# Enable optimization
export NCCL_PROXY_LOCKFREE_QUEUE=1

# Run all unit tests
make test

# Run specific collective tests
./test/single/all_reduce_test -b 8 -e 128M -f 2 -g 2
./test/single/all_gather_test -b 8 -e 128M -f 2 -g 2
./test/single/broadcast_test -b 8 -e 128M -f 2 -g 2
./test/single/reduce_test -b 8 -e 128M -f 2 -g 2

# Run multi-node tests (if available)
# mpirun -np 8 ./test/mpi/all_reduce_test
```

### Application Testing

```bash
# PyTorch DDP test
export NCCL_PROXY_LOCKFREE_QUEUE=1
python3 -m torch.distributed.run --nproc_per_node=8 your_training_script.py

# Monitor improvements:
# 1. Reduced training step time
# 2. Better GPU utilization
# 3. Lower communication overhead
```

---

## Rollback Instructions

If you encounter issues and need to rollback:

```bash
cd /home/dn/amd-dev/amd/rccl

# Revert patch
git apply -R tools/docs/rccl/proxy-lockfree-queue.patch

# Or reset to original state
git checkout src/proxy.cc src/include/proxy.h

# Rebuild
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## Next Steps

After validating the lock-free queue optimization:

1. **Implement Batched Polling** - See optimization proposal document
2. **Add Fast Path for Small Messages** - Specialized code path for <4KB messages  
3. **Tune for Your Workload** - Adjust environment variables based on profiling
4. **Consider Upstreaming** - Submit successful optimizations back to RCCL project

---

## Support and Resources

- **Full Optimization Proposal:** `tools/docs/rccl/proxy-thread-optimizations.md`
- **Technical Internals:** `tools/docs/rccl/rccl-technical-internals.md`
- **RCCL GitHub:** https://github.com/ROCmSoftwarePlatform/rccl
- **ROCm Documentation:** https://rocm.docs.amd.com/

---

## Appendix: Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_PROXY_LOCKFREE_QUEUE` | 1 | Enable lock-free queue optimization |
| `NCCL_PROXY_SPIN_COUNT` | 100 | Tight spin iterations before nanosleep |
| `NCCL_PROXY_YIELD_THRESHOLD` | 1000 | Idle cycles before yielding CPU |
| `NCCL_PROXY_APPEND_BATCH_SIZE` | 16 | Number of ops to process per batch |
| `NCCL_PROGRESS_APPENDOP_FREQ` | 8 | Progress iterations per append check |
| `NCCL_DEBUG` | WARN | Debug output level (WARN/INFO/TRACE) |
| `NCCL_DEBUG_SUBSYS` | ALL | Debug subsystems (PROXY,NET,etc.) |

**Recommended Settings for Different Scenarios:**

**Low Latency (Trading, HPC):**
```bash
export NCCL_PROXY_LOCKFREE_QUEUE=1
export NCCL_PROXY_SPIN_COUNT=500
export NCCL_PROXY_YIELD_THRESHOLD=5000
export NCCL_PROGRESS_APPENDOP_FREQ=4
```

**High Throughput (ML Training):**
```bash
export NCCL_PROXY_LOCKFREE_QUEUE=1
export NCCL_PROXY_SPIN_COUNT=100
export NCCL_PROXY_YIELD_THRESHOLD=1000
export NCCL_PROXY_APPEND_BATCH_SIZE=32
export NCCL_PROGRESS_APPENDOP_FREQ=16
```

**Power Efficient (Shared Systems):**
```bash
export NCCL_PROXY_LOCKFREE_QUEUE=1
export NCCL_PROXY_SPIN_COUNT=10
export NCCL_PROXY_YIELD_THRESHOLD=100
export NCCL_PROGRESS_APPENDOP_FREQ=32
```

---

**Document Version:** 1.0  
**Last Updated:** November 4, 2025


