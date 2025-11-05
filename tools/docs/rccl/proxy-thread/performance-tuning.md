# RCCL Proxy Performance Tuning

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Performance Metrics](#performance-metrics)
4. [Tuning Strategies](#tuning-strategies)
5. [Common Performance Issues](#common-performance-issues)
6. [Hardware-Specific Tuning](#hardware-specific-tuning)
7. [Monitoring and Profiling](#monitoring-and-profiling)

## Overview

The RCCL proxy thread system has several tunable parameters that affect performance. This document provides guidance on tuning these parameters for different workloads and hardware configurations.

## Environment Variables

### Progress Thread Parameters

#### NCCL_PROGRESS_APPENDOP_FREQ

**Purpose**: Controls how frequently the progress thread fetches new operations from the ops pool.

**Default**: `8`

**Valid Range**: `1` - `100+`

**Impact**:
- **Lower values** (1-4):
  - Lower latency (ops start progressing sooner)
  - Higher CPU usage (more frequent pool access)
  - More lock contention (frequent mutex acquisition)
  - Better for latency-sensitive workloads

- **Higher values** (16-32):
  - Higher latency (ops may wait longer to start)
  - Lower CPU usage (less frequent pool access)
  - Less lock contention
  - Better for throughput-oriented workloads

**Usage**:
```bash
export NCCL_PROGRESS_APPENDOP_FREQ=4  # Low latency
export NCCL_PROGRESS_APPENDOP_FREQ=16 # High throughput
```

**Tuning Guidelines**:
- **Small messages** (<1KB): Use 1-4 for lowest latency
- **Large messages** (>1MB): Use 8-16 for best throughput
- **Many small ops**: Increase to reduce overhead
- **Few large ops**: Decrease to start progress faster

#### NCCL_PROXY_APPEND_BATCH_SIZE

**Purpose**: Maximum number of operations fetched from pool in one batch.

**Default**: `16`

**Valid Range**: `1` - `128`

**Impact**:
- **Smaller batches** (4-8):
  - Lower latency per operation
  - More frequent pool access
  - Better load balancing across operations

- **Larger batches** (32-64):
  - Higher throughput
  - Less overhead per operation
  - May increase latency for late operations in batch

**Usage**:
```bash
export NCCL_PROXY_APPEND_BATCH_SIZE=8   # Low latency
export NCCL_PROXY_APPEND_BATCH_SIZE=32  # High throughput
```

**Tuning Guidelines**:
- Match to typical number of concurrent operations
- Larger for multi-GPU collective operations
- Smaller for P2P or sparse communication patterns

#### NCCL_PROXY_DUMP_SIGNAL

**Purpose**: Enable proxy state dumping on signal (for debugging).

**Default**: `-1` (disabled)

**Valid Values**: Any signal number (e.g., `10` for SIGUSR1, `12` for SIGUSR2)

**Usage**:
```bash
export NCCL_PROXY_DUMP_SIGNAL=10
# Then send signal: kill -10 <pid>
```

**Output**: Dumps all active operations, states, and progress counters to stderr.

**Use Case**: Debugging hangs, understanding operation progress

#### NCCL_CREATE_THREAD_CONTEXT

**Purpose**: Create dedicated CUDA context for proxy threads.

**Default**: `0` (disabled)

**Valid Values**: `0`, `1`

**Impact**:
- **Enabled** (1):
  - Dedicated context per proxy thread
  - Better isolation
  - Potentially better performance with many GPUs
  - Requires CUDA 11.3+ and recent drivers

- **Disabled** (0):
  - Uses `cudaSetDevice()`
  - Simpler, more compatible
  - Shares context with main thread

**Usage**:
```bash
export NCCL_CREATE_THREAD_CONTEXT=1
```

**Tuning Guidelines**:
- Try enabling on systems with many GPUs (8+)
- May help with CUDA context contention
- Disable if seeing context-related errors

### Network Transport Parameters

These affect proxy behavior for network operations:

#### NCCL_NET_SHARED_BUFFERS

**Purpose**: Enable shared buffers across connections.

**Default**: `-2` (auto-detect)

**Valid Values**: `-2` (auto), `0` (disabled), `1` (enabled)

**Impact**:
- **Enabled**: Reduces memory usage, may increase contention
- **Disabled**: More memory, better concurrency

#### NCCL_GDRCOPY_SYNC_ENABLE

**Purpose**: Use GPU memory for network tail pointer (GDRCOPY optimization).

**Default**: `1` (enabled)

**Valid Values**: `0`, `1`

**Impact on Proxy**:
- **Enabled**: Proxy reads tail from GPU, may add latency
- **Disabled**: Tail in host memory, faster proxy access

#### NCCL_GDRCOPY_FLUSH_ENABLE

**Purpose**: Use PCIe read to flush GDRDMA buffers.

**Default**: `0` (disabled)

**Valid Values**: `0`, `1`

**Impact on Proxy**:
- **Enabled**: Proxy performs PCIe read before network send
- **Disabled**: Relies on HDP flush (AMD) or other mechanisms

### Shared Memory Transport Parameters

#### NCCL_SHM_USE_CUDA_MEMCPY

**Purpose**: Enable proxy-assisted shared memory copies.

**Default**: `0` (disabled)

**Valid Values**: `0`, `1`

**Impact**:
- **Enabled**: Proxy performs cudaMemcpy, frees GPU
- **Disabled**: Direct GPU-to-GPU copy, no proxy

**Tuning**:
- Try enabling if GPU compute resources are constrained
- Disable for lowest latency intra-node communication

#### NCCL_SHM_MEMCPY_MODE

**Purpose**: Control which direction uses proxy.

**Default**: `3` (both)

**Valid Values**: 
- `1`: Send side only
- `2`: Receive side only
- `3`: Both sides

#### NCCL_SHM_LOCALITY

**Purpose**: Choose which side allocates shared memory.

**Default**: `2` (receive side)

**Valid Values**:
- `1`: Send side (sender-side allocation)
- `2`: Receive side (receiver-side allocation)

**Impact**:
- Affects NUMA placement
- Can impact proxy thread memory access latency

### P2P Transport Parameters

#### NCCL_P2P_USE_CUDA_MEMCPY

**Purpose**: Enable proxy-assisted P2P copies.

**Default**: `0` (disabled)

**Valid Values**: `0`, `1`

**Impact**: Similar to SHM_USE_CUDA_MEMCPY but for P2P

## Performance Metrics

### Key Metrics to Monitor

1. **Operation Latency**:
   - Time from operation post to completion
   - Measure: End-to-end collective time

2. **Bandwidth**:
   - Bytes transferred per second
   - Measure: Large message throughput

3. **CPU Usage**:
   - Proxy thread CPU utilization
   - Measure: `top`, `htop`, or profiler

4. **Lock Contention**:
   - Time waiting on opsPool mutex
   - Measure: Thread profiling tools

5. **Progress Rate**:
   - Operations progressed per second
   - Measure: NCCL debug logs

### Measuring Performance

#### Latency Measurement

```bash
# Small message latency
./nccl-tests/build/all_reduce_perf -b 8 -e 128 -f 2 -g <ngpus>

# Look for "Avg bus bandwidth" and "Time" columns
```

#### Bandwidth Measurement

```bash
# Large message bandwidth
./nccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g <ngpus>

# Look for peak bandwidth values
```

#### CPU Usage Monitoring

```bash
# While running NCCL application
top -H -p <pid>  # Shows per-thread CPU usage

# Look for threads named "NCCL Progress*"
```

#### Proxy State Dumping

```bash
# Setup
export NCCL_PROXY_DUMP_SIGNAL=10

# Run application, then in another terminal:
kill -10 <pid>

# Check stderr for proxy state dump
```

## Tuning Strategies

### Strategy 1: Latency Optimization

**Goal**: Minimize time from operation start to completion

**Parameters**:
```bash
export NCCL_PROGRESS_APPENDOP_FREQ=1
export NCCL_PROXY_APPEND_BATCH_SIZE=4
export NCCL_SHM_USE_CUDA_MEMCPY=0
export NCCL_P2P_USE_CUDA_MEMCPY=0
```

**Best For**:
- Small message collectives (<1KB)
- Latency-sensitive applications
- Sparse communication patterns
- Interactive workloads

**Trade-offs**:
- Higher CPU usage
- More lock contention
- Lower peak bandwidth

### Strategy 2: Throughput Optimization

**Goal**: Maximize bytes transferred per second

**Parameters**:
```bash
export NCCL_PROGRESS_APPENDOP_FREQ=16
export NCCL_PROXY_APPEND_BATCH_SIZE=32
export NCCL_NET_SHARED_BUFFERS=1  # If memory constrained
```

**Best For**:
- Large message collectives (>1MB)
- Batch training workloads
- Bandwidth-bound applications
- Multi-GPU data parallel training

**Trade-offs**:
- Higher latency for individual operations
- More memory usage (if shared buffers disabled)

### Strategy 3: CPU Efficiency

**Goal**: Minimize CPU overhead of proxy threads

**Parameters**:
```bash
export NCCL_PROGRESS_APPENDOP_FREQ=16
export NCCL_PROXY_APPEND_BATCH_SIZE=64
export NCCL_SHM_USE_CUDA_MEMCPY=0
export NCCL_P2P_USE_CUDA_MEMCPY=0
```

**Best For**:
- CPU-bound applications
- Oversubscribed systems
- Multi-process/multi-tenant environments

**Trade-offs**:
- Higher latency
- May underutilize network

### Strategy 4: Balanced (Default)

**Goal**: Good balance of latency, throughput, and CPU usage

**Parameters**:
```bash
export NCCL_PROGRESS_APPENDOP_FREQ=8
export NCCL_PROXY_APPEND_BATCH_SIZE=16
# Other parameters at defaults
```

**Best For**:
- General-purpose workloads
- Mixed message sizes
- Most deep learning training

## Common Performance Issues

### Issue 1: High Latency for Small Messages

**Symptoms**:
- Small message collectives slower than expected
- Latency increases with number of ranks
- Good bandwidth but poor latency

**Diagnosis**:
```bash
# Check if proxy is batching too much
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=PROXY
# Run and check time between operation post and progress
```

**Solutions**:
1. Reduce `NCCL_PROGRESS_APPENDOP_FREQ`:
   ```bash
   export NCCL_PROGRESS_APPENDOP_FREQ=1
   ```

2. Reduce batch size:
   ```bash
   export NCCL_PROXY_APPEND_BATCH_SIZE=4
   ```

3. Check CPU affinity:
   ```bash
   # Ensure proxy thread not sharing core with application
   taskset -c <dedicated_cores> <app>
   ```

### Issue 2: Low Bandwidth

**Symptoms**:
- Large message bandwidth below expected
- Network utilization low
- GPU utilization low during communication

**Diagnosis**:
```bash
# Check if operations progressing slowly
kill -<SIGNAL> <pid>  # If NCCL_PROXY_DUMP_SIGNAL set
# Look for operations stuck in progress

# Check for network issues
# Run network benchmarks independently
```

**Solutions**:
1. Increase operation batching:
   ```bash
   export NCCL_PROGRESS_APPENDOP_FREQ=16
   export NCCL_PROXY_APPEND_BATCH_SIZE=32
   ```

2. Check GDR configuration:
   ```bash
   # Ensure GDR enabled and working
   export NCCL_NET_GDR_LEVEL=3  # Or appropriate for your system
   ```

3. Verify network configuration:
   ```bash
   # Check RDMA, routing, etc.
   # Use ib_write_bw or similar tools
   ```

### Issue 3: High CPU Usage

**Symptoms**:
- Proxy threads consuming excessive CPU
- System overloaded
- Other processes affected

**Diagnosis**:
```bash
# Monitor proxy thread CPU
top -H -p <pid>

# Check for busy-waiting
perf record -g -p <pid>
perf report
# Look for spin loops
```

**Solutions**:
1. Increase batching to reduce wakeup frequency:
   ```bash
   export NCCL_PROGRESS_APPENDOP_FREQ=32
   ```

2. If proxy mostly idle:
   ```bash
   # Should already yield CPU, check for driver issues
   # May need system-level tuning
   ```

3. Bind to specific cores:
   ```bash
   # Use numactl or taskset to control placement
   ```

### Issue 4: Deadlocks or Hangs

**Symptoms**:
- Application hangs during communication
- Some operations complete, others don't
- No progress after certain point

**Diagnosis**:
```bash
# Enable proxy dump
export NCCL_PROXY_DUMP_SIGNAL=10

# Dump state when hung
kill -10 <pid>

# Look for:
# - Operations stuck in same state
# - Mismatched counters
# - Missing operations
```

**Solutions**:
1. Check for abort conditions:
   ```bash
   # Look in logs for errors before hang
   export NCCL_DEBUG=WARN
   ```

2. Verify network connectivity:
   ```bash
   # Ensure all ranks can communicate
   # Check firewalls, routing, etc.
   ```

3. Check resource limits:
   ```bash
   ulimit -a
   # Ensure sufficient locked memory, open files, etc.
   ```

### Issue 5: Memory Pressure

**Symptoms**:
- High memory usage
- OOM errors
- Swapping

**Diagnosis**:
```bash
# Check proxy memory usage
pmap <pid> | grep -i rccl
cat /proc/<pid>/status | grep -i vm
```

**Solutions**:
1. Enable buffer sharing:
   ```bash
   export NCCL_NET_SHARED_BUFFERS=1
   ```

2. Reduce buffer sizes (advanced):
   ```bash
   export NCCL_BUFFSIZE=<size>  # Default varies
   ```

3. Reduce concurrent operations:
   ```bash
   # Application-level: Use smaller groups or more sequential operations
   ```

## Hardware-Specific Tuning

### AMD GPUs

#### HDP Flush

AMD GPUs require HDP (Host Data Path) flush for GDR:

```bash
# Automatically handled by proxy, but verify:
export NCCL_DEBUG=INFO
# Look for "HDP register" messages
```

**Tuning**:
- Ensure flush happens once per batch, not per operation
- Check `args->hdp_flushed` logic in proxy

#### GFX Architecture

Different architectures have different characteristics:

- **gfx908** (MI100): Good GDR support
- **gfx90a** (MI200): Enhanced GDR, requires HDP flush
- **gfx942** (MI300): Advanced features, check latest RCCL version

**Tuning**:
```bash
# Check architecture detection
export NCCL_DEBUG=INFO
# Look for GPU architecture in logs
```

### InfiniBand

#### RDMA Configuration

```bash
# Ensure optimal RDMA settings
export NCCL_IB_GID_INDEX=3  # RoCE v2
export NCCL_IB_TC=<value>   # Traffic class
export NCCL_IB_TIMEOUT=<value>  # Timeout value
```

#### Multi-Rail

```bash
# Use multiple HCAs
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

### High Core Count Systems

**Issue**: Many CPUs may cause scheduling delays

**Solutions**:
1. Pin proxy threads:
   ```bash
   # Use NUMA-aware placement
   numactl --cpunodebind=<node> --membind=<node> <app>
   ```

2. Dedicate cores:
   ```bash
   # Reserve cores for proxy threads
   isolcpus=<cores>  # In kernel command line
   ```

3. Real-time priority (advanced):
   ```bash
   # Requires privileges
   chrt -f 50 <app>  # SCHED_FIFO
   ```

### Cloud Environments

**Challenges**:
- Variable network performance
- Shared resources
- Virtual NICs

**Tuning**:
```bash
# Be more conservative
export NCCL_PROGRESS_APPENDOP_FREQ=8  # Default
export NCCL_PROXY_APPEND_BATCH_SIZE=16  # Default

# Monitor performance variability
# May need dynamic adjustment (not directly supported)
```

## Monitoring and Profiling

### Built-in NCCL Logging

```bash
# Basic logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,PROXY

# Detailed logging (verbose)
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# Log to file
export NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log
```

### Proxy Tracing

RCCL includes proxy tracing support (Facebook RCCL contribution):

```bash
# Enable proxy tracing
export RCCL_PROXY_TRACE=1

# Generates detailed operation traces
# Useful for understanding operation timing
```

### System Profiling

#### CPU Profiling

```bash
# Sample proxy threads
perf record -F 99 -p <pid> -g sleep 10
perf report

# Look for:
# - Time in progressOps()
# - Lock contention
# - Network polling
```

#### Memory Profiling

```bash
# Track memory allocations
valgrind --tool=massif <app>
ms_print massif.out.<pid>

# Or use heaptrack:
heaptrack <app>
heaptrack_gui heaptrack.<app>.<pid>.gz
```

#### Network Profiling

```bash
# Monitor network traffic
iftop -i <interface>

# Or use netdata for system-wide monitoring
```

### Application-Level Instrumentation

```cpp
// Add timing around collectives
auto start = std::chrono::high_resolution_clock::now();
ncclAllReduce(/*...*/);
cudaStreamSynchronize(stream);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
printf("AllReduce took %ld us\n", duration.count());
```

## Performance Tuning Workflow

### Step 1: Baseline Measurement

1. Run with default settings
2. Measure latency, bandwidth, CPU usage
3. Identify bottleneck (latency, bandwidth, or CPU)

### Step 2: Initial Tuning

1. **If latency-bound**: Reduce `NCCL_PROGRESS_APPENDOP_FREQ`
2. **If bandwidth-bound**: Increase batching parameters
3. **If CPU-bound**: Increase batching, reduce progress frequency

### Step 3: Iterative Refinement

1. Change one parameter at a time
2. Measure impact
3. Keep changes that improve target metric
4. Revert changes that harm performance

### Step 4: Validation

1. Test with full workload
2. Verify no regressions
3. Check stability over time
4. Document final settings

### Step 5: Production Monitoring

1. Monitor key metrics continuously
2. Watch for performance degradation
3. Re-tune if workload characteristics change

## Tuning Checklist

- [ ] Measured baseline performance
- [ ] Identified primary bottleneck
- [ ] Tuned `NCCL_PROGRESS_APPENDOP_FREQ`
- [ ] Tuned `NCCL_PROXY_APPEND_BATCH_SIZE`
- [ ] Verified GDR configuration
- [ ] Checked CPU affinity
- [ ] Tested with representative workload
- [ ] Documented final settings
- [ ] Validated stability
- [ ] Setup monitoring

## Summary

Effective proxy performance tuning requires:

1. **Understanding**: Know your workload characteristics
2. **Measurement**: Use appropriate metrics and tools
3. **Iteration**: Try different settings systematically
4. **Validation**: Verify improvements with real workloads
5. **Monitoring**: Track performance over time

Key trade-offs:
- **Latency vs Throughput**: Lower latency often means lower peak bandwidth
- **CPU vs Performance**: Less CPU usage may reduce network utilization
- **Memory vs Performance**: Shared buffers save memory but may add contention

Most applications work well with default settings, but tuning can provide significant improvements for specific use cases.


