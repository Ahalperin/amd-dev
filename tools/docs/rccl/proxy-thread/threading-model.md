# RCCL Proxy Threading Model

## Table of Contents

1. [Overview](#overview)
2. [Thread Architecture](#thread-architecture)
3. [Thread Lifecycle](#thread-lifecycle)
4. [Synchronization](#synchronization)
5. [Thread Interactions](#thread-interactions)
6. [Performance Characteristics](#performance-characteristics)

## Overview

RCCL employs a multi-threaded architecture where specialized proxy threads handle communication operations independently from the main application threads and GPU kernels. This document details the threading model, lifecycle management, and interaction patterns.

## Thread Architecture

### Thread Hierarchy

```
Application Process
│
├─ Main Application Thread(s)
│   ├─ Launch GPU kernels
│   ├─ Post operations to proxy
│   └─ Poll for completion
│
├─ GPU Kernel Threads
│   ├─ Execute collective operations
│   └─ Update shared memory counters
│
└─ RCCL Proxy Threads (per communicator)
    │
    ├─ Progress Thread
    │   ├─ Progress active operations
    │   ├─ Fetch new operations
    │   └─ Call transport progress functions
    │
    ├─ Service Thread
    │   ├─ Accept connections
    │   ├─ Handle setup messages
    │   └─ Process async operations
    │
    └─ UDS Service Thread
        ├─ Handle Unix domain socket
        └─ Support cuMem operations
```

### Thread Types

| Thread | Count | Purpose | Main Loop | Wake Condition |
|--------|-------|---------|-----------|----------------|
| Progress | 1 per comm | Progress operations | Continuous | New ops or timeout |
| Service | 1 per comm | Connection mgmt | Event-driven | Socket events |
| UDS Service | 1 per comm | cuMem support | Event-driven | UDS events |

## Thread Lifecycle

### Progress Thread Lifecycle

#### Creation

**Location**: `ncclProxyProgressCreate()` in `src/proxy.cc` (line 1016)

```c
static ncclResult_t ncclProxyProgressCreate(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (!state->thread) {
    PTHREADCHECK(pthread_create(&state->thread, NULL, 
                                 ncclProxyProgress, proxyState), 
                 "pthread_create");
    ncclSetThreadName(state->thread, "NCCL Progress%2d", 
                      proxyState->tpLocalnRanks);
  }
  return ncclSuccess;
}
```

**Trigger**: Called during `proxyProgressInit()` when first connection with `proxyProgress` is established.

#### Initialization

```c
void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;
  
  // 1. Set CUDA context (optional, for GPU memory access)
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Set CUDA context on device %d", 
         proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
    WARN("[Proxy Progress] Failed to set CUDA device %d", 
         proxyState->cudaDev);
  }
  
  // 2. Set thread name for debugging
  char threadName[NCCL_THREAD_NAMELEN];
  snprintf(threadName, NCCL_THREAD_NAMELEN, "NCCL Progress%2d", 
           proxyState->cudaDev);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);
  
  // 3. Initialize state
  struct ncclProxyProgressState* state = &proxyState->progressState;
  state->nextOps = -1;
  
  // 4. Setup signal handler (optional, for debugging)
  const int sig = ncclParamProxyDumpSignal();
  if (sig != -1) signal(sig, ncclDumpProxyState);
  
  // 5. Enter main loop
  // ...
}
```

#### Main Loop

**Location**: `ncclProxyProgress()` in `src/proxy.cc` (line 939)

```c
int lastIdle = 0;
int proxyOpAppendCounter = 0;

do {
  int idle = 1;
  
  // 1. Progress all active operations
  ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
  if (ret != ncclSuccess) {
    __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
    break;
  }
  
  // 2. Update profiler with idle/active transitions
  if ((lastIdle == 0 && idle == 1) || (lastIdle == 1 && idle == 0)) {
    void* eHandle;
    ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
    if (lastIdle == 0 && idle == 1) 
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlIdle);
    if (lastIdle == 1 && idle == 0) 
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlActive);
    ncclProfilerStopProxyCtrlEvent(eHandle);
  }
  
  // 3. Fetch new operations (periodically or when idle)
  if (idle || !state->active || 
      (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
    int added = 0;
    proxyOpAppendCounter = 0;
    ret = ncclProxyGetPostedOps(proxyState, &added);
    if (ret != ncclSuccess) {
      __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
    }
    if (added == 0) {
      sched_yield(); // Yield CPU if no work
    }
  }
  lastIdle = idle;
  
} while ((state->stop == 0 || (state->stop == 1 && state->active)) && 
         __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) == 0);

return NULL;
```

**Loop Characteristics**:
- **Continuous**: Runs until stop flag set and no active ops
- **Adaptive**: Yields CPU when idle
- **Batched**: Fetches operations in batches (controlled by `NCCL_PROGRESS_APPENDOP_FREQ`)
- **Non-blocking**: Uses try-lock for ops pool access when active

#### Termination

**Location**: `ncclProxyProgressDestroy()` in `src/proxy.cc` (line 1025)

```c
ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  
  // 1. Request stop and wake thread
  if (state->opsPool) {
    pthread_mutex_lock(&state->opsPool->mutex);
    state->stop = 1;
    pthread_cond_signal(&state->opsPool->cond);
    pthread_mutex_unlock(&state->opsPool->mutex);
    
    // 2. Wait for thread to complete
    PTHREADCHECK(pthread_join(state->thread, NULL), "pthread_join");
  }
  
  // 3. Free memory pools
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    free(state->pools);
    state->pools = next;
  }
  
  return ncclSuccess;
}
```

**Termination Sequence**:
1. Set `state->stop = 1`
2. Signal condition variable to wake thread
3. Thread completes active operations
4. Thread exits main loop
5. Main thread joins proxy thread
6. Resources cleaned up

### Service Thread Lifecycle

#### Creation

**Location**: `ncclProxyCreate()` in `src/proxy.cc` (line 1903)

```c
PTHREADCHECK(pthread_create(&comm->proxyState->thread, NULL, 
                             ncclProxyService, comm->proxyState), 
             "pthread_create");
```

**Trigger**: Called during communicator initialization (`ncclProxyCreate()`).

#### Main Loop

**Location**: `ncclProxyService()` in `src/proxy.cc` (line 1617)

```c
void* ncclProxyService(void* _args) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*) _args;
  
  // 1. Initialize CUDA context
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service] Created CUDA context on device %d", 
         proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
    WARN("[Proxy Service] Failed to set CUDA device %d", proxyState->cudaDev);
  }
  
  // 2. Setup poll descriptors
  struct pollfd pollfds[NCCL_MAX_PROXY_CONNECTIONS+1];
  struct ncclProxyLocalPeer peers[NCCL_MAX_PROXY_CONNECTIONS];
  memset(&peers, 0, sizeof(struct ncclProxyLocalPeer)*NCCL_MAX_PROXY_CONNECTIONS);
  
  for (int s=0; s<NCCL_MAX_PROXY_CONNECTIONS; s++) {
    pollfds[s].fd = -1;
    pollfds[s].events = POLLHUP|POLLIN;
  }
  
  // 3. Add listen socket
  ncclSocketGetFd(proxyState->listenSock, 
                  &pollfds[NCCL_MAX_PROXY_CONNECTIONS].fd);
  pollfds[NCCL_MAX_PROXY_CONNECTIONS].events = POLLIN;
  
  int maxnpeers = 0;
  int npeers = 0;
  int stop = PROXY_RUNNING;
  int asyncOpCount = 0;
  
  // 4. Main service loop
  while (stop == PROXY_RUNNING || npeers > 0) {
    // Check abort flag
    if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0) 
      stop = PROXY_ABORT;
    
    // Poll with timeout (never block forever)
    int ret;
    do {
      ret = poll(pollfds, NCCL_MAX_PROXY_CONNECTIONS+1, 
                 asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      return NULL;
    }
    
    // Handle new connections on listen socket
    if (pollfds[NCCL_MAX_PROXY_CONNECTIONS].revents) {
      // Find free slot and accept connection
      // ...
    }
    
    // Process each peer connection
    for (int s=0; s<maxnpeers; s++) {
      struct ncclProxyLocalPeer* peer = peers+s;
      if (pollfds[s].fd == -1) continue;
      
      // Progress async operations for this peer
      ncclProxyAsyncOp* op = peer->asyncOps;
      while (op != nullptr) {
        ncclProxyAsyncOp* opnext = op->next;
        res = proxyProgressAsync(op, proxyState, &asyncOpCount, 
                                 peer, &connectionPool);
        // ...
      }
      
      // Receive and handle new messages
      if (pollfds[s].revents & POLLIN) {
        // Receive message header
        // Dispatch based on message type
        // Send response
      }
      
      // Handle connection errors
      if (closeConn) {
        // Clean up connection
        // ...
      }
    }
  }
  
  // 5. Cleanup
  // ...
  return NULL;
}
```

**Loop Characteristics**:
- **Event-driven**: Uses `poll()` to wait for socket events
- **Timeout**: 500ms timeout to check abort flag
- **Persistent**: Continues until all peers disconnect
- **Async processing**: Handles async operations in each iteration

#### Termination

**Location**: `ncclProxyStop()` in `src/proxy.cc`

```c
// Set stop flag (checked by service thread)
__atomic_store_n(&proxyState->stop, 1, __ATOMIC_RELEASE);

// Service thread will:
// 1. See stop flag or abort flag
// 2. Complete processing current messages
// 3. Wait for all peers to disconnect
// 4. Exit main loop
// 5. Join with main thread
```

### UDS Service Thread Lifecycle

#### Creation

**Location**: `ncclProxyCreate()` in `src/proxy.cc` (line 1908)

```c
PTHREADCHECK(pthread_create(&comm->proxyState->threadUDS, NULL, 
                             ncclProxyServiceUDS, comm->proxyState), 
             "pthread_create");
```

**Trigger**: Called during communicator initialization if cuMem support enabled.

#### Main Loop

**Location**: `ncclProxyServiceUDS()` in `src/proxy.cc` (line 1821)

```c
void* ncclProxyServiceUDS(void* _args) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*) _args;
  struct pollfd pollfds[1];
  
  // 1. Set CUDA context
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service UDS] Set CUDA context on device %d", 
         proxyState->cudaDev);
  }
  
  // 2. Get UDS file descriptor
  if (ncclIpcSocketGetFd(&proxyState->ipcSock, &pollfds[0].fd) != ncclSuccess) {
    WARN("[Proxy Service UDS] Get listenSock fd fails");
    return NULL;
  }
  pollfds[0].events = POLLIN|POLLHUP;
  
  // 3. Main loop
  while (1) {
    // Poll with timeout
    int ret;
    do {
      ret = poll(pollfds, 1, 500);
    } while (ret < 0 && errno == EINTR);
    
    if (ret < 0) {
      WARN("[Proxy Service UDS] Poll failed: %s", strerror(errno));
      return NULL;
    }
    
    // Check for stop/abort
    if (__atomic_load_n(&proxyState->stop, __ATOMIC_ACQUIRE) || 
        __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE)) 
      break;
    
    // Handle UDS request
    if (pollfds[0].revents) {
      proxyUDSRecvReq(proxyState, pollfds[0].fd);
    }
  }
  
  // 4. Cleanup
  (void)ncclIpcSocketClose(&proxyState->ipcSock);
  INFO(NCCL_PROXY, "[Proxy Service UDS] exit: stop %d abortFlag %d", 
       proxyState->stop, *proxyState->abortFlag);
  return NULL;
}
```

**Loop Characteristics**:
- **Specialized**: Handles cuMem operations only
- **Simple**: Single socket, straightforward poll loop
- **Independent**: Operates independently of service thread

## Synchronization

### Synchronization Mechanisms

#### 1. Mutex + Condition Variable (Ops Pool)

**Purpose**: Coordinate operation posting between main and proxy threads

```c
// Main thread posting operation
pthread_mutex_lock(&pool->mutex);
// Add operation to pool
pool->ops[index] = *op;
pool->nextOps = index;
pthread_cond_signal(&pool->cond);  // Wake progress thread
pthread_mutex_unlock(&pool->mutex);

// Progress thread fetching operations
pthread_mutex_lock(&pool->mutex);
if (pool->nextOps == -1 && !state->stop) {
  pthread_cond_wait(&pool->cond, &pool->mutex);  // Sleep until signaled
}
state->nextOps = pool->nextOps;
pool->nextOps = -1;
pthread_mutex_unlock(&pool->mutex);
```

**Characteristics**:
- **Blocking**: Progress thread blocks when no operations
- **Efficient**: Avoids busy-waiting
- **Batched**: Multiple operations can be posted before thread wakes

#### 2. Atomic Operations (Flags and Counters)

**Purpose**: Lock-free communication of state and progress

```c
// Abort flag (written by main, read by proxies)
__atomic_store_n(proxyState->abortFlag, 1, __ATOMIC_RELEASE);
if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE)) { ... }

// Stop flag (written by main, read by proxies)
__atomic_store_n(&proxyState->stop, 1, __ATOMIC_RELEASE);

// Progress counters (written by proxy, read by GPU/main)
__atomic_store_n(&sub->transmitted, value, __ATOMIC_RELEASE);
uint64_t val = __atomic_load_n(&sub->done, __ATOMIC_ACQUIRE);
```

**Memory Ordering**:
- `__ATOMIC_RELEASE`: Ensures all previous writes visible before this store
- `__ATOMIC_ACQUIRE`: Ensures this load completes before subsequent reads
- Critical for cross-thread visibility

#### 3. Socket-Based Synchronization (Service Thread)

**Purpose**: Block service thread until messages arrive

```c
int ret = poll(pollfds, nfds, timeout);
if (ret > 0) {
  // Data available on socket(s)
  for (int i = 0; i < nfds; i++) {
    if (pollfds[i].revents & POLLIN) {
      // Process message
    }
  }
}
```

**Characteristics**:
- **Event-driven**: Efficient waiting for I/O
- **Multiple sockets**: Can monitor many connections
- **Timeout**: Allows periodic abort check

### Lock Ordering Rules

To prevent deadlocks, locks must be acquired in this order:

1. `opsPool->mutex` (if needed)
2. Transport connection locks (if any)
3. Memory registration locks (if any)

**Never hold multiple locks unless necessary.**

### Thread-Safe Data Structures

| Structure | Protection | Writer(s) | Reader(s) |
|-----------|-----------|-----------|-----------|
| `ncclProxyOpsPool` | Mutex | Main threads | Progress thread |
| `progressState->active` | None (owned by progress) | Progress thread | Progress thread |
| `proxyState->abortFlag` | Atomic | Main thread | All proxy threads |
| `proxyState->stop` | Atomic | Main thread | All proxy threads |
| Progress counters | Atomic | Proxy threads | GPU kernels, main |
| `expectedResponses` | Mutex (implicit) | Service thread | Main thread |

## Thread Interactions

### Main Thread → Progress Thread

**Mechanism**: Shared memory ops pool

```
Main Thread                                Progress Thread
    │                                           │
    │ 1. Create ncclProxyOp                     │
    │                                           │
    │ 2. Lock pool->mutex                       │
    │                                           │
    │ 3. Add to pool->ops[]                     │
    │                                           │
    │ 4. Signal pool->cond                      │
    ├──────────────────────────────────────────▶│
    │                                           │ 5. Wake from cond_wait
    │ 6. Unlock pool->mutex                     │
    │                                           │ 7. Lock pool->mutex
    │                                           │
    │                                           │ 8. Fetch operations
    │                                           │
    │                                           │ 9. Unlock pool->mutex
    │                                           │
    │                                           │ 10. Convert to ProxyArgs
    │                                           │
    │                                           │ 11. Progress operations
    │                                           │
    │ 12. Poll completion counters              │
    │◀──────────────────────────────────────────┤ 13. Update counters (atomic)
    │                                           │
```

### Main Thread → Service Thread

**Mechanism**: Socket-based RPC

```
Main Thread                                Service Thread
    │                                           │
    │ 1. ncclProxyCallAsync()                   │
    │                                           │
    │ 2. Create async op                        │
    │                                           │
    │ 3. Send message on socket                 │
    ├──────────────────────────────────────────▶│
    │                                           │ 4. poll() returns
    │                                           │
    │                                           │ 5. Receive message
    │                                           │
    │                                           │ 6. Process request
    │                                           │    (e.g., setup connection)
    │                                           │
    │                                           │ 7. Send response
    │ 8. ncclPollProxyResponse()                │
    │◀──────────────────────────────────────────┤
    │                                           │
    │ 9. Receive response                       │
    │                                           │
    │ 10. Return to caller                      │
    │                                           │
```

### Progress Thread → GPU Kernel

**Mechanism**: Shared memory counters

```
Progress Thread                            GPU Kernel
    │                                           │
    │ 1. Post network send                      │
    │                                           │
    │ 2. Wait for completion                    │
    │                                           │
    │ 3. Network send complete                  │
    │                                           │
    │ 4. Update transmitted counter (atomic)    │
    ├──────────────────────────────────────────▶│
    │                                           │ 5. Load transmitted (atomic)
    │                                           │
    │                                           │ 6. Check if >= expected
    │                                           │
    │                                           │ 7. Proceed with computation
    │                                           │
    │                                           │ 8. Update done counter (atomic)
    │ 9. Load done counter (atomic)             │
    │◀──────────────────────────────────────────┤
    │                                           │
    │ 10. Check if operation complete           │
    │                                           │
```

### Inter-Proxy Thread Communication

**Minimal**: Proxy threads typically don't communicate directly with each other. They coordinate through shared state in `ncclProxyState`.

**Exception**: Service thread may update connection state that progress thread reads (but not concurrently - happens during setup phase before progress starts).

## Performance Characteristics

### CPU Usage

**Progress Thread**:
- **Active**: High CPU usage (continuous polling)
- **Idle**: Low CPU usage (yields after timeout)
- **Typical**: 50-100% of one CPU core when active

**Service Thread**:
- **Setup phase**: Moderate CPU usage
- **Steady state**: Very low CPU usage (blocked in poll)
- **Typical**: <1% CPU when no setup activity

**UDS Thread**:
- **Always**: Very low CPU usage (blocked in poll)
- **Typical**: <0.1% CPU

### Latency Impact

**Operation Posting Latency**:
- Lock acquisition: ~100-500 ns
- Memory copy: ~50-200 ns
- Signal: ~100-500 ns
- **Total**: ~300-1500 ns

**Operation Progress Latency**:
- Detection by progress thread: 0-500 μs (depends on batching frequency)
- Network operation initiation: 1-10 μs
- **Total added latency**: 1-510 μs

### Scalability

**Per-Communicator Resources**:
- 3 threads (2 if no cuMem support)
- ~1-5 MB memory
- 1 shared memory segment

**Multiple Communicators**:
- Resources scale linearly
- Limited by system thread limit
- Typically support 10s to 100s of communicators

### Tuning Trade-offs

| Parameter | Setting | Throughput | Latency | CPU Usage |
|-----------|---------|------------|---------|-----------|
| `NCCL_PROGRESS_APPENDOP_FREQ` | Low (1-4) | Lower | Lower | Higher |
| `NCCL_PROGRESS_APPENDOP_FREQ` | High (16-32) | Higher | Higher | Lower |
| `NCCL_PROXY_APPEND_BATCH_SIZE` | Low (8-16) | Lower | Lower | Higher |
| `NCCL_PROXY_APPEND_BATCH_SIZE` | High (64-128) | Higher | Higher | Lower |

**Recommendations**:
- **Low latency**: Set `NCCL_PROGRESS_APPENDOP_FREQ=1`, `NCCL_PROXY_APPEND_BATCH_SIZE=8`
- **High throughput**: Use defaults or increase both parameters
- **Balanced**: Default values (8 and 16) work well for most cases

## Thread Affinity and NUMA

### CPU Affinity

RCCL supports CPU affinity for proxy threads:

```c
// Can be set via comm->cpuAffinity
if (CPU_COUNT(&comm->cpuAffinity)) 
  sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
```

**Recommendations**:
- Pin progress thread to cores near NIC
- Pin service thread to any available core
- Avoid cores used by application or GPU kernels

### NUMA Considerations

1. **Allocate proxy structures on correct NUMA node**:
   - Allocated by proxy thread ensures local memory
   - Reduces memory access latency

2. **Network buffer placement**:
   - Should be on NUMA node close to NIC
   - Affects GDR performance significantly

3. **Shared memory placement**:
   - System-dependent
   - May want to use `numactl` for control

## Debugging Threading Issues

### Common Problems

1. **Deadlock**:
   - Check lock ordering
   - Look for mutex held while waiting on condition variable
   - Use `pstack` or `gdb` to examine thread stacks

2. **Race Condition**:
   - Verify atomic operation usage
   - Check memory ordering
   - Use thread sanitizer (`-fsanitize=thread`)

3. **Thread Starvation**:
   - Check CPU affinity settings
   - Verify no priority inversion
   - Monitor CPU usage per thread

### Debugging Tools

1. **Thread dumps**:
   ```bash
   kill -SIGUSR1 <pid>  # If NCCL_PROXY_DUMP_SIGNAL=10
   ```

2. **GDB examination**:
   ```
   (gdb) info threads
   (gdb) thread <n>
   (gdb) bt
   ```

3. **Logging**:
   ```bash
   NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=PROXY
   ```

## Summary

The RCCL proxy threading model provides:

1. **Asynchronous Progress**: Dedicated threads for communication
2. **Efficient Synchronization**: Mix of blocking and non-blocking mechanisms
3. **Scalability**: Per-communicator thread pools
4. **Flexibility**: Tunable parameters for different workloads
5. **Robustness**: Clean shutdown and error handling

Understanding the threading model is crucial for:
- Performance tuning
- Debugging hangs and race conditions
- Adding new features
- Optimizing for specific hardware configurations


