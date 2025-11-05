# RCCL Proxy Thread Architecture

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [System Architecture](#system-architecture)
4. [Thread Model](#thread-model)
5. [Operation Flow](#operation-flow)
6. [Design Patterns](#design-patterns)
7. [Memory Model](#memory-model)

## Overview

The RCCL proxy thread architecture implements a producer-consumer model where GPU kernels and main application threads (producers) post communication operations, and dedicated proxy threads (consumers) execute these operations asynchronously. This design enables efficient overlap of computation and communication.

## Design Philosophy

### Core Principles

1. **Asynchronous Progress**: Network operations proceed independently of GPU execution
2. **Lock-Free Where Possible**: Minimize contention between main and proxy threads
3. **Batching**: Group operations for efficiency
4. **Transport Abstraction**: Uniform interface across different transport mechanisms
5. **Fault Tolerance**: Graceful handling of errors and aborts

### Why Proxy Threads?

Network communication operations can be slow compared to GPU computation. Without proxy threads:

- GPU kernels would need to poll network interfaces
- Application threads would block on network I/O
- Communication and computation couldn't overlap efficiently

With proxy threads:

- GPU kernels post operations and continue computing
- Proxy threads handle all network interaction
- Maximum overlap of communication and computation
- Simplified programming model for users

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RCCL Communicator                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌─────────────────────────────┐     │
│  │  GPU Kernels │────────▶│   Main Application Thread   │     │
│  └──────────────┘         └────────────┬────────────────┘     │
│                                         │                       │
│                                         │ Post Operations       │
│                                         ▼                       │
│                            ┌────────────────────────┐          │
│                            │  ncclProxyOpsPool      │          │
│                            │  (Shared Memory)       │          │
│                            └────────┬───────────────┘          │
│                                     │                           │
│                                     │ Consume Operations        │
│                                     ▼                           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Proxy Thread System                         │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │                                                          │ │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ Progress Thread │  │Service Thread│  │ UDS Thread │ │ │
│  │  │                 │  │              │  │            │ │ │
│  │  │ • Progress ops  │  │ • Setup conn │  │ • cuMem    │ │ │
│  │  │ • Poll network  │  │ • Handle RPC │  │ • FD pass  │ │ │
│  │  │ • Batch mgmt    │  │ • Async ops  │  │            │ │ │
│  │  └─────────────────┘  └──────────────┘  └────────────┘ │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                     │                           │
│                                     ▼                           │
│              ┌──────────────────────────────────────┐          │
│              │   Transport Layer (Net/SHM/P2P)     │          │
│              └──────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
ncclComm
  └── ncclProxyState (one per communicator)
      ├── ncclProxyProgressState (progress thread state)
      │   ├── opsPool (shared with main threads)
      │   ├── active (linked list of active operations)
      │   ├── pool (free list of ncclProxyArgs)
      │   └── thread (pthread handle)
      │
      ├── Service Thread
      │   ├── listenSock (TCP socket)
      │   ├── peerSocks (connections to local ranks)
      │   └── connectionPool (transport connections)
      │
      └── UDS Service Thread
          └── ipcSock (Unix domain socket)
```

## Thread Model

### Progress Thread (`ncclProxyProgress`)

**Purpose**: Continuously progress active network operations

**Responsibilities**:
1. Poll for new operations from the ops pool
2. Progress all active operations by calling transport-specific progress functions
3. Manage operation state transitions (Ready → Progress → Done)
4. Handle operation batching and scheduling
5. Yield CPU when idle

**Main Loop**:
```
while (!stop) {
    1. Progress all active operations
    2. If idle or threshold reached:
       - Get new posted operations from pool
       - Append to active list
    3. If no work: yield CPU
}
```

**Key Characteristics**:
- Runs continuously until shutdown
- Lock-free operation progress (uses atomic operations)
- Batches operation fetching for efficiency
- Sets CUDA device context for GPU memory access

### Service Thread (`ncclProxyService`)

**Purpose**: Handle setup, connections, and asynchronous RPC requests

**Responsibilities**:
1. Accept connections from local rank processes
2. Process setup/connect/register messages
3. Establish transport connections
4. Handle deferred/asynchronous operations
5. Manage connection lifecycle

**Main Loop**:
```
while (!stop || has_peers) {
    1. Poll socket file descriptors
    2. Accept new connections
    3. Process async operations for each peer
    4. Handle incoming messages
    5. Clean up closed connections
}
```

**Key Characteristics**:
- Uses `poll()` on socket file descriptors
- Non-blocking with timeout for abort detection
- Manages per-peer async operation queues
- Stays alive until all peers disconnect

### UDS Service Thread (`ncclProxyServiceUDS`)

**Purpose**: Handle CUDA unified memory (cuMem) operations via Unix domain sockets

**Responsibilities**:
1. Listen on Unix domain socket
2. Handle file descriptor passing (cuMem support)
3. Process cuMem-specific requests (GetFd, QueryFd)
4. Support CUDA memory handle operations

**Main Loop**:
```
while (!stop) {
    1. Poll UDS file descriptor
    2. Receive requests with ancillary data
    3. Process cuMem operations
    4. Send responses
}
```

**Key Characteristics**:
- Specialized for CUDA memory operations
- Uses `sendmsg`/`recvmsg` for FD passing
- Separate from main service thread for isolation

## Operation Flow

### 1. Operation Posting (Main Thread)

```
User Calls ncclAllReduce()
    ↓
Kernel Launched (with operation descriptors)
    ↓
ncclProxySaveOp() called
    ↓
Operation added to ncclProxyOpsPool
    ↓
ncclProxyStart() signals progress thread
```

### 2. Operation Consumption (Progress Thread)

```
Progress thread wakes
    ↓
ncclProxyGetPostedOps()
    - Lock ops pool mutex
    - Fetch batch of operations
    - Unlock mutex
    ↓
For each operation:
    ProxyAppend()
        - Convert ncclProxyOp → ncclProxyArgs
        - Group operations by peer/opCount
        - Add to active list
    ↓
progressOps()
    - Call transport-specific progress function
    - Update operation state
    - Remove completed operations
```

### 3. Network Progress (Transport Layer)

```
For each ncclProxyArgs:
    args->progress(proxyState, args)
        ↓
    Transport-specific progress (e.g., sendProxyProgress):
        State: Ready
            - Initialize sub-operations
            - Record start counters
            → Progress
        
        State: Progress
            - Post network sends/receives
            - Poll for completions
            - Update progress counters
            - Flush GPU write buffers if needed
            → Done (when all complete)
        
        State: Done
            - Clean up resources
            - Return for removal from active list
```

### 4. Operation Completion

```
All sub-operations complete
    ↓
Operation state = ncclProxyOpNone
    ↓
removeOp() called
    - Remove from active list
    - Return to free pool
    ↓
GPU kernel detects completion (via shared counters)
    ↓
User operation returns
```

## Design Patterns

### 1. Producer-Consumer Pattern

**Producer**: Main application threads and GPU kernels
**Consumer**: Progress thread
**Shared Queue**: `ncclProxyOpsPool`
**Synchronization**: Mutex + condition variable

**Benefits**:
- Decouples operation posting from execution
- Enables batching
- Minimizes contention

### 2. Object Pool Pattern

**Pools**:
- `ncclProxyOpsPool`: Pre-allocated operation descriptors
- `ncclProxyArgs` pool: Pre-allocated progress structures

**Benefits**:
- Avoids allocation in critical path
- Reduces memory fragmentation
- Predictable memory usage

### 3. State Machine Pattern

**States**: `ncclProxyOpReady` → `ncclProxyOpProgress` → `ncclProxyOpNone`

**Benefits**:
- Clear operation lifecycle
- Simplified error handling
- Easy to debug and trace

### 4. Strategy Pattern

**Strategy**: Transport-specific progress functions
**Interface**: `proxyProgressFunc_t`
**Implementations**: `sendProxyProgress`, `recvProxyProgress` per transport

**Benefits**:
- Uniform interface across transports
- Easy to add new transports
- Transport-specific optimizations

### 5. Batch Processing Pattern

**Batching Points**:
- Operation fetching (`NCCL_PROXY_APPEND_BATCH_SIZE`)
- Operation grouping (by peer and opCount)
- Network operations (multiple steps per progress call)

**Benefits**:
- Amortizes overhead
- Reduces lock contention
- Improves cache locality

## Memory Model

### Shared Memory Regions

1. **ncclProxyOpsPool**:
   - Shared between all local ranks
   - Lock-protected
   - Contains pre-allocated operation descriptors

2. **GPU Buffers**:
   - Registered with network adapters (if GDR enabled)
   - Accessible by both GPU and proxy threads
   - Managed per-connection

3. **Communication Buffers**:
   - Ring/pipeline buffers for protocols
   - Shared or dedicated per connection
   - Memory-mapped if in different processes

### Synchronization Primitives

1. **Mutexes**:
   - `opsPool->mutex`: Protects operation queue
   - Per-connection mutexes for shared resources

2. **Condition Variables**:
   - `opsPool->cond`: Wakes progress thread when operations posted

3. **Atomic Operations**:
   - Operation counters (posted, transmitted, done)
   - Abort flag
   - Stop flag

4. **Memory Fences**:
   - HDP (Host Data Path) flush on AMD GPUs
   - PCIe ordering for RDMA operations

### Memory Visibility

```
Main Thread                    Progress Thread
    │                                 │
    │ Write operation to opsPool      │
    │ (protected by mutex)            │
    ├─────────────────────────────────▶
    │                                 │ Read operation
    │                                 │ Convert to active args
    │                                 │
    │                                 │ Progress operation
    │ GPU writes to buffer            │ Network reads buffer
    │ (GPU visible)                   │ (via GDR or host copy)
    │◀─────────────────────────────────
    │ Read completion counters        │ Write completion counters
    │ (atomic loads)                  │ (atomic stores)
```

## Thread Safety

### Lock-Free Paths

1. **Operation Progress**: Uses atomic counters, no locks needed
2. **Completion Detection**: GPU and proxy use atomic operations
3. **Abort Detection**: Atomic load of abort flag

### Lock-Protected Paths

1. **Operation Posting**: Mutex-protected addition to ops pool
2. **Connection Setup**: Serialized through service thread
3. **Resource Allocation**: Protected by transport-specific locks

### Lock Ordering

To prevent deadlocks, locks are acquired in this order:
1. `opsPool->mutex`
2. Transport-specific connection locks
3. Memory registration locks

## Error Handling

### Error Propagation

1. **Progress Thread Errors**:
   - Store in `proxyState->asyncResult`
   - Main thread polls this on next operation
   - All pending operations aborted

2. **Service Thread Errors**:
   - Close affected connection
   - Mark operations as failed
   - Continue serving other connections

3. **Transport Errors**:
   - Returned through progress function
   - Handled by removeOp()
   - Resources cleaned up

### Abort Handling

When `abortFlag` is set:
1. Progress thread completes current operations
2. Service thread closes connections
3. UDS thread terminates
4. All threads join cleanly
5. Resources deallocated

## Performance Considerations

### Optimization Techniques

1. **Batching**: Reduces lock acquisition frequency
2. **Prefetching**: Prefetch next operation in queue
3. **NUMA Awareness**: Allocate proxy structures on network thread
4. **CPU Affinity**: Can bind proxy threads to specific cores
5. **Polling**: Prefer polling over blocking for low latency

### Tuning Parameters

- `NCCL_PROGRESS_APPENDOP_FREQ`: Balance between latency and overhead
- `NCCL_PROXY_APPEND_BATCH_SIZE`: Larger = better throughput, higher latency
- Thread scheduling priority: Can use real-time scheduling for proxies

### Performance Trade-offs

- **More batching**: Higher throughput, higher latency
- **More progress calls**: Lower latency, higher CPU usage
- **Lock-free operations**: Better scalability, more complex code

## Debugging and Diagnostics

### Proxy State Dumping

Set `NCCL_PROXY_DUMP_SIGNAL=<signal>` to dump proxy state on signal:
- Shows all active operations
- Displays state and progress counters
- Identifies stuck operations

### Logging

Proxy operations logged with `NCCL_PROXY` debug flag:
- Operation posting
- Progress updates
- Completions
- Errors

### Common Issues

1. **Proxy thread hangs**: Check network connectivity, look for stuck operations
2. **Performance regression**: Check batching parameters, CPU affinity
3. **Memory corruption**: Check for race conditions, verify lock ordering


