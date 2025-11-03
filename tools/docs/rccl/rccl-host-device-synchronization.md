# RCCL Host-Device Synchronization Architecture

**Date:** November 3, 2025  
**Purpose:** Comprehensive explanation of synchronization mechanisms between RCCL host (CPU) and device (GPU) sides  
**Target Audience:** RCCL developers, performance engineers, system architects

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Work FIFO - Host-to-GPU Communication](#work-fifo---host-to-gpu-communication)
4. [Device Communication Structure](#device-communication-structure)
5. [HIP Streams and Events](#hip-streams-and-events)
6. [Proxy Threads - Async Network Operations](#proxy-threads---async-network-operations)
7. [Memory Ordering and Visibility](#memory-ordering-and-visibility)
8. [Event Callbacks - GPU-to-Host Notification](#event-callbacks---gpu-to-host-notification)
9. [Complete Synchronization Flow](#complete-synchronization-flow)
10. [Performance Considerations](#performance-considerations)
11. [Code References](#code-references)

---

## Overview

RCCL's host-device synchronization architecture enables **asynchronous, high-performance communication** between CPU and GPU while maintaining correctness guarantees. The design balances three critical requirements:

1. **Minimal CPU overhead**: Avoid blocking the host thread
2. **Maximum GPU utilization**: Keep GPU busy with work
3. **Correct ordering**: Ensure operations complete in proper sequence

### Key Mechanisms

```
Host (CPU)                           Device (GPU)
──────────                           ────────────
Application
    ↓
ncclAllReduce()
    ↓
Task Enqueuing ─────────────────→   (queued)
    ↓
Group Launch
    ↓
Kernel Planning
    ↓
Work FIFO Upload ─────────────→     Work FIFO (GPU memory)
    ↓                                    ↓
hipExtLaunchKernel() ──────────→     Kernel reads FIFO
    ↓                                    ↓
Return (non-blocking)                Executes work
    ↓                                    ↓
Event Callbacks ←─────────────       Completion signal
    ↓
Resource Cleanup
```

---

## Architecture Components

### Host-Side Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ncclComm` | CPU Memory | Main communicator state |
| Work FIFO Buffer | CPU Memory (host-pinned) | Staging area for GPU work |
| Kernel Args | CPU Memory | Parameters for kernel launch |
| Event Callbacks | CPU Memory | Async cleanup tasks |
| Proxy Threads | CPU Threads | Handle network operations |

### Device-Side Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ncclDevComm` | GPU Memory | Device-visible communicator state |
| Work FIFO Buffer | GPU Memory | GPU-readable work queue |
| Channel State | GPU Memory | Per-channel execution state |
| FIFO Buffers | GPU Memory | Inter-GPU communication buffers |

### Shared/Mapped Components

| Component | Access Pattern | Purpose |
|-----------|----------------|---------|
| Work FIFO | CPU Write, GPU Read | Work submission |
| Abort Flag | CPU Write, GPU Read | Emergency abort signal |
| Step Counters | GPU Write, GPU Read (peer) | Inter-GPU sync |
| Profiler Counters | GPU Write, CPU Read | Performance tracking |

---

## Work FIFO - Host-to-GPU Communication

### Purpose

The **Work FIFO** (First-In-First-Out queue) is the primary mechanism for the host to communicate work to the GPU kernel **without blocking or synchronization overhead**.

### Structure

**Host-Side State** (`src/include/comm.h:576-585`):
```c
struct ncclComm {
  // ...
  uint32_t workArgsBytes;          // Max size of kernel args
  uint32_t workFifoBytes;          // Size of FIFO buffer (power of 2)
  void* workFifoBuf;               // Host-accessible pointer
  void* workFifoBufDev;            // Device-accessible pointer
  void* workFifoBufGdrHandle;      // GPUDirect RDMA handle (if enabled)
  
  uint32_t workFifoProduced;       // Bytes produced (mod 2^32)
  uint32_t workFifoProducedLastRecorded;  // Last recorded position
  uint32_t workFifoConsumed;       // Bytes consumed (mod 2^32)
  // ...
};
```

**Key Properties**:
- **Power-of-2 size**: Enables fast modulo via masking (`offset & (size-1)`)
- **Circular buffer**: Reuses same memory region
- **Monotonic counters**: 32-bit wraparound is safe due to difference checks
- **Host-pinned memory**: Enables fast CPU writes and GPU reads

### Work FIFO Size

**Default**: Calculated based on requirements, typically 64 KiB - 256 KiB

**Configuration**:
```bash
NCCL_WORK_FIFO_BYTES=262144  # 256 KiB
```

**Calculation Logic**:
```c
// Minimum size based on operations
minSize = maxSimultaneousOps × avgWorkSize

// Round up to power of 2
workFifoBytes = roundUpPow2(minSize)
```

### Work Structures

**Device Kernel Arguments** (`src/include/device.h:610-618`):
```c
struct alignas(16) ncclDevKernelArgs {
  struct ncclDevComm* comm;        // Device communicator
  struct channelMasks channelMask; // Which channels are active
  enum ncclDevWorkStorageType workStorageType;  // Where work is stored
  uint32_t workMask;               // Size mask (for circular buffer)
  void* workBuf;                   // Pointer to work buffer
  // Followed by: struct ncclDevWorkBatch batches[];
};
```

**Work Batch** (`src/include/device.h`):
```c
struct ncclDevWorkBatch {
  uint32_t offsetBase;      // Base offset in work buffer
  uint16_t nWorkItems;      // Number of work items
  uint16_t nextJump;        // Offset to next batch (for linked list)
  uint8_t workType;         // P2P or Collective
  uint8_t funcId;           // Function type (AllReduce, etc.)
  // Followed by work items
};
```

**Work Items**:

For **Collective Operations** (`src/include/device.h:334-380`):
```c
struct alignas(16) ncclDevWorkColl {
  uint32_t channelLo:8, channelHi:8;  // Channel range
  uint32_t nWarps:8;                   // Warps to use
  uint32_t flags;                      // Various control flags
  uint16_t root;                       // Root rank (for rooted ops)
  void* recvbuff;                      // Receive buffer
  void* sendbuff;                      // Send buffer
  // ... protocol-specific fields ...
  size_t count;                        // Element count
  uint32_t algorithm;                  // RING, TREE, etc.
  uint32_t protocol;                   // SIMPLE, LL, LL128
  ncclDevRedOp_t redOp;               // Reduction operation
  // ...
};
```

For **P2P Operations** (`src/include/device.h:287-332`):
```c
struct alignas(16) ncclDevWorkP2p {
  void *sendAddr, *recvAddr;    // Buffer addresses
  size_t sendBytes, recvBytes;  // Transfer sizes
  int sendRank, recvRank;       // Peer ranks
  uint64_t sendOpCount, recvOpCount;  // Operation IDs
  uint8_t nP2pChannels;         // Channels to use
  uint8_t channelBase;          // First channel
  uint8_t nSendChannels, nRecvChannels;  // Active channels
  // ... protocol and registration flags ...
};
```

### Uploading Work to FIFO

**Process** (`src/enqueue.cc:1322-1390`):

```c
static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  size_t workBytes = plan->workBytes;
  void* fifoBufHost = comm->workFifoBuf;
  uint32_t fifoCursor = comm->workFifoProduced;
  uint32_t fifoMask = comm->workFifoBytes - 1;
  
  // 1. Wait for space in FIFO
  NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
  
  // 2. Copy work batches and items to FIFO
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode + 1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16),
        __builtin_assume_aligned(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }
  
  // 3. Update producer counter (visible to GPU)
  comm->workFifoProduced = fifoCursor;
  
  // 4. Memory fence if using GPUDirect RDMA
  if (comm->workFifoBufGdrHandle != nullptr)
    wc_store_fence();  // Write-combining memory barrier
  
  return ncclSuccess;
}
```

**Key Steps**:
1. **Check availability**: Ensure `(produced - consumed) <= bufferSize`
2. **Write work items**: Copy to circular buffer using mask
3. **Update producer**: Atomic/visible write of new position
4. **Fence if needed**: Ensure visibility for GPUDirect access

### Waiting for FIFO Space

**Backpressure Mechanism** (`src/enqueue.cc:1281-1304`):
```c
static ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  
  if (!hasRoom) {
    while (true) {
      // Process event callbacks (may update workFifoConsumed)
      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
      
      // Check again
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
      if (hasRoom) break;
      
      // Yield CPU to avoid busy-wait
      sched_yield();
    }
  }
  return ncclSuccess;
}
```

**When FIFO fills up**:
- Host **blocks** until GPU consumes work
- Event callbacks update `workFifoConsumed`
- Rare in normal operation (indicates GPU is slower than CPU)

---

## Device Communication Structure

### ncclDevComm - GPU-Visible State

The GPU kernel accesses a **simplified, device-side copy** of communicator state.

**Structure** (`src/include/device.h:547-589`):
```c
struct ncclDevComm {
  int rank;                    // This rank's ID
  int nRanks;                  // Total ranks
  int node;                    // Node ID
  int nNodes;                  // Total nodes
  int buffSizes[NCCL_NUM_PROTOCOLS];  // Buffer sizes
  int p2pChunkSize;           // P2P chunk size
  int isAllNvlink;            // Topology flag
  int p2pnChannelsPerPeer;   // P2P channels
  
  int* collNetDenseToUserRank;  // Rank mapping
  volatile uint32_t* abortFlag; // Abort signal from host
  
  struct ncclDevChannel* channels/*[MAXCHANNELS]*/;  // Channel array
  int* rankToLocalRank;       // Rank translation
  
  // Profiling counters (CPU can read)
  struct ncclDevProfiler* workStarted/*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted/*[MAXCHANNELS]*/;
  
  // Optional features...
};
```

**Initialization**: Copied from `ncclComm` during setup, resides in GPU memory.

**Access Pattern**:
- **GPU**: Read-only during kernel execution
- **Host**: Write during initialization, read for profiling
- **Update**: Rebuilt when communicator state changes

### ncclDevCommAndChannels - Full Device State

**Structure** (`src/include/device.h:595-598`):
```c
struct alignas(16) ncclDevCommAndChannels {
  struct ncclDevComm comm;              // Main device comm
  struct ncclDevChannel channels[MAXCHANNELS];  // All channels
};
```

**Memory Layout**:
```
GPU Memory:
├── ncclDevComm (communicator state)
└── ncclDevChannel[MAXCHANNELS]
    ├── Channel 0
    │   ├── Peer connections
    │   ├── Ring/Tree topology
    │   └── Work FIFO state
    ├── Channel 1
    │   └── ...
    └── ...
```

**Host Reference** (`src/include/comm.h:573`):
```c
struct ncclComm {
  // ...
  struct ncclDevComm* devComm;  // Points to GPU memory
  // ...
};
```

---

## HIP Streams and Events

### Stream Management

RCCL uses **HIP streams** for asynchronous kernel launches and operation ordering.

**Per-Communicator Streams** (`src/include/comm.h:632, 650`):
```c
struct ncclComm {
  // ...
  hipStream_t sideStream;   // Non-captured operations
  hipStream_t lastStream;   // Most recent user stream
  // ...
};
```

**Stream Types**:

1. **User Streams**: Provided by application in collective calls
2. **Side Stream**: RCCL-managed for non-captured work
3. **Planner Streams**: Tracked during group operations

**Stream Ordering** (`src/enqueue.cc:2695-2719`):
```c
// Track streams used in a group
if (info->stream != planner->streamRecent || planner->streams == nullptr) {
  planner->streamRecent = info->stream;
  
  // Check for CUDA graph capture
  struct ncclCudaGraph graph;
  NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream));
  
  // All streams in group must be consistent
  if (planner->streams != nullptr && 
      !ncclCudaGraphSame(planner->capturingGraph, graph)) {
    WARN("Streams in group must all be captured or all uncaptured");
    return ncclInvalidUsage;
  }
  
  // Add to stream list
  struct ncclCudaStreamList* l = ncclMemoryStackAlloc<...>(&comm->memScoped);
  l->stream = info->stream;
  l->next = planner->streams;
  planner->streams = l;
  planner->numStreams++;
}
```

### Event-Based Synchronization

RCCL uses **HIP events** for lightweight synchronization without blocking.

**Done Event** (`src/include/comm.h:649`):
```c
struct ncclComm {
  // ...
  hipEvent_t doneEvent;   // Signals kernel completion
  // ...
};
```

**Kernel Launch with Event** (`src/enqueue.cc:1759`):
```c
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // ...
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  void* extra[] = {plan->kernelArgs, &plan->kernelArgsSize};
  
  // Launch kernel with completion event
  CUDACHECK(hipExtLaunchKernel(
    plan->kernelFn,      // Kernel function
    grid, block,         // Dimensions
    extra,               // Args
    0,                   // Shared memory
    launchStream,        // Stream
    NULL,                // Start event (unused)
    comm->doneEvent,     // Completion event
    0                    // Flags
  ));
  // ...
}
```

**Benefits**:
- **Non-blocking**: Host doesn't wait for kernel
- **Lightweight**: GPU hardware signals completion
- **Callback trigger**: Enables async cleanup

---

## Proxy Threads - Async Network Operations

### Purpose

**Proxy threads** handle network operations asynchronously, allowing GPU kernels to focus on computation and local data movement.

### Architecture

```
GPU Kernel ←──────────┐
    ↓                  │
Writes to FIFO        │ (via shared memory)
    ↓                  │
Proxy Thread ─────────┘
    ↓
Network Send/Recv
    ↓
Remote GPU
```

**Why Proxies?**:
1. **GPUs can't do network I/O**: No system calls from GPU
2. **Async progress**: Network ops don't block GPU
3. **Overlap**: Network transfers while GPU computes

### Proxy State

**Per-Communicator** (`src/include/comm.h:598-599`):
```c
struct ncclComm {
  // ...
  struct ncclProxyState* proxyState;  // Proxy thread state
  int proxyRefCountOld;               // Lifecycle management
  // ...
};
```

**Proxy Operations** (`src/include/proxy.h`):
```c
struct ncclProxyOp {
  int channelId;              // Which channel
  int nsteps;                 // Total steps
  uint64_t opCount;           // Operation ID
  enum ncclDevWorkType type;  // Send, Recv, etc.
  
  void* sendbuff;            // Source buffer
  void* recvbuff;            // Dest buffer
  size_t nbytes;             // Transfer size
  
  int peer;                  // Remote rank
  // Network-specific fields...
};
```

### Proxy Communication Flow

**GPU → Proxy**:
1. GPU kernel writes to **ConnFifo** (per-connection FIFO)
2. Proxy thread **polls** ConnFifo for new work
3. Proxy executes network operation
4. Proxy updates completion counter

**Proxy → GPU**:
1. Proxy completes network receive
2. Updates **step counter** (GPU-visible)
3. GPU kernel reads counter, proceeds

**Synchronization** (`src/device/prims_simple.h:140-150`):
```c
// GPU waits for proxy to complete network operation
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
  __builtin_amdgcn_s_sleep(1);
  connStepCache = loadStepValue(connStepPtr);  // Read proxy's counter
  if (checkAbort(flags, Aborted, spins)) break;
}
```

---

## Memory Ordering and Visibility

### Memory Types and Ordering

| Memory Type | CPU Ordering | GPU Ordering | Visibility |
|-------------|--------------|--------------|------------|
| Host Memory | Program order | Via PCIe | CPU writes → GPU reads slow |
| Device Memory | Via PCIe | Coalesced | GPU writes → CPU reads slow |
| Host-Pinned | DMA-able | Via PCIe | Faster CPU↔GPU |
| Fine-Grained | Coherent | Coherent | Slowest, but coherent |
| GPUDirect RDMA | Write-combining | Direct | NIC can read GPU memory |

### Critical Ordering Points

#### 1. Work FIFO Upload

**Host Side** (`src/enqueue.cc:1389-1391`):
```c
comm->workFifoProduced = fifoCursor;

// If using GPUDirect RDMA, ensure visibility
if (comm->workFifoBufGdrHandle != nullptr)
  wc_store_fence();  // Write-combining store fence
```

**Purpose**: Ensure all work data is visible before updating producer counter.

#### 2. GPU Kernel Reads Work FIFO

**Device Side** (kernel startup):
```c
// Read kernel args (includes work buffer pointer)
__shared__ ncclDevKernelArgs args;
if (threadIdx.x == 0)
  args = *kernelArgs;  // Implicit memory fence via shared mem
__syncthreads();
```

**Ordering**: Shared memory write + syncthreads ensures all threads see consistent data.

#### 3. Inter-GPU Step Counters

**Producer (GPU)** (`src/device/prims_simple.h:223-224`):
```c
step += StepPerSlice;
STORE(connStepPtr, step);  // Atomic store with release semantics
```

**Consumer (Peer GPU)** (`src/device/prims_simple.h:142`):
```c
connStepCache = loadStepValue(connStepPtr);

// loadStepValue uses __atomic_load_n with ACQUIRE or RELAXED
#if defined(__gfx1200__) || defined(__gfx1201__)
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
#endif
```

**Memory Model**:
- **Release store**: All prior writes visible before counter update
- **Acquire load**: All subsequent reads see prior writes
- **RDMA**: Hardware ensures visibility across GPUs

#### 4. Abort Flag

**Host Sets** (`src/group.cc:453-460`):
```c
if (!job->destroyFlag && (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) || errorJobAbortFlag)) {
  __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
  __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
  if (job->childAbortFlag) {
    __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
    __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
  }
}
```

**GPU Checks** (`src/device/primitives.h:180`):
```c
int abort = __atomic_load_n((ncclShmem.comm.abortFlag), __ATOMIC_SEQ_CST);
if (abort) {
  __atomic_store_n(&ncclShmem.aborted, abort, __ATOMIC_SEQ_CST);
  return abort;
}
```

**Ordering**: Sequential consistency ensures all GPUs see abort.

### Memory Fences

**Types Used**:
```c
// Device-side fences
__threadfence_system();   // System-wide (CPU + all GPUs visible)
__threadfence_block();    // Workgroup-local
__threadfence();          // GPU-global

// Host-side fences
wc_store_fence();         // Write-combining memory barrier
__atomic_thread_fence(__ATOMIC_RELEASE);  // Release barrier
```

**When to Use**:
- **System fence**: Before signaling remote GPU or CPU
- **Block fence**: Before shared memory sync
- **WC fence**: Before NIC reads GPU memory (GPUDirect)

---

## Event Callbacks - GPU-to-Host Notification

### Purpose

Event callbacks enable **asynchronous resource cleanup** without blocking the application thread.

### Event Callback Queue

**Structure** (`src/include/comm.h:118-122, 639`):
```c
struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;  // Linked list
  cudaEvent_t event;                   // HIP event to query
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

struct ncclComm {
  // ...
  struct ncclIntruQueue<struct ncclCommEventCallback, 
                        &ncclCommEventCallback::next> eventCallbackQueue;
  // ...
};
```

**Purpose**: Queue callbacks tied to HIP events.

### Callback Processing

**Polling Loop** (`src/include/comm.h:753-777`):
```c
static inline ncclResult_t ncclCommPollEventCallbacks(
    struct ncclComm* comm, bool waitSome
  ) {
  ncclResult_t result = ncclSuccess;
  
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    
    cudaError_t ok;
    if (waitSome) {
      // Block until first event completes
      ok = cudaEventSynchronize(cb->event);
      waitSome = false;
    } else {
      // Non-blocking query
      ok = cudaEventQuery(cb->event);
      if (ok == cudaErrorNotReady) break;  // Not ready yet
    }
    
    // Event completed, execute callback
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      CUDACHECKGOTO(ok, result, finish);
    }
  }
  
finish:
  return result;
}
```

**When Called**:
1. **During FIFO wait**: While waiting for GPU to consume work
2. **During group operations**: Between kernel launches
3. **Explicitly**: By application via ncclCommGetAsyncError()

### Example: Work FIFO Cleanup

**Callback Registration** (`src/enqueue.cc:1394-1419`):
```c
// For persistent kernel plans
struct uploadWork_cleanup_t* cleanup = nullptr;
NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
cleanup->base.fn = uploadWork_cleanup_fn;  // Cleanup function
cleanup->hostBuf = fifoBufHost;            // Buffer to free

// Record event after kernel completes
cudaEvent_t event;
CUDACHECKGOTO(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), result, fail);
CUDACHECKGOTO(cudaEventRecord(event, deviceStream), result, fail);
cleanup->base.event = event;

// Enqueue callback
ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cleanup->base);
```

**Cleanup Function** (`src/enqueue.cc:1311-1319`):
```c
ncclResult_t uploadWork_cleanup_fn(
    struct ncclComm* comm, struct ncclCommEventCallback* cb
  ) {
  struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
  free(me->hostBuf);                    // Free host buffer
  CUDACHECK(cudaEventDestroy(me->base.event));  // Destroy event
  free(me);                             // Free callback struct
  return ncclSuccess;
}
```

**Flow**:
1. Host uploads work, records event
2. Host returns (non-blocking)
3. GPU executes kernel
4. Event signals completion
5. Next FIFO wait polls events
6. Callback executes, frees memory

### Benefits

- **Non-blocking**: Host doesn't wait for GPU
- **Memory efficient**: Resources freed when safe
- **Low overhead**: Polling is fast (event query)
- **Ordered**: Callbacks execute in completion order

---

## Complete Synchronization Flow

### Scenario: 256MB AllReduce Between 2 Nodes

Let's trace the complete host-device synchronization for a multi-node AllReduce.

#### Phase 1: Host Enqueuing (CPU)

```
Thread: Application
──────────────────────────────────────────────────────────────
T0: ncclAllReduce(sendbuf, recvbuf, count, ..., stream)
    │
    ├─→ ncclGroupStartInternal() (implicit)
    │   └─→ Increment ncclGroupDepth (0 → 1)
    │
    ├─→ taskAppend(comm, info)
    │   ├─→ ncclGroupCommJoin(comm)
    │   │   └─→ Add comm to thread-local group list
    │   │
    │   ├─→ Allocate ncclTaskColl from memory pool
    │   └─→ ncclTaskCollSorterInsert(&planner->collSorter, task)
    │
    └─→ ncclGroupEndInternal()
        └─→ Depth: 1 → 0, TRIGGER EXECUTION
            │
            ├─→ Create ncclGroupJob
            ├─→ groupLaunch(job)
            │   │
            │   ├─→ [Async Thread] Preconnect network connections
            │   │
            │   ├─→ ncclPrepareTasks(comm)
            │   │   ├─→ Algorithm selection: RING
            │   │   ├─→ Protocol selection: SIMPLE
            │   │   ├─→ Calculate: stepSize, chunkSize, sliceSize
            │   │   └─→ Build work structures
            │   │
            │   └─→ doLaunches(groupCommHead)
```

#### Phase 2: Plan Building and Upload (CPU)

```
Thread: Application (in doLaunches)
──────────────────────────────────────────────────────────────
T1: ncclLaunchPrepare(comm)
    │
    ├─→ Build ncclKernelPlan
    │   ├─→ Schedule collective tasks to channels
    │   ├─→ Create ncclDevWorkColl structures
    │   └─→ Calculate workBytes, threadPerBlock
    │
    └─→ Plan ready
    
T2: ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan)
    │
    └─→ uploadWork(comm, plan)
        │
        ├─→ waitWorkFifoAvailable(comm, needed)
        │   └─→ Check: (produced - consumed) <= fifoBytes
        │       [If full: Poll event callbacks until space available]
        │
        ├─→ Copy work batches to workFifoBuf
        │   └─→ memcpy to (cursor & fifoMask) position
        │
        ├─→ Update comm->workFifoProduced = new_cursor
        │
        └─→ if (GPUDirect): wc_store_fence()
            [Ensures NIC can see work data]
```

#### Phase 3: Kernel Launch (CPU)

```
Thread: Application (in doLaunches)
──────────────────────────────────────────────────────────────
T3: ncclLaunchKernel(comm, plan)
    │
    ├─→ Setup kernel args
    │   ├─→ plan->kernelArgs->comm = devComm
    │   ├─→ plan->kernelArgs->workBuf = workFifoBufDev
    │   └─→ plan->kernelArgs->workMask = fifoMask
    │
    ├─→ dim3 grid = {nChannels, 1, 1}
    ├─→ dim3 block = {threadPerBlock, 1, 1}
    │
    └─→ hipExtLaunchKernel(
        kernelFn,         // ncclDevKernel_Generic_X
        grid, block,
        extra,            // kernelArgs
        0,                // smem
        stream,           // User stream
        NULL,             // Start event
        comm->doneEvent,  // Completion event
        0                 // Flags
    )
    [Returns immediately, kernel queued]

T4: ncclLaunchFinish(comm)
    │
    ├─→ Register event callback for cleanup
    └─→ Return to application

T5: Application continues (non-blocking)
```

#### Phase 4: GPU Execution (GPU)

```
Device: GPU Kernel (per channel)
──────────────────────────────────────────────────────────────
K0: Kernel startup (block for channel N)
    │
    ├─→ Load kernelArgs to shared memory
    │   └─→ __syncthreads() [All threads see same args]
    │
    ├─→ Read work FIFO
    │   ├─→ workBuf = kernelArgs->workBuf
    │   ├─→ Load ncclDevWorkBatch for this channel
    │   └─→ Load ncclDevWorkColl work item
    │
    └─→ Parse work
        ├─→ algorithm = RING
        ├─→ protocol = SIMPLE
        ├─→ sendbuff, recvbuff, count
        └─→ Initialize primitives

K1: Execute Ring AllReduce (Reduce-Scatter phase)
    │
    ├─→ Loop: slices to send
    │   │
    │   ├─→ waitPeer<Recv=1, Send=1>()
    │   │   ├─→ while (connStepCache < step + StepPerSlice)
    │   │   │   └─→ connStepCache = loadStepValue(peer->stepPtr)
    │   │   │       [ATOMIC_ACQUIRE from peer GPU]
    │   │   │
    │   │   └─→ Calculate slice buffer pointer
    │   │       ptr = connEltsFifo + (step % NCCL_STEPS) * connStepSize
    │   │
    │   ├─→ directRecvReduceCopySend(inpIx, outIx, nelem)
    │   │   ├─→ Read from peer (via RDMA or shared mem)
    │   │   ├─→ Reduce with local data
    │   │   ├─→ Write to next peer
    │   │   └─→ [All memory ops use GPU L2 cache]
    │   │
    │   └─→ postPeer<Recv=1, Send=1>()
    │       ├─→ __threadfence_system() [if multi-node]
    │       ├─→ step += StepPerSlice
    │       └─→ STORE(connStepPtr, step)
    │           [ATOMIC_RELEASE, visible to peer GPU]
    │
    └─→ Reduce-Scatter complete

K2: Execute Ring AllReduce (All-Gather phase)
    │
    └─→ Similar to reduce-scatter, but:
        ├─→ Send reduced chunks
        └─→ No reduction, just copy

K3: Kernel completion
    │
    ├─→ Update profiling counters
    │   └─→ workCompleted[channelId]++ [CPU can read]
    │
    └─→ Return
        [HIP runtime signals doneEvent]
```

#### Phase 5: Proxy Thread (if network involved)

```
Thread: Proxy (runs concurrently with GPU)
──────────────────────────────────────────────────────────────
P0: proxyProgressThread(proxyState)
    │
    └─→ while (!stop) {
        │
        ├─→ Poll for proxy operations from GPU
        │   └─→ Check ConnFifo[step % NCCL_STEPS]
        │
        ├─→ if (new work):
        │   ├─→ Execute network send/recv
        │   │   ├─→ ncclNet->isend/irecv (InfiniBand, RoCE, etc.)
        │   │   └─→ [DMA from/to GPU memory via GPUDirect]
        │   │
        │   └─→ On completion:
        │       └─→ Update step counter (GPU-visible)
        │           [GPU can now proceed]
        │
        └─→ sched_yield() [Yield CPU]
    }
```

#### Phase 6: Completion and Cleanup (CPU)

```
Thread: Application (later)
──────────────────────────────────────────────────────────────
C0: [Time passes, application continues]

C1: Next RCCL call or explicit check
    │
    ├─→ waitWorkFifoAvailable(...) 
    │   └─→ ncclCommPollEventCallbacks(comm, waitSome=true)
    │       │
    │       ├─→ Query doneEvent
    │       │   └─→ cudaEventQuery(doneEvent) == cudaSuccess
    │       │
    │       ├─→ Execute callbacks
    │       │   ├─→ uploadWork_cleanup_fn()
    │       │   │   ├─→ free(hostBuf)
    │       │   │   └─→ Update comm->workFifoConsumed
    │       │   │       [Frees FIFO space]
    │       │   │
    │       │   └─→ Other cleanup callbacks
    │       │
    │       └─→ Return
    │
    └─→ FIFO space available for next operation

C2: Application can verify completion:
    │
    └─→ hipStreamSynchronize(stream)
        [Or wait for downstream kernels to complete]
```

### Synchronization Points Summary

| Point | Location | Mechanism | Purpose |
|-------|----------|-----------|---------|
| **Task Queue** | Host | Thread-local list | Batch operations |
| **Work Upload** | Host→GPU | Work FIFO + fence | Send work to kernel |
| **Kernel Launch** | Host | hipExtLaunchKernel | Start GPU execution |
| **Work Read** | GPU | Shared mem + sync | All threads see work |
| **Inter-GPU** | GPU↔GPU | Step counter + atomic | Synchronize peers |
| **Proxy Comm** | GPU↔Proxy | ConnFifo | Network operations |
| **Completion** | GPU→Host | HIP event | Signal finished |
| **Cleanup** | Host | Event callback | Free resources |

---

## Performance Considerations

### Optimization Strategies

#### 1. Minimize Host-Device Synchronization

**Bad**:
```c
for (int i = 0; i < 1000; i++) {
  ncclAllReduce(...);
  hipStreamSynchronize(stream);  // BLOCKS!
}
```

**Good**:
```c
for (int i = 0; i < 1000; i++) {
  ncclAllReduce(...);  // Queued, non-blocking
}
hipStreamSynchronize(stream);  // Sync once at end
```

**Impact**: ~100× faster (microseconds vs. milliseconds per call)

#### 2. Work FIFO Sizing

**Default**: Usually sufficient (64-256 KiB)

**When to increase**:
- **Large-scale AllToAll**: Many simultaneous P2P ops
- **High iteration count**: Back-to-back collectives
- **Persistent kernels**: Long-running GPU work

**Tuning**:
```bash
# Increase to 1 MB
export NCCL_WORK_FIFO_BYTES=1048576
```

**Trade-off**: More memory vs. less blocking

#### 3. Event Callback Overhead

**Fast Path**: No callbacks pending
```c
// Hot path: quick check
if (ncclIntruQueueEmpty(&comm->eventCallbackQueue))
  return;  // No overhead
```

**Slow Path**: Process callbacks
- **cudaEventQuery()**: ~1-5 μs per event
- **Callback execution**: Varies (memory free, etc.)

**Best Practice**: Use callbacks for cleanup, not critical path

#### 4. Memory Ordering Cost

**Operation Costs** (approximate):
| Operation | Latency | When Needed |
|-----------|---------|-------------|
| Store (normal) | ~1 cycle | Local writes |
| __threadfence_block() | ~10 cycles | Shared mem sync |
| __threadfence() | ~100 cycles | GPU-global visibility |
| __threadfence_system() | ~1000 cycles | CPU/remote GPU visibility |
| wc_store_fence() | ~100-1000 cycles | GPUDirect RDMA |

**Minimize system fences**: Only use when crossing domains (GPU→CPU, GPU→NIC)

#### 5. Proxy Thread Efficiency

**Polling overhead**: Minimal (~1-2% CPU)
- Yields CPU when idle
- Wakes on new work

**Network latency**: Dominates (10-100 μs)
- RDMA offload minimizes proxy work
- Pipelining hides latency

---

## Code References

### Key Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/enqueue.cc` | Work management | `uploadWork()`, `ncclLaunchKernel()` |
| `src/group.cc` | Group operations | `groupLaunch()`, `doLaunches()` |
| `src/include/comm.h` | Host structures | `ncclComm`, event callbacks |
| `src/include/device.h` | Device structures | `ncclDevComm`, `ncclDevWorkColl` |
| `src/device/prims_simple.h` | GPU primitives | `waitPeer()`, `postPeer()` |
| `src/proxy.cc` | Proxy threads | `proxyProgressThread()` |
| `src/include/proxy.h` | Proxy structures | `ncclProxyState`, `ncclProxyOp` |

### Critical Functions

#### Host Side
```c
// Work FIFO management
ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced);

// Kernel launch
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);

// Event callbacks
ncclResult_t ncclCommPollEventCallbacks(struct ncclComm* comm, bool waitSome);
```

#### Device Side
```c
// GPU kernel entry
__global__ void ncclDevKernel_Generic_X(struct ncclDevKernelArgs args);

// Synchronization primitives
template<...> void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts);
template<...> void postPeer(bool dataStored);

// Memory ordering
uint64_t loadStepValue(uint64_t* ptr);  // Atomic load with acquire
```

---

## Summary

### Key Takeaways

1. **Work FIFO** enables host-to-GPU communication without blocking
   - Circular buffer in host-pinned memory
   - Producer-consumer with 32-bit wraparound counters
   - GPU reads work asynchronously

2. **HIP Streams/Events** provide lightweight synchronization
   - Non-blocking kernel launches
   - Event callbacks for async cleanup
   - Minimal host overhead

3. **Proxy Threads** handle network operations
   - Async progress independent of GPU
   - RDMA for low-latency data transfer
   - Step counter synchronization with GPU

4. **Memory Ordering** ensures correctness
   - Atomic operations for inter-GPU sync
   - Memory fences for CPU↔GPU visibility
   - Write-combining barriers for GPUDirect

5. **Event Callbacks** enable async resource management
   - Non-blocking cleanup
   - Execution on completion
   - Ordered by event sequence

### Performance Impact

Well-designed host-device synchronization achieves:
- **Sub-microsecond overhead** for operation enqueuing
- **Zero blocking** on host thread (async everywhere)
- **Maximum GPU utilization** via pipelining
- **Low CPU usage** (~1-2% for proxy threads)

### Design Principles

1. **Async Everything**: Never block unless necessary
2. **Zero-Copy**: Share memory, don't copy
3. **Hardware Offload**: Use DMA, RDMA, GPU atomics
4. **Minimal Fences**: Only when crossing domains
5. **Event-Driven**: React to completion, don't poll actively

---

**Document Version:** 1.0  
**Last Updated:** November 3, 2025  
**Author:** RCCL Documentation Team

