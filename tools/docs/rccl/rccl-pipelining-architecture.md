# RCCL Pipelining Architecture: Groups, Steps, Chunks, and Slices

**Date:** November 3, 2025  
**Purpose:** Comprehensive explanation of RCCL's pipelining and data partitioning mechanisms  
**Target Audience:** RCCL developers, performance engineers, system architects

---

## Table of Contents

1. [Overview](#overview)
2. [Group Operations](#group-operations)
3. [Steps - The FIFO Pipeline](#steps---the-fifo-pipeline)
4. [Chunks - Rank-Based Partitioning](#chunks---rank-based-partitioning)
5. [Slices - Pipeline Units](#slices---pipeline-units)
6. [Complete Example: 256MB AllReduce](#complete-example-256mb-allreduce)
7. [Configuration and Tuning](#configuration-and-tuning)
8. [Code References](#code-references)

---

## Overview

RCCL (ROCm Communication Collectives Library) uses a sophisticated multi-level pipelining architecture to maximize bandwidth and minimize latency in GPU-to-GPU communication. Understanding four key concepts is essential:

- **Groups**: Batch multiple operations for efficient execution
- **Steps**: FIFO-based pipeline for overlapping communication
- **Chunks**: Data division across ranks in the ring algorithm
- **Slices**: Fine-grained pipelining units within chunks

### The Hierarchy

```
Total Data (256 MB)
    ↓
Divided across Channels (16 channels = 16 MB each)
    ↓
Divided into Chunks per Rank (2 ranks = 8 MB per chunk)
    ↓
Divided into Slices for Pipelining (2 slices = 4 MB per slice)
    ↓
Transferred via Steps in FIFO (8 step circular buffer)
    ↓
Multiple Operations batched in Groups
```

---

## Group Operations

### Purpose

Group operations allow **batching multiple collective/P2P operations** for efficient execution and proper synchronization.

### API

```c
ncclGroupStart();
// Multiple operations here
ncclAllReduce(..., comm0, stream0);
ncclAllReduce(..., comm1, stream1);
ncclSend(...);
ncclRecv(...);
ncclGroupEnd();  // Triggers execution
```

### Why Groups Are Necessary

#### 1. **Multi-GPU Single-Thread Management**
When managing multiple GPUs from one thread, operations may require inter-CPU synchronization. Without grouping, deadlocks can occur.

```c
// WITHOUT GROUP - May deadlock!
ncclAllReduce(..., comm0, stream0);  // May sync with other ranks
ncclAllReduce(..., comm1, stream1);  // May sync, causing deadlock

// WITH GROUP - Safe
ncclGroupStart();
ncclAllReduce(..., comm0, stream0);  // Queued
ncclAllReduce(..., comm1, stream1);  // Queued
ncclGroupEnd();  // All execute together
```

#### 2. **Concurrent P2P Operations**
Send/Recv operations that must progress concurrently **require** grouping:

```c
ncclGroupStart();
ncclSend(sendbuff, count, datatype, peer, comm, stream);
ncclRecv(recvbuff, count, datatype, peer, comm, stream);
ncclGroupEnd();
```

#### 3. **Operation Fusion**
Groups allow RCCL to:
- Aggregate multiple operations on the same device
- Optimize kernel launches by batching work
- Reduce overhead by launching fewer kernels

### Implementation Details

**Thread-Local State** (`src/include/group.h`):
```c
__thread int ncclGroupDepth = 0;           // Nesting depth
__thread ncclResult_t ncclGroupError;      // Error tracking
__thread struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum];
__thread struct ncclComm* ncclGroupCommPreconnectHead;
```

**Operation Flow**:
1. `ncclGroupStart()` → Increment `ncclGroupDepth`
2. Operations are **queued**, not executed
3. `ncclGroupEnd()` → When depth reaches 0, trigger execution:
   - Preconnect network connections
   - Prepare tasks (algorithm/protocol selection)
   - Register buffers for RDMA
   - Build kernel plans
   - Launch GPU kernels

**Automatic Grouping**: Even single operations are wrapped in implicit groups:
```c
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  NCCLCHECK(ncclGroupStartInternal());  // Implicit start
  // ... validation and task append ...
  NCCLCHECK(ncclGroupEndInternal());    // Implicit end (triggers if depth=0)
}
```

### Benefits

- **Deadlock Prevention**: Ensures concurrent operations progress together
- **Performance**: Batch processing reduces overhead
- **Correctness**: Proper synchronization across multiple communicators
- **Network Efficiency**: Connection setup happens once per group

---

## Steps - The FIFO Pipeline

### Core Concept

A "**step**" is a slot in a **circular FIFO buffer** that enables pipelining by allowing multiple in-flight transfers between sender and receiver.

### NCCL_STEPS = 8

**Definition** (`src/include/device.h`):
```c
#define NCCL_STEPS 8
```

This constant defines:
- **FIFO depth**: 8 slots in the circular buffer
- **Maximum in-flight transfers**: Up to 8 simultaneous data transfers
- **Buffer partitioning**: Total buffer divided into 8 steps

### Step Size Calculation

For **SIMPLE protocol**:

```c
stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS
         = 4 MiB / 8
         = 512 KiB
```

Where:
- `buffSizes[NCCL_PROTO_SIMPLE]` = 4 MiB (default, see `src/rccl_wrap.cc:438`)
- Each step can hold up to 512 KiB of data

### Circular FIFO Structure

**Data Structure** (`src/include/collectives.h`):
```c
struct ncclConnFifo {
  int mode;        // NORMAL, OFFSET, or PTR mode
  int offset;      // Buffer offset for this step
  ssize_t size;    // Size of data in this step
  void* ptr;       // Pointer to data (in PTR mode)
};
```

**Per-Connection FIFO Array** (`src/include/comm.h`):
```c
struct ncclRecvMem {
  uint64_t tail;
  struct ncclConnFifo connFifo[NCCL_STEPS];  // 8 slots
  int flush;
};
```

### Step Counter and Synchronization

**Step Variables** (`src/device/prims_simple.h`):
```c
uint64_t step;              // Current step (monotonically increasing)
uint64_t *connStepPtr;      // Pointer to peer's step counter
uint64_t connStepCache;     // Cached peer step value
int connStepSize;           // Elements per step
```

### How Steps Enable Pipelining

#### Circular Buffer Mapping
```
Step Counter:     0  1  2  3  4  5  6  7  8  9  10 11 12 ...
FIFO Slot:        0  1  2  3  4  5  6  7  0  1  2  3  4  ...
                  └────────────────────┘  └──────────────┘
                     One full cycle         Next cycle
```

Slot index = `step % NCCL_STEPS`

#### Synchronization Protocol

**1. waitPeer() - Wait for FIFO space** (`src/device/prims_simple.h:140`):
```c
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
  __builtin_amdgcn_s_sleep(1);
  connStepCache = loadStepValue(connStepPtr);
}
```

**Condition explained**:
- **Sender waits**: Until `receiverStep + 8 >= senderStep`
- **Ensures**: Sender doesn't overwrite unconsumed data
- **Max distance**: 8 steps (FIFO depth)

**2. postPeer() - Signal completion** (`src/device/prims_simple.h:222`):
```c
step += StepPerSlice;
STORE(connStepPtr, step);  // Atomic write visible to peer
```

### Step Timeline Example

```
Time  Sender Step  Receiver Step  FIFO Slot  Status
────────────────────────────────────────────────────
t0    0            0              -          Empty
t1    1→           0              0          Sender writes slot 0
t2    2→           0              1          Sender writes slot 1
t3    2            0→1            -          Receiver reads slot 0
t4    3→           1              2          Sender writes slot 2
t5    4→           1→2            3          Both active
t6    5→           2              4          Sender writes slot 4
t7    6→           2→3            5          Pipeline flowing
t8    7→           3              6          Sender writes slot 6
t9    8→           3→4            7          Sender writes slot 7
t10   9→           4              0          Sender wraps to slot 0
t11   9 WAIT       4              -          WAIT: 9 - 4 = 5 < 8, OK
t12   9            4→5            -          Receiver advances
t13   10→          5              1          Sender can proceed
```

**Key insight**: Maximum 8 transfers in flight, creating a continuous pipeline.

### Benefits of 8 Steps

- **Overlap**: Communication and computation overlap efficiently
- **Bandwidth**: Keeps network and GPU busy continuously
- **Latency hiding**: Multiple messages in transit hide individual latencies
- **Memory efficient**: 8 × 512 KiB = 4 MiB buffer size is reasonable

---

## Chunks - Rank-Based Partitioning

### Purpose

In **Ring algorithms** (AllReduce, ReduceScatter, AllGather), data is divided into **chunks**, with each rank responsible for one chunk.

### Chunk Size Calculation

**Parameters**:
```c
#define ALLREDUCE_CHUNKSTEPS 4  // Steps per chunk
```

**Calculation** (`src/enqueue.cc:2273`):
```c
int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
int chunkSteps = ALLREDUCE_CHUNKSTEPS;  // 4 for SIMPLE+RING
int chunkSize = stepSize * chunkSteps;
              = 512 KiB × 4
              = 2 MiB
```

### Chunk Allocation Per Rank

For **Ring AllReduce** with N ranks:

```
Total Data per Channel
        ↓
Divided into N chunks
        ↓
Each chunk = (Channel Data) / N ranks
```

**Example: 2 ranks, 16 MB per channel**:
```
Rank 0: Chunk 0 = 8 MB (responsible for reducing this portion)
Rank 1: Chunk 1 = 8 MB (responsible for reducing this portion)
```

### Ring Algorithm with Chunks

**AllReduce = Reduce-Scatter + AllGather**

#### Reduce-Scatter Phase (N-1 steps):
```
Initial state (2 ranks):
GPU0: [Chunk0] [Chunk1]
GPU1: [Chunk0] [Chunk1]

Step 1: Exchange and reduce
GPU0: [Chunk0] [Chunk1] → Send Chunk0 to GPU1
GPU1: [Chunk0] [Chunk1] → Send Chunk1 to GPU0

Result:
GPU0: [Chunk0]         [Reduced Chunk1]
GPU1: [Reduced Chunk0] [Chunk1]
```

#### AllGather Phase (N-1 steps):
```
Step 1: Exchange reduced chunks
GPU0: Send Reduced Chunk1 → GPU1
GPU1: Send Reduced Chunk0 → GPU0

Result:
GPU0: [Reduced Chunk0] [Reduced Chunk1]
GPU1: [Reduced Chunk0] [Reduced Chunk1]
```

### Chunk Addressing

**Ring Algorithm Implementation** (`src/include/collectives.h:148`):
```c
chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
chunkOffset = chunkId * curChunkSize;
nelem = std::min(remSize - chunkOffset, curChunkSize);
```

**For 2 ranks**:
- Rank 0 (ringIndex=0): Processes chunks in order 0, 1
- Rank 1 (ringIndex=1): Processes chunks in order 1, 0

### Why Chunks?

- **Work Distribution**: Each rank owns one chunk for reduction
- **Bandwidth Optimal**: Ring algorithm is optimal for bandwidth
- **Scalability**: Scales to any number of ranks
- **Load Balance**: Equal chunk sizes ensure balanced work

---

## Slices - Pipeline Units

### Purpose

Each **chunk** is further divided into **slices** to enable fine-grained pipelining within the chunk.

### Slice Size Calculation

**Parameters**:
```c
#define ALLREDUCE_SLICESTEPS 2  // Multi-node (NCCL_STEPS/4)
#define ALLREDUCE_SLICESTEPS_SINGLE_NODE 4  // Single-node
```

**Calculation** (`src/enqueue.cc:2397`):
```c
int chunkSize = 2 MiB;
int chunkSteps = 4;
int sliceSteps = 2;
int sliceSize = (chunkSize / chunkSteps) * sliceSteps;
              = (2 MiB / 4) × 2
              = 512 KiB × 2
              = 1 MiB
```

### Slices Per Chunk

```
slicesPerChunk = chunkSteps / sliceSteps
               = 4 / 2
               = 2 slices per chunk
```

**Chunk decomposition**:
```
Chunk (2 MiB)
    ├── Slice 0 (1 MiB)
    └── Slice 1 (1 MiB)
```

### Dynamic Slice Sizing

Actual slice size may vary based on data alignment and remaining data:

**From** `src/include/collectives.h:151`:
```c
curSliceSize = std::max(
  divUp(nelem / elemSize, 16 * slicePerChunk) * 16,  // Align to 16 elements
  sliceSize / elemSize / 32                           // Minimum size
) * elemSize;
```

**Ensures**:
- **16-element alignment**: Efficient memory access
- **Minimum slice size**: At least `sliceSize / 32` elements
- **Even distribution**: Slices divide chunk evenly

### Slice Timeline in Ring Algorithm

**Per Channel, Per Chunk**:
```
Time    Operation                          Data Transferred
─────────────────────────────────────────────────────────────
t0      GPU0 sends Slice 0 → GPU1         1 MiB
        GPU1 sends Slice 0 → GPU0         1 MiB

t1      GPU0 sends Slice 1 → GPU1         1 MiB
        GPU1 sends Slice 1 → GPU0         1 MiB
        (GPU0 & GPU1 reduce received Slice 0)

t2      Reduction completes
        (GPU0 & GPU1 reduce received Slice 1)
```

**Key**: While Slice 1 is transferring, Slice 0 is being reduced (overlap!).

### Why Slices?

1. **Pipeline Overlap**: Transfer next slice while processing current
2. **Lower Latency**: Smaller messages start faster
3. **Memory Efficiency**: Don't need huge staging buffers
4. **Network Utilization**: Steady stream of data
5. **GPU Efficiency**: GPU reduces data while network transfers next

### Slice vs Step Relationship

```
Slice: Logical unit of work (1 MiB)
  ↓
Transferred via multiple Steps in FIFO
  ↓
Each slice uses StepPerSlice = 2 steps
```

**During transfer** (`src/device/prims_simple.h:199`):
```c
step += StepPerSlice;  // Advance by 2 steps per slice
```

---

## Complete Example: 256MB AllReduce

### Scenario
- **Operation**: ncclAllReduce
- **Data Size**: 256 MB
- **Data Type**: float32 (4 bytes)
- **Ranks**: 2 (GPU0 on Node0, GPU1 on Node1)
- **Algorithm**: Ring
- **Protocol**: SIMPLE

### Step-by-Step Breakdown

#### 1. Group Operation

```c
// User calls (implicit group)
ncclAllReduce(sendbuff, recvbuff, 67108864, ncclFloat32, ncclSum, comm, stream);

// Internally:
ncclGroupStartInternal();          // depth: 0 → 1
  taskAppend(comm, info);          // Queue task
ncclGroupEndInternal();            // depth: 1 → 0, TRIGGER EXECUTION
  groupLaunch();
    ncclPrepareTasks();            // Algorithm/protocol selection
    doLaunches();                  // Build plans, launch kernels
```

#### 2. Channel Division

Assume 16 channels (typical for MI300X):

```
256 MB total
  ↓
16 channels
  ↓
16 MB per channel
```

#### 3. Chunk Division (Per Channel)

```
16 MB per channel
  ↓
2 ranks
  ↓
8 MB per chunk
  ↓
Chunk 0: 8 MB (GPU0's responsibility)
Chunk 1: 8 MB (GPU1's responsibility)
```

#### 4. Slice Division (Per Chunk)

```
8 MB chunk
  ↓
chunkSize = 2 MiB (base unit)
  ↓
Need 4 loops to transfer 8 MB
  ↓
Each 2 MiB chunk divided into 2 slices
  ↓
Slice 0: 1 MiB
Slice 1: 1 MiB
```

#### 5. Step Allocation

```
sliceSize = 1 MiB
stepSize = 512 KiB
  ↓
Steps per slice = 1 MiB / 512 KiB = 2 steps
  ↓
StepPerSlice = 2
```

#### 6. Complete Data Flow

**Loop 1** (First 2 MiB of 8 MB chunk):

```
Reduce-Scatter Phase:
  Slice 0 (1 MiB):
    Step 0: GPU0 → GPU1 (512 KiB, FIFO slot 0)
    Step 1: GPU0 → GPU1 (512 KiB, FIFO slot 1)
    GPU1 receives and reduces into Chunk0
    
  Slice 1 (1 MiB):
    Step 2: GPU0 → GPU1 (512 KiB, FIFO slot 2)
    Step 3: GPU0 → GPU1 (512 KiB, FIFO slot 3)
    GPU1 receives and reduces into Chunk0

Simultaneously:
  Slice 0 (1 MiB):
    Step 0: GPU1 → GPU0 (512 KiB, FIFO slot 0)
    Step 1: GPU1 → GPU0 (512 KiB, FIFO slot 1)
    GPU0 receives and reduces into Chunk1
    
  Slice 1 (1 MiB):
    Step 2: GPU1 → GPU0 (512 KiB, FIFO slot 2)
    Step 3: GPU1 → GPU0 (512 KiB, FIFO slot 3)
    GPU0 receives and reduces into Chunk1
```

**Loops 2-4**: Repeat for remaining 6 MB of the 8 MB chunk

**All-Gather Phase**: Similar pattern, but broadcasting reduced data

#### 7. Total Operations Per Channel

```
Data per channel: 16 MB
Chunk size: 2 MiB
Loops needed: 16 MB / (2 ranks × 2 MiB) = 4 loops

Per loop:
  - 2 slices per chunk
  - 2 chunks (one per rank)
  - Total: 4 slices × 2 phases (reduce-scatter + all-gather) = 8 slices

Total slices per channel: 4 loops × 8 slices = 32 slices
Total data transferred: 32 slices × 1 MiB = 32 MB per channel

Across 16 channels: 16 × 32 MB = 512 MB
  (= 2 × 256 MB, as expected for AllReduce traffic)
```

### Timeline Visualization

```
Channel 0, Loop 1, Reduce-Scatter Phase:

Time(μs)  GPU0                           GPU1
─────────────────────────────────────────────────────────
0         Send Slice0[0:512K] step0 ───→ Recv step0
5         Send Slice0[512K:1M] step1 ──→ Recv step1, Reduce Slice0[0:512K]
10        Send Slice1[0:512K] step2 ───→ Recv step2, Reduce Slice0[512K:1M]
15        Send Slice1[512K:1M] step3 ──→ Recv step3, Reduce Slice1[0:512K]
20        Process next chunk            → Reduce Slice1[512K:1M]

         ←─── Simultaneous reverse direction ────
```

**Key**: Steps, slices, and chunks enable continuous pipelining with minimal idle time.

---

## Configuration and Tuning

### Environment Variables

#### Buffer Size (affects stepSize)
```bash
# Default: 4 MiB
NCCL_BUFFSIZE=8388608  # 8 MiB

# Result:
# stepSize = 8 MiB / 8 = 1 MiB
# chunkSize = 1 MiB × 4 = 4 MiB
# sliceSize = 4 MiB / 4 × 2 = 2 MiB
```

#### Chunk/Slice Steps
Cannot be changed via environment (compile-time constants):
```c
// src/include/collectives.h
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)  // 4
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)  // 2 (multi-node)
```

#### P2P Chunk Size
```bash
# Network P2P (multi-node)
NCCL_P2P_NET_CHUNKSIZE=131072  # 128 KiB (default)

# NVLink/xGMI
NCCL_P2P_NVL_CHUNKSIZE=524288  # 512 KiB (default)

# PCIe
NCCL_P2P_PCI_CHUNKSIZE=131072  # 128 KiB (default)
```

### Tuning Guidelines

#### Large Messages (> 1 GB)
```bash
# Increase buffer size for better pipelining
NCCL_BUFFSIZE=16777216  # 16 MiB
```

Benefits:
- Larger chunks reduce loop overhead
- Fewer kernel launches
- Better amortization of setup costs

#### Small Messages (< 1 MB)
Default settings are optimal. Consider:
- Using LL or LL128 protocols (automatically selected)
- These protocols have different step/chunk/slice characteristics

#### High Latency Networks
```bash
# Larger chunks help hide latency
NCCL_BUFFSIZE=8388608  # 8 MiB
```

#### Low Latency Networks (NVLink, xGMI)
```bash
# Default is good, or slightly smaller for responsiveness
NCCL_BUFFSIZE=4194304  # 4 MiB (default)
```

---

## Code References

### Key Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `src/include/device.h` | Constants and structures | `NCCL_STEPS`, `ncclConnFifo` |
| `src/include/collectives.h` | Algorithm constants | `ALLREDUCE_CHUNKSTEPS`, `ALLREDUCE_SLICESTEPS`, `RingARAlgorithm` |
| `src/include/group.h` | Group operation API | `ncclGroupStart/End`, thread-local state |
| `src/group.cc` | Group implementation | `groupLaunch()`, `doLaunches()` |
| `src/enqueue.cc` | Task queuing & planning | `ncclEnqueueCheck()`, `ncclPrepareTasks()`, size calculations |
| `src/device/prims_simple.h` | GPU primitives | `waitPeer()`, `postPeer()`, step management |
| `src/rccl_wrap.cc` | RCCL-specific config | `rcclSetDefaultBuffSizes()` |

### Critical Functions

#### Group Management
```c
// src/group.cc
ncclResult_t ncclGroupStartInternal()     // Increment depth
ncclResult_t ncclGroupEndInternal()       // Trigger execution
static ncclResult_t groupLaunch()         // Orchestrate execution
static ncclResult_t doLaunches()          // Launch kernels
```

#### Task Preparation
```c
// src/enqueue.cc
ncclResult_t ncclEnqueueCheck()           // API entry point
static ncclResult_t taskAppend()          // Queue task
ncclResult_t ncclPrepareTasks()           // Algo/proto selection
static ncclResult_t setupWork()           // Calculate sizes
```

#### GPU Primitives
```c
// src/device/prims_simple.h
template<...> void waitPeer()             // Wait for FIFO space
template<...> void postPeer()             // Signal completion
```

#### Ring Algorithm
```c
// src/include/collectives.h
class RingARAlgorithm                     // AllReduce ring
  void getNextSendAddr()                  // Calculate send pointer
  void getNextRecvAddr()                  // Calculate recv pointer
```

---

## Summary

### The Big Picture

RCCL's pipelining architecture uses four hierarchical levels:

1. **Groups** (millisecond scale)
   - Batch multiple operations
   - Enable proper synchronization
   - Reduce kernel launch overhead

2. **Chunks** (megabyte scale)
   - Divide work among ranks
   - Enable ring algorithm
   - Balance load across GPUs

3. **Slices** (sub-megabyte scale)
   - Fine-grained pipelining
   - Overlap computation and communication
   - Hide latency

4. **Steps** (microsecond scale)
   - FIFO-based pipelining
   - Enable multiple in-flight transfers
   - Maximize bandwidth utilization

### Key Takeaways

- **Groups are required** for correct multi-GPU and P2P operations
- **8 steps** provide optimal pipeline depth for most networks
- **Chunks** enable the bandwidth-optimal ring algorithm
- **Slices** hide latency through fine-grained pipelining
- All four mechanisms work together to achieve high performance

### Performance Impact

Well-tuned pipelining can achieve:
- **Near-theoretical bandwidth**: 95%+ of peak network bandwidth
- **Low latency**: Sub-microsecond per-message overhead
- **High throughput**: Sustained GB/s across multiple GPUs
- **Efficient overlap**: Communication hidden by computation

---

**Document Version:** 1.0  
**Last Updated:** November 3, 2025  
**Author:** RCCL Documentation Team

