# RCCL Reduce-Scatter Algorithm Documentation

**Date:** November 10, 2025  
**Source File:** `src/device/reduce_scatter.h`  
**Purpose:** Comprehensive documentation of reduce-scatter collective operation algorithms  
**Target Audience:** RCCL developers, performance engineers, algorithm researchers

---

## Table of Contents
1. [Overview](#overview)
2. [Reduce-Scatter Operation Semantics](#reduce-scatter-operation-semantics)
3. [Algorithm Variants](#algorithm-variants)
4. [Ring Algorithm](#ring-algorithm)
5. [PAT Algorithm](#pat-algorithm-parallel-algorithm-template)
6. [NVLS Algorithm](#nvls-algorithm-nvlink-sharp)
7. [CollNet Direct Algorithm](#collnet-direct-algorithm)
8. [Protocol Variants](#protocol-variants)
9. [Performance Analysis](#performance-analysis)
10. [Implementation Details](#implementation-details)
11. [Tuning and Optimization](#tuning-and-optimization)

---

## Overview

**Reduce-Scatter** is a fundamental collective communication operation that combines reduction and scattering in a single operation. It's widely used in distributed machine learning, particularly in data-parallel training where gradients need to be reduced and distributed.

### Key Characteristics

- **Type:** Collective operation (all ranks participate)
- **Input:** Each rank provides a full array of `N×count` elements
- **Output:** Each rank receives `count` elements (its portion of reduced result)
- **Reduction:** Supports multiple operations (Sum, Max, Min, Prod, etc.)
- **Complexity:** O(N) communication steps for N ranks (ring algorithm)

### Why Reduce-Scatter Matters

In distributed training:
1. **Forward pass:** Compute on local data shard
2. **Backward pass:** Compute gradients
3. **Reduce-Scatter:** Each GPU gets reduced gradients for its parameter shard
4. **Update:** Apply optimizer update
5. **All-Gather:** Reconstruct full model (inverse operation)

This is more efficient than All-Reduce when followed by parameter updates on sharded data.

---

## Reduce-Scatter Operation Semantics

### Mathematical Definition

Given:
- **N ranks:** Ranks 0 through N-1
- **Input buffers:** Each rank `i` has buffer `sendbuf[i]` with `N×count` elements
- **Reduction operation:** ⊕ (e.g., sum, max, min)
- **Output buffers:** Each rank `i` has buffer `recvbuf[i]` with `count` elements

Operation:
```
For rank r:
  recvbuf[r] = sendbuf[0][r×count : (r+1)×count] ⊕ 
               sendbuf[1][r×count : (r+1)×count] ⊕ 
               ... ⊕ 
               sendbuf[N-1][r×count : (r+1)×count]
```

### Visual Example (4 Ranks, 4 Elements Each, Sum Operation)

```
Input State:
┌─────────────────────────────────────┐
│ Rank 0: [10, 20, 30, 40]           │  Segments: A0 B0 C0 D0
│ Rank 1: [ 1,  2,  3,  4]           │  Segments: A1 B1 C1 D1
│ Rank 2: [ 5, 10, 15, 20]           │  Segments: A2 B2 C2 D2
│ Rank 3: [ 2,  4,  6,  8]           │  Segments: A3 B3 C3 D3
└─────────────────────────────────────┘

After Reduce-Scatter (Sum):
┌─────────────────────────────────────┐
│ Rank 0: [18]              ← A0+A1+A2+A3 = 10+1+5+2 = 18
│ Rank 1: [36]              ← B0+B1+B2+B3 = 20+2+10+4 = 36
│ Rank 2: [54]              ← C0+C1+C2+C3 = 30+3+15+6 = 54
│ Rank 3: [72]              ← D0+D1+D2+D3 = 40+4+20+8 = 72
└─────────────────────────────────────┘
```

### Supported Reduction Operations

From `src/collectives.h`:
- **Sum:** Element-wise addition
- **Prod:** Element-wise multiplication  
- **Max:** Element-wise maximum
- **Min:** Element-wise minimum
- **PreMulSum:** Pre-multiplication followed by sum (custom)
- **SumPostDiv:** Sum followed by division (averaging)

### Supported Data Types

- **Integer:** int8, uint8, int32, uint32, int64, uint64
- **Floating Point:** half (fp16), float (fp32), double (fp64)
- **AMD-Specific:** bfloat16, fp8, bfloat8

---

## Algorithm Variants

RCCL implements four distinct reduce-scatter algorithms, each optimized for different scenarios:

| Algorithm | Best For | Network | Complexity | Latency |
|-----------|----------|---------|------------|---------|
| **Ring** | Large messages, regular topology | Any | O(N) steps | High |
| **PAT** | Complex topologies, medium messages | Any | O(log N) phases | Medium |
| **NVLS** | Single-node or multi-node with NVLink Sharp | NVLink/xGMI | O(1) or O(N) | Low |
| **CollNet Direct** | Hardware offload available | InfiniBand SHARP, Slingshot | O(log N) | Very Low |

Selection is automatic based on:
- Message size
- Network topology
- Hardware capabilities
- Tuning thresholds

---

## Ring Algorithm

The ring algorithm is the most common reduce-scatter implementation, providing bandwidth-optimal communication for large messages.

### Algorithm Overview

**Key Idea:** Organize ranks in a logical ring and pass data segments around the ring, progressively reducing each segment.

**Properties:**
- **Steps:** N-1 communication rounds
- **Data Movement:** Each rank sends/receives (N-1)×(count/N)×element_size per round
- **Bandwidth Utilization:** Near-optimal (approaches 100% for large messages)
- **Latency:** O(N) × message_latency

### Ring Topology Setup

**Source Code Location:** Lines 15-60 in `reduce_scatter.h`

```c
ncclRing *ring = &ncclShmem.channel.ring;
int const *ringRanks = ring->userRanks;
const int nranks = ncclShmem.comm.nRanks;
```

**Ring Structure:**
```
Rank 0 → Rank 1 → Rank 2 → Rank 3 → Rank 0
  ↑                                      ↓
  └──────────────────────────────────────┘
```

Each rank knows:
- **prev:** Previous rank in ring (recv from here)
- **next:** Next rank in ring (send to here)
- **ringRanks:** Array mapping ring positions to actual ranks

### Three-Phase Algorithm

#### **Phase 1: Initial Send (Step 0)**

**Code:** Lines 74-82

```c
// step 0: push data to next GPU
rankDest = ringRanks[nranks-1];
offset = dataOffset + rankDest * count;
prims.send(offset, nelem);
```

**Purpose:** Each rank sends its last segment to the next rank.

**Visual (4 ranks):**
```
Round 0: Initial Send
┌─────────┬─────────┬─────────┬─────────┐
│ Rank 0  │ Rank 1  │ Rank 2  │ Rank 3  │
├─────────┼─────────┼─────────┼─────────┤
│   D0 ───┼→       │         │         │
│         │   D1 ───┼→       │         │
│         │         │   D2 ───┼→       │
│    ←────┼─────────┼─────────┼── D3   │
└─────────┴─────────┴─────────┴─────────┘

After Round 0:
Rank 0 has: [A0, B0, C0, D0] + received D3
Rank 1 has: [A1, B1, C1, D1] + received D0
Rank 2 has: [A2, B2, C2, D2] + received D1
Rank 3 has: [A3, B3, C3, D3] + received D2
```

#### **Phase 2: Reduce-Send Loop (Steps 1 to N-2)**

**Code:** Lines 89-100

```c
// k-2 steps: reduce and copy to next GPU
for (int j=2; j<nranks; ++j) {
  rankDest = ringRanks[nranks-j];
  offset = dataOffset + rankDest * count;
  prims.recvReduceSend(offset, nelem);
}
```

**Purpose:** Iteratively receive, reduce, and forward data segments.

**Detailed Operation:**
1. **Receive** data segment from previous rank
2. **Reduce** with local data using reduction operation (e.g., sum)
3. **Send** reduced result to next rank
4. Repeat for N-2 rounds

**Visual (4 ranks, Round 1):**
```
Round 1: Recv-Reduce-Send (processing segment index 2)
┌──────────────────────────────────────────────────────┐
│ Rank 0: recv D3, reduce D0←D0+D3, send C0 → Rank 1 │
│ Rank 1: recv D0, reduce D1←D1+D0, send C1 → Rank 2 │
│ Rank 2: recv D1, reduce D2←D2+D1, send C2 → Rank 3 │
│ Rank 3: recv D2, reduce D3←D3+D2, send C3 → Rank 0 │
└──────────────────────────────────────────────────────┘

After Round 1:
Rank 0: [A0, B0, C0, D0+D3] received C3
Rank 1: [A1, B1, C1, D1+D0] received C0
Rank 2: [A2, B2, C2, D2+D1] received C1
Rank 3: [A3, B3, C3, D3+D2] received C2
```

**Round 2 (processing segment index 1):**
```
Rank 0: recv C3, reduce C0←C0+C3, send B0 → Rank 1
Rank 1: recv C0, reduce C1←C1+C0, send B1 → Rank 2
Rank 2: recv C1, reduce C2←C2+C1, send B2 → Rank 3
Rank 3: recv C2, reduce C3←C3+C2, send B3 → Rank 0

After Round 2:
Rank 0: [A0, B0, C0+C3+C2+C1, D0+D3] received B3
Rank 1: [A1, B1, C1+C0+C3+C2, D1+D0] received B0
Rank 2: [A2, B2, C2+C1+C0+C3, D2+D1] received B1
Rank 3: [A3, B3, C3+C2+C1+C0, D3+D2] received B2
```

#### **Phase 3: Final Reduce-Copy (Step N-1)**

**Code:** Lines 109-117

```c
// step k-1: reduce this buffer and data, which will produce the final result
rankDest = ringRanks[0];
offset = dataOffset + rankDest * count;
prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
```

**Purpose:** Final reduction and copy to output buffer.

**Detailed Operation:**
1. **Receive** last incoming data
2. **Reduce** with local data (now fully accumulated)
3. **Copy** result to output buffer (`recvbuff`)
4. Apply **post-operation** if configured (e.g., division for averaging)

**Visual (4 ranks, Round 3 - Final):**
```
Round 3 (Final): Recv-Reduce-Copy to output
┌────────────────────────────────────────────────────────────┐
│ Rank 0: recv B3+B2+B1, reduce A0←A0+B3+B2+B1, copy to out│
│ Rank 1: recv B0+B3+B2, reduce B1←B1+B0+B3+B2, copy to out│
│ Rank 2: recv B1+B0+B3, reduce C2←C2+B1+B0+B3, copy to out│
│ Rank 3: recv B2+B1+B0, reduce D3←D3+B2+B1+B0, copy to out│
└────────────────────────────────────────────────────────────┘

Actually, let me correct the segments being processed...
```

**Corrected Final Round:**
```
Round 3 (Final): Each rank gets its designated segment fully reduced
Rank 0: recv (A1+A2+A3), reduce A0←A0+A1+A2+A3 → output[A]
Rank 1: recv (B0+B2+B3), reduce B1←B1+B0+B2+B3 → output[B]
Rank 2: recv (C0+C1+C3), reduce C2←C2+C0+C1+C3 → output[C]
Rank 3: recv (D0+D1+D2), reduce D3←D3+D0+D1+D2 → output[D]
```

### Complete Example with Numbers

**Initial:**
```
Rank 0: [10, 20, 30, 40]  (A0=10, B0=20, C0=30, D0=40)
Rank 1: [ 1,  2,  3,  4]  (A1= 1, B1= 2, C1= 3, D1= 4)
Rank 2: [ 5, 10, 15, 20]  (A2= 5, B2=10, C2=15, D2=20)
Rank 3: [ 2,  4,  6,  8]  (A3= 2, B3= 4, C3= 6, D3= 8)
```

**After 3 rounds:**
```
Rank 0 output: [18]  (A0+A1+A2+A3 = 10+1+5+2 = 18)
Rank 1 output: [36]  (B0+B1+B2+B3 = 20+2+10+4 = 36)
Rank 2 output: [54]  (C0+C1+C2+C3 = 30+3+15+6 = 54)
Rank 3 output: [72]  (D0+D1+D2+D3 = 40+4+20+8 = 72)
```

### Pipelining with Chunks

**Code:** Lines 68-70

```c
for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
  nelem = min(chunkCount, channelCount - elemOffset);
  dataOffset = gridOffset + elemOffset;
  // ... run algorithm on this chunk
}
```

**Purpose:** Break large messages into smaller chunks for better pipelining.

**Benefits:**
1. **Overlapping Communication:** While chunk N is being reduced, chunk N+1 is in transit
2. **Lower Latency:** Don't wait for entire message to arrive before processing
3. **Better GPU Utilization:** Keeps compute and communication overlapped

**Visual:**
```
Time →
Chunk 0: [Send] [Recv] [Reduce] [Send] ...
Chunk 1:        [Send] [Recv] [Reduce] [Send] ...
Chunk 2:               [Send] [Recv] [Reduce] [Send] ...
                ↑ Overlap reduces latency
```

### Multi-Channel Parallelism

RCCL uses multiple channels to parallelize the operation:

```c
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), 
                &count, &gridOffset, &channelCount, &chunkCount);
```

**Example with 4 channels:**
```
Channel 0 handles: Elements [0 : N/4)
Channel 1 handles: Elements [N/4 : N/2)
Channel 2 handles: Elements [N/2 : 3N/4)
Channel 3 handles: Elements [3N/4 : N)
```

Each channel runs independently on different GPU blocks, maximizing throughput.

---

## PAT Algorithm (Parallel Algorithm Template)

**Source Code:** Lines 171-234

The PAT algorithm is a more sophisticated approach that can adapt to different network topologies and achieve better performance on irregular networks or medium-sized messages.

### Algorithm Overview

**Key Idea:** Use a dynamic algorithm computed by an "algo thread" that coordinates worker threads to perform optimized communication patterns.

**Architecture:**
- **1 Algorithm Thread:** Computes optimal communication schedule
- **N Worker Threads/Warps:** Execute the schedule
- **Shared Memory:** Communication via `ncclPatShmem` structure

### Thread Organization

```c
static constexpr int nworkers = NCCL_PAT_NWORKERS;
struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
```

**Roles:**
- **Thread 0 to nworkers-1:** Worker threads (execute operations)
- **Thread nworkers:** Algorithm computation thread
- **Other threads:** Idle

### Algorithm Computation (Thread nworkers)

**Code:** Lines 188-203

```c
if (tid == nworkers) { // Algo computation thread
  PatRSAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, 
                             NCCL_PAT_NWORKERS/WARP_SIZE, 
                             channelOffset, channelOffset + channelCount, 
                             count, chunkCount, rank, nranks);
  int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
  
  int step = 0;
  while (1) {
    struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
    int* poll = &ps->flags;
    
    // Wait for workers to finish previous step
    while (__hip_atomic_load(poll, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != 0);
    
    // Compute next operation
    patAlgo.getNextOp(ps);
    
    int last = ps->last;
    step++;
    if (last == 2) break;
  }
}
```

**Function:**
1. Initialize `PatRSAlgorithm` with problem parameters
2. Compute parallel execution factor
3. For each step:
   - Wait for workers to complete previous step
   - Compute next operation parameters
   - Store in shared memory step structure
   - Signal workers to proceed
4. Continue until algorithm complete

### Worker Execution

**Code:** Lines 204-232

```c
else if (tid < nworkers) { // Worker threads
  T *inputBuf = (T*)work->sendbuff;
  T *outputBuf = (T*)work->recvbuff;
  
  int parallelFactor = 0;
  volatile int* pfPtr = &shmem->parallelFactor;
  while (parallelFactor == 0) parallelFactor = *pfPtr;
  
  int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
  int group = tid / groupSize;
  int nGroups = nworkers / groupSize;
  int tidInGroup = tid - group*groupSize;
  
  // Initialize primitives
  Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, false, 0, Pipeline> prims
    (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, 
     inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, 
     primsModePatRs);
  
  int step = group;
  while(1) {
    struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
    int* poll = &ps->flags;
    
    // Wait for algo thread to compute step
    while (__hip_atomic_load(poll, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) == 0);
    
    int last = ps->last;
    prims.patReduce(ps, shmem);
    
    // Signal completion back to algo thread
    if (tidInGroup == 0) 
      __hip_atomic_store(poll, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    
    if (last) break;
    step += nGroups;
  }
}
```

**Function:**
1. Wait for algorithm thread to compute parallel factor
2. Divide workers into groups
3. Initialize communication primitives
4. For each step:
   - Wait for algorithm thread to compute next operation
   - Execute the reduction operation via `prims.patReduce()`
   - Signal completion to algorithm thread
5. Continue until marked as last step

### Synchronization Pattern

```
Algorithm Thread          Worker Threads
      |                         |
      | Compute Step 0          |
      |─────────────────────→   |
      | (set flags=1)           | Wait (spin on flags)
      |                         | Execute Step 0
      |                         | Clear flags
      | Wait (spin on flags)  ←─|
      | Compute Step 1          |
      |─────────────────────→   |
      ...                      ...
```

Uses atomic operations for lock-free synchronization in shared memory.

### Advantages of PAT

1. **Adaptive:** Algorithm computes optimal schedule based on topology
2. **Flexible:** Can handle irregular network topologies
3. **Parallel:** Multiple worker groups can operate simultaneously
4. **Lower Latency:** Better than ring for medium-sized messages

### When PAT is Selected

- Medium message sizes (between LL and Simple thresholds)
- Complex network topologies
- When tuning indicates PAT is faster
- Typically automatic selection via RCCL's auto-tuner

---

## NVLS Algorithm (NVLink Sharp)

**Source Code:** Lines 237-441

NVLS is AMD's/NVIDIA's hardware-accelerated reduce-scatter using NVLink Sharp technology, providing extremely low latency for single-node and multi-node operations.

### Algorithm Overview

**Key Idea:** Use hardware multicast and reduction capabilities in NVLink fabric to perform operations with minimal CPU/GPU involvement.

**Architecture:**
- **Single-Node Path:** Direct NVLS reduction through shared memory
- **Multi-Node Path:** Combines NVLS with network operations
- **Thread Specialization:** Different thread groups handle different phases

### Thread Allocation

**Code:** Lines 313-318

```c
const int nThreadsNetRecv = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE : 6*WARP_SIZE);
const int nThreadsScatter = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8*WARP_SIZE;
const int nThreadsReduce = NCCL_MAX_NTHREADS - nThreadsNetRecv - nThreadsScatter;
const int tidEndNetRecv = nThreadsNetRecv;
const int tidEndScatter = tidEndNetRecv + nThreadsScatter;
const int tidEndReduce = tidEndScatter + nThreadsReduce;
```

**Thread Groups:**
1. **Network Receive:** Threads [0, tidEndNetRecv) - Receive from network
2. **Scatter:** Threads [tidEndNetRecv, tidEndScatter) - Scatter to NVLS heads
3. **Reduce:** Threads [tidEndScatter, tidEndReduce) - Reduce through NVLS

### Single-Node NVLS (Non-Registered)

**Code:** Lines 326-349

```c
if (tid < tidEndScatter) {
  // Scatter phase
  using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
  Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
    prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
          work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
  
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.scatter(offset, nvls->nHeads * count, nelem, count, -1, 0);
  }
} else if (tid < tidEndReduce) {
  // Reduce phase
  using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL, 1, 0>;
  Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
    prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, NULL, 
          NULL, work->recvbuff, work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
  
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.recv(offset, nelem);
  }
}
```

**Operation:**
1. **Scatter Phase:**
   - Each rank scatters its data to all NVLS heads
   - Uses `FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>` (0 inputs, multiple outputs)
   - Data distributed across NVLS fabric

2. **Reduce Phase:**
   - Hardware performs reduction in NVLS memory
   - Each rank receives its reduced segment
   - Uses `FanAsymmetric<1, 0>` (1 input, 0 outputs - just receive)

### Single-Node NVLS (Registered Memory)

**Code:** Lines 351-382

```c
if (tid < tidEndScatter) {
  // Scatter for sync only
  Primitives<...> prims(...);
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    prims.scatter(0, 0, 0, 0, -1, 0);  // Sync only
  }
  prims.gather(0, 0, 0, 0, -1, 0);  // Sync
  
} else if (tid < tidEndReduce) {
  // Direct reduction from registered memory
  Primitives<..., /*Direct=*/1, ...> prims(...);
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    size_t outOffset = gridOffset + elemOffset;
    size_t inpOffset = outOffset + rank * count;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.directRecvCopy(inpOffset, outOffset, nelem);
  }
  prims.send(0, 0);  // Sync
}
```

**Optimization:**
- When buffers are pre-registered with NVLS
- Hardware can directly access memory
- Scatter/gather used only for synchronization
- Significantly lower latency

### Multi-Node NVLS

**Code:** Lines 390-438

**Three-Phase Operation:**

1. **Network Receive Phase** (Lines 390-411):
```c
if (tid < tidEndNetRecv) {
  // Receive reduced data from network
  Primitives<..., FanAsymmetric<1, 0>, ...> prims(...);
  for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; 
       railGridOffset += nChannels * chunkCount) {
    // Calculate which portion belongs to this node
    ssize_t beg = max(railAllBeg, railOneBeg);
    ssize_t end = min(railAllEnd, railOneEnd);
    prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
  }
}
```

2. **Scatter Phase** (Lines 413-424):
```c
if (tid < tidEndScatter) {
  // Scatter local data to NVLS heads
  Primitives<..., FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/1, ...> prims(...);
  for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; 
       railGridOffset += nChannels * chunkCount) {
    Scatterer</*ReduceSendNotRecv=*/true> scat;
    scat.work = work;
    scat.chunkCount = chunkCount;
    scat.railGridOffset = railGridOffset;
    prims.template process</*Recv=*/0, /*Send=*/1>(scat);
  }
}
```

3. **Reduce Phase** (Lines 425-437):
```c
if (tid < tidEndReduce) {
  // Reduce from NVLS and send to network
  Primitives<..., FanSymmetric<1>, /*Direct=*/1, ...> prims(...);
  for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; 
       railGridOffset += nChannels * chunkCount) {
    Scatterer</*ReduceSendNotRecv=*/false> scat;
    scat.work = work;
    scat.chunkCount = chunkCount;
    scat.railGridOffset = railGridOffset;
    prims.template process</*Recv=*/1, /*Send=*/1>(scat);
  }
}
```

**Multi-Node Flow:**
```
Network → Receive → Local NVLS Scatter → NVLS Reduce → Send to Network
   ↓         ↓            ↓                  ↓              ↓
  Data   Unpack     Distribute         Hardware       Pack & Send
                    to Heads          Reduction
```

### NVLS Advantages

1. **Ultra-Low Latency:** Hardware reduction faster than software
2. **Bandwidth Efficient:** Multicast capabilities reduce data movement
3. **CPU Offload:** Minimal CPU/GPU involvement
4. **Scalability:** Excellent for single-node and tightly-coupled multi-node

### When NVLS is Used

- `work->oneNode` flag indicates single-node
- NVLS support available (`nvlsSupport > 0`)
- Appropriate message size
- NVLink/xGMI fabric present

---

## CollNet Direct Algorithm

**Source Code:** Lines 444-604

CollNet Direct uses network hardware offload (e.g., InfiniBand SHARP, HPE Slingshot) to perform reduction operations in the network fabric itself.

### Algorithm Overview

**Key Idea:** Offload the reduction operation to smart switches or NICs, minimizing GPU involvement.

**Architecture:**
- **Multi-Rail Support:** Can use multiple network rails simultaneously
- **Three-Phase Pipeline:** Scatter → Reduce → Receive
- **Hardware Acceleration:** Network performs actual reduction

### Thread Allocation (Dynamic)

**Code:** Lines 527-533

```c
bool isMultiRail = (direct->nHeads > 1);
int nWarps1 = (isMultiRail ? 2 : 0);   // Scatter phase
int nWarps2 = (isMultiRail ? 2 : 1);   // Reduce phase
int nWarps3 = 1;                        // Receive phase

float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
nWarps3 = int(denom*nWarps3);
nWarps2 = int(denom*nWarps2);
nWarps1 = work->nWarps - (nWarps2+nWarps3);
```

**Dynamic allocation based on:**
- Number of rails (single vs multi-rail)
- Total available warps
- Proportional distribution

### Phase 1: Scatter to Network (Lines 538-551)

```c
int tn = nWarps1*WARP_SIZE;
if (tid < tn) {
  // Scatter local inputs to network peers
  Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
    prims(tid, tn, nullptr, direct->heads+1, work->sendbuff, nullptr,
          work->redOpArg, 0*Proto::MaxGroupWidth, 1, 1);
  
  for (ssize_t railGridOffset=0; railGridOffset < nNodes*countPerRank; 
       railGridOffset += nChannels*chunkSize) {
    Scatterer</*ReduceSendNotRecv=*/true> scat;
    scat.work = work;
    scat.chunkSize = chunkSize;
    scat.railGridOffset = railGridOffset;
    prims.template process</*Recv=*/0, /*Send=*/1>(scat, 0, 0);
  }
  return;
}
```

**Function:**
- Send data to network heads (smart switch ports)
- Each rail handles different portion
- Uses custom `Scatterer` to handle multi-rail distribution

### Phase 2: Reduce in Network (Lines 555-575)

```c
tn = nWarps2*WARP_SIZE;
if (tid < tn) {
  if (work->netRegUsed && !hasDn) {
    // Registered memory: just notify
    if (tid == 0) {
      Primitives<...>::sendPeerNotify(direct->out, 1, 1);
    }
  } else {
    // Reduce from peers + local → send to network
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
      prims(tid, tn, direct->heads + 1, &direct->out, nullptr, nullptr,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
    
    for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; 
         railGridOffset += nChannels * chunkSize) {
      Scatterer</*ReduceSendNotRecv=*/false> scat;
      scat.work = work;
      scat.chunkSize = chunkSize;
      scat.railGridOffset = railGridOffset;
      prims.template process</*Recv=*/1, /*Send=*/1>(scat, 0, 0);
    }
  }
  return;
}
```

**Function:**
- Receive from multiple network heads
- Reduce locally if needed
- Send to network for hardware reduction
- Network switch/NIC performs final reduction

### Phase 3: Receive from Network (Lines 578-602)

```c
tn = nWarps3*WARP_SIZE;
if (tid < tn) {
  if (work->netRegUsed) {
    // Registered: just notify
    if (tid == 0) {
      int steps = hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize) : 1;
      Primitives<...>::recvPeerNotify(direct->out, 0, steps);
    }
  } else {
    // Receive reduced result from network
    Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
      prims(tid, tn, &direct->out, nullptr, nullptr, work->recvbuff,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0);
    
    for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; 
         railGridOffset += nChannels * chunkSize) {
      ssize_t railAllBeg = railGridOffset + part * chunkSize;
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
      ssize_t railOneEnd = railOneBeg + countPerRank;
      ssize_t beg = max(railAllBeg, railOneBeg);
      ssize_t end = min(railAllEnd, railOneEnd);
      prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
    }
  }
  return;
}
```

**Function:**
- Receive reduced results from network
- Write to output buffer
- Apply post-operations (e.g., averaging)

### CollNet Pipeline

```
GPU 0 ────┐
GPU 1 ────┤
GPU 2 ────┼──→ Smart Switch ──→ Reduction ──→ Result ──→ GPUs
GPU 3 ────┤      (CollNet)      (Hardware)     (Scatter)
GPU N ────┘

Timeline:
  Scatter Phase  →  Network Reduction  →  Receive Phase
  (GPU-to-NIC)      (Switch Hardware)     (NIC-to-GPU)
```

### CollNet Advantages

1. **Lowest Latency:** Hardware reduction is extremely fast
2. **GPU Offload:** GPU freed for computation
3. **Network Efficiency:** Reduces traffic (reduction at source)
4. **Scalability:** O(log N) complexity in switch tree

### When CollNet is Used

- InfiniBand with SHARP support
- HPE Slingshot with hardware collectives
- Message size above threshold
- Network topology supports it
- Automatically detected and enabled

---

## Protocol Variants

Each algorithm can use different protocols optimized for different message sizes.

### Simple Protocol

**Code Reference:** Lines 150-154, etc.

```c
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    rcclReduceScatterRunRingSimpleProtoImpl(tid, nthreads, work);
  }
};
```

**Characteristics:**
- **Message Size:** Large messages (typically > 1MB)
- **Chunk Size:** Large chunks for bandwidth optimization
- **Slicing:** Multiple slices for pipelining
- **Buffer:** Uses dedicated protocol buffers

**Macro for AMD GPUs (Lines 134-147):**
```c
#if defined(__gfx942__) || defined(__gfx950__)
#define rcclReduceScatterRunRingSimpleProtoImpl(tid, nthreads, work) \
  if(work->rcclUseOneSlice){ \
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS_SINGLE_NODE, 
                              REDUCESCATTER_SLICESTEPS_SINGLE_NODE>; \
    runRing<T, RedOp, Proto>(tid, nthreads, work); \
  } else{ \
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, 
                              REDUCESCATTER_SLICESTEPS>; \
    runRing<T, RedOp, Proto>(tid, nthreads, work); \
  }
#endif
```

**GFX942/GFX950 Optimization:**
- Single-node: Use fewer slices (lower overhead)
- Multi-node: Use more slices (better pipelining)

### LL Protocol (Low Latency)

**Code Reference:** Lines 157-161

```c
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};
```

**Characteristics:**
- **Message Size:** Small messages (typically < 8KB)
- **Latency Optimized:** Minimal protocol overhead
- **Single Slice:** No pipelining overhead
- **In-Order:** Guarantees in-order delivery
- **Reliability:** Built-in error detection

**Use Case:** Latency-sensitive small messages, control messages

### LL128 Protocol

**Code Reference:** Lines 164-168

```c
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
```

**Characteristics:**
- **Message Size:** Medium messages (8KB - 1MB)
- **128-bit Transfers:** Efficient for PCIe Gen3/Gen4
- **Balanced:** Good latency and bandwidth
- **Pipelining:** Some pipelining support

**Use Case:** Most common for typical workloads

---

## Performance Analysis

### Complexity Analysis

| Algorithm | Steps | Data per Link | Latency | Bandwidth |
|-----------|-------|---------------|---------|-----------|
| Ring | N-1 | (N-1)×M/N | α×(N-1) + β×M×(N-1)/N | ~Optimal |
| PAT | O(log N) | Varies | α×O(log N) + β×M | Good |
| NVLS (Single) | O(1) | M/N | α + β×M/N | Excellent |
| NVLS (Multi) | O(N) | M/N | α×N + β×M/N | Excellent |
| CollNet | O(log N) | M/N | α×log(N) + β×M/N | Excellent |

Where:
- N = number of ranks
- M = total message size
- α = latency per message
- β = inverse bandwidth

### Bandwidth Efficiency

**Ring Algorithm:**
```
Effective Bandwidth = Link Bandwidth × (N-1)/N
```

For large N, approaches 100% of link bandwidth.

**Example (8 GPUs, 400 GB/s links):**
```
Effective Bandwidth = 400 × 7/8 = 350 GB/s per link
Aggregate = 350 × 8 = 2.8 TB/s total
```

### Latency Analysis

**Ring Algorithm:**
```
Total Time = (N-1) × (Latency + ChunkSize/Bandwidth)
```

**NVLS Algorithm (Single-Node):**
```
Total Time = Latency + TotalSize/(Bandwidth × NumHeads)
```

**CollNet Algorithm:**
```
Total Time = 2 × Latency + TotalSize/Bandwidth
```

### Message Size Thresholds (Typical)

Based on empirical tuning on AMD MI300:

| Size Range | Preferred Algorithm | Protocol |
|------------|-------------------|----------|
| 0 - 8KB | Ring | LL |
| 8KB - 256KB | Ring or PAT | LL128 |
| 256KB - 2MB | Ring | LL128 or Simple |
| 2MB+ | Ring or NVLS | Simple |

These are configurable via environment variables:
- `RCCL_REDUCE_SCATTER_KERNEL_THRESHOLD`
- `RCCL_LL_THRESHOLD`
- `RCCL_LL128_THRESHOLD`

---

## Implementation Details

### Memory Access Patterns

**Input Buffer (sendbuff):**
```
Layout: [Segment 0 | Segment 1 | ... | Segment N-1]
Size: N × count × sizeof(T)

Ring Algorithm Access Pattern:
Round 0: Read segment (rank-1) mod N
Round 1: Read segment (rank-2) mod N
...
Round N-2: Read segment (rank-(N-1)) mod N
```

**Output Buffer (recvbuff):**
```
Layout: [count elements]
Size: count × sizeof(T)

Write Pattern:
Final round: Write fully reduced segment
```

### Communication Primitives

**Located in:** `src/device/primitives.h`

**Key Operations:**
1. **`send(offset, nelem)`** - Send data to next rank
2. **`recv(offset, nelem)`** - Receive data from previous rank
3. **`recvReduceSend(offset, nelem)`** - Receive, reduce, forward
4. **`recvReduceCopy(offset, dataOffset, nelem)`** - Receive, reduce, copy to output

**Example Usage:**
```c
Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, false, 0, Pipeline>
  prims(tid, nthreads, &ring->prev, &ring->next, 
        work->sendbuff, work->recvbuff, work->redOpArg, 
        0, work->connIndex, work->connIndex);

prims.send(offset, nelem);
prims.recvReduceSend(offset, nelem);
prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
```

### Work Partitioning

**Function:** `ncclCollCbdPart`

```c
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), 
                &count, &gridOffset, &channelCount, &chunkCount);
```

**Calculates:**
- `count`: Total elements per rank
- `gridOffset`: Starting offset for this channel
- `channelCount`: Number of elements this channel handles
- `chunkCount`: Chunk size for pipelining

**Example (1M elements, 4 channels):**
```
Channel 0: offset=0,      count=250K
Channel 1: offset=250K,   count=250K
Channel 2: offset=500K,   count=250K
Channel 3: offset=750K,   count=250K
```

### NPKit Profiling Integration

**Events Collected:**
- `NPKIT_EVENT_TIME_SYNC_CPU` - CPU timestamp sync
- `NPKIT_EVENT_TIME_SYNC_GPU` - GPU timestamp sync
- `NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY` - Algorithm start
- `NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_ENTRY/EXIT` - Send phase
- `NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_ENTRY/EXIT` - Main loop
- `NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_ENTRY/EXIT` - Final phase
- `NPKIT_EVENT_REDUCE_SCATTER_RING_EXIT` - Algorithm complete

**Example:**
```c
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY)
if (tid == 0) {
  NpKit::CollectGpuEvent(NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY, 
                         count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
                         ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
}
#endif
```

Provides nanosecond-precision timing for performance analysis.

---

## Tuning and Optimization

### Auto-Tuning

RCCL automatically selects algorithm and protocol based on:
1. **Message size**
2. **Network topology**
3. **Hardware capabilities**
4. **Measured performance**

**Selection Logic (Simplified):**
```
if (size < LL_THRESHOLD) {
  protocol = LL;
} else if (size < LL128_THRESHOLD) {
  protocol = LL128;
} else {
  protocol = Simple;
}

if (nvlsSupport && oneNode && size > NVLS_THRESHOLD) {
  algorithm = NVLS;
} else if (collNetSupport && size > COLLNET_THRESHOLD) {
  algorithm = CollNet;
} else if (size < PAT_THRESHOLD) {
  algorithm = PAT;
} else {
  algorithm = Ring;
}
```

### Environment Variables

**Algorithm Selection:**
- `NCCL_ALGO=RING` - Force ring algorithm
- `NCCL_ALGO=TREE` - Force tree (not for reduce-scatter)
- `RCCL_FORCE_ENABLE_MSCCL=1` - Enable MSCCL algorithms

**Protocol Selection:**
- `NCCL_PROTO=SIMPLE` - Force simple protocol
- `NCCL_PROTO=LL` - Force low-latency protocol
- `NCCL_PROTO=LL128` - Force LL128 protocol

**Thresholds:**
- `NCCL_LL_THRESHOLD=<bytes>` - LL protocol threshold
- `NCCL_LL128_THRESHOLD=<bytes>` - LL128 protocol threshold
- `RCCL_REDUCE_SCATTER_KERNEL_THRESHOLD=<bytes>` - Kernel selection threshold

**Channel Configuration:**
- `NCCL_NCHANNELS=<n>` - Number of channels to use
- `NCCL_MIN_NCHANNELS=<n>` - Minimum channels
- `NCCL_MAX_NCHANNELS=<n>` - Maximum channels

**NVLS Configuration:**
- `NCCL_NVLS_ENABLE=1` - Enable NVLS
- `NCCL_NVLS_CHUNKSIZE=<bytes>` - NVLS chunk size

**Debug:**
- `NCCL_DEBUG=INFO` - Print algorithm selection
- `NCCL_DEBUG_SUBSYS=COLL` - Collective-specific debug
- `ENABLE_NPKIT=1` - Enable NPKit profiling (compile-time)

### Performance Tips

**For Large Messages (> 1MB):**
1. Use Simple protocol
2. Increase number of channels (up to 32-64)
3. Enable NVLS if single-node
4. Consider CollNet for multi-node

**For Small Messages (< 8KB):**
1. Use LL protocol
2. Reduce number of channels (4-8)
3. Consider batching multiple small operations

**For Medium Messages (8KB - 1MB):**
1. Use LL128 protocol
2. 8-16 channels typically optimal
3. PAT algorithm may help on irregular topologies

**Memory Considerations:**
1. Align buffers to 128 bytes
2. Use contiguous memory when possible
3. Register buffers with NVLS if available
4. Avoid small, frequent allocations

---

## Algorithm Selection Decision Tree

```
┌─────────────────────────────────────────────┐
│  Reduce-Scatter Operation Requested         │
└────────────────┬────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Message Size? │
         └───┬───────┬───┘
             │       │
    < 8KB ───┘       └─── > 8KB
     │                     │
     ▼                     ▼
┌─────────┐         ┌──────────────┐
│LL Proto │         │ Size > 1MB?  │
└────┬────┘         └───┬──────┬───┘
     │                  │      │
     │             Yes ─┘      └─ No
     │              │            │
     │              ▼            ▼
     │      ┌──────────┐   ┌─────────┐
     │      │ Simple   │   │ LL128   │
     │      │ Protocol │   │ Protocol│
     │      └────┬─────┘   └────┬────┘
     │           │              │
     └───────────┴──────┬───────┘
                        │
                        ▼
                 ┌─────────────┐
                 │Single Node? │
                 └──┬────────┬──┘
                    │        │
               Yes ─┘        └─ No
                │              │
                ▼              ▼
         ┌────────────┐  ┌──────────────┐
         │NVLS        │  │CollNet       │
         │Available?  │  │Available?    │
         └──┬─────┬───┘  └──┬───────┬───┘
            │     │         │       │
       Yes ─┘     └─ No Yes─┘       └─ No
        │            │    │            │
        ▼            │    ▼            │
   ┌─────────┐      │ ┌──────────┐    │
   │  NVLS   │      │ │ CollNet  │    │
   │Algorithm│      │ │ Direct   │    │
   └─────────┘      │ └──────────┘    │
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   Medium    │
                      │  Message?   │
                      └──┬──────┬───┘
                         │      │
                    Yes ─┘      └─ No
                     │            │
                     ▼            ▼
              ┌──────────┐  ┌─────────┐
              │   PAT    │  │  Ring   │
              │Algorithm │  │Algorithm│
              └──────────┘  └─────────┘
```

---

## Summary

### Key Takeaways

1. **Ring Algorithm:** Bandwidth-optimal, works everywhere, best for large messages
2. **PAT Algorithm:** Adaptive, good for complex topologies and medium messages
3. **NVLS Algorithm:** Ultra-low latency for NVLink/xGMI systems
4. **CollNet Direct:** Hardware offload for lowest latency on supported networks

### Performance Characteristics

- **Ring:** O(N) steps, near-optimal bandwidth
- **PAT:** O(log N) phases, adaptive scheduling
- **NVLS:** O(1) single-node, hardware-accelerated
- **CollNet:** O(log N), in-network reduction

### Implementation Highlights

- **Pipelining:** Chunks enable overlapping communication and computation
- **Multi-Channel:** Parallel execution across channels maximizes throughput
- **Protocol Variants:** Different protocols optimized for different message sizes
- **NPKit Integration:** Nanosecond-precision profiling for optimization

### Optimization Guidelines

1. **Profile First:** Use NPKit to understand actual performance
2. **Tune Thresholds:** Adjust based on your workload characteristics
3. **Consider Topology:** Algorithm selection depends on network
4. **Memory Matters:** Alignment and registration significantly impact performance
5. **Batch Operations:** Combine small reduce-scatters when possible

---

## Additional Resources

- **Source Files:**
  - `src/device/reduce_scatter.h` - Main implementation
  - `src/device/primitives.h` - Communication primitives
  - `src/device/common.h` - Shared memory structures
  - `src/collectives.cc` - Host-side enqueue logic

- **Related Documentation:**
  - [ncclShmemData.md](../data-types/ncclShmemData.md) - Shared memory structure
  - [RCCL Design Overview](../rccl-design-overview.md) - Overall architecture
  - [NPKit Profiling Guide](../profiling/rccl-profiling-guide.md) - Performance analysis

- **Reference Papers:**
  - "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"
  - "Hierarchical Collectives for Multi-GPU Systems"
  - "In-Network Computing"

---

**Document Version:** 1.0  
**Last Updated:** November 10, 2025  
**Maintainer:** RCCL Documentation Team

