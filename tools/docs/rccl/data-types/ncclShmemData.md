# ncclShmemData Structure Documentation

**Date:** November 10, 2025  
**Source File:** `src/device/common.h`  
**Purpose:** GPU shared memory data structure for RCCL device kernels  
**Target Audience:** RCCL kernel developers, performance engineers, GPU programming experts

---

## Table of Contents
1. [Overview](#overview)
2. [Purpose and Usage](#purpose-and-usage)
3. [Structure Definition](#structure-definition)
4. [Memory Layout and Alignment](#memory-layout-and-alignment)
5. [Detailed Property Documentation](#detailed-property-documentation)
6. [Initialization Process](#initialization-process)
7. [Usage in Collective Operations](#usage-in-collective-operations)
8. [Related Structures](#related-structures)
9. [Performance Considerations](#performance-considerations)

---

## Overview

The `ncclShmemData` structure (accessed via the global `ncclShmem` variable) is the **central GPU shared memory data structure** used by all RCCL device kernels. It serves as a high-speed communication hub for all threads within a CUDA/HIP thread block during collective operations.

### Key Characteristics
- **Shared Memory Location:** Lives in GPU shared memory (fast on-chip memory)
- **Block-Scoped:** One instance per thread block
- **Read-Mostly:** Initialized once at kernel start, then read by all threads
- **Performance Critical:** Avoids expensive global memory accesses
- **Large Structure:** ~2KB+ depending on configuration

### Declaration
```c
extern __shared__ ncclShmemData ncclShmem;
```

This declares `ncclShmem` as an external dynamically-sized shared memory allocation that is shared across all threads in a thread block.

---

## Purpose and Usage

### Primary Purposes

1. **Fast Data Access**
   - Provides threads with fast access to communicator and channel information
   - Avoids repeated global memory reads (100x+ slower than shared memory)
   - Enables efficient coordination between threads in a block

2. **Communication Hub**
   - Centralizes all kernel state in one location
   - Provides topology information (ring, tree, NVLS)
   - Stores work descriptors for current operations

3. **Thread Coordination**
   - Synchronization barriers for groups
   - Shared scratch space for algorithms
   - Work distribution information

4. **Profiling and Debug**
   - NPKit event collection contexts
   - CollTrace for operation tracking
   - Performance profiler integration

### Lifecycle

```
Kernel Launch
     ↓
Stage 1: Copy kernel args to ncclShmem.args
     ↓
Stage 2: Calculate ncclShmem.channelId from blockIdx
     ↓
Stage 3: Load ncclShmem.comm from global memory (Warp 0)
     ↓
Stage 4: Load ncclShmem.channel from global memory (Warp 1)
     ↓
Stage 5: Load work descriptors to ncclShmem.workStorage (Warps 2+)
     ↓
__syncthreads() - Publish all data
     ↓
Run collective operations (all threads read from ncclShmem)
     ↓
Kernel Exit
```

---

## Structure Definition

Located at lines 135-169 in `src/device/common.h`:

```c
struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];

  alignas(16) char workStorage[1024];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
#ifdef ENABLE_COLLTRACE
  struct ncclCollTrace* collTrace;
  union ncclCollTraceTail* collTraceTail;
#endif
#ifdef ENABLE_PROFILING
  struct ncclProf prof;
#endif
#ifdef ENABLE_FAULT_INJECTION
  uint64_t faults;
#endif
  uint64_t barrier_pat;
};

extern __shared__ ncclShmemData ncclShmem;
```

---

## Memory Layout and Alignment

### Alignment Requirements

The structure uses careful alignment to ensure optimal memory access patterns:

- **16-byte alignment** for `comm`, `channel`, `workStorage`, and `devicePlugin`
  - Enables efficient vector loads (128-bit loads on GPUs)
  - Matches cache line boundaries

### Size Considerations

Typical size breakdown:
- `ncclDevKernelArgs`: ~128 bytes
- `ncclDevComm`: ~512 bytes (depends on configuration)
- `ncclDevChannel`: ~512 bytes (depends on configuration)
- `workStorage`: 1024 bytes
- `groups`: Variable (NCCL_MAX_GROUPS * group size)
- Other fields: ~100 bytes
- **Total**: ~2KB+ (varies by configuration)

### Shared Memory Limitations

GPUs have limited shared memory per block:
- **AMD MI300 series**: 64KB per CU
- **NVIDIA H100**: 228KB per SM
- RCCL must balance shared memory usage with occupancy

---

## Detailed Property Documentation

### 1. Kernel Arguments

#### `args` (struct ncclDevKernelArgs)
- **Line:** 136
- **Purpose:** Copy of kernel launch arguments
- **Size:** ~128 bytes
- **Contains:**
  - `comm`: Pointer to device communicator
  - `channelMask`: Bitmask of active channels
  - `workStorageType`: Where work descriptors are stored
  - `workMask` / `workBuf`: Work FIFO information
- **Initialization:** Copied cooperatively by all threads in first stage
- **Usage:** Avoids accessing parameter space (slow on some architectures)

**Code Reference (Initialization):**
```c
// Lines 495-497 in common.h
if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
  ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
}
```

---

### 2. Channel Identification

#### `channelId` (int)
- **Line:** 137
- **Purpose:** Which channel this thread block is processing
- **Range:** 0 to MAXCHANNELS-1
- **Calculation:** Computed from `blockIdx.x` and `args.channelMask`
- **Initialization:** Warp 0 calculates this using population count
- **Usage:** Used to index into device communicator's channel array

**Code Reference (Calculation):**
```c
// Lines 506-527 in common.h
// Maps blockIdx.x to channelId using channelMask bitmask
for (int i = 0; i < num; i++) {
  if (args->channelMask.masks[i] & (1ull<<x)) {
    y = __popcll(args->channelMask.masks[i] & ((1ull<<x)-1));
    if (blockIdx.x == y) {
      ncclShmem.channelId = x + total;
      break;
    }
  }
  total = total + __popcll(args->channelMask.masks[i]);
}
```

---

### 3. Abort Handling

#### `aborted` (int)
- **Line:** 138
- **Purpose:** Flag indicating if operations should abort
- **Values:** 0 = normal, non-zero = abort requested
- **Initialization:** Set to 0 at kernel start
- **Usage:** Checked in main loop to exit early on errors
- **Thread Safety:** Written by thread 0, read by all threads

**Code Reference (Usage):**
```c
// Line 598 in common.h
while (ncclShmem.aborted == 0) {
  // Run collective operations
}
```

---

### 4. Device Communicator (Core Data)

#### `comm` (struct ncclDevComm)
- **Line:** 139
- **Alignment:** 16 bytes
- **Purpose:** Device-side communicator information
- **Size:** ~512 bytes
- **Contains:**
  - `rank`: This GPU's rank
  - `nRanks`: Total number of ranks
  - `node`: Node ID
  - `nNodes`: Total nodes
  - `buffSizes`: Buffer sizes per protocol
  - `channels`: Pointer to all channels
  - `collNetDenseToUserRank`: Rank mappings
  - `abortFlag`: Device pointer to abort flag
  - NPKit profiling contexts
- **Initialization:** Loaded by Warp 0 from global memory
- **Usage:** Accessed throughout collective operations

**Code Reference (Initialization):**
```c
// Lines 557-563 in common.h
case 0:
  { void* dst = &ncclShmem.comm;
    void* src = ncclShmem.args.comm;
    int bytes = sizeof(ncclDevComm);
    copyToShmem16(tid, dst, src, bytes);
  } break;
```

**Example Usage in Reduce-Scatter:**
```c
// Lines 21 in reduce_scatter.h
const int nranks = ncclShmem.comm.nRanks;
```

---

### 5. Device Channel (Topology Data)

#### `channel` (struct ncclDevChannel)
- **Line:** 140
- **Alignment:** 16 bytes
- **Purpose:** Channel-specific topology and connection information
- **Size:** ~512 bytes
- **Contains:**
  - `peers`: Array of peer connections
  - `ring`: Ring algorithm topology (prev/next ranks, userRanks)
  - `tree`: Tree algorithm topology
  - `nvls`: NVLS (NVLink Sharp) topology
  - `collnetDirect`: CollNet direct topology
  - `workFifoDone`: Work completion tracking
  - `workCounter`: Completed work counter
- **Initialization:** Loaded by Warp 1 from global memory
- **Usage:** Provides topology for communication patterns

**Code Reference (Initialization):**
```c
// Lines 565-571 in common.h
case 1:
  { void* dst = &ncclShmem.channel;
    void* src = &((ncclDevCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
    int bytes = sizeof(ncclDevChannel);
    copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
  } break;
```

**Example Usage in Reduce-Scatter:**
```c
// Lines 19-20 in reduce_scatter.h
ncclRing *ring = &ncclShmem.channel.ring;
int const *ringRanks = ring->userRanks;
```

---

### 6. Work Batch Management

#### `batchIx` (int)
- **Line:** 142
- **Purpose:** Current batch index being processed
- **Usage:** Tracks which batch of work is active

#### `nextBatchIx` (int)
- **Line:** 142
- **Purpose:** Next batch index to process
- **Special Value:** -1 indicates no more batches
- **Usage:** Controls kernel execution loop

**Code Reference (Usage):**
```c
// Lines 621-622 in common.h
if (ncclShmem.nextBatchIx == -1) break;
int batchIx = ncclShmem.nextBatchIx;
```

---

### 7. Work Type and Function

#### `workType` (enum ncclDevWorkType)
- **Line:** 143
- **Purpose:** Type of work being performed
- **Values:**
  - `ncclDevWorkTypeP2p`: Point-to-point operations
  - `ncclDevWorkTypeColl`: Collective operations
  - `ncclDevWorkTypeCollReg`: Collective with registration
- **Usage:** Determines work descriptor structure

#### `funcId` (uint16_t)
- **Line:** 145
- **Purpose:** Function ID identifying the collective operation
- **Usage:** Indexes into function dispatch table
- **Values:** Enumeration of all RCCL functions

**Code Reference (Function Dispatch):**
```c
// Lines 601-618 in common.h
if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
  SpecializedRunWorkBatch().run();
} else {
#ifdef USE_INDIRECT_FUNCTION_CALL
  ncclDevFuncTable[ncclShmem.funcId]();
#else
  NCCL_CALL_FUNCTIONS(ncclShmem.funcId);
#endif
}
```

---

### 8. Work Management

#### `nWorks` (int)
- **Line:** 146
- **Purpose:** Number of work items in current batch
- **Usage:** Iteration count for processing work items

#### `workSize` (int)
- **Line:** 147
- **Purpose:** Size of each work descriptor in bytes
- **Values:**
  - `sizeof(ncclDevWorkP2p)` for P2P
  - `sizeof(ncclDevWorkColl)` for collectives
- **Usage:** Pointer arithmetic to access work items

#### `workCounter` (uint64_t)
- **Line:** 148
- **Purpose:** Counter of completed work items
- **Usage:** Profiler synchronization

---

### 9. Work Storage (Work Descriptors)

#### `workStorage` (char[1024])
- **Line:** 153
- **Alignment:** 16 bytes
- **Purpose:** Storage for work descriptors
- **Capacity:** Up to ~16 small work items or 2-3 large ones
- **Contains:** `ncclDevWorkColl` or `ncclDevWorkP2p` structures
- **Initialization:** Loaded by Warps 2+ from parameter space or work FIFO
- **Usage:** Holds all parameters for current operations

**Structure Layout:**
```
workStorage[0..workSize-1]:        Work item 0
workStorage[workSize..2*workSize]: Work item 1
...
```

**Code Reference (Access):**
```c
// Lines 438-439 in common.h
struct ncclDevWorkColl* work = 
  (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
```

**Example ncclDevWorkColl contents:**
- `sendbuff`, `recvbuff`: Buffer pointers
- `count`: Element count
- `channelLo`, `channelHi`: Channel range
- `nWarps`: Warps to use
- `redOpArg`: Reduction operation argument
- Protocol-specific parameters

---

### 10. Thread Groups

#### `groups` (struct ncclShmemGroup[NCCL_MAX_GROUPS])
- **Line:** 150
- **Purpose:** Per-group scratch space and synchronization
- **Typical NCCL_MAX_GROUPS:** 8
- **Usage:** Complex algorithms with multiple thread groups

**ncclShmemGroup structure:**
```c
struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* userAcc;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  void* acc;
  uint64_t barrier;
  int32_t dstSizes[NCCL_MAX_ARITY+1];
};
```

**Initialization:**
```c
// Lines 530-533 in common.h
if (tid < WARP_SIZE + NCCL_MAX_GROUPS) {
  if (tid == WARP_SIZE) ncclShmem.barrier_pat = 0;
  ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
}
```

---

### 11. Reduction Operation Arguments

#### `redOpArgs` (uint64_t[NCCL_MAX_NVLS_ARITY+1])
- **Line:** 151
- **Purpose:** Storage for reduction operation arguments
- **Size:** Array sized for maximum NVLS arity
- **Usage:** Custom reduction operations with arguments

---

### 12. Profiling Support

#### `profilerEnabled` (bool)
- **Line:** 149
- **Purpose:** Whether profiler is active for this work
- **Usage:** Conditional profiler instrumentation

#### `prof` (struct ncclProf) [Conditional]
- **Line:** 163
- **Conditional:** Only if `ENABLE_PROFILING` is defined
- **Purpose:** In-kernel profiling data collection
- **Contains:**
  - `count`: Number of recorded events
  - `seq`: Sequence number
  - `elem[]`: Array of timestamp entries

**Code Reference (Usage):**
```c
// Lines 245-251 in common.h
#ifdef ENABLE_PROFILING
#define __insert_timestamp(line_num) do { \
  if (ncclShmem.prof.count < PROFILE_NUM_ITEMS) { \
    ncclShmem.prof.elem[ncclShmem.prof.count].line = line_num; \
    ncclShmem.prof.elem[ncclShmem.prof.count].timeStamp = wall_clock64(); \
    ncclShmem.prof.count++; \
  } \
} while(0);
#endif
```

---

### 13. CollTrace Support

#### `collTrace` (struct ncclCollTrace*) [Conditional]
- **Line:** 159
- **Conditional:** Only if `ENABLE_COLLTRACE` is defined
- **Purpose:** Pointer to collective operation trace buffer
- **Usage:** Detailed operation tracking for debugging

#### `collTraceTail` (union ncclCollTraceTail*) [Conditional]
- **Line:** 160
- **Purpose:** Tail pointer for trace circular buffer
- **Usage:** Lock-free circular buffer management

**Code Reference (Initialization):**
```c
// Lines 582-585 in common.h
#ifdef ENABLE_COLLTRACE
if (tid == 0) {
  ncclShmem.collTrace = args->comm->collTrace + 
                        COLLTRACE_NUM_ITEMS*ncclShmem.channelId;
  ncclShmem.collTraceTail = args->comm->collTraceTail + ncclShmem.channelId;
}
#endif
```

---

### 14. NPKit Profiling

NPKit (Network Profiling Kit) provides detailed performance analysis. The contexts are accessed through `ncclShmem.comm.npKitEventCollectContexts`:

**Usage Example from Reduce-Scatter:**
```c
// Lines 32-41 in reduce_scatter.h
#if defined(ENABLE_NPKIT)
int npKitCtxIdx = ncclShmem.channelId;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
if (tid == 0) {
  NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, 
                         NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
                         ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
}
#endif
```

---

### 15. Device Plugin Extension

#### `devicePlugin` (union)
- **Line:** 155-157
- **Alignment:** 16 bytes
- **Purpose:** Extension point for device plugins
- **Current member:** `unpackShmem` for network unpack operations
- **Usage:** Plugin-specific shared memory

---

### 16. Fault Injection

#### `faults` (uint64_t) [Conditional]
- **Line:** 166
- **Conditional:** Only if `ENABLE_FAULT_INJECTION` is defined
- **Purpose:** Control flags for fault injection testing
- **Usage:** Testing error handling paths

**Code Reference (Initialization):**
```c
// Lines 536-540 in common.h
#ifdef ENABLE_FAULT_INJECTION
if (tid == 2*WARP_SIZE) {
  ncclShmem.faults = args->comm->faults;
}
#endif
```

---

### 17. Barriers and Synchronization

#### `barrier_pat` (uint64_t)
- **Line:** 168
- **Purpose:** Barrier for PAT (Parallel Algorithm Template) operations
- **Usage:** Complex multi-phase algorithms

#### `directMode` (uint8_t)
- **Line:** 144
- **Purpose:** Whether direct mode is enabled
- **Usage:** Optimization flag for direct peer access

---

## Initialization Process

The initialization of `ncclShmem` is a carefully orchestrated multi-stage process that happens at the start of every RCCL kernel.

### Stage-by-Stage Breakdown

#### **Stage 1: Copy Kernel Arguments (All Threads)**
```c
// Lines 495-497
if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
  ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
}
```
- **Participants:** All threads cooperate
- **Action:** Copy args from parameter space to shared memory
- **Why:** Parameter space is not generically addressable; copying enables pointer arithmetic

#### **Stage 2: Calculate Channel ID (Warp 0)**
```c
// Lines 504-527
switch (tid/WARP_SIZE) {
  case 0:
    // Calculate ncclShmem.channelId from blockIdx.x and channelMask
    // Uses population count to find nth set bit
```
- **Participants:** Warp 0
- **Action:** Maps `blockIdx.x` to actual `channelId`
- **Algorithm:** Population count on bitmask

#### **Stage 3: Initialize Barriers (Warp 1)**
```c
// Lines 530-534
case 1:
  if (tid < WARP_SIZE + NCCL_MAX_GROUPS) {
    if (tid == WARP_SIZE) ncclShmem.barrier_pat = 0;
    ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
  }
```
- **Participants:** Warp 1
- **Action:** Zero out barrier counters

#### **Stage 4: Load Communicator (Warp 0)**
```c
// Lines 557-563
case 0:
  void* dst = &ncclShmem.comm;
  void* src = ncclShmem.args.comm;
  copyToShmem16(tid, dst, src, sizeof(ncclDevComm));
```
- **Participants:** Warp 0
- **Action:** Load device communicator from global memory
- **Size:** ~512 bytes, loaded 16 bytes per thread

#### **Stage 5: Load Channel (Warp 1)**
```c
// Lines 565-571
case 1:
  void* dst = &ncclShmem.channel;
  void* src = &((ncclDevCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
  copyToShmem16(tid-WARP_SIZE, dst, src, sizeof(ncclDevChannel));
```
- **Participants:** Warp 1
- **Action:** Load this channel's data from global memory
- **Size:** ~512 bytes

#### **Stage 6: Load Work Descriptors (Warps 2+)**
```c
// Lines 572-580
default:
  int subtid = tid - 2*WARP_SIZE;
  int subtn = tn - 2*WARP_SIZE;
  loadWorkBatchToShmem(subtid, subtn, args, blockIdx.x);
```
- **Participants:** All remaining warps
- **Action:** Load work descriptors into `workStorage`
- **Source:** Parameter space or work FIFO depending on `workStorageType`

#### **Stage 7: Synchronization**
```c
// Line 587
__syncthreads(); // publish shmem
```
- **Participants:** All threads
- **Action:** Ensure all data is visible to all threads
- **Critical:** Memory fence ensures no thread reads before writes complete

#### **Stage 8: Additional Initialization**
```c
// Lines 550-553
if (tid == 0) {
  ncclShmem.aborted = 0;
  ncclShmem.channel.workCounter = 
    ((ncclDevCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter;
}
```
- **Participants:** Thread 0
- **Action:** Initialize abort flag and work counter

---

## Usage in Collective Operations

### Reduce-Scatter Example

From `src/device/reduce_scatter.h`:

```c
template<typename T, typename RedOp, typename Proto>
__device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
  // Access ring topology
  ncclRing *ring = &ncclShmem.channel.ring;
  int const *ringRanks = ring->userRanks;
  
  // Get rank information
  const int nranks = ncclShmem.comm.nRanks;
  
  // NPKit profiling
  int npKitCtxIdx = ncclShmem.channelId;
  NpKit::CollectGpuEvent(NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY, 
                         count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
                         ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
  
  // Create primitives using channel data
  Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
    prims(tid, nthreads, &ring->prev, &ring->next, 
          work->sendbuff, work->recvbuff, work->redOpArg, ...);
  
  // Execute reduce-scatter rounds
  for (int j=2; j<nranks; ++j) {
    rankDest = ringRanks[nranks-j];
    prims.recvReduceSend(offset, nelem);
  }
}
```

**Key Access Patterns:**
1. **Topology:** `ncclShmem.channel.ring` provides peer connections
2. **Global Info:** `ncclShmem.comm.nRanks` provides communicator size
3. **Profiling:** `ncclShmem.comm.npKitEventCollectContexts` for event logging
4. **Channel ID:** `ncclShmem.channelId` for identifying this block's channel

---

## Related Structures

### Host-Side Structures

#### `ncclComm` (Host Communicator)
- **Location:** `src/include/comm.h`
- **Relationship:** Host-side counterpart to device communicator
- **Connection:** `comm->devComm` points to `ncclDevComm` in GPU memory

#### `ncclChannel` (Host Channel)
- **Location:** `src/include/comm.h`
- **Relationship:** Host-side channel structure
- **Connection:** Copied to device as `ncclDevChannel`

---

### Device-Side Structures

#### `ncclDevComm` (Device Communicator)
- **Location:** `src/include/device.h` (lines 547-594)
- **Relationship:** Stored in `ncclShmem.comm`
- **Contains:**
  ```c
  struct ncclDevComm {
    int rank;
    int nRanks;
    int node;
    int nNodes;
    int buffSizes[NCCL_NUM_PROTOCOLS];
    int p2pChunkSize;
    int isAllNvlink;
    int p2pnChannelsPerPeer;
    int* collNetDenseToUserRank;
    volatile uint32_t* abortFlag;
    struct ncclDevChannel* channels;
    int* rankToLocalRank;
    struct ncclDevProfiler* workStarted;
    struct ncclDevProfiler* workCompleted;
    #if defined(ENABLE_NPKIT)
    NpKitEventCollectContext* npKitEventCollectContexts;
    uint64_t* cpuTimestamp;
    #endif
    // ... more fields
  };
  ```

#### `ncclDevChannel` (Device Channel)
- **Location:** `src/include/device.h` (lines 528-537)
- **Relationship:** Stored in `ncclShmem.channel`
- **Contains:**
  ```c
  struct ncclDevChannel {
    struct ncclDevChannelPeer** peers;
    struct ncclRing ring;
    struct ncclTree tree;
    struct ncclTree collnetChain;
    struct ncclDirect collnetDirect;
    struct ncclTree binTree;
    struct ncclNvls nvls;
    uint32_t* workFifoDone;
    uint64_t workCounter;
  };
  ```

#### `ncclDevKernelArgs` (Kernel Arguments)
- **Location:** `src/include/device.h` (lines 610-618)
- **Relationship:** Stored in `ncclShmem.args`
- **Contains:**
  ```c
  struct ncclDevKernelArgs {
    struct ncclDevComm* comm;
    struct channelMasks channelMask;
    enum ncclDevWorkStorageType workStorageType;
    uint32_t workMask;
    void* workBuf;
  };
  ```

#### `ncclDevWorkColl` (Collective Work Descriptor)
- **Location:** `src/include/device.h`
- **Relationship:** Stored in `ncclShmem.workStorage`
- **Contains:** Buffer pointers, counts, channel ranges, reduction ops

---

### Topology Structures

#### `ncclRing`
- **Purpose:** Ring algorithm topology
- **Location:** In `ncclDevChannel`
- **Access:** `ncclShmem.channel.ring`
- **Contains:**
  - `prev`, `next`: Neighbor ranks
  - `userRanks`: Array of ranks in ring order

#### `ncclTree`
- **Purpose:** Tree algorithm topology
- **Location:** In `ncclDevChannel`
- **Access:** `ncclShmem.channel.tree`
- **Contains:** Parent, children ranks

#### `ncclNvls`
- **Purpose:** NVLS (NVLink Sharp) topology
- **Location:** In `ncclDevChannel`
- **Access:** `ncclShmem.channel.nvls`
- **Contains:** NVLS-specific routing information

---

### Helper Structures

#### `ncclShmemGroup`
- **Location:** `src/device/common.h` (lines 119-133)
- **Relationship:** Array stored in `ncclShmem.groups`
- **Purpose:** Per-group scratch space
- **Size:** `NCCL_MAX_GROUPS` instances

---

## Performance Considerations

### Why Shared Memory Matters

**Speed Comparison (Approximate):**
- **Shared Memory:** ~1-2 cycles latency, ~19 TB/s bandwidth (MI300)
- **L1 Cache:** ~5-10 cycles
- **L2 Cache:** ~50-100 cycles
- **Global Memory:** ~200-400 cycles, ~5.2 TB/s bandwidth (MI300)

**Impact:** Accessing `ncclShmem` is ~100x faster than global memory!

### Optimization Strategies

#### 1. **Read-Mostly Pattern**
- Initialize once at kernel start
- All threads read throughout execution
- Minimizes bank conflicts

#### 2. **Cooperative Loading**
- Multiple warps load different parts in parallel
- `copyToShmem16()` uses 16-byte vector loads
- Maximizes memory bandwidth

#### 3. **Alignment**
- 16-byte alignment enables vector operations
- Reduces number of memory transactions
- Improves cache efficiency

#### 4. **Bank Conflict Avoidance**
- Structure layout designed to minimize conflicts
- Different threads access different cache lines
- AMD GPUs: 64 banks, 4-byte wide

### Memory Footprint Management

**Strategies to reduce footprint:**

1. **Conditional Compilation**
   - NPKit, CollTrace, Profiling only when enabled
   - Saves 100s of bytes in production builds

2. **Union for Plugins**
   - Multiple plugins share same space
   - Only one active at a time

3. **Work Storage Sizing**
   - Fixed 1KB size balances capacity vs occupancy
   - Holds 2-3 typical work items

4. **Dynamic Shared Memory**
   - Additional per-warp scratch space allocated dynamically
   - Accessed via `ncclScratchForWarp(warpId)`

### Occupancy Trade-offs

Higher shared memory usage → Lower occupancy:

**Example (MI300 with 64KB shared memory per CU):**
- 2KB per block → 32 blocks per CU (max)
- 4KB per block → 16 blocks per CU
- 8KB per block → 8 blocks per CU

RCCL uses ~2-3KB typically, allowing good occupancy while maintaining performance.

---

## Access Patterns in Different Algorithms

### Ring Algorithm
```c
// Primary accesses
ncclShmem.channel.ring.prev
ncclShmem.channel.ring.next
ncclShmem.channel.ring.userRanks[i]
ncclShmem.comm.nRanks
```

### Tree Algorithm
```c
// Primary accesses
ncclShmem.channel.tree.up
ncclShmem.channel.tree.down[]
ncclShmem.comm.rank
```

### NVLS Algorithm
```c
// Primary accesses
ncclShmem.channel.nvls.up[]
ncclShmem.channel.nvls.down
ncclShmem.channel.nvls.nHeads
ncclShmem.channel.nvls.headRank
ncclShmem.comm.nNodes
```

### PAT (Parallel Algorithm Template)
```c
// Uses groups for scratch space
ncclShmem.groups[group].srcs[]
ncclShmem.groups[group].dsts[]
ncclShmem.groups[group].barrier
ncclShmem.barrier_pat
```

---

## Debugging and Profiling

### CollTrace Usage

When `ENABLE_COLLTRACE` is defined:

```c
#define INC_COLL_TRACE \
  uint32_t pos = __hip_atomic_fetch_add(&ncclShmem.collTraceTail->tail, 1, ...); \
  struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos; \
  collTrace->timeStamp = wall_clock64(); \
  collTrace->channelId = ncclShmem.channelId;
```

Captures:
- Function ID
- Batch index
- Channel ID
- Operation counts
- Timestamps

### NPKit Integration

```c
int npKitCtxIdx = ncclShmem.channelId;
NpKit::CollectGpuEvent(
  event_type, 
  data_size, 
  additional_data,
  NPKIT_GET_GPU_TIMESTAMP(),
  ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx
);
```

Enables per-channel, per-event profiling with nanosecond precision.

### Profiler Plugin

```c
#ifdef ENABLE_PROFILING
if (ncclShmem.prof.count < PROFILE_NUM_ITEMS) {
  ncclShmem.prof.elem[ncclShmem.prof.count].line = __LINE__;
  ncclShmem.prof.elem[ncclShmem.prof.count].timeStamp = wall_clock64();
  ncclShmem.prof.count++;
}
#endif
```

---

## Common Pitfalls and Best Practices

### ❌ Common Mistakes

1. **Reading before `__syncthreads()`**
   ```c
   // WRONG: Reading before initialization complete
   int rank = ncclShmem.comm.rank; // May read garbage!
   __syncthreads();
   ```

2. **Writing from multiple threads**
   ```c
   // WRONG: Race condition
   ncclShmem.channelId = myValue; // Multiple threads write
   ```

3. **Assuming zero-initialization**
   ```c
   // WRONG: Shared memory is NOT zero-initialized
   if (ncclShmem.someField == 0) // Undefined behavior
   ```

### ✅ Best Practices

1. **Always wait for initialization**
   ```c
   __syncthreads(); // Wait for initialization
   int rank = ncclShmem.comm.rank; // Now safe
   ```

2. **Single-writer pattern**
   ```c
   if (tid == 0) {
     ncclShmem.someField = value; // Only one thread writes
   }
   __syncthreads(); // Others wait
   ```

3. **Explicit initialization**
   ```c
   if (tid == 0) {
     ncclShmem.counter = 0; // Explicitly initialize
   }
   __syncthreads();
   ```

4. **Read-mostly pattern**
   ```c
   // Cache frequently-used values in registers
   int nranks = ncclShmem.comm.nRanks; // Read once
   for (int i = 0; i < nranks; i++) {  // Use register value
     // ...
   }
   ```

---

## Key Takeaways

### Understanding ncclShmemData

1. **Central Communication Hub:** All kernel threads access data through this structure
2. **Performance Critical:** Shared memory access is 100x+ faster than global memory
3. **Carefully Initialized:** Multi-stage cooperative loading by different warps
4. **Read-Mostly Pattern:** Initialized once, read many times
5. **Topology Provider:** Contains all connection and routing information
6. **Profiling Gateway:** Enables detailed performance analysis

### When Working with ncclShmem

1. **Always sync before reading** after kernel start
2. **Understand initialization stages** when debugging early kernel execution
3. **Use read-mostly pattern** for best performance
4. **Cache values in registers** when used repeatedly
5. **Be aware of memory footprint** impact on occupancy
6. **Leverage profiling infrastructure** for optimization

### Design Principles

1. **Minimize global memory access** by caching in shared memory
2. **Cooperative loading** maximizes bandwidth
3. **Alignment matters** for vector loads
4. **Conditional compilation** reduces footprint in production
5. **Single-writer pattern** avoids race conditions

---

## Additional Resources

- **Source Files:**
  - `src/device/common.h` - Structure definition and initialization
  - `src/device/reduce_scatter.h` - Usage example in reduce-scatter
  - `src/device/allreduce.h` - Usage in all-reduce
  - `src/include/device.h` - Device structure definitions

- **Related Documentation:**
  - [ncclComm.md](ncclComm.md) - Host-side communicator
  - RCCL Device Programming Guide
  - AMD GPU Programming Guide - Shared Memory

- **Profiling Tools:**
  - NPKit for detailed event tracing
  - CollTrace for operation tracking
  - rocProfiler for GPU profiling

---

**Document Version:** 1.0  
**Last Updated:** November 10, 2025  
**Maintainer:** RCCL Documentation Team

