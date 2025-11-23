# RCCL Multi-Level Parallelism Architecture

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Hierarchy of Parallelism](#hierarchy-of-parallelism)
4. [Key Terms: Channel, Chunk, and Slice](#key-terms-channel-chunk-and-slice)
5. [Concrete Example: AllReduce with Ring Algorithm on 4 GPUs](#concrete-example-allreduce-with-ring-algorithm-on-4-gpus)
6. [Implementation Details](#implementation-details)
7. [Performance Implications](#performance-implications)

## Overview

RCCL (ROCm Communication Collectives Library) implements a sophisticated multi-level parallelism model to maximize bandwidth utilization and minimize latency in collective communication operations. This architecture enables efficient data movement across multiple GPUs by exploiting parallelism at multiple granularities simultaneously.

The parallelism model is hierarchical, with each level optimized for different aspects of the communication workload:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Level Parallelism                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: Channel Parallelism (Coarse-grained)                  │
│           ├── Channel 0 ─────────────┐                          │
│           ├── Channel 1 ─────────────┤ Up to 128 channels       │
│           └── Channel N ─────────────┘                          │
│                    │                                            │
│  Level 2: Thread Block Parallelism                              │
│           └── Each channel → 1 thread block                     │
│                    │                                            │
│  Level 3: Warp Parallelism (Fine-grained)                       │
│           └── Each block → Multiple warps (32-64 threads)       │
│                    │                                            │
│  Level 4: SIMD Parallelism                                      │
│           └── Each warp → SIMD execution                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Communication Primitives

RCCL operations decompose data movement into **primitives** - fundamental communication patterns like send, receive, reduce-send, etc. These primitives operate on subdivisions of the input data to enable pipelining and parallel processing.

### Data Decomposition

To achieve parallelism and pipelining, RCCL decomposes the input data into progressively smaller units:

```
Total Data
    └── Per-Channel Data (divided across channels)
         └── Chunks (logical division for algorithm)
              └── Slices (pipelining unit)
```

## Hierarchy of Parallelism

### Level 1: Channel-Level Parallelism

**Channels** are independent communication pathways between GPUs. They represent the coarsest level of parallelism.

- **Maximum channels**: 128 (`MAXCHANNELS`)
- **Typical usage**: 2-32 channels depending on message size and topology
- **Purpose**: Maximize aggregate bandwidth by using multiple parallel communication paths
- **Independence**: Each channel operates completely independently with its own:
  - GPU thread block
  - Communication buffers
  - Ring/tree topology connections

### Level 2: Thread Block Parallelism

Each channel is serviced by one GPU thread block:

- **Mapping**: 1 channel = 1 thread block
- **Block size**: 128-256 threads (`NCCL_MAX_NTHREADS`)
- **Coordination**: Threads within a block cooperate via shared memory
- **Synchronization**: Thread blocks synchronize only at kernel boundaries

### Level 3: Warp-Level Parallelism

Within each thread block, multiple warps work cooperatively:

- **Warp size**: 32 threads (CDNA) or 64 threads (older GFX9)
- **Multiple warps**: Typically 4-8 warps per channel (`nWarps`)
- **Cooperation**: Warps process different portions of the data chunk
- **Efficiency**: Enables hiding of memory latency

### Level 4: SIMD Parallelism

Within each warp, threads execute in SIMD (Single Instruction, Multiple Data) fashion:

- **Execution**: All threads in warp execute same instruction
- **Data**: Each thread operates on different data elements
- **Vectorization**: Enables efficient memory coalescing

## Key Terms: Channel, Chunk, and Slice

### Channel

A **channel** is an independent communication pathway between GPUs that operates in parallel with other channels.

**Key Properties:**
- Each channel has its own ring or tree topology
- Processes a disjoint subset of the total data
- Serviced by one GPU thread block
- Has dedicated send/receive buffers

**Code Reference:**
```c
// From src/include/comm.h
struct ncclChannel {
    struct ncclChannelPeer** peers;      // Per-peer connections
    struct ncclRing ring;                // Ring topology for this channel
    struct ncclTree tree;                // Tree topology for this channel
    int id;                              // Channel identifier
    // ...
};

// Maximum channels allowed
#define MAXCHANNELS 128

// Typical channel configuration
int nChannels;      // Total connection channels
int collChannels;   // Active channels for this collective
```

**Important:** Different channels can have different ring topologies (controlled by `sameChannels` flag in `ncclTopoGraph`). This allows RCCL to utilize more physical links in fully-connected topologies.

**Example from real log (6 GPUs, 8 channels):**
```
Channel 00/08 : 0 1 2 3 4 5         ← Ring topology 1
Channel 01/08 : 0 2 4 1 5 3         ← Ring topology 2 (different!)
Channel 02/08 : 0 1 2 3 4 5         ← Ring topology 1
Channel 03/08 : 0 2 4 1 5 3         ← Ring topology 2
Channel 04/08 : 0 1 2 3 4 5         ← Ring topology 1
Channel 05/08 : 0 2 4 1 5 3         ← Ring topology 2
Channel 06/08 : 0 1 2 3 4 5         ← Ring topology 1
Channel 07/08 : 0 2 4 1 5 3         ← Ring topology 2
```

By using different ring paths for different channels, RCCL can utilize more physical links and distribute traffic more evenly across the network.

### Chunk

A **chunk** is a logical subdivision of the per-channel data where each chunk is **assigned to a specific rank** as the location where that rank will produce the final reduced result.

**Key Properties:**
- For ring algorithms, there are `N` chunks (where N = number of GPUs)
- **Chunk i is assigned to Rank i** - each rank "owns" one chunk for reduction
- **All ranks initially have all chunks** in their local buffer (for AllReduce)
- During reduce-scatter: each chunk circulates through all ranks, accumulating reductions
- After reduce-scatter: each rank holds the fully-reduced version of its assigned chunk
- During allgather: each rank distributes its reduced chunk to all other ranks
- Size: `chunkCount = channelCount / nRanks` (approximately)

A chunk is not "data that originates from one rank" - rather it's a portion of the data that a specific rank is responsible for reducing. All ranks contribute to the reduction of all chunks, but each rank completes the reduction for its assigned chunk.

**Code Reference:**
```c
// From src/include/collectives.h
#define NCCL_STEPS 8                              // Pipeline depth
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)      // 4 steps = 1 chunk
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)      // 2 steps = 1 slice

// Chunk calculation in device code
// From src/device/all_reduce.h
ncclCollCbdPart(work, channelId, Proto::Id, sizeof(T), 
                &size, &gridOffset, &channelCount, &chunkCount);

// For allReduce ring:
const ssize_t loopCount = nranks * chunkCount;  // Total chunks to process
```

**Example:**
For 1 MB data on 4 GPUs with 2 channels:
- Total data per channel: 512 KB
- Chunk size: 512 KB / 4 GPUs = 128 KB per chunk
- Each GPU processes 4 chunks (one from each GPU)

### Slice

A **slice** is the smallest unit of data transfer in the pipeline, enabling fine-grained pipelining and latency hiding.

**Key Properties:**
- Multiple slices compose one chunk
- Ratio: `CHUNKSTEPS / SLICESTEPS` slices per chunk
- Each slice is transferred independently in the pipeline
- Enables overlapping communication and computation
- Typically 2-4 slices per chunk for allReduce

**Code Reference:**
```c
// From src/include/collectives.h
// Standard configuration (multi-node)
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)      // 2 pipeline steps per slice
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)      // 4 pipeline steps per chunk
// Result: 2 slices per chunk (4/2 = 2)

// Protocol definition with chunk/slice steps
template<int CHUNKSTEPS, int SLICESTEPS>
struct ProtoSimple {
    static constexpr int ChunkSteps = CHUNKSTEPS;
    static constexpr int SliceSteps = SLICESTEPS;
    static constexpr int StepPerSlice = CHUNKSTEPS / SLICESTEPS;
};
```

#### Understanding Pipeline Steps (STEPS)

A **STEP** is a unit in RCCL's pipeline that represents:
1. One slot in a circular buffer array
2. A synchronization point between sender and receiver
3. A fixed amount of buffer space (stepSize bytes)

RCCL uses 8 STEPs (`NCCL_STEPS = 8`), meaning there are 8 buffer slots arranged in a circular array that wrap around.

#### How Slicing Enables Overlap

**Without slicing:**
```
Step: 0    1    2    3    4    5    6    7
      ├────┼────┼────┼────┼────┼────┼────┼────┤
      [========Chunk 0========]
                               [========Chunk 1========]
```
Chunk 1 waits for Chunk 0 to completely finish - no overlap.

**With slicing (2 slices per chunk):**
```
Step: 0    1    2    3    4    5    6    7
      ├────┼────┼────┼────┼────┼────┼────┼────┤
      [Chunk 0, Slice 0]
                [Chunk 0, Slice 1]
                [Chunk 1, Slice 0]    ← STARTS EARLY!
                          [Chunk 1, Slice 1]
                                    [Chunk 2, Slice 0]
```
Chunk 1 starts at Step 2 while Chunk 0 is still finishing - overlap achieved!

The circular buffer allows sender and receiver to coordinate: receiver consumes data from buffer slots and frees them for reuse, allowing the sender to continue without waiting for the entire chunk to complete.

**Benefits:**
- Without slicing: Ring steps are sequential
- With slicing: Ring steps overlap (20-50% faster)

### Relationship Summary

```
┌─────────────────────────────────────────────────────────────┐
│                        Total Data                           │
│                         1 MB                                │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌────────▼───────┐
│  Channel 0     │  ...  │  Channel N     │  (Channel-level)
│   512 KB       │       │   512 KB       │
└───────┬────────┘       └────────────────┘
        │
  ┌─────┴─────┬─────┬─────┬─────┐
  │           │     │     │     │
┌─▼──┐  ┌───▼─┐ ┌─▼──┐ ┌─▼──┐   │
│C0  │  │C1   │ │C2  │ │C3  │   │  (Chunk-level)
│128K│  │128K │ │128K│ │128K│   │
└─┬──┘  └─────┘ └────┘ └────┘   │
  │                             │
┌─┴──┬─────┐                    │
│    │     │                    │
│S0  │ S1  │                    │  (Slice-level)
│64K │ 64K │                    │
└────┴─────┘                    │
```

## Concrete Example: AllReduce with Ring Algorithm on 4 GPUs

### Scenario Setup

**Configuration:**
- **Operation**: AllReduce (sum)
- **Algorithm**: Ring
- **GPUs**: 4 GPUs in a single node (GPU0, GPU1, GPU2, GPU3)
- **Data**: 1 MB of float32 data (256K elements)
- **Channels**: 2 channels for simplicity
- **Datatype**: float32 (4 bytes per element)

### Understanding Ring Topology

A ring topology connects N GPUs in a circular pattern where:
- Each GPU has exactly **2 neighbors** (one to send to, one to receive from)
- GPU i sends to GPU (i+1) mod N
- GPU i receives from GPU (i-1) mod N
- Forms a closed loop

**Why Use Ring Topology?**

1. **Bandwidth Efficiency** - Each GPU uses only its link to the next neighbor, with all GPUs working in parallel. No bottleneck at any single GPU.

2. **Balanced Load** - Every GPU does equal work: each sends (N-1) chunks during reduce-scatter and (N-1) chunks during allgather.

3. **Optimal Complexity** - Total time is 2(N-1) steps, which is optimal for allReduce.

4. **Maximum Bandwidth Utilization** - All network links are active simultaneously.

**Example ring for 4 GPUs:**
```
GPU0 → GPU1 → GPU2 → GPU3 → GPU0
```

### How Channels Map to Ring Topologies

Each channel can have its own ring topology to maximize network utilization:

```
┌──────────────────────────────────────────────────────────────┐
│ Channel 0:  GPU0 → GPU1 → GPU2 → GPU3 → GPU0                │
│             Processes elements [0 ... 127999]                 │
│                                                               │
│ Channel 1:  GPU0 → GPU1 → GPU2 → GPU3 → GPU0                │
│             Processes elements [128000 ... 255999]            │
└──────────────────────────────────────────────────────────────┘
```

In a fully-connected topology (where all GPUs can communicate directly), RCCL may assign different ring paths to different channels to utilize more physical links:

```
┌──────────────────────────────────────────────────────────────┐
│ Channel 0:  GPU0 → GPU1 → GPU2 → GPU3 → GPU0                │
│             Uses links: 0-1, 1-2, 2-3, 3-0                   │
│                                                               │
│ Channel 1:  GPU0 → GPU2 → GPU1 → GPU3 → GPU0                │
│             Uses links: 0-2, 2-1, 1-3, 3-0                   │
│                                                               │
│ Result: More physical links utilized!                         │
└──────────────────────────────────────────────────────────────┘
```

This is controlled by the `sameChannels` flag in RCCL's topology search:
- `sameChannels = 1`: All channels use the same ring topology
- `sameChannels = 0`: Channels can have different ring topologies

### Data Distribution Across Channels

Channels divide the data into **contiguous sequential sections**, determined by the `ncclCollCbdPart()` function:

```c
// From src/include/device.h
__device__ inline void ncclCollCbdPart(
    struct ncclDevWorkColl* work, uint32_t channelId, int proto, int eltSize,
    Int* count, Int* partOffset, Int* partCount, Int* chunkCount
) {
    // For middle channels:
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid*work->cbd.countMid;  // Sequential offset
    *partCount = work->cbd.countMid;                            // Contiguous count
    // ...
}
```

**Example:**
```
Total Data: 256K elements (1 MB)

Channel 0: Elements [0 ... 127999]       (first half, 512 KB)
Channel 1: Elements [128000 ... 255999]  (second half, 512 KB)
```

### Data Flow Example

For simplicity, we'll show Channel 0 only (processing elements 0-127999):

#### Per-Channel Data Division

Channel 0's 512 KB is divided into 4 chunks (one per GPU):

```
Channel 0 (512 KB = 128K elements):
├─ Chunk 0: Elements [0 ... 31999]       (128 KB) - Assigned to GPU0
├─ Chunk 1: Elements [32000 ... 63999]   (128 KB) - Assigned to GPU1
├─ Chunk 2: Elements [64000 ... 95999]   (128 KB) - Assigned to GPU2
└─ Chunk 3: Elements [96000 ... 127999]  (128 KB) - Assigned to GPU3
```

#### Initial State

All GPUs start with ALL chunks in their buffers (since this is AllReduce). The letters A, B, C, D represent data from each GPU's initial buffer:

```
         Chunk 0   Chunk 1   Chunk 2   Chunk 3
GPU0:    [  A  ]   [  A  ]   [  A  ]   [  A  ]    ← GPU0's initial data
GPU1:    [  B  ]   [  B  ]   [  B  ]   [  B  ]    ← GPU1's initial data
GPU2:    [  C  ]   [  C  ]   [  C  ]   [  C  ]    ← GPU2's initial data
GPU3:    [  D  ]   [  D  ]   [  D  ]   [  D  ]    ← GPU3's initial data
```

**Chunk Assignment** (who will produce the final reduced result):
- **Chunk 0 is assigned to GPU0** (GPU0 will hold A+B+C+D for Chunk 0 first)
- **Chunk 1 is assigned to GPU1** (GPU1 will hold A+B+C+D for Chunk 1 first)
- **Chunk 2 is assigned to GPU2** (GPU2 will hold A+B+C+D for Chunk 2 first)
- **Chunk 3 is assigned to GPU3** (GPU3 will hold A+B+C+D for Chunk 3 first)

#### Phase 1: Reduce-Scatter (3 steps for 4 GPUs)

In the reduce-scatter phase, each GPU reduces and forwards chunks. After N-1 steps (3 steps for 4 GPUs), each GPU has one fully-reduced chunk.

**Step 0:**
```
GPU0 sends Chunk 3[A] → GPU1 (GPU1: Chunk 3 = A+B)
GPU1 sends Chunk 0[B] → GPU2 (GPU2: Chunk 0 = B+C)
GPU2 sends Chunk 1[C] → GPU3 (GPU3: Chunk 1 = C+D)
GPU3 sends Chunk 2[D] → GPU0 (GPU0: Chunk 2 = D+A)
```

**Step 1:**
```
GPU0 sends Chunk 2[D+A] → GPU1 (GPU1: Chunk 2 = D+A+B)
GPU1 sends Chunk 3[A+B] → GPU2 (GPU2: Chunk 3 = A+B+C)
GPU2 sends Chunk 0[B+C] → GPU3 (GPU3: Chunk 0 = B+C+D)
GPU3 sends Chunk 1[C+D] → GPU0 (GPU0: Chunk 1 = C+D+A)
```

**Step 2:**
```
GPU0 sends Chunk 1[C+D+A] → GPU1 (GPU1: Chunk 1 = C+D+A+B ✓ COMPLETE)
GPU1 sends Chunk 2[D+A+B] → GPU2 (GPU2: Chunk 2 = D+A+B+C ✓ COMPLETE)
GPU2 sends Chunk 3[A+B+C] → GPU3 (GPU3: Chunk 3 = A+B+C+D ✓ COMPLETE)
GPU3 sends Chunk 0[B+C+D] → GPU0 (GPU0: Chunk 0 = B+C+D+A ✓ COMPLETE)
```

**After Reduce-Scatter:**
```
         Chunk 0     Chunk 1     Chunk 2     Chunk 3
GPU0:    [A+B+C+D]   [partial]   [partial]   [partial]
GPU1:    [partial]   [A+B+C+D]   [partial]   [partial]
GPU2:    [partial]   [partial]   [A+B+C+D]   [partial]
GPU3:    [partial]   [partial]   [partial]   [A+B+C+D]
```

Each GPU now has one fully reduced chunk!

#### Phase 2: AllGather (3 steps for 4 GPUs)

In the allgather phase, each GPU forwards its reduced chunk around the ring. After N-1 steps, all GPUs have all reduced chunks.

**Step 3:**
```
GPU0 sends Chunk 0[A+B+C+D] → GPU1
GPU1 sends Chunk 1[A+B+C+D] → GPU2
GPU2 sends Chunk 2[A+B+C+D] → GPU3
GPU3 sends Chunk 3[A+B+C+D] → GPU0
```

**Step 4:**
```
GPU0 sends Chunk 3[A+B+C+D] → GPU1
GPU1 sends Chunk 0[A+B+C+D] → GPU2
GPU2 sends Chunk 1[A+B+C+D] → GPU3
GPU3 sends Chunk 2[A+B+C+D] → GPU0
```

**Step 5:**
```
GPU0 sends Chunk 2[A+B+C+D] → GPU1
GPU1 sends Chunk 3[A+B+C+D] → GPU2
GPU2 sends Chunk 0[A+B+C+D] → GPU3
GPU3 sends Chunk 1[A+B+C+D] → GPU0
```

**Final State:**
```
         Chunk 0     Chunk 1     Chunk 2     Chunk 3
GPU0:    [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   ✓ Complete
GPU1:    [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   ✓ Complete
GPU2:    [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   ✓ Complete
GPU3:    [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   [A+B+C+D]   ✓ Complete
```

All GPUs now have the complete reduced result!

### Parallelism at Work

During the above process, multiple levels of parallelism operate simultaneously:

1. **Channel Parallelism**: Channel 0 and Channel 1 execute independently, processing different data subsets
2. **Thread Block Parallelism**: Each channel uses one thread block with 128-256 threads
3. **Warp Parallelism**: Multiple warps within each block process different elements
4. **SIMD Parallelism**: Threads within a warp execute in parallel on different data

**Total parallelism example:**
- 2 channels × 256 threads per channel = **512 threads** working simultaneously on a single GPU
- 4 GPUs × 512 threads = **2048 threads** across all GPUs
- All network links active simultaneously

## Implementation Details

### Channel Assignment and Data Distribution

```c
// From src/enqueue.cc and device code
// Channels are assigned based on thread block ID
int channelId = blockIdx.x;

// Data is divided sequentially (NOT interleaved) across channels
// Each channel gets a contiguous section determined by ncclCollCbdPart()
// For 2 channels with total count N:
//   Channel 0: elements [0 ... N/2-1]
//   Channel 1: elements [N/2 ... N-1]
```

### Chunk Size Calculation

```c
// From src/device/all_reduce.h
ssize_t chunkSize = sliceSize * nSlicesPerChunk;
ssize_t gridOffset, channelCount, chunkCount;

// Get this channel's data partition
ncclCollCbdPart(work, channelId, Proto::Id, sizeof(T),
                &size, &gridOffset, &channelCount, &chunkCount);
```

### Ring Algorithm Implementation

```c
// From src/device/prims_ll.h and all_reduce.h
template<typename T, typename RedOp>
__device__ void runRing(int tid, int nthreads, 
                        int nranks, ncclRing* ring,
                        T* sendbuff, T* recvbuff, ssize_t count) {
  const int nSlices = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS;
  const ssize_t sliceSize = chunkSize / nSlices;
  
  // Reduce-scatter phase
  for (int chunk = 0; chunk < nranks-1; chunk++) {
    for (int slice = 0; slice < nSlices; slice++) {
      ssize_t offset = calcOffset(chunk, slice);
      prims.recvReduceSend(sendbuff + offset, sliceSize);
    }
  }
  
  // Allgather phase
  for (int chunk = 0; chunk < nranks-1; chunk++) {
    for (int slice = 0; slice < nSlices; slice++) {
      ssize_t offset = calcOffset(chunk, slice);
      prims.recvCopySend(sendbuff + offset, sliceSize);
    }
  }
}
```

### Kernel Launch

```c
// From src/enqueue.cc
void ncclEnqueueKernel(ncclComm_t comm, ...) {
    // One thread block per channel
    int gridDim = nChannels;
    int blockDim = nThreads;
    
    ncclKernel<<<gridDim, blockDim, 0, stream>>>(args);
}
```

### Network Resource Utilization

RCCL maximizes network resource utilization through:

1. **Topology-Aware Ring Selection**: RCCL's search algorithm (`ncclTopoCompute`) analyzes the physical network topology and selects ring paths that maximize bandwidth.

2. **Multiple Ring Topologies**: In fully-connected topologies, RCCL can assign different ring paths to different channels (when `sameChannels = 0`), allowing more physical links to be utilized simultaneously.

3. **Algorithm Selection**: Different algorithms (Ring, Tree, CollNet) for different network types and message sizes.

4. **Channel Parallelism**: Multiple channels saturate each selected link with parallel traffic.

**Example from topology search:**
```c
// From src/graph/search.cc
int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
graph->sameChannels = trySameChannels;

// Try having different channels (except when going through AMD CPUs)
if (tmpGraph.sameChannels == 1 &&
    !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD)) {
  tmpGraph.sameChannels = 0;  // Try different rings per channel
  goto search;
}
```

## Performance Implications

### Benefits of Multi-Level Parallelism

1. **Bandwidth Saturation**
   - Multiple channels saturate available network links
   - Aggregate bandwidth = nChannels × single-channel bandwidth (up to physical limit)

2. **Latency Hiding**
   - Pipelining through slices hides communication latency
   - Different chunks overlap in flight
   - 20-50% performance improvement from pipelining

3. **GPU Utilization**
   - Multiple warps per channel hide memory access latency
   - SIMD execution maximizes compute throughput
   - Thread-level parallelism keeps GPU busy

4. **Scalability**
   - Ring algorithm scales linearly: O(N) complexity
   - Time ≈ 2(N-1) × (chunk_size / bandwidth)
   - For large N, time ≈ 2S / bandwidth (constant!)

### Tuning Parameters

Key parameters that affect performance:

- **Number of channels** (`NCCL_NCHANNELS`): More channels = higher bandwidth (up to saturation point)
- **Number of threads** (`NCCL_NTHREADS`): More threads per channel = better GPU utilization
- **Protocol selection**: LL (low-latency), LL128, Simple - chosen based on message size
- **Chunk/Slice steps**: Control pipelining granularity

### Example Performance

For a 4-GPU system with 50 GB/s links:

**Single channel:**
- Bandwidth: ~45 GB/s (90% efficiency)
- Latency: ~10 µs

**8 channels:**
- Bandwidth: ~190 GB/s (95% of 4×50 GB/s theoretical)
- Latency: ~12 µs (slight increase due to coordination)

**With pipelining:**
- Effective latency reduced by 30-40%
- Ramp-up time improved significantly

### Comparison with Naive Approaches

| Approach | Bandwidth | Scalability | Complexity |
|----------|-----------|-------------|------------|
| **Ring (RCCL)** | O(N) links used, 100% | Linear | Optimal: 2(N-1) steps |
| Central reduction | 2 links used, bottleneck | O(N²) | Poor: N steps |
| All-to-all | N² links needed | O(N²) | Complex |

## Summary

RCCL's multi-level parallelism architecture achieves high performance through:

1. **Channel Parallelism**: Independent communication pathways with potentially different ring topologies to maximize network utilization
2. **Thread Block Parallelism**: One block per channel for hardware mapping
3. **Warp Parallelism**: Multiple warps hide latency and maximize throughput
4. **SIMD Parallelism**: Vectorized execution for efficiency

Combined with intelligent algorithms (Ring, Tree, CollNet) and pipelining through slices, RCCL achieves near-optimal bandwidth utilization and minimal latency for collective operations across multiple GPUs.
