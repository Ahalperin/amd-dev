# RCCL Design Overview

**Date:** October 30, 2025  
**Purpose:** Comprehensive architectural analysis for performance optimization and bottleneck identification  
**Target Audience:** System engineers, performance engineers, and RCCL developers

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Communication Patterns and Algorithms](#communication-patterns-and-algorithms)
5. [Data Flow and Execution Pipeline](#data-flow-and-execution-pipeline)
6. [Network and Transport Layer](#network-and-transport-layer)
7. [Topology Discovery and Graph Optimization](#topology-discovery-and-graph-optimization)
8. [Memory Management](#memory-management)
9. [Synchronization and Threading Model](#synchronization-and-threading-model)
10. [Performance Optimization Areas](#performance-optimization-areas)
11. [Known Bottlenecks and Limitations](#known-bottlenecks-and-limitations)
12. [Profiling and Debugging Infrastructure](#profiling-and-debugging-infrastructure)

---

## Executive Summary

**RCCL (ROCm Communication Collectives Library)** is AMD's implementation of collective communication primitives optimized for AMD GPUs. It is functionally compatible with NVIDIA's NCCL and provides high-performance collective operations for both single-node and multi-node GPU communication.

### Key Characteristics
- **Collective Operations:** AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, Gather, Scatter, AllToAll
- **Point-to-Point:** GPU-to-GPU Send/Recv operations
- **Interconnects:** PCIe, xGMI (AMD Infinity Fabric), InfiniBand, TCP/IP sockets
- **Algorithms:** Ring, Tree, CollNet (Direct/Chain), MSCCL, MSCCLPP
- **Protocols:** Simple, LL (Low Latency), LL128 (Low Latency 128-bit)
- **Scale:** Arbitrary number of GPUs (single/multi-node)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Application Layer                       │
│                    (PyTorch, TensorFlow, MPI)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RCCL Public API                             │
│    ncclAllReduce, ncclAllGather, ncclBroadcast, ncclSend/Recv   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Communicator Management                        │
│        ncclComm, Group Operations, Channel Management            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │   Algorithm    │  │   Topology     │  │   Transport    │
   │   Selection    │  │   Discovery    │  │   Layer        │
   │  (Ring/Tree)   │  │   & Graph      │  │  (P2P/NET)     │
   └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Enqueue & Execution                    │
│              Device Kernels, Primitives, Memory Ops              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Hardware Layer                               │
│         AMD GPUs, xGMI, PCIe, InfiniBand, NICs                  │
└─────────────────────────────────────────────────────────────────┘
```

### Source Code Organization

```
rccl/
├── src/
│   ├── init.cc                    # Library initialization
│   ├── enqueue.cc                 # Operation enqueuing
│   ├── collectives.cc             # Collective operation implementations
│   ├── channel.cc                 # Channel management
│   ├── group.cc                   # Group operation handling
│   ├── proxy.cc                   # Proxy thread for network operations
│   ├── bootstrap.cc               # Bootstrap communication
│   ├── transport.cc               # Transport layer coordination
│   │
│   ├── device/                    # GPU kernel implementations
│   │   ├── all_reduce.h
│   │   ├── all_gather.h
│   │   ├── reduce_scatter.h
│   │   ├── primitives.h           # Core GPU primitives
│   │   ├── prims_*.h              # Protocol-specific primitives
│   │   └── common.cu              # Common device code
│   │
│   ├── graph/                     # Topology and algorithm selection
│   │   ├── topo.cc                # Topology discovery
│   │   ├── xml.cc                 # Topology XML parsing
│   │   ├── search.cc              # Path search algorithms
│   │   ├── rings.cc               # Ring algorithm
│   │   ├── trees.cc               # Tree algorithm
│   │   ├── tuning.cc              # Algorithm tuning
│   │   └── rome_models.cc         # GPU-specific models
│   │
│   ├── transport/                 # Transport implementations
│   │   ├── p2p.cc                 # Peer-to-peer (xGMI/PCIe)
│   │   ├── shm.cc                 # Shared memory
│   │   ├── net.cc                 # Network transport
│   │   ├── net_ib.cc              # InfiniBand
│   │   └── net_socket.cc          # TCP/IP sockets
│   │
│   └── include/                   # Internal headers
│       ├── comm.h                 # Communicator structures
│       ├── channel.h              # Channel structures
│       ├── graph.h                # Topology graph structures
│       └── ...
│
├── ext-net/                       # Network plugin API
├── ext-profiler/                  # Profiler plugin API
└── ext-tuner/                     # Tuner plugin API
```

---

## Core Components

### 1. Communicator (`ncclComm`)

The **communicator** is the central data structure that encapsulates all state for a collective communication group.

**Key Responsibilities:**
- Maintain GPU rank and total number of ranks
- Manage channels (parallel execution paths)
- Store topology information
- Coordinate shared resources across ranks
- Handle collective operation state

**Location:** `src/include/comm.h`

**Structure Highlights:**
```c
struct ncclComm {
  int rank;                      // This rank's ID
  int nRanks;                    // Total number of ranks
  int cudaDev;                   // CUDA/HIP device ID
  int nChannels;                 // Number of channels
  struct ncclChannel channels[MAXCHANNELS];
  struct ncclTopoSystem* topo;   // Topology information
  struct ncclSharedResources* sharedRes;
  // ... many more fields
};
```

**Initialization Flow:**
1. `ncclCommInitRank()` → Entry point for communicator creation
2. Bootstrap communication setup (rank coordination)
3. Topology discovery and graph construction
4. Channel initialization
5. Transport setup (P2P, network)
6. Algorithm selection and tuning

---

### 2. Channels

**Channels** are independent execution paths that allow parallel collective operations. Multiple channels enable better hardware utilization and improved bandwidth.

**Key Concepts:**
- Each channel has its own ring or tree structure
- Channels operate independently and in parallel
- Data is striped across channels for large transfers
- Channel count is tuned based on GPU architecture and topology

**Location:** `src/channel.cc`, `src/include/channel.h`

**Per-Channel State:**
- Send/receive rings
- Send/receive trees
- Connection information (peers)
- Work FIFOs for GPU kernels

**Typical Channel Counts:**
- Single node: 4-16 channels
- Multi-node: Depends on network topology and NIC count

---

### 3. Bootstrap Communication

The **bootstrap** layer provides out-of-band communication for initial setup and coordination before collective operations can begin.

**Location:** `src/bootstrap.cc`

**Responsibilities:**
- Rank-to-rank address exchange
- Topology information sharing
- Synchronization during initialization
- Error propagation during setup

**Mechanisms:**
- Socket-based communication
- Root rank acts as rendezvous point
- All ranks connect to root for initial coordination

---

### 4. Topology Discovery

**Topology discovery** is critical for optimal algorithm selection and data routing.

**Location:** `src/graph/topo.cc`, `src/graph/xml.cc`

**Discovery Process:**
1. **GPU Detection:** Enumerate all GPUs and their properties
2. **Interconnect Detection:** 
   - xGMI links (Infinity Fabric)
   - PCIe topology
   - NIC associations
3. **CPU/NUMA Topology:** Map GPUs to CPU sockets
4. **Path Computation:** Calculate bandwidth and latency for all paths
5. **Graph Construction:** Build topology graph with weighted edges

**Path Types (by decreasing performance):**
- `PATH_LOC` (0): Local (same GPU)
- `PATH_NVL` (1): xGMI/NVLink direct
- `PATH_C2C` (3): Chip-to-chip
- `PATH_PIX` (4): Single PCIe bridge
- `PATH_PXB` (5): Multiple PCIe bridges
- `PATH_PHB` (8): Through CPU
- `PATH_SYS` (9): Cross-NUMA
- `PATH_NET` (10): Network

**Bandwidth Estimates:**
- xGMI (MI300): 48 GT/s per direction
- xGMI (MI200): 36 GT/s per direction
- PCIe Gen4 x16: ~12 GB/s
- PCIe Gen5 x16: ~24 GB/s

---

### 5. Algorithm Engine

RCCL supports multiple collective algorithms, selected based on message size, topology, and operation type.

**Location:** `src/graph/rings.cc`, `src/graph/trees.cc`, `src/graph/search.cc`

#### Ring Algorithm
- **Best for:** Large messages, high bandwidth
- **Complexity:** O(N) steps for N ranks
- **Bandwidth:** Near-optimal for large messages
- **Pattern:** Each rank sends to next neighbor in ring

#### Tree Algorithm
- **Best for:** Small messages, low latency
- **Complexity:** O(log N) steps for N ranks
- **Bandwidth:** Lower than ring, but lower latency
- **Pattern:** Binary/N-ary tree structure

#### CollNet (Collective Network)
- **Best for:** Multi-node with SHARP/similar offload
- **Requires:** Network support for collective offload
- **Variants:** Direct, Chain

#### MSCCL (Microsoft Collective Communication Library)
- **Purpose:** Custom, optimized algorithms via XML definitions
- **Flexibility:** Algorithm can be defined per topology
- **Location:** `src/msccl.cc`, `src/misc/msccl/`

#### MSCCLPP
- **Purpose:** Next-generation MSCCL with C++ API
- **Status:** Optional, enabled via `RCCL_MSCCLPP_ENABLE`
- **Location:** `ext-src/mscclpp/`

---

### 6. Protocol Selection

RCCL uses three main protocols, selected based on message size:

#### Simple Protocol
- **Use Case:** Large messages (>512 KB)
- **Mechanism:** Direct bulk transfers
- **Overhead:** Minimal per-operation overhead
- **Throughput:** Highest for large messages

#### LL (Low Latency) Protocol
- **Use Case:** Small messages (<8 KB)
- **Mechanism:** 32-bit flags for synchronization
- **Overhead:** Higher per-byte, but lower latency
- **Throughput:** Optimized for latency

#### LL128 (Low Latency 128-bit) Protocol
- **Use Case:** Medium messages (8 KB - 512 KB)
- **Mechanism:** 128-bit operations with inline flags
- **Overhead:** Balanced
- **Throughput:** Good for medium-sized transfers

**Selection Logic:** `src/graph/tuning.cc`

---

## Communication Patterns and Algorithms

### AllReduce

**Most Common Operation:** Reduce data from all ranks and broadcast result to all ranks.

```
Phase 1 (Reduce-Scatter):
Rank 0: [A0] → [A0+A1+A2+A3]  (chunk 0)
Rank 1: [A1] → [B0+B1+B2+B3]  (chunk 1)
Rank 2: [A2] → [C0+C1+C2+C3]  (chunk 2)
Rank 3: [A3] → [D0+D1+D2+D3]  (chunk 3)

Phase 2 (AllGather):
All ranks gather all reduced chunks
```

**Ring Implementation:**
- N-1 reduce-scatter steps
- N-1 allgather steps
- Total: 2(N-1) steps

**Tree Implementation:**
- log(N) reduce steps (up the tree)
- log(N) broadcast steps (down the tree)
- Total: 2*log(N) steps

---

### AllGather

**Purpose:** Gather data from all ranks and distribute to all ranks.

**Ring Pattern:**
- Each rank starts with 1/N of data
- N-1 send-receive steps
- At end, all ranks have all data

**Bandwidth Optimal:** Sends exactly (N-1)/N of data

---

### ReduceScatter

**Purpose:** Reduce data from all ranks, scatter result chunks to all ranks.

**Relationship:** First half of AllReduce
**Output:** Each rank receives 1/N of reduced result

---

### Broadcast

**Purpose:** One rank sends data to all other ranks.

**Tree Pattern:**
- Root sends to children
- Children forward to their children
- log(N) steps for balanced tree

---

## Data Flow and Execution Pipeline

### Collective Operation Flow

```
1. User calls ncclAllReduce()
   │
   ▼
2. Group launch (if in ncclGroupStart/End)
   │
   ▼
3. ncclEnqueueCheck() - Validation
   │
   ▼
4. Work enqueuing - ncclEnqueueEvents()
   │
   ├─→ Algorithm selection (ring/tree)
   ├─→ Protocol selection (Simple/LL/LL128)
   ├─→ Channel assignment
   └─→ Chunk size calculation
   │
   ▼
5. Proxy activation (for network operations)
   │
   ▼
6. Kernel launch - ncclLaunchKernel()
   │
   ▼
7. GPU kernel execution
   │
   ├─→ Load data
   ├─→ Send to next rank
   ├─→ Receive from previous rank
   ├─→ Reduce operation (if applicable)
   └─→ Store result
   │
   ▼
8. Completion (stream synchronization)
```

### GPU Kernel Architecture

**Location:** `src/device/`

**Kernel Structure:**
```c
// Simplified structure
__global__ void ncclKernel(struct ncclWorkElem work) {
  // 1. Initialize primitives (send/recv)
  // 2. Loop over chunks
  //    - Send data to next peer
  //    - Receive data from previous peer
  //    - Apply reduction (if needed)
  //    - Wait for completion
  // 3. Finalize
}
```

**Primitives:**
- `prims_simple.h`: Simple protocol primitives
- `prims_ll.h`: LL protocol primitives
- `prims_ll128.h`: LL128 protocol primitives

**Key Operations:**
- `Send()`: Send data to peer
- `Recv()`: Receive data from peer
- `Reduce()`: Apply reduction operation
- `Directxxx()`: Direct memory operations

---

## Network and Transport Layer

### Transport Types

#### 1. P2P (Peer-to-Peer)
**Location:** `src/transport/p2p.cc`

**Mechanisms:**
- **xGMI (Infinity Fabric):** Direct GPU-to-GPU on MI200/MI300
- **PCIe:** Standard PCIe transfers
- **Direct RDMA:** GPU memory registered for RDMA

**Characteristics:**
- Lowest latency for intra-node
- Highest bandwidth on xGMI systems
- Used for most single-node communication

#### 2. SHM (Shared Memory)
**Location:** `src/transport/shm.cc`

**Use Case:** 
- CPU-mediated transfers when direct P2P not available
- Fallback mechanism

**Mechanism:**
- GPU writes to CPU shared memory
- Other GPU reads from CPU shared memory

#### 3. Network (NET)
**Location:** `src/transport/net.cc`, `src/transport/net_ib.cc`, `src/transport/net_socket.cc`

**Supported Networks:**
- **InfiniBand:** `net_ib.cc`, uses IBVerbs
- **TCP/IP Sockets:** `net_socket.cc`, fallback
- **Plugins:** External network plugins (e.g., AMD ANP)

**Network Plugin Architecture:**
- Plugin API: `ext-net/`
- Dynamically loaded at runtime
- Examples: AMD ANP (AINIC Network Plugin)

**Proxy Threads:**
- Dedicated CPU threads for network operations
- Offload network work from GPU kernels
- One proxy thread per channel per network peer

---

### AMD ANP (AINIC Network Plugin)

**Location:** `amd-dev/amd/amd-anp/`

**Purpose:** 
- Optimized network path for AMD AINIC (AI Network Interface Card)
- RDMA-based communication
- Hardware-accelerated collectives

**Key Features:**
- Direct GPU-to-NIC communication
- GPUDirect support
- Optimized for InfiniBand networks

---

## Topology Discovery and Graph Optimization

### Topology Graph Structure

**Nodes:**
- `GPU`: GPU devices
- `PCI`: PCIe switches/bridges
- `CPU`: NUMA nodes
- `NIC`: Network interface cards
- `NVS`: NVSwitch (NVIDIA-specific)
- `NET`: Network endpoints

**Edges:**
- Bandwidth (GB/s)
- Latency (us)
- Type (xGMI, PCIe, etc.)

### Path Search Algorithm

**Location:** `src/graph/search.cc`

**Goal:** Find optimal paths for each collective operation

**Algorithm:**
1. Enumerate all possible ring/tree structures
2. Compute bandwidth and latency for each
3. Score based on:
   - Minimum bandwidth link (bottleneck)
   - Total latency
   - Path balance
4. Select best scoring structure

**Caching:**
- Topology is computed once at initialization
- Cached in communicator structure
- Reused for all subsequent operations

---

## Memory Management

### Memory Types

#### 1. Device Memory
- Allocated on GPU via `hipMalloc()`
- User buffers (input/output)
- Internal buffers for multi-step operations

#### 2. Host Memory
- Pinned memory via `hipHostMalloc()`
- Bootstrap communication buffers
- Proxy thread buffers

#### 3. Shared Memory (SHM)
- System V shared memory
- Used for inter-process communication
- Channel connection buffers

#### 4. Registered Memory
- GPU memory registered for RDMA
- Required for GPUDirect
- Managed by network plugins

### RCCL Allocator

**Location:** `src/allocator.cc`

**Features:**
- Custom memory pool for RCCL internal buffers
- Reduces allocation overhead
- Reference counting for shared buffers

---

## Synchronization and Threading Model

### User Thread
- Calls RCCL API
- Enqueues work to GPU streams
- Returns control to user

### GPU Kernels
- Execute on user stream
- Perform collective communication
- Use atomic operations for synchronization

### Proxy Threads
- One proxy thread per network connection per channel
- Handle network send/receive
- Coordinate with GPU kernels via shared memory

### Helper Thread
- Graph helper thread (optional)
- Manages CUDA graph resources
- IPC handle cleanup

---

## Performance Optimization Areas

### 1. **Algorithm Selection**
**Current:** Heuristic-based selection in `src/graph/tuning.cc`
**Opportunity:** Machine learning-based tuning

### 2. **Channel Count**
**Current:** Fixed or environment variable
**Opportunity:** Dynamic adjustment based on workload

### 3. **Chunk Size**
**Current:** Fixed chunking strategy
**Opportunity:** Adaptive chunking based on buffer size and network latency

### 4. **Protocol Thresholds**
**Current:** Fixed thresholds (Simple: 512KB, LL128: 8KB)
**Opportunity:** Hardware-specific tuning

### 5. **Network Proxy Overhead**
**Impact:** Proxy thread CPU usage and latency
**Opportunity:** GPU-driven network operations, zero-copy paths

### 6. **Memory Copies**
**Current:** Some operations require staging buffers
**Opportunity:** In-place operations, reduced copies

### 7. **Kernel Launch Latency**
**Impact:** Small message latency
**Opportunity:** Kernel fusion, persistent kernels

### 8. **Topology Discovery**
**Current:** Done once at init
**Opportunity:** Runtime topology updates, better xGMI detection

---

## Known Bottlenecks and Limitations

### 1. **Small Message Latency**
- **Issue:** High overhead for small (<1KB) messages
- **Root Cause:** Kernel launch overhead, protocol overhead
- **Impact:** Training with high communication frequency

### 2. **Cross-NUMA Communication**
- **Issue:** Lower bandwidth when GPUs span NUMA domains
- **Root Cause:** CPU interconnect (Infinity Fabric) bandwidth
- **Impact:** Large multi-GPU nodes

### 3. **Network Proxy Scalability**
- **Issue:** Proxy thread CPU overhead increases with rank count
- **Root Cause:** One proxy thread per connection
- **Impact:** Large-scale multi-node jobs

### 4. **PCIe Contention**
- **Issue:** PCIe bandwidth shared across GPUs
- **Root Cause:** Physical topology constraints
- **Impact:** Systems without xGMI

### 5. **Collective Synchronization**
- **Issue:** Stragglers impact overall collective time
- **Root Cause:** Synchronous collective nature
- **Impact:** Heterogeneous systems or variable workloads

### 6. **Memory Fragmentation**
- **Issue:** Long-running jobs may fragment GPU memory
- **Root Cause:** Multiple allocations/deallocations
- **Impact:** Memory allocation failures

### 7. **Bootstrap Overhead**
- **Issue:** Slow initialization for large rank counts
- **Root Cause:** Sequential bootstrap protocol
- **Impact:** Job startup time

---

## Profiling and Debugging Infrastructure

### 1. **NCCL_DEBUG Environment Variable**
**Levels:**
- `VERSION`: Library version
- `WARN`: Warnings and errors
- `INFO`: Initialization and configuration
- `TRACE`: Detailed operation tracing

**Usage:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
```

### 2. **Collective Trace**
**Location:** `src/include/latency_profiler/`

**Features:**
- Per-collective latency tracking
- Detailed event timeline
- Enabled with `RCCL_COLL_TRACE_ENABLE=1`

### 3. **NPKit Profiling**
**Location:** `src/misc/npkit.cc`

**Purpose:**
- Fine-grained kernel profiling
- Event-based timeline
- Requires compile-time flag: `--npkit-enable`

### 4. **Proxy Trace**
**Location:** `src/misc/proxy_trace/`

**Purpose:**
- Trace proxy thread activity
- Network operation profiling

### 5. **RCCL Test Suite**
**Location:** `amd-dev/amd/rccl-tests/`

**Tests:**
- `all_reduce_perf`: AllReduce bandwidth/latency
- `all_gather_perf`: AllGather performance
- `sendrecv_perf`: Point-to-point performance
- Verifiable tests for correctness

**Usage:**
```bash
./all_reduce_perf -b 8 -e 128M -f 2 -g 8
# -b: start size (bytes)
# -e: end size (bytes)
# -f: size factor (multiply by 2 each iteration)
# -g: number of GPUs
```

### 6. **ROCm Profiling Tools**
- **rocprof:** GPU kernel profiling
- **roctracer:** API tracing
- **Perfetto:** Timeline visualization

---

## Next Steps for Bottleneck Analysis

### Phase 1: Profiling and Measurement
1. **Run rccl-tests with various message sizes and GPU counts**
   - Identify performance cliffs
   - Compare against theoretical peak bandwidth
   
2. **Enable detailed tracing**
   - `NCCL_DEBUG=TRACE`
   - Collective trace
   - NPKit profiling

3. **Profile specific collectives**
   - Focus on AllReduce (most common)
   - Measure time breakdown: kernel launch, data transfer, synchronization

### Phase 2: Code Analysis
1. **Algorithm selection logic** (`src/graph/tuning.cc`)
   - Are algorithms selected optimally?
   - Are thresholds tuned for MI300X?

2. **Kernel primitives** (`src/device/primitives.h`)
   - Are memory operations optimal?
   - Are atomic operations efficient?

3. **Transport layer** (`src/transport/`)
   - Network proxy overhead
   - Memory registration costs
   - RDMA path efficiency

### Phase 3: Optimization Targets
Based on profiling results:
1. **Kernel optimization** (if GPU-bound)
2. **Network optimization** (if network-bound)
3. **Algorithm tuning** (if sub-optimal routing)
4. **Protocol tuning** (if threshold issues)

---

## References

- **RCCL Documentation:** https://rocm.docs.amd.com/projects/rccl/
- **NCCL Paper:** "NCCL: Fast Multi-GPU Collective Communication"
- **MSCCL Paper:** "MSCCLang: Microsoft Collective Communication Language"
- **xGMI Whitepaper:** AMD Infinity Fabric documentation
- **Source Code:** `/Users/ahalperin/xai/amd-dev/amd/rccl/`

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-10-30 | AI Assistant | Initial comprehensive design overview |


