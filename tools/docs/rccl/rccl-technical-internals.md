# RCCL Technical Internals and Implementation Details

**Date:** October 30, 2025  
**Purpose:** Deep technical dive into RCCL implementation for optimization work  
**Related:** [RCCL Design Overview](rccl-design-overview.md), [Bottleneck Analysis](rccl-bottleneck-analysis.md)

---

## Table of Contents
1. [Data Structures Deep Dive](#data-structures-deep-dive)
2. [GPU Kernel Implementation](#gpu-kernel-implementation)
3. [Transport Layer Internals](#transport-layer-internals)
4. [Memory Management Details](#memory-management-details)
5. [Synchronization Mechanisms](#synchronization-mechanisms)
6. [Algorithm Implementation](#algorithm-implementation)
7. [Network Proxy Architecture](#network-proxy-architecture)
8. [Topology Graph and Path Search](#topology-graph-and-path-search)
9. [AMD-Specific Optimizations](#amd-specific-optimizations)

---

## Data Structures Deep Dive

### ncclComm Structure

**Location:** `src/include/comm.h` (lines ~150-500)

```c
struct ncclComm {
  // Basic identity
  int rank;                           // This rank's ID (0 to nRanks-1)
  int nRanks;                         // Total number of ranks in communicator
  int cudaDev;                        // GPU device ID
  int nvmlDev;                        // NVML device ID
  int compCap;                        // Compute capability
  
  // Channels (parallel execution paths)
  int nChannels;                      // Number of channels (typically 4-32)
  int p2pnChannels;                   // Number of P2P channels
  struct ncclChannel channels[MAXCHANNELS];  // Channel array
  struct ncclChannel* channelsShmem;  // Shared memory view of channels
  
  // Topology
  struct ncclTopoSystem* topo;        // Full topology graph
  int nNodes;                         // Number of nodes
  int node;                           // This node's ID
  int localRank;                      // Local rank within node
  int localRanks;                     // Number of local ranks
  
  // Shared resources
  struct ncclSharedResources* sharedRes;  // Shared across local ranks
  
  // Work management
  struct ncclTasks tasks;             // Pending work queue
  struct ncclKernelPlan* planner;     // Kernel launch planner
  
  // Streams
  hipStream_t userStream;             // User-provided stream
  hipStream_t groupStream;            // Internal group stream
  
  // State
  uint64_t opCount;                   // Operation counter
  ncclResult_t asyncResult;           // Async error state
  
  // Tuning parameters
  int* peerInfo;                      // Peer connectivity info
  struct ncclTopoRanks topoRanks;     // Rank topology mapping
  
  // Bootstrap
  struct ncclBootstrapHandle* bootstrap;  // Bootstrap communication
  
  // Proxy
  struct ncclProxyState* proxyState;  // Network proxy threads
  
  // Tuner plugin
  struct ncclTuner* tuner;            // External tuner plugin
  
  // ... many more fields
};
```

**Key Fields for Optimization:**

1. **nChannels**: Controls parallelism. More channels = higher bandwidth (up to a point)
2. **topo**: Contains all topology information for routing decisions
3. **sharedRes**: Shared between local ranks to reduce memory overhead
4. **tasks**: Work queue - reducing queue overhead can improve latency

---

### ncclChannel Structure

**Location:** `src/include/channel.h`

```c
struct ncclChannel {
  // Ring structure
  struct ncclRing {
    int prev;                         // Previous rank in ring
    int next;                         // Next rank in ring
    
    // Per-peer connections
    struct ncclChannelPeer* devPeers;  // Device-side peer structures
    
    // User buffers (for direct operations)
    int userRanks[NCCL_MAX_TREE_ARITY];
  } ring;
  
  // Tree structure
  struct ncclTree {
    int depth;                        // Tree depth
    int up;                           // Parent in tree
    int down[NCCL_MAX_TREE_ARITY];   // Children in tree
    
    struct ncclChannelPeer* devPeers;
  } tree;
  
  // CollNet structure
  struct ncclCollNet {
    int rank;
    int nHeads;
    // ... CollNet-specific fields
  } collNet;
  
  // Work tracking
  uint64_t workFifoHead;              // Work FIFO head pointer
  uint64_t workFifoTail;              // Work FIFO tail pointer
  
  // Device memory
  void* devMemBase;                   // Base pointer for device buffers
  size_t devMemSize;                  // Size of device memory allocation
};
```

**Connection Details:**

Each channel has connections to neighboring ranks based on the chosen algorithm (ring/tree). The `ncclChannelPeer` structure contains:
- Send/receive buffers
- Synchronization flags
- Connection state

---

### ncclChannelPeer Structure

**Location:** `src/include/channel.h`

```c
struct ncclChannelPeer {
  // Send side
  struct ncclConnector send;
  
  // Receive side  
  struct ncclConnector recv;
};

struct ncclConnector {
  // Transport type
  int transportComm;                  // Transport implementation
  
  // Buffers
  void* buffs[NCCL_NUM_PROTOCOLS];   // One buffer per protocol
  int buffSizes[NCCL_NUM_PROTOCOLS]; // Buffer sizes
  
  // Connection state
  struct ncclConnInfo* conn;          // Connection information
  
  // Transport-specific data
  void* transportResources;           // P2P, NET, SHM specific data
  
  // Memory handles
  struct ncclMemoryHandle* memHandle; // For memory registration
};
```

---

## GPU Kernel Implementation

### Kernel Launch Flow

**Entry Point:** `src/enqueue.cc:ncclLaunchKernel()` (~line 200)

```c
ncclResult_t ncclLaunchKernel(ncclComm_t comm, struct ncclWork* work) {
  // 1. Select kernel based on collective type and protocol
  void* kernelFn = selectKernel(work->coll, work->proto);
  
  // 2. Calculate grid/block dimensions
  int blockSize = NCCL_THREADS_PER_BLOCK;
  int gridSize = work->nChannels;
  
  // 3. Setup kernel arguments
  void* args[] = { &work };
  
  // 4. Launch
  hipLaunchKernel(kernelFn, gridSize, blockSize, args, 0, comm->userStream);
  
  return ncclSuccess;
}
```

### Kernel Structure

**Location:** `src/device/common.cu` (~line 500)

```cpp
template<typename T, typename RedOp, typename Proto>
__global__ void ncclKernel(struct ncclDevWorkColl work) {
  // Each thread block handles one channel
  int channelId = blockIdx.x;
  
  // Initialize primitives for this channel
  ncclPrimitives<T, RedOp, Proto> prims(work, channelId);
  
  // Calculate this channel's data slice
  size_t offset = work.count / work.nChannels * channelId;
  size_t count = work.count / work.nChannels;
  
  // Main transfer loop
  const int sliceSteps = work.sliceSteps;
  const int chunkSteps = work.chunkSteps;
  
  for (int slice = 0; slice < sliceSteps; slice++) {
    for (int chunk = 0; chunk < chunkSteps; chunk++) {
      size_t chunkOffset = offset + chunk * work.chunkSize;
      size_t chunkCount = min(work.chunkSize, count - chunk * work.chunkSize);
      
      // Execute transfer based on collective type
      if (work.redOp == ncclSum) {
        prims.send(work.sendbuff + chunkOffset, chunkCount);
        prims.recv(work.recvbuff + chunkOffset, chunkCount);
      }
    }
  }
  
  // Wait for all transfers to complete
  prims.wait();
}
```

### Primitive Implementation

**Location:** `src/device/primitives.h` (~line 100-800)

```cpp
template<typename T, typename RedOp, typename Proto>
class ncclPrimitives {
private:
  // Connection pointers
  T* sendBuffer;
  T* recvBuffer;
  volatile uint64_t* sendHead;
  volatile uint64_t* sendTail;
  volatile uint64_t* recvHead;
  volatile uint64_t* recvTail;
  
  // Synchronization
  __device__ void barrier() {
    __syncthreads();
  }
  
  __device__ void waitSend(int nelem) {
    // Wait until send buffer has space
    while ((*sendHead - *sendTail) * sizeof(T) < nelem * sizeof(T)) {
      // Busy wait or yield
    }
  }
  
  __device__ void waitRecv(int nelem) {
    // Wait until receive buffer has data
    while ((*recvHead - *recvTail) * sizeof(T) < nelem * sizeof(T)) {
      // Busy wait
    }
  }

public:
  __device__ void send(const T* src, int nelem) {
    // Wait for buffer space
    waitSend(nelem);
    
    // Copy data to send buffer
    for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
      sendBuffer[i] = src[i];
    }
    
    // Memory fence
    __threadfence_system();
    
    // Update head pointer (visible to receiver)
    if (threadIdx.x == 0) {
      atomicAdd((unsigned long long*)sendHead, nelem * sizeof(T));
    }
    
    barrier();
  }
  
  __device__ void recv(T* dst, int nelem) {
    // Wait for data availability
    waitRecv(nelem);
    
    // Copy data from receive buffer
    for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
      dst[i] = recvBuffer[i];
    }
    
    // Memory fence
    __threadfence();
    
    // Update tail pointer (signal to sender)
    if (threadIdx.x == 0) {
      atomicAdd((unsigned long long*)recvTail, nelem * sizeof(T));
    }
    
    barrier();
  }
  
  __device__ void recvReduceSend(T* dst, int nelem) {
    // Combined operation: recv, reduce, send
    // More efficient than separate operations
    
    waitRecv(nelem);
    waitSend(nelem);
    
    // Receive, reduce, and send in one pass
    for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
      T val = recvBuffer[i];
      T local = dst[i];
      T result = RedOp::reduce(local, val);
      dst[i] = result;
      sendBuffer[i] = result;
    }
    
    __threadfence_system();
    
    if (threadIdx.x == 0) {
      atomicAdd((unsigned long long*)recvTail, nelem * sizeof(T));
      atomicAdd((unsigned long long*)sendHead, nelem * sizeof(T));
    }
    
    barrier();
  }
};
```

**Performance Notes:**
1. **Memory coalescing**: Loop iteration `i = threadIdx.x; i < nelem; i += blockDim.x` must ensure coalesced access
2. **Atomic operations**: `atomicAdd` on head/tail can be bottleneck for small messages
3. **Thread synchronization**: `__syncthreads()` overhead increases with block size
4. **Memory fences**: `__threadfence_system()` required for P2P visibility, but has overhead

---

### Protocol Variants

#### Simple Protocol (`prims_simple.h`)

**Characteristics:**
- Minimal overhead
- Large buffers (typically 1 MB per channel)
- Best for large messages (>512 KB)
- No inline flags

```cpp
// Simplified structure
template<typename T, typename RedOp>
class SimplePrimitives {
  // Simple uses dedicated send/recv buffers
  T* sendBuff[NCCL_STEPS];  // Multiple buffers for pipelining
  T* recvBuff[NCCL_STEPS];
  
  // Step tracking
  int sendStep;
  int recvStep;
  
  __device__ void send(const T* src, int nelem) {
    // Direct copy to peer's memory
    int step = sendStep % NCCL_STEPS;
    T* dst = sendBuff[step];
    
    // Bulk copy
    copyToShmem(dst, src, nelem);
    
    // Signal peer
    postSend(step, nelem);
    sendStep++;
  }
};
```

#### LL Protocol (`prims_ll.h`)

**Characteristics:**
- 32-bit flags embedded in data
- Small buffers
- Best for small messages (<8 KB)
- Low latency synchronization

```cpp
// LL uses flag-embedded data
struct llData {
  uint32_t flag;   // Upper bits for synchronization
  uint32_t data;   // Lower bits for actual data
};

template<typename T, typename RedOp>
class LLPrimitives {
  __device__ void send(const T* src, int nelem) {
    // Pack data with flags
    for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
      llData packed;
      packed.flag = currentFlag;
      packed.data = src[i];
      
      // Write packed data
      sendBuff[i] = packed;
    }
    
    // No separate signaling needed - flags are inline
  }
  
  __device__ void recv(T* dst, int nelem) {
    // Wait for correct flag
    for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
      llData packed;
      do {
        packed = recvBuff[i];
      } while (packed.flag != expectedFlag);
      
      dst[i] = packed.data;
    }
  }
};
```

#### LL128 Protocol (`prims_ll128.h`)

**Characteristics:**
- 128-bit operations
- Flags in separate locations
- Best for medium messages (8 KB - 512 KB)
- Balance between LL and Simple

```cpp
template<typename T, typename RedOp>
class LL128Primitives {
  // Uses 128-bit loads/stores
  using PackType = uint4;  // 128 bits
  
  __device__ void send(const T* src, int nelem) {
    // Vectorized copy
    PackType* src4 = (PackType*)src;
    PackType* dst4 = (PackType*)sendBuff;
    int nelem4 = nelem / 4;
    
    for (int i = threadIdx.x; i < nelem4; i += blockDim.x) {
      dst4[i] = src4[i];  // 128-bit load/store
    }
  }
};
```

---

## Transport Layer Internals

### P2P Transport

**Location:** `src/transport/p2p.cc`

**Setup Flow:**

```c
ncclResult_t p2pSetup(struct ncclComm* comm, 
                      struct ncclConnect* connectInfo,
                      struct ncclConnector* send,
                      struct ncclConnector* recv) {
  // 1. Check P2P capability
  int p2p = 0;
  hipDeviceCanAccessPeer(&p2p, comm->cudaDev, remoteDev);
  if (!p2p) return ncclInternalError;
  
  // 2. Enable peer access
  hipDeviceEnablePeerAccess(remoteDev, 0);
  
  // 3. Exchange buffer pointers via bootstrap
  void* localBuff = send->buffs[NCCL_PROTO_SIMPLE];
  bootstrapSend(comm->bootstrap, remotePeer, &localBuff, sizeof(void*));
  
  void* remoteBuff;
  bootstrapRecv(comm->bootstrap, remotePeer, &remoteBuff, sizeof(void*));
  
  // 4. Store remote pointer (can now access directly)
  recv->buffs[NCCL_PROTO_SIMPLE] = remoteBuff;
  
  // 5. Setup synchronization
  volatile uint64_t* sendHead = allocShm(sizeof(uint64_t));
  volatile uint64_t* recvTail = allocShm(sizeof(uint64_t));
  
  // Exchange sync pointers
  exchangeSyncPointers(comm, sendHead, recvTail);
  
  return ncclSuccess;
}
```

**Data Transfer:**

For P2P/xGMI transfers, data movement is direct:
```
GPU 0 kernel writes to → GPU 1's memory
                          ↓
                     GPU 1 kernel reads
```

No CPU involvement, no copies - this is why xGMI is so fast.

**Key AMD Feature: xGMI**

xGMI (Infinity Fabric) provides direct GPU-to-GPU connectivity:
- MI200 (gfx90a): 36 GT/s per direction
- MI300X (gfx942): 48 GT/s per direction
- Point-to-point full-duplex
- Low latency (~1 us)

**Detection:**

```c
// src/graph/topo.cc:~line 800
ncclResult_t ncclTopoDetectXGMI(struct ncclTopoSystem* system) {
  // Read from sysfs
  for (int gpu1 = 0; gpu1 < nGpus; gpu1++) {
    for (int gpu2 = 0; gpu2 < nGpus; gpu2++) {
      char path[PATH_MAX];
      sprintf(path, "/sys/class/drm/card%d/device/link_type_%d", gpu1, gpu2);
      
      char linkType[16];
      if (readFile(path, linkType, sizeof(linkType)) == 0) {
        if (strcmp(linkType, "XGMI") == 0) {
          // Add xGMI link to topology graph
          addLink(system, gpu1, gpu2, LINK_C2C, width);
        }
      }
    }
  }
}
```

---

### Network Transport (InfiniBand)

**Location:** `src/transport/net_ib.cc`

**IB Verbs Flow:**

```c
// Initialization
struct ibv_context* context = ibv_open_device(device);
struct ibv_pd* pd = ibv_alloc_pd(context);
struct ibv_cq* cq = ibv_create_cq(context, cqSize, NULL, NULL, 0);
struct ibv_qp* qp = ibv_create_qp(pd, &qpInitAttr);

// Memory registration (for RDMA)
struct ibv_mr* mr = ibv_reg_mr(pd, buffer, size, 
                               IBV_ACCESS_LOCAL_WRITE | 
                               IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ);

// Post send
struct ibv_send_wr wr = {
  .wr_id = requestId,
  .sg_list = &sge,
  .num_sge = 1,
  .opcode = IBV_WR_RDMA_WRITE,  // RDMA write
  .send_flags = IBV_SEND_SIGNALED,
  .wr.rdma.remote_addr = remoteAddr,
  .wr.rdma.rkey = remoteKey
};

ibv_post_send(qp, &wr, &bad_wr);

// Poll completion
struct ibv_wc wc;
while (ibv_poll_cq(cq, 1, &wc) == 0) {
  // Busy wait
}
```

**GPUDirect RDMA:**

When enabled, GPU memory is registered directly:
```c
// Register GPU memory
hipDevicePtr_t gpuBuff;
hipMalloc(&gpuBuff, size);

// Register with IB
struct ibv_mr* mr = ibv_reg_mr(pd, gpuBuff, size, flags);

// Now RDMA can write directly to GPU memory!
```

**Proxy Thread Role:**

Network operations are handled by proxy threads to avoid blocking GPU kernels:

```c
// src/proxy.cc:~line 600
void* proxyThread(void* arg) {
  struct ncclProxyArgs* args = arg;
  
  while (true) {
    // 1. Check for GPU requests
    struct ncclProxyOp* op = checkWorkQueue();
    if (op == NULL) continue;
    
    // 2. Execute network operation
    if (op->type == ncclProxyOpSend) {
      ibv_post_send(op->qp, &op->wr, NULL);
      pollCompletion(op->cq);
    } else if (op->type == ncclProxyOpRecv) {
      ibv_post_recv(op->qp, &op->wr, NULL);
      pollCompletion(op->cq);
    }
    
    // 3. Signal GPU completion
    op->done = 1;
    __atomic_thread_fence(__ATOMIC_RELEASE);
  }
}
```

---

## Memory Management Details

### Buffer Allocation

**Location:** `src/allocator.cc`

RCCL uses a custom allocator for internal buffers:

```c
struct ncclMemoryPool {
  void* base;                    // Base address
  size_t size;                   // Total size
  size_t allocated;              // Currently allocated
  struct ncclMemoryBlock* blocks;  // Free list
};

ncclResult_t ncclMemoryPoolAlloc(struct ncclMemoryPool* pool, 
                                 size_t size, void** ptr) {
  // Round up to alignment
  size = ALIGN_UP(size, MEM_ALIGN);
  
  // Find free block
  struct ncclMemoryBlock* block = findFreeBlock(pool, size);
  if (block == NULL) {
    // Allocate new block
    hipMalloc(&block->ptr, size);
    block->size = size;
  }
  
  *ptr = block->ptr;
  pool->allocated += size;
  
  return ncclSuccess;
}
```

### Channel Buffers

Each channel allocates buffers for each protocol:

```c
// src/channel.cc:~line 200
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  
  // Allocate buffers for each protocol
  size_t buffSizes[] = {
    [NCCL_PROTO_LL] = NCCL_LL_BUFF_SIZE,        // 32 KB
    [NCCL_PROTO_LL128] = NCCL_LL128_BUFF_SIZE,  // 64 KB
    [NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_BUFF_SIZE // 1 MB
  };
  
  for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
    // Allocate per peer, per protocol
    for (int peer = 0; peer < nPeers; peer++) {
      void* buff;
      hipMalloc(&buff, buffSizes[proto] * NCCL_STEPS);
      channel->peers[peer].send.buffs[proto] = buff;
      
      hipMalloc(&buff, buffSizes[proto] * NCCL_STEPS);
      channel->peers[peer].recv.buffs[proto] = buff;
    }
  }
  
  return ncclSuccess;
}
```

**Memory Consumption:**

For 8 GPUs, 8 channels, 3 protocols:
- Per channel, per peer: (32KB + 64KB + 1MB) × 2 (send/recv) × NCCL_STEPS(8) ≈ 18 MB
- Total per channel: 18 MB × 7 peers = 126 MB
- Total for 8 channels: 126 MB × 8 = ~1 GB

This is why memory optimization is important for many-channel configurations.

---

## Synchronization Mechanisms

### Device-Side Synchronization

**1. Intra-Block Synchronization:**
```cpp
__syncthreads();  // All threads in block must reach this point
```

**2. Inter-Block Synchronization:**
```cpp
// No native inter-block sync on GPU
// Must use atomic flags in global memory

__device__ void barrierAcrossBlocks(int nBlocks, int* flag) {
  if (threadIdx.x == 0) {
    atomicAdd(flag, 1);
    
    // Wait for all blocks
    while (atomicAdd(flag, 0) < nBlocks) {
      // Busy wait
    }
  }
  __syncthreads();
}
```

**3. Inter-GPU Synchronization:**
```cpp
// Using atomic flags in peer-accessible memory

__device__ void signalPeer(volatile uint64_t* peerFlag, uint64_t value) {
  atomicExch((unsigned long long*)peerFlag, value);
  __threadfence_system();  // Ensure visibility to peer GPU
}

__device__ void waitPeer(volatile uint64_t* localFlag, uint64_t expected) {
  while (atomicAdd((unsigned long long*)localFlag, 0) != expected) {
    // Busy wait
  }
  __threadfence_system();
}
```

### Host-Device Synchronization

**1. Stream Synchronization:**
```cpp
hipStreamSynchronize(stream);  // Wait for all operations on stream
```

**2. Event-Based:**
```cpp
hipEvent_t event;
hipEventCreate(&event);
hipEventRecord(event, stream);
hipEventSynchronize(event);
```

**3. Async Checks:**
```cpp
// RCCL uses async error checking
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  *asyncError = comm->asyncResult;
  return ncclSuccess;
}
```

---

## Algorithm Implementation

### Ring AllReduce

**Location:** `src/graph/rings.cc`

**Ring Construction:**

```c
ncclResult_t ncclTopoComputeRings(struct ncclTopoSystem* system) {
  // Goal: Find N rings (one per channel) that maximize bandwidth
  
  for (int channel = 0; channel < nChannels; channel++) {
    int* ring = malloc(nRanks * sizeof(int));
    
    // Start with rank 0
    ring[0] = 0;
    int prev = 0;
    
    for (int step = 1; step < nRanks; step++) {
      // Find next rank with best path from prev
      int next = -1;
      float bestBw = 0;
      
      for (int candidate = 0; candidate < nRanks; candidate++) {
        if (inRing(candidate, ring, step)) continue;
        
        float bw = getPathBandwidth(system, prev, candidate);
        if (bw > bestBw) {
          bestBw = bw;
          next = candidate;
        }
      }
      
      ring[step] = next;
      prev = next;
    }
    
    // Close the ring
    verifyPath(system, ring[nRanks-1], ring[0]);
    
    // Store ring
    storeRing(comm, channel, ring);
  }
}
```

**Ring Execution (Device Side):**

```cpp
// AllReduce = ReduceScatter + AllGather

__device__ void ringAllReduce(T* data, int count) {
  int nRanks = comm->nRanks;
  int rank = comm->rank;
  
  // Phase 1: Reduce-Scatter
  // Each rank reduces 1/N of data
  int chunkSize = count / nRanks;
  
  for (int step = 0; step < nRanks - 1; step++) {
    int sendChunk = (rank - step + nRanks) % nRanks;
    int recvChunk = (rank - step - 1 + nRanks) % nRanks;
    
    T* sendPtr = data + sendChunk * chunkSize;
    T* recvPtr = data + recvChunk * chunkSize;
    
    // Send current chunk, receive next chunk
    prims.send(sendPtr, chunkSize);
    prims.recvReduce(recvPtr, chunkSize);  // Receive and reduce
  }
  
  // Phase 2: AllGather
  // Distribute reduced chunks
  for (int step = 0; step < nRanks - 1; step++) {
    int sendChunk = (rank - step + 1 + nRanks) % nRanks;
    int recvChunk = (rank - step + nRanks) % nRanks;
    
    T* sendPtr = data + sendChunk * chunkSize;
    T* recvPtr = data + recvChunk * chunkSize;
    
    prims.send(sendPtr, chunkSize);
    prims.recv(recvPtr, chunkSize);
  }
}
```

---

## Network Proxy Architecture

**Location:** `src/proxy.cc`

### Proxy State Machine

```c
struct ncclProxyOp {
  ncclProxyOpType_t type;         // Send, Recv, etc.
  struct ncclProxyConnection* connection;
  void* reqBuff;                  // Request buffer
  void* respBuff;                 // Response buffer
  size_t reqSize;
  size_t respSize;
  volatile int done;              // Completion flag
  struct ncclProxyOp* next;       // Next in queue
};

struct ncclProxyState {
  pthread_t thread;               // Proxy thread
  struct ncclProxyOp* opQueue;    // Operation queue
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int stop;                       // Stop signal
};
```

### Proxy Thread Loop

```c
void* ncclProxyFunc(void* arg) {
  struct ncclProxyState* state = arg;
  
  while (!state->stop) {
    // 1. Get next operation from queue
    pthread_mutex_lock(&state->mutex);
    struct ncclProxyOp* op = state->opQueue;
    if (op) {
      state->opQueue = op->next;
    }
    pthread_mutex_unlock(&state->mutex);
    
    if (op == NULL) {
      // No work, sleep briefly
      usleep(1);
      continue;
    }
    
    // 2. Execute operation
    if (op->type == ncclProxyOpSend) {
      executeSend(op);
    } else if (op->type == ncclProxyOpRecv) {
      executeRecv(op);
    }
    
    // 3. Mark done
    __atomic_store_n(&op->done, 1, __ATOMIC_RELEASE);
  }
  
  return NULL;
}
```

### GPU-Proxy Communication

```cpp
// GPU kernel requests proxy operation
__device__ void requestProxySend(void* data, size_t size) {
  // 1. Allocate operation slot
  int slot = atomicAdd(&proxyReqHead, 1);
  struct ncclProxyOp* op = &proxyOps[slot % MAX_OPS];
  
  // 2. Fill operation
  op->type = ncclProxyOpSend;
  op->reqBuff = data;
  op->reqSize = size;
  op->done = 0;
  
  // 3. Signal proxy
  __threadfence_system();
  
  // 4. Wait for completion (or continue and check later)
  while (!op->done) {
    // GPU busy-waits or does other work
  }
}
```

---

## Topology Graph and Path Search

### Graph Representation

**Location:** `src/graph/topo.h`, `src/graph/topo.cc`

```c
struct ncclTopoSystem {
  int nNodes;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
  
  // Path cache
  struct ncclTopoPath paths[NCCL_TOPO_MAX_NODES][NCCL_TOPO_MAX_NODES];
};

struct ncclTopoNode {
  ncclTopoNodeType_t type;  // GPU, PCI, CPU, NIC, NET
  int id;
  
  // Links to other nodes
  int nLinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  
  // Node properties
  union {
    struct { int dev; } gpu;
    struct { int domain; int bus; } pci;
    struct { int numa; } cpu;
    struct { int port; char* devName; } nic;
  };
};

struct ncclTopoLink {
  int remoteNode;           // Target node index
  int type;                 // LINK_C2C, LINK_PCI, etc.
  float bandwidth;          // GB/s
  float latency;            // us
};

struct ncclTopoPath {
  int type;                 // PATH_C2C, PATH_PHB, etc.
  int count;                // Number of hops
  float bandwidth;          // Bottleneck bandwidth
  float latency;            // Total latency
  int hops[NCCL_TOPO_MAX_HOPS];
};
```

### Path Search Algorithm

**Location:** `src/graph/search.cc`

```c
ncclResult_t ncclTopoSearchPath(struct ncclTopoSystem* system, 
                                int fromNode, int toNode,
                                struct ncclTopoPath* path) {
  // Dijkstra-like search
  
  float dist[NCCL_TOPO_MAX_NODES];
  int prev[NCCL_TOPO_MAX_NODES];
  int visited[NCCL_TOPO_MAX_NODES] = {0};
  
  // Initialize
  for (int i = 0; i < system->nNodes; i++) {
    dist[i] = INFINITY;
  }
  dist[fromNode] = 0;
  
  // Main loop
  for (int iter = 0; iter < system->nNodes; iter++) {
    // Find minimum distance unvisited node
    int u = -1;
    float minDist = INFINITY;
    for (int i = 0; i < system->nNodes; i++) {
      if (!visited[i] && dist[i] < minDist) {
        minDist = dist[i];
        u = i;
      }
    }
    
    if (u == -1 || u == toNode) break;
    visited[u] = 1;
    
    // Update neighbors
    struct ncclTopoNode* node = &system->nodes[u];
    for (int l = 0; l < node->nLinks; l++) {
      int v = node->links[l].remoteNode;
      float weight = 1.0 / node->links[l].bandwidth;  // Lower BW = higher cost
      
      if (dist[u] + weight < dist[v]) {
        dist[v] = dist[u] + weight;
        prev[v] = u;
      }
    }
  }
  
  // Reconstruct path
  path->count = 0;
  path->bandwidth = INFINITY;
  path->latency = 0;
  
  int current = toNode;
  while (current != fromNode) {
    path->hops[path->count++] = current;
    
    // Find link
    int p = prev[current];
    struct ncclTopoLink* link = findLink(system, p, current);
    
    // Update metrics
    path->bandwidth = min(path->bandwidth, link->bandwidth);
    path->latency += link->latency;
    
    current = p;
  }
  
  // Classify path type
  path->type = classifyPath(system, path);
  
  return ncclSuccess;
}
```

---

## AMD-Specific Optimizations

### 1. xGMI-Aware Path Selection

```c
// Prefer xGMI links over PCIe
int scorePath(struct ncclTopoPath* path) {
  int score = 0;
  
  if (path->type == PATH_C2C) {
    score += 1000;  // xGMI path - highest priority
  } else if (path->type == PATH_PIX) {
    score += 500;   // Single PCIe bridge
  } else if (path->type == PATH_PHB) {
    score += 100;   // Through CPU
  }
  
  // Add bandwidth score
  score += (int)path->bandwidth;
  
  return score;
}
```

### 2. MI300-Specific Tuning

```c
// src/graph/tuning.cc
if (comm->arch == GFX942) {  // MI300X
  // Larger buffer sizes for higher bandwidth
  comm->buffSizes[NCCL_PROTO_SIMPLE] = 2 * 1024 * 1024;  // 2 MB
  
  // More channels for better parallelism
  if (comm->nChannels < 16) comm->nChannels = 16;
  
  // Protocol thresholds
  comm->llThreshold = 8192;
  comm->ll128Threshold = 524288;
  
  // Enable cheap fence by default
  if (!rcclParamGfx9CheapFenceOff()) {
    comm->cheapFenceEnabled = 1;
  }
}
```

### 3. ROCm-Specific Memory Operations

```cpp
// Use ROCm-specific memory copy for efficiency
__device__ void copyData(void* dst, void* src, size_t size) {
  #if defined(__HIP_PLATFORM_AMD__)
    // AMD-optimized memory copy
    __builtin_amdgcn_s_memtime();  // Timestamp
    
    // Use LDS for staging if beneficial
    if (size < LDS_SIZE) {
      __shared__ char lds[LDS_SIZE];
      // Copy through LDS for better coalescing
      copyThroughLDS(dst, src, size, lds);
    } else {
      // Direct copy with vector instructions
      copyVectorized(dst, src, size);
    }
  #else
    memcpy(dst, src, size);
  #endif
}
```

---

## Performance Tuning Parameters

### Critical Environment Variables for Optimization

| Variable | Purpose | Recommended Values |
|----------|---------|-------------------|
| `NCCL_NCHANNELS` | Number of channels | 8-32 (test for optimal) |
| `NCCL_MIN_NCHANNELS` | Minimum channels | 4-8 |
| `NCCL_MAX_NCHANNELS` | Maximum channels | 16-32 |
| `NCCL_P2P_LEVEL` | P2P preference | `SYS` (full system P2P) |
| `NCCL_PROTO` | Force protocol | `Simple`, `LL`, `LL128` |
| `NCCL_ALGO` | Force algorithm | `Ring`, `Tree` |
| `NCCL_BUFFSIZE` | Buffer size | 1048576-8388608 (1-8MB) |
| `RCCL_FORCE_XGMI` | Force xGMI detection | 1 (enable) |

### Tuning for Different Workloads

**1. Large Message Throughput (>1 MB):**
```bash
export NCCL_NCHANNELS=16
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=8388608  # 8 MB
export NCCL_ALGO=Ring
```

**2. Small Message Latency (<64 KB):**
```bash
export NCCL_NCHANNELS=4
export NCCL_PROTO=LL
export NCCL_ALGO=Tree
```

**3. Multi-Node:**
```bash
export NCCL_NCHANNELS=8
export NCCL_NET="IB"
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Use specific IB devices
export NCCL_IB_GID_INDEX=3         # RoCE mode
export NCCL_IB_TIMEOUT=22          # Increase timeout
```

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-10-30 | AI Assistant | Initial technical internals documentation |


