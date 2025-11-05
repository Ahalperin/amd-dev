# RCCL Proxy Transport Integration

## Table of Contents

1. [Overview](#overview)
2. [Transport Interface](#transport-interface)
3. [Network Transport](#network-transport)
4. [Shared Memory Transport](#shared-memory-transport)
5. [P2P Transport](#p2p-transport)
6. [CollNet Transport](#collnet-transport)
7. [Adding New Transports](#adding-new-transports)

## Overview

RCCL's proxy thread system integrates with multiple transport mechanisms to enable communication across different hardware configurations. Each transport provides specialized proxy progress functions that handle the specific requirements of that transport type.

## Transport Interface

### Transport Structure

**Location**: `src/include/transport.h`

```c
struct ncclTransport {
  const char name[4];
  
  // Setup functions
  ncclResult_t (*canConnect)(int* ret, struct ncclTopoSystem* topo, 
                             struct ncclTopoGraph* graph, 
                             struct ncclPeerInfo* info1,
                             struct ncclPeerInfo* info2);
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph,
                        struct ncclPeerInfo* myInfo, 
                        struct ncclPeerInfo* peerInfo,
                        struct ncclConnect* connectInfo, 
                        struct ncclConnector* send, int channelId);
  
  // Connection functions
  struct {
    ncclResult_t (*connect)(struct ncclComm* comm, 
                            struct ncclConnect* connectInfo,
                            int nranks, int rank, 
                            struct ncclConnector* send);
    ncclResult_t (*free)(struct ncclConnector* send);
    
    // Proxy functions
    ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection,
                               struct ncclProxyState* proxyState,
                               void* reqBuff, int reqSize, 
                               void* respBuff, int respSize);
    ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection,
                                 struct ncclProxyState* proxyState,
                                 void* reqBuff, int reqSize, 
                                 void* respBuff, int respSize);
    ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection,
                              struct ncclProxyState* proxyState);
    proxyProgressFunc_t proxyProgress;  // Progress function
  } send;
  
  struct {
    // Similar structure for recv
  } recv;
};
```

### Key Integration Points

1. **Setup Phase**: `proxySetup()` - Allocate resources, setup buffers
2. **Connection Phase**: `proxyConnect()` - Establish connection with peer
3. **Progress Phase**: `proxyProgress()` - Execute communication operations
4. **Cleanup Phase**: `proxyFree()` - Release resources

### Progress Function Signature

```c
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState* proxyState,
                                            struct ncclProxyArgs* args);
```

**Called By**: Progress thread in `progressOps()`

**Frequency**: Every iteration of progress loop while operation active

**Responsibility**: 
- Post network operations
- Poll for completions
- Update progress counters
- Signal completion when done

## Network Transport

### Overview

The network transport (`src/transport/net.cc`) handles communication over network adapters (InfiniBand, RoCE, TCP, etc.) using the NCCL network plugin interface.

### Resources Structure

**Send Resources**:
```c
struct sendNetResources {
  ncclNetDeviceHandle_t* netDeviceHandle;  // Network device handle
  void* netSendComm;                       // Network send communicator
  struct ncclSendMem* sendMem;             // Send memory (host)
  struct ncclRecvMem* recvMem;             // Receive memory (host)
  
  uint64_t step;                           // Current step counter
  void* mhandles[NCCL_NUM_PROTOCOLS];      // Memory handles per protocol
  void* buffs[NCCL_NUM_PROTOCOLS];         // Buffers per protocol
  int buffSizes[NCCL_NUM_PROTOCOLS];       // Buffer sizes per protocol
  
  void* reqFifo[NCCL_STEPS];               // Request FIFO
  bool useGdr;                             // GPU Direct RDMA enabled
  uint32_t* gdcSync;                       // GDR sync pointer
  struct ncclProxyConnection* proxyConn;   // Associated connection
};
```

**Receive Resources**: Similar structure for receive direction

### Send Progress Function

**Location**: `src/transport/net.cc` (line 1245)

```c
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, 
                                      struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    // Initialize sub-operations
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources = 
        (struct sendNetResources*)(sub->connection->transportResources);
      
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
      
      if (!sub->reg)
        sub->sendMhandle = resources->mhandles[args->protocol];
    }
    args->state = ncclProxyOpProgress;
    args->hdp_flushed = 0;
  }
  
  args->idle = 1;
  
  // Progress each sub-operation
  for (int s=0; s<args->nsubs; s++) {
    struct ncclProxySubArgs* sub = args->subs+s;
    struct sendNetResources* resources = 
      (struct sendNetResources*)(sub->connection->transportResources);
    
    // Check if we can post more sends
    if (sub->posted < sub->nsteps && 
        sub->posted < sub->done + NCCL_STEPS) {
      
      // Wait for GPU to write data
      if (sub->posted >= sub->transmitted) {
        sub->transmitted = *sub->connection->head;
        if (sub->posted >= sub->transmitted) {
          args->idle = 0;
          continue; // GPU hasn't written data yet
        }
      }
      
      // Flush HDP if needed (AMD GPU)
      if (resources->useGdr && !args->hdp_flushed) {
        // HDP flush code
        args->hdp_flushed = 1;
      }
      
      // Calculate buffer and size
      int buffSlot = sub->posted % NCCL_STEPS;
      int stepSize = sub->chunkSize;
      void* buff = NCCL_NET_MAP_GET_POINTER(/*...*/, buffs[args->protocol]);
      
      // Post send
      NCCLCHECK(proxyState->ncclNet->isend(
        resources->netSendComm,
        buff,
        stepSize,
        sub->sendMhandle,
        &sub->requests[buffSlot]));
      
      sub->posted++;
      args->idle = 0;
    }
    
    // Poll for completion
    if (sub->done < sub->posted) {
      int done = 0;
      NCCLCHECK(proxyState->ncclNet->test(
        sub->requests[sub->done % NCCL_STEPS],
        &done, NULL));
      
      if (done) {
        sub->done++;
        args->idle = 0;
      }
    }
  }
  
  // Check if all sub-operations complete
  if (args->idle) {
    for (int s=0; s<args->nsubs; s++) {
      if (args->subs[s].done < args->subs[s].nsteps) {
        return ncclSuccess;
      }
    }
    args->state = ncclProxyOpNone;
  }
  
  return ncclSuccess;
}
```

### Receive Progress Function

**Location**: `src/transport/net.cc`

Similar structure to send but handles:
- Posting irecv operations
- Flushing received data to GPU memory
- Updating receive counters
- Handling GDRCOPY optimizations

### Key Features

1. **GPU Direct RDMA (GDR)**:
   - Direct GPU memory access by NIC
   - Requires HDP flush on AMD GPUs
   - Reduces CPU involvement

2. **Pipelining**:
   - Multiple in-flight operations (NCCL_STEPS)
   - Overlaps network and GPU work
   - Improves bandwidth utilization

3. **Memory Registration**:
   - Buffers registered with network adapter
   - Handles stored in `mhandles`
   - Supports multiple protocols

4. **Flow Control**:
   - Tracks posted vs transmitted (send)
   - Tracks received vs flushed (recv)
   - Prevents buffer overruns

## Shared Memory Transport

### Overview

The shared memory transport (`src/transport/shm.cc`) handles intra-node communication using CPU memory or CUDA memcpy operations.

### When Proxy Used

**Normal SHM**: No proxy (direct GPU-to-GPU copy)

**SHM with cudaMemcpy**: Proxy used for CPU-assisted copies

**Enabled By**: `NCCL_SHM_USE_CUDA_MEMCPY=1` and `NCCL_SHM_MEMCPY_MODE`

### Resources Structure

```c
struct shmProxyInfo {
  struct ncclSendMem* ceRecvMem;    // Send memory
  void* devFifo;                     // Device FIFO
  int ceRecvMemSize;                 // Size
};
```

### Send Progress Function

**Location**: `src/transport/shm.cc`

```c
static ncclResult_t shmSendProxyProgress(struct ncclProxyState* proxyState,
                                         struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    args->state = ncclProxyOpProgress;
  }
  
  args->idle = 1;
  
  for (int s=0; s<args->nsubs; s++) {
    struct ncclProxySubArgs* sub = args->subs+s;
    
    // Check if can copy more chunks
    if (sub->posted < sub->nsteps && 
        sub->posted < sub->done + NCCL_STEPS) {
      
      // Wait for GPU to signal data ready
      uint64_t *tail = sub->connection->tail;
      if (sub->posted >= *tail) {
        args->idle = 0;
        continue;
      }
      
      // Calculate source and destination
      int slot = sub->posted % NCCL_STEPS;
      void* src = /* GPU buffer */;
      void* dst = /* Host/other GPU buffer */;
      size_t size = sub->chunkSize;
      
      // Perform copy
      CUDACHECK(cudaMemcpyAsync(dst, src, size, 
                                cudaMemcpyDeviceToDevice,
                                /* stream */));
      
      sub->posted++;
      args->idle = 0;
    }
    
    // Check for completion
    if (sub->done < sub->posted) {
      // Poll CUDA stream or check completion flag
      int done = checkCopyComplete(sub, sub->done % NCCL_STEPS);
      if (done) {
        sub->done++;
        args->idle = 0;
      }
    }
  }
  
  // Mark complete when all done
  if (args->idle) {
    for (int s=0; s<args->nsubs; s++) {
      if (args->subs[s].done < args->subs[s].nsteps) {
        return ncclSuccess;
      }
    }
    args->state = ncclProxyOpNone;
  }
  
  return ncclSuccess;
}
```

### Key Features

1. **CPU-Assisted Copies**:
   - Uses cudaMemcpy in proxy thread
   - Frees GPU from copy overhead
   - Can overlap with computation

2. **Locality Control**:
   - `NCCL_SHM_LOCALITY`: Choose send or receive side
   - Affects NUMA placement
   - Impacts performance

3. **Optional**: Only used when configured, otherwise direct GPU copies

## P2P Transport

### Overview

The P2P transport (`src/transport/p2p.cc`) handles direct GPU-to-GPU communication within a node using PCIe or NVLink/Infinity Fabric.

### When Proxy Used

**Normal P2P**: No proxy (direct GPU access)

**P2P with cudaMemcpy**: Proxy used for CPU-assisted copies

**Enabled By**: `NCCL_P2P_USE_CUDA_MEMCPY=1`

### Progress Function

**Location**: `src/transport/p2p.cc`

Similar to SHM proxy progress:
- Performs cudaMemcpy operations
- Manages pipelining
- Updates progress counters

### Key Features

1. **Direct GPU Access**:
   - GPUs can access each other's memory
   - No proxy needed in most cases
   - Lowest latency path

2. **Fallback Mode**:
   - Proxy used when direct access problematic
   - Can work around driver issues
   - Provides compatibility option

## CollNet Transport

### Overview

The CollNet transport (`src/transport/coll_net.cc`) handles collective operations offloaded to network hardware (e.g., InfiniBand SHARP, AWS EFA).

### Resources Structure

```c
struct ncclCollNetSharedRes {
  void* collNetComm;                    // CollNet communicator
  void* sendResources[MAXCHANNELS];     // Per-channel send resources
  void* recvResources[MAXCHANNELS];     // Per-channel recv resources
  int nChannels;                        // Number of channels
  int buffSize;                         // Buffer size
  void* buffs[NCCL_NUM_PROTOCOLS];      // Protocol buffers
  void* mhandles[NCCL_NUM_PROTOCOLS];   // Memory handles
};
```

### Send Progress Function

**Location**: `src/transport/coll_net.cc`

```c
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState,
                                      struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    // Setup for collective operation
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      // Initialize sub-operation
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  
  args->idle = 1;
  
  for (int s=0; s<args->nsubs; s++) {
    struct ncclProxySubArgs* sub = args->subs+s;
    struct ncclCollNetSharedRes* collRes = 
      sub->connection->collNet;
    
    // Post collective send (reduce, allreduce, etc.)
    if (sub->posted < sub->nsteps) {
      // Wait for GPU data
      if (sub->posted >= sub->transmitted) {
        sub->transmitted = *sub->connection->head;
        if (sub->posted >= sub->transmitted) {
          args->idle = 0;
          continue;
        }
      }
      
      // Post to CollNet
      void* sendBuff = /* ... */;
      void* recvBuff = /* ... */;
      size_t count = sub->nbytes / dtypeSize;
      
      NCCLCHECK(proxyState->ncclCollNet->iallreduce(
        collRes->collNetComm,
        sendBuff,
        recvBuff,
        count,
        args->dtype,
        args->redOp,
        &sub->requests[sub->posted % NCCL_STEPS]));
      
      sub->posted++;
      args->idle = 0;
    }
    
    // Poll for completion
    if (sub->done < sub->posted) {
      int done = 0;
      NCCLCHECK(proxyState->ncclCollNet->test(
        sub->requests[sub->done % NCCL_STEPS],
        &done));
      
      if (done) {
        // Flush to GPU if needed
        sub->done++;
        args->idle = 0;
      }
    }
  }
  
  // Mark complete
  if (args->idle) {
    for (int s=0; s<args->nsubs; s++) {
      if (args->subs[s].done < args->subs[s].nsteps) {
        return ncclSuccess;
      }
    }
    args->state = ncclProxyOpNone;
  }
  
  return ncclSuccess;
}
```

### Key Features

1. **Hardware Offload**:
   - Collective operations executed by NIC
   - Reduces GPU and CPU involvement
   - Lower latency, higher efficiency

2. **Hierarchical Collectives**:
   - Combines with intra-node operations
   - Used for inter-node communication
   - Transparent to user

3. **Vendor-Specific**:
   - Requires hardware support
   - Plugin-based implementation
   - Falls back to regular network if unavailable

## Adding New Transports

### Step 1: Define Transport Structure

```c
// In src/transport/mytransport.cc

static struct ncclTransport myTransport = {
  .name = "MYT",
  .canConnect = myCanConnect,
  .setup = mySetup,
  .send = {
    .connect = mySendConnect,
    .free = mySendFree,
    .proxySetup = mySendProxySetup,
    .proxyConnect = mySendProxyConnect,
    .proxyFree = mySendProxyFree,
    .proxyProgress = mySendProxyProgress,  // May be NULL
  },
  .recv = {
    // Similar for receive
  }
};

// Register transport
NCCL_PARAM(MyTransportEnable, "MY_TRANSPORT_ENABLE", 1);

ncclResult_t ncclMyTransportInit(struct ncclComm* comm) {
  if (ncclParamMyTransportEnable()) {
    comm->transports[TRANSPORT_MY] = &myTransport;
  }
  return ncclSuccess;
}
```

### Step 2: Implement Proxy Setup

```c
static ncclResult_t mySendProxySetup(
    struct ncclProxyConnection* connection,
    struct ncclProxyState* proxyState,
    void* reqBuff, int reqSize,
    void* respBuff, int respSize) {
  
  // Parse request
  struct setupReq* req = (struct setupReq*)reqBuff;
  
  // Allocate resources
  struct mySendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  
  // Setup buffers
  NCCLCHECK(allocateBuffers(resources, req));
  
  // Initialize transport
  NCCLCHECK(myTransportInit(resources, req));
  
  // Store in connection
  connection->transportResources = resources;
  
  // Fill response
  struct setupResp* resp = (struct setupResp*)respBuff;
  resp->buffAddr = resources->buff;
  resp->buffSize = resources->buffSize;
  
  return ncclSuccess;
}
```

### Step 3: Implement Proxy Connect

```c
static ncclResult_t mySendProxyConnect(
    struct ncclProxyConnection* connection,
    struct ncclProxyState* proxyState,
    void* reqBuff, int reqSize,
    void* respBuff, int respSize) {
  
  struct mySendResources* resources = connection->transportResources;
  struct connectReq* req = (struct connectReq*)reqBuff;
  
  // Establish connection with peer
  NCCLCHECK(myTransportConnect(resources, req->peerAddr));
  
  // Exchange handles
  struct connectResp* resp = (struct connectResp*)respBuff;
  resp->myHandle = resources->localHandle;
  
  return ncclSuccess;
}
```

### Step 4: Implement Progress Function

```c
static ncclResult_t mySendProxyProgress(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args) {
  
  // Initialize on first call
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct mySendResources* resources = 
        sub->connection->transportResources;
      
      // Initialize counters
      sub->base = resources->step;
      resources->step += sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  
  args->idle = 1;
  
  // Progress each sub-operation
  for (int s=0; s<args->nsubs; s++) {
    struct ncclProxySubArgs* sub = args->subs+s;
    struct mySendResources* resources = 
      sub->connection->transportResources;
    
    // Post operations
    while (sub->posted < sub->nsteps && 
           sub->posted < sub->done + NCCL_STEPS) {
      
      // Wait for GPU
      if (sub->posted >= sub->transmitted) {
        sub->transmitted = *sub->connection->head;
        if (sub->posted >= sub->transmitted) {
          args->idle = 0;
          break;
        }
      }
      
      // Post send
      NCCLCHECK(myTransportPost(resources, sub, sub->posted));
      sub->posted++;
      args->idle = 0;
    }
    
    // Poll for completion
    while (sub->done < sub->posted) {
      int done;
      NCCLCHECK(myTransportTest(resources, sub, sub->done, &done));
      if (!done) break;
      
      sub->done++;
      args->idle = 0;
    }
  }
  
  // Check completion
  if (args->idle) {
    for (int s=0; s<args->nsubs; s++) {
      if (args->subs[s].done < args->subs[s].nsteps) {
        return ncclSuccess;
      }
    }
    args->state = ncclProxyOpNone;
  }
  
  return ncclSuccess;
}
```

### Step 5: Implement Cleanup

```c
static ncclResult_t mySendProxyFree(
    struct ncclProxyConnection* connection,
    struct ncclProxyState* proxyState) {
  
  struct mySendResources* resources = connection->transportResources;
  if (resources == NULL) return ncclSuccess;
  
  // Close connections
  NCCLCHECK(myTransportClose(resources));
  
  // Free buffers
  NCCLCHECK(freeBuffers(resources));
  
  // Free structure
  free(resources);
  connection->transportResources = NULL;
  
  return ncclSuccess;
}
```

### Step 6: Register Transport

```c
// In src/init.cc or transport registration
NCCLCHECK(ncclMyTransportInit(comm));
```

### Best Practices

1. **Resource Management**:
   - Allocate in proxySetup
   - Free in proxyFree
   - Handle partial allocation failures

2. **Error Handling**:
   - Check all return codes
   - Clean up on errors
   - Return meaningful error codes

3. **Thread Safety**:
   - Progress function called only by progress thread
   - No locks needed within progress
   - Use atomics for shared state

4. **Performance**:
   - Minimize work in progress function
   - Batch operations when possible
   - Avoid memory allocation in hot path

5. **Testing**:
   - Test with various message sizes
   - Test error conditions
   - Verify cleanup on abort

## Summary

RCCL's proxy transport integration provides:

1. **Flexible Architecture**: Support for multiple transport types
2. **Unified Interface**: Common proxy progress model across transports
3. **Optimized Paths**: Transport-specific optimizations
4. **Extensibility**: Easy to add new transports
5. **Performance**: Efficient progress and pipelining

Understanding transport integration is essential for:
- Performance optimization
- Adding hardware support
- Debugging communication issues
- Tuning for specific networks


