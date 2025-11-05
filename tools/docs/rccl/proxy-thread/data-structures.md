# RCCL Proxy Thread Data Structures

## Table of Contents

1. [Overview](#overview)
2. [Core Structures](#core-structures)
3. [Operation Structures](#operation-structures)
4. [Connection Structures](#connection-structures)
5. [Memory Structures](#memory-structures)
6. [State Machines](#state-machines)
7. [Relationships and Flow](#relationships-and-flow)

## Overview

The RCCL proxy thread system uses a hierarchy of data structures to manage asynchronous communication operations. This document details each structure, its purpose, and relationships with other components.

## Core Structures

### ncclProxyState

**Location**: `src/include/proxy.h` (line 333)

**Purpose**: Main proxy state per communicator, contains all proxy-related resources.

```c
struct ncclProxyState {
  int refCount;                          // Reference counter
  int tpRank;                            // Top-parent rank
  int tpnRanks;                          // Total top-parent ranks
  int tpLocalnRanks;                     // Local top-parent ranks
  int cudaDev;                           // CUDA device ID
  int p2pnChannels;                      // P2P channel count
  int p2pChunkSize;                      // P2P chunk size
  int nChannels;                         // Total channels
  int buffSizes[NCCL_NUM_PROTOCOLS];     // Buffer sizes per protocol
  bool allocP2pNetLLBuffers;             // Allocate LL buffers flag
  bool dmaBufSupport;                    // DMA-BUF support flag
  
  ncclNet_t* ncclNet;                    // Network transport interface
  ncclCollNet_t* ncclCollNet;            // Collective network interface
  uint32_t* abortFlag;                   // Abort flag (shared)
  bool directMode;                       // Direct mode flag
  
  // Service threads
  pthread_t thread;                      // Main service thread
  pthread_t threadUDS;                   // UDS service thread
  struct ncclSocket* listenSock;         // Listening socket
  struct ncclIpcSocket ipcSock;          // IPC socket (UDS)
  int stop;                              // Stop flag
  CUcontext cudaCtx;                     // CUDA context
  ncclResult_t asyncResult;              // Async error result
  
  // Main thread resources
  union ncclSocketAddress* peerAddresses; // Peer TCP addresses
  struct ncclSocket* peerSocks;          // Peer sockets
  struct ncclProxyOps* proxyOps;         // Per-rank proxy ops
  void** sharedDevMems;                  // Shared device memories
  struct ncclIpcSocket peerIpcSock;      // Peer IPC socket
  uint64_t *peerAddressesUDS;            // Peer UDS addresses
  
  // Progress thread
  struct ncclProxyProgressState progressState;
  
  // Profiler support
  void* profilerContext;
  
  // Expected responses queue
  struct ncclExpectedProxyResponse* expectedResponses;
  
  // Proxy tracing
  std::unique_ptr<facebook_rccl::ProxyTrace> proxyTrace;
};
```

**Key Relationships**:
- One per `ncclComm` (communicator)
- Contains one `ncclProxyProgressState`
- Owns all proxy threads
- Manages all proxy connections

### ncclProxyProgressState

**Location**: `src/include/proxy.h` (line 272)

**Purpose**: State for the progress thread, manages active operations and pools.

```c
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;      // Shared ops pool
  ncclShmHandle_t handle;                // Shared memory handle
  char opsPoolShmSuffix[6];              // SHM name suffix
  
  pthread_t thread;                      // Progress thread handle
  volatile int stop;                     // Stop flag
  
  struct ncclProxyPeer** localPeers;     // Local peer info
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];  // Network comms
  
  struct ncclProxyArgs* active;          // Active operations list
  struct ncclProxyArgs* pool;            // Free args pool
  struct ncclProxyPool* pools;           // Memory pools
  
  int nextOps;                           // Next op to process
};
```

**Key Relationships**:
- Owned by `ncclProxyState`
- Shares `opsPool` with main threads
- Maintains `active` list of in-flight operations
- Manages memory pools for operation structures

## Operation Structures

### ncclProxyOp

**Location**: `src/include/proxy.h` (line 58)

**Purpose**: Describes a single communication operation posted by main thread.

```c
struct ncclProxyOp {
  struct ncclProxyConnection* connection;  // Connection to use
  ssize_t nbytes;                         // Operation size
  uint64_t opCount;                       // Operation counter
  int root:30;                            // Root rank (collectives)
  uint32_t connIndex:2;                   // Connection index
  int next;                               // Next op in list
  int nsteps;                             // Pipeline steps
  size_t chunkSize;                       // Chunk size
  size_t sliceSize;                       // Slice size
  size_t loopSize;                        // Loop size
  size_t loopOffset;                      // Loop offset
  size_t channelSize;                     // Channel size
  uint8_t sliceSteps;                     // Steps per slice
  uint8_t chunkSteps;                     // Steps per chunk
  uint8_t channelId;                      // Channel ID
  uint8_t dtype;                          // Data type
  uint8_t redOp;                          // Reduction operation
  uint8_t coll;                           // Collective type
  uint8_t pattern;                        // Communication pattern
  uint8_t protocol;                       // Protocol (Simple/LL/LL128)
  uint8_t algorithm;                      // Algorithm (Ring/Tree/etc)
  uint8_t reg;                            // Registration flag
  
  // Memory handles
  void* sendMhandle;                      // Send memory handle
  void* recvMhandle;                      // Recv memory handle
  uint8_t* sendbuff;                      // Send buffer
  uint8_t* recvbuff;                      // Recv buffer
  
  // Ring algorithm info
  int isOneRPN;                           // One rank per node
  RingAlgorithm *ringAlgo;                // Ring algorithm descriptor
  int nextRank;                           // Next rank in ring
  int prevRank;                           // Previous rank in ring
  
  union ncclProxyOpSpecifics specifics;   // Operation-specific data
  
  // Profiler support
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;
  bool incWorkCounter;                    // Increment work counter
  int eActivationMask;                    // Event activation mask
  void* taskEventHandle;                  // Task event handle
  int rank;                               // Rank
  int peer;                               // Peer rank
  pid_t pid;                              // Process ID
  void* profilerContext;                  // Profiler context
  uint64_t workCounter;                   // Work counter
  
  struct ncclProxyOp *enqNext;            // Enqueue list next
  
  // Tracing
  uint32_t totalBytes;                    // Total bytes
  facebook_rccl::ProxyTraceRecordKey traceKey;
  facebook_rccl::ProxyTraceExtraInfo traceInfo;
};
```

**Key Relationships**:
- Posted by main thread to `ncclProxyOpsPool`
- Converted to `ncclProxyArgs` by progress thread
- References `ncclProxyConnection` for transport

### ncclProxyArgs

**Location**: `src/include/proxy.h` (line 182)

**Purpose**: Batched operation arguments used by progress thread.

```c
struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];  // Sub-operations
  proxyProgressFunc_t progress;           // Progress function pointer
  int nsubs;                              // Number of sub-operations
  int done;                               // Done flag
  int onePPN;                             // One process per node
  uint64_t opCount;                       // Operation counter
  int sliceSteps;                         // Slice steps
  int chunkSteps;                         // Chunk steps
  size_t chunkSize;                       // Chunk size
  size_t totalSendSize;                   // Total send size
  size_t totalRecvSize;                   // Total receive size
  size_t sendSizePerRound;                // Send per round
  size_t recvSizePerRound;                // Receive per round
  uint8_t dtype;                          // Data type
  uint8_t redOp;                          // Reduction op
  uint8_t pattern;                        // Pattern
  uint8_t coll;                           // Collective
  uint8_t protocol;                       // Protocol
  uint8_t algorithm;                      // Algorithm
  int state;                              // Operation state
  char* sharedBuff[NCCL_STEPS];           // Shared buffers
  int sharedSize[NCCL_STEPS];             // Shared sizes
  
  int idle;                               // Idle flag
  uint64_t hdp_flushed;                   // HDP flush counter
  
  // Linked list management
  struct ncclProxyArgs* next;             // Next in active list
  struct ncclProxyArgs* nextPeer;         // Next for same peer
  struct ncclProxyArgs** proxyAppendPtr;  // Append pointer
  
  union ncclProxyOpSpecifics specifics;   // Op-specific data
  
  int prevRank;                           // Previous rank
  int nextRank;                           // Next rank
  int send;                               // Send direction
  int retry_total;                        // Retry counter
};
```

**Key Relationships**:
- Created from one or more `ncclProxyOp` by `ProxyAppend()`
- Contains array of `ncclProxySubArgs` for multi-channel operations
- Linked in `progressState->active` list
- Progress function pointer set by transport

### ncclProxySubArgs

**Location**: `src/include/proxy.h` (line 128)

**Purpose**: Per-channel sub-operation within a `ncclProxyArgs`.

```c
struct ncclProxySubArgs {
  struct ncclProxyConnection* connection; // Connection
  int reg;                                // Registration flag
  void* sendMhandle;                      // Send memory handle
  void* recvMhandle;                      // Recv memory handle
  uint8_t* sendbuff;                      // Send buffer
  uint8_t* recvbuff;                      // Recv buffer
  size_t offset;                          // Buffer offset
  ssize_t loopSize;                       // Loop size
  ssize_t loopOffset;                     // Loop offset
  int channelId;                          // Channel ID
  int nsteps;                             // Steps
  ssize_t nbytes;                         // Bytes to transfer
  ssize_t chunkSize;                      // Chunk size
  int peer;                               // Peer rank
  int isOneRPN;                           // One rank per node
  RingAlgorithm *ringAlgo;                // Ring algorithm
  int groupSize;                          // Group size
  
  // Progress tracking
  uint64_t base;                          // Base step
  uint64_t posted;                        // Posted steps
  uint64_t received;                      // Received steps
  uint64_t flushed;                       // Flushed steps
  uint64_t transmitted;                   // Transmitted steps
  uint64_t done;                          // Done steps
  uint64_t end;                           // End step
  int regBufferReady;                     // Reg buffer ready flag
  void* requests[NCCL_STEPS];             // Network requests
  
  // Profiler support
  int eActivationMask;
  int rank;
  pid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* kernelEventHandle;
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];
  size_t transSize;
  uint64_t workCounter;
  
  // Request caching
  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
  
#if defined(ENABLE_NPKIT)
  int npKitSizesFifo[NCCL_STEPS];
  uint64_t timestamp[NCCL_STEPS];
#endif
  
  // Tracing
  facebook_rccl::ProxyTraceRecordKey traceKey;
  facebook_rccl::ProxyTraceExtraInfo traceInfo;
};
```

**Key Relationships**:
- Multiple sub-args per `ncclProxyArgs` (one per channel/connection)
- Progress tracked independently via counters
- References transport connection

### ncclProxyOpsPool

**Location**: `src/include/proxy.h` (line 229)

**Purpose**: Shared memory pool for passing operations between main and proxy threads.

```c
struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;                   // Head of posted ops
  volatile int nextOpsEnd;                // Tail of posted ops
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS]; // Free lists per rank
  pthread_mutex_t mutex;                  // Protection mutex
  pthread_cond_t cond;                    // Wake-up condition variable
};
```

**Key Relationships**:
- Allocated in shared memory
- Shared between all local ranks and proxy threads
- Mutex-protected for concurrent access
- Circular buffer with free lists

## Connection Structures

### ncclProxyConnection

**Location**: `src/include/proxy.h` (line 388)

**Purpose**: Represents a communication connection to a peer.

```c
struct ncclProxyConnection {
  int send;                               // Send (1) or recv (0)
  int transport;                          // Transport type
  int shared;                             // Shared connection flag
  int tpLocalRank;                        // TP local rank
  int sameProcess;                        // Same process flag
  struct ncclSocket* sock;                // Socket (if remote)
  struct ncclTransportComm* tcomm;        // Transport comm
  struct ncclProxyArgs *proxyAppend;      // Append pointer
  struct ncclProxyArgs **proxyAppendPtr;  // Append pointer ptr
  void* transportResources;               // Transport-specific resources
  ncclNetDeviceHandle_t* netDeviceHandle; // Network device handle
  void* mhandles[NCCL_NUM_PROTOCOLS];     // Memory handles
  proxyConnectState state;                // Connection state
  struct ncclCollNetSharedRes* collNet;   // CollNet shared resources
  int needsProxyProgress;                 // Needs proxy progress flag
};
```

**Key Relationships**:
- Created during connection setup
- References transport-specific resources
- Maintains state machine for connection lifecycle
- Links to active operations via `proxyAppend`

### ncclProxyPeer

**Location**: `src/include/proxy.h` (line 257)

**Purpose**: Manages shared P2P resources for a peer.

```c
struct ncclProxyPeer {
  struct ncclProxySharedP2p send;         // Send shared resources
  struct ncclProxySharedP2p recv;         // Recv shared resources
};
```

### ncclProxySharedP2p

**Location**: `src/include/proxy.h` (line 247)

**Purpose**: Shared P2P resources between ranks.

```c
struct ncclProxySharedP2p {
  int refcount;                           // Reference count
  int64_t size;                           // Buffer size
  char* cudaBuff;                         // CUDA buffer
  char* hostBuff;                         // Host buffer
  ncclIpcDesc ipcDesc;                    // IPC descriptor
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS];  // Per-channel append
};
```

**Key Relationships**:
- Shared between multiple connections
- Reference counted for lifecycle management
- Used for P2P and shared buffer modes

## Memory Structures

### ncclProxyPool

**Location**: `src/proxy.cc` (line 44)

**Purpose**: Pool of pre-allocated `ncclProxyArgs` structures.

```c
struct ncclProxyPool {
  struct ncclProxyPool *next;             // Next pool in list
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];  // Elements
};
```

**Key Relationships**:
- Multiple pools linked together
- Elements allocated from pool as needed
- Returned to pool when operation completes

### ncclSharedNetComms

**Location**: `src/include/proxy.h` (line 262)

**Purpose**: Shared network communicators across channels.

```c
struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];         // Active connect flags
  int activeAccept[MAXCHANNELS];          // Active accept flags
  void* sendComm[MAXCHANNELS];            // Send communicators
  void* recvComm[MAXCHANNELS];            // Recv communicators
  int sendRefCount[MAXCHANNELS];          // Send ref counts
  int recvRefCount[MAXCHANNELS];          // Recv ref counts
};
```

**Key Relationships**:
- Shared across channels on same network device
- Reference counted for resource sharing
- Managed by service thread

## State Machines

### Proxy Operation State

**Enumeration**: `ncclProxyOpState`

```c
enum ncclProxyOpState {
  ncclProxyOpNone = 0,       // Not active / completed
  ncclProxyOpReady,          // Ready to start
  ncclProxyOpProgress        // In progress
};
```

**State Transitions**:
```
       ProxyAppend()
         ↓
    ncclProxyOpReady
         ↓
    progress() first call
         ↓
    ncclProxyOpProgress
         ↓
    progress() returns done
         ↓
    ncclProxyOpNone
         ↓
    removeOp()
```

### Proxy Connection State

**Enumeration**: `proxyConnectState`

```c
enum proxyConnectState {
  connUninitialized = 0,      // Not initialized
  connInitialized,            // Basic initialization
  connSharedInitialized,      // Shared resources ready
  connSetupDone,              // Setup complete
  connConnected,              // Fully connected
  numConnStates
};
```

**State Transitions**:
```
connUninitialized
    ↓ ncclProxyMsgInit
connInitialized
    ↓ ncclProxyMsgSharedInit (if shared)
connSharedInitialized
    ↓ ncclProxyMsgSetup
connSetupDone
    ↓ ncclProxyMsgConnect
connConnected
```

## Relationships and Flow

### Operation Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Thread                                  │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ 1. Create ncclProxyOp
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                ncclProxyOpsPool (Shared Memory)                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                   │
│  │ ProxyOp 1 │─▶│ ProxyOp 2 │─▶│ ProxyOp 3 │─▶ ...              │
│  └───────────┘  └───────────┘  └───────────┘                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ 2. Consume ops
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Progress Thread (ncclProxyProgress)                │
│                                                                 │
│  ncclProxyGetPostedOps()                                        │
│      ↓                                                          │
│  ProxyAppend() → Convert to ncclProxyArgs                       │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Active Operations List              │          │
│  │  ┌─────────────┐   ┌─────────────┐              │          │
│  │  │ ProxyArgs 1 │──▶│ ProxyArgs 2 │──▶ ...       │          │
│  │  │ • nsubs: 2  │   │ • nsubs: 1  │              │          │
│  │  │ • state     │   │ • state     │              │          │
│  │  └──────┬──────┘   └─────────────┘              │          │
│  │         │                                        │          │
│  │  ┌──────┴──────┐                                │          │
│  │  │ SubArgs[0]  │                                │          │
│  │  │ SubArgs[1]  │                                │          │
│  │  └─────────────┘                                │          │
│  └──────────────────────────────────────────────────┘          │
│      ↓                                                          │
│  progressOps()                                                  │
│      ↓                                                          │
│  args->progress() (transport-specific)                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ 3. Network I/O
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Transport Layer                               │
│  • sendProxyProgress()                                          │
│  • recvProxyProgress()                                          │
│  • Post/Poll network operations                                │
│  • Update counters in SubArgs                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Connection Setup Flow

```
Main Thread                Service Thread           Transport
    │                            │                      │
    │ ncclProxyCallAsync()       │                      │
    │  (ncclProxyMsgSetup)       │                      │
    ├───────────────────────────▶│                      │
    │                            │                      │
    │                            │ proxySetup()         │
    │                            ├─────────────────────▶│
    │                            │                      │
    │                            │◀─────────────────────┤
    │                            │  Setup Resources     │
    │                            │                      │
    │                            │ Send Response        │
    │◀───────────────────────────┤                      │
    │                            │                      │
    │ ncclPollProxyResponse()    │                      │
    │  (blocks until complete)   │                      │
    │                            │                      │
```

### Memory Pool Management

```
┌───────────────────────────────────────────────────────────┐
│         ncclProxyProgressState                            │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  pools ──▶ [Pool 1] ──▶ [Pool 2] ──▶ [Pool 3]            │
│             ↓            ↓            ↓                   │
│           elems[]      elems[]      elems[]               │
│                                                           │
│  pool ──▶ Free List                                       │
│           ↓                                               │
│       [Free Args] ──▶ [Free Args] ──▶ [Free Args]        │
│                                                           │
│  active ──▶ Active List                                   │
│             ↓                                             │
│         [Active Args] ──▶ [Active Args] ──▶ ...          │
│                                                           │
└───────────────────────────────────────────────────────────┘

Allocation:
  1. Need new ncclProxyArgs
  2. Take from 'pool' free list
  3. If empty, allocate new Pool, add to 'pools'
  4. Add to 'active' list

Deallocation:
  1. Operation completes
  2. Remove from 'active' list
  3. Return to 'pool' free list
```

## Size and Memory Considerations

### Key Constants

```c
#define NCCL_PROXY_MAX_SUBS     MAXCHANNELS  // Max sub-operations
#define MAX_OPS_PER_PEER        (2*MAXCHANNELS*2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH)
#define NCCL_MAX_PROXY_CONNECTIONS  (NCCL_MAX_LOCAL_RANKS+1)
#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS  // Per pool allocation
#define NCCL_STEPS              8             // Pipeline depth
```

### Memory Footprint

**ncclProxyOpsPool** (shared memory):
- Size: `sizeof(ncclProxyOp) * MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS`
- Typically: ~500 KB - 1 MB depending on configuration

**ncclProxyArgs pools** (per communicator):
- Initial: `sizeof(ncclProxyArgs) * PROXYARGS_ALLOCATE_SIZE`
- Grows dynamically as needed
- Each: ~10-20 KB per pool

**Total per communicator**:
- ~1-5 MB depending on configuration and active operations

## Performance Implications

### Cache Considerations

1. **ncclProxyArgs**: Frequently accessed, benefits from cache locality
2. **ncclProxySubArgs**: Array iteration, good for prefetching
3. **Progress counters**: Hot path, atomic operations may cause cache-line bouncing

### Lock Contention

1. **opsPool->mutex**: Contended between all posting threads and progress thread
   - Batching reduces frequency
2. **Connection-specific locks**: Usually low contention

### Memory Bandwidth

1. **Operation copying**: `ncclProxyOp` → `ncclProxyArgs` conversion
2. **Counter updates**: Atomic operations on shared counters
3. **Buffer access**: GPU/host memory access patterns

## Summary

The RCCL proxy data structures form a sophisticated system for managing asynchronous communication:

1. **Operation Flow**: `ncclProxyOp` → `ncclProxyArgs` → `ncclProxySubArgs`
2. **Memory Management**: Pool-based allocation for efficiency
3. **State Tracking**: Multiple state machines for lifecycle management
4. **Concurrency**: Lock-free where possible, protected where necessary
5. **Flexibility**: Supports multiple transports, protocols, and patterns

Understanding these structures is essential for:
- Debugging proxy-related issues
- Performance optimization
- Adding new features or transports
- Understanding operation flow and timing


