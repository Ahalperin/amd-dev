# RCCL Proxy Communication Protocol

## Table of Contents

1. [Overview](#overview)
2. [Message Types](#message-types)
3. [Operation Flow](#operation-flow)
4. [Asynchronous RPC Protocol](#asynchronous-rpc-protocol)
5. [UDS Protocol](#uds-protocol)
6. [Progress Protocol](#progress-protocol)

## Overview

The RCCL proxy system uses multiple communication protocols for different purposes:

1. **Shared Memory Protocol**: Main thread → Progress thread (operation posting)
2. **Socket-Based RPC**: Main thread ↔ Service thread (setup and control)
3. **UDS Protocol**: Main thread ↔ UDS thread (cuMem operations)
4. **Progress Protocol**: Progress thread ↔ Transport layer (network I/O)

## Message Types

### Proxy Message Types

**Location**: `src/include/proxy.h` (line 418)

```c
// NB: ncclProxyMsgTypeStr[] in proxy.cc needs to match
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,          // Initialize connection
  ncclProxyMsgSharedInit = 2,    // Initialize shared resources
  ncclProxyMsgSetup = 3,         // Setup connection
  ncclProxyMsgConnect = 4,       // Connect to peer
  ncclProxyMsgStart = 5,         // Start operation
  ncclProxyMsgClose = 6,         // Close connection
  ncclProxyMsgAbort = 7,         // Abort operations
  ncclProxyMsgStop = 8,          // Stop proxy
  ncclProxyMsgGetFd = 9,         // Get file descriptor (cuMem)
  ncclProxyMsgQueryFd = 10,      // Query file descriptor
  ncclProxyMsgRegister = 11,     // Register memory
  ncclProxyMsgDeregister = 12    // Deregister memory
};
```

### Message Type Details

#### ncclProxyMsgInit (1)

**Purpose**: Initialize a new proxy connection

**Direction**: Main thread → Service thread

**Request**:
```c
struct {
  int tpLocalRank;      // Top-parent local rank
  int tpRank;           // Top-parent rank
  // Transport-specific init data
};
```

**Response**: Transport-specific initialization data

**When Used**: First time a connection is needed

**State Transition**: `connUninitialized` → `connInitialized`

#### ncclProxyMsgSharedInit (2)

**Purpose**: Initialize shared resources for a connection

**Direction**: Main thread → Service thread

**Request**: Transport-specific shared resource info

**Response**: Shared resource handles/addresses

**When Used**: When connection uses shared buffers

**State Transition**: `connInitialized` → `connSharedInitialized`

#### ncclProxyMsgSetup (3)

**Purpose**: Setup connection parameters (buffers, channels, etc.)

**Direction**: Main thread → Service thread

**Request**:
```c
struct setupReq {
  int netDev;           // Network device ID
  int useGdr;           // Use GPU Direct RDMA
  int channelId;        // Channel ID
  int connIndex;        // Connection index
  int shared;           // Shared buffer flag
  uint32_t curr_hdp_reg; // HDP register (AMD)
  // Plus transport-specific fields
};
```

**Response**: Connection handle, buffer addresses

**When Used**: After Init, before Connect

**State Transition**: `connInitialized/connSharedInitialized` → `connSetupDone`

#### ncclProxyMsgConnect (4)

**Purpose**: Establish connection with peer

**Direction**: Main thread → Service thread

**Request**: Peer address/handle information

**Response**: Connection confirmation, peer handles

**When Used**: After Setup, ready to communicate

**State Transition**: `connSetupDone` → `connConnected`

#### ncclProxyMsgRegister (11)

**Purpose**: Register memory with network adapter

**Direction**: Main thread → Service thread

**Request**:
```c
struct {
  void* data;           // Memory address
  size_t size;          // Size to register
  int type;             // Memory type
};
```

**Response**:
```c
struct {
  void* mhandle;        // Memory handle from network plugin
};
```

**When Used**: Before using buffers for network operations

#### ncclProxyMsgDeregister (12)

**Purpose**: Deregister memory from network adapter

**Direction**: Main thread → Service thread

**Request**:
```c
struct {
  void* mhandle;        // Memory handle to deregister
};
```

**Response**: Status

**When Used**: During cleanup

#### ncclProxyMsgClose (6)

**Purpose**: Close a connection

**Direction**: Main thread → Service thread

**Request**: Connection identifier

**Response**: Acknowledgment

**When Used**: During communicator destruction

#### ncclProxyMsgAbort (7)

**Purpose**: Abort all operations

**Direction**: Main thread → Service thread

**Request**: None

**Response**: None (best effort)

**When Used**: On error or user abort

#### ncclProxyMsgStop (8)

**Purpose**: Stop proxy service

**Direction**: Main thread → Service thread

**Request**: None

**Response**: None

**When Used**: During communicator destruction

#### ncclProxyMsgGetFd (9)

**Purpose**: Get file descriptor for cuMem handle

**Direction**: Main thread → UDS thread

**Request**:
```c
struct {
  void* handle;         // cuMem handle
};
```

**Response**:
```c
struct {
  int fd;               // File descriptor
};
```

**When Used**: When using CUDA unified memory

#### ncclProxyMsgQueryFd (10)

**Purpose**: Query remote file descriptor

**Direction**: Main thread → UDS thread

**Request**:
```c
struct {
  int localFd;          // Local file descriptor
};
```

**Response**:
```c
struct {
  int remoteFd;         // Remote file descriptor
};
```

**When Used**: When mapping cuMem across processes

## Operation Flow

### Connection Setup Flow

```
Main Thread                Service Thread                Transport
    │                            │                          │
    │──── ncclProxyMsgInit ─────▶│                          │
    │                            │─── transport->init ─────▶│
    │                            │◀─────────────────────────┤
    │◀──── Response ─────────────┤                          │
    │                            │                          │
    │─ ncclProxyMsgSharedInit ──▶│                          │
    │  (if needed)               │─ transport->sharedInit ─▶│
    │                            │◀─────────────────────────┤
    │◀──── Response ─────────────┤                          │
    │                            │                          │
    │──── ncclProxyMsgSetup ────▶│                          │
    │                            │─── transport->setup ────▶│
    │                            │  • Allocate buffers      │
    │                            │  • Setup channels        │
    │                            │◀─────────────────────────┤
    │◀──── Response ─────────────┤                          │
    │                            │                          │
    │──── ncclProxyMsgConnect ──▶│                          │
    │                            │─── transport->connect ──▶│
    │                            │  • Exchange handles      │
    │                            │  • Establish connection  │
    │                            │◀─────────────────────────┤
    │◀──── Response ─────────────┤                          │
    │                            │                          │
    │  Connection Ready          │                          │
    │                            │                          │
```

### Memory Registration Flow

```
Main Thread                Service Thread                Network Plugin
    │                            │                          │
    │─── ncclProxyMsgRegister ──▶│                          │
    │                            │─── net->regMr() ────────▶│
    │                            │  • Register GPU memory   │
    │                            │  • Get mhandle           │
    │                            │◀─────────────────────────┤
    │◀──── mhandle ──────────────┤                          │
    │                            │                          │
    │  Use mhandle in operations │                          │
    │                            │                          │
    │  ...later...               │                          │
    │                            │                          │
    │─ ncclProxyMsgDeregister ──▶│                          │
    │                            │─── net->deregMr() ──────▶│
    │                            │◀─────────────────────────┤
    │◀──── Response ─────────────┤                          │
    │                            │                          │
```

### Operation Execution Flow

```
Main Thread          Ops Pool          Progress Thread       Transport
    │                   │                    │                  │
    │ 1. Post op        │                    │                  │
    ├──────────────────▶│                    │                  │
    │                   │                    │                  │
    │ 2. Signal         │                    │                  │
    ├──────────────────▶│                    │                  │
    │                   │                    │                  │
    │                   │ 3. Fetch ops       │                  │
    │                   │◀───────────────────┤                  │
    │                   │                    │                  │
    │                   │                    │ 4. ProxyAppend   │
    │                   │                    │    (create args) │
    │                   │                    │                  │
    │                   │                    │ 5. Progress op   │
    │                   │                    ├─────────────────▶│
    │                   │                    │                  │
    │                   │                    │ 6. Network I/O   │
    │                   │                    │◀─────────────────┤
    │                   │                    │                  │
    │ 7. Poll counters  │                    │ 8. Update        │
    │◀───────────────────────────────────────┤    counters      │
    │                   │                    │                  │
```

## Asynchronous RPC Protocol

### Request-Response Model

The service thread implements an asynchronous RPC protocol:

**Caller Side** (Main Thread):

```c
// 1. Async call - returns immediately
void* opId = uniqueId();
ncclProxyCallAsync(comm, proxyConn, msgType, reqBuff, reqSize, 
                   respSize, opId);

// 2. Poll for response
ncclResult_t result;
do {
  result = ncclPollProxyResponse(comm, proxyConn, respBuff, opId);
} while (result == ncclInProgress);

// 3. Handle response
if (result == ncclSuccess) {
  // Process respBuff
}
```

**Service Side** (Service Thread):

```c
// 1. Receive message
struct ncclIpcHdr hdr;
recv(sock, &hdr, sizeof(hdr), 0);

// 2. Receive request data
char reqBuff[hdr.reqSize];
recv(sock, reqBuff, hdr.reqSize, 0);

// 3. Process request
char respBuff[hdr.respSize];
ncclResult_t res = handleMessage(hdr.type, reqBuff, respBuff);

// 4. Send response header
struct ncclProxyRpcResponseHeader respHdr;
respHdr.opId = hdr.opId;
respHdr.res = res;
respHdr.respSize = hdr.respSize;
send(sock, &respHdr, sizeof(respHdr), 0);

// 5. Send response data
if (hdr.respSize > 0) {
  send(sock, respBuff, hdr.respSize, 0);
}
```

### Message Format

#### Request Message

```
┌───────────────────────────────────────┐
│       ncclIpcHdr                      │
│  • type: ncclProxyMsgType             │
│  • rank: sender rank                  │
│  • reqSize: request data size         │
│  • respSize: expected response size   │
│  • opId: operation identifier         │
│  • data[16]: inline data (optional)   │
├───────────────────────────────────────┤
│       Request Data                    │
│  (reqSize bytes, type-specific)       │
└───────────────────────────────────────┘
```

#### Response Message

```
┌───────────────────────────────────────┐
│  ncclProxyRpcResponseHeader           │
│  • opId: operation identifier         │
│  • res: ncclResult_t                  │
│  • respSize: response data size       │
├───────────────────────────────────────┤
│       Response Data                   │
│  (respSize bytes, type-specific)      │
└───────────────────────────────────────┘
```

### Blocking vs. Async Calls

#### Blocking Call

**Function**: `ncclProxyCallBlocking()`

**Location**: `src/proxy.cc`

**Implementation**:
```c
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, 
                                   struct ncclProxyConnector* proxyConn,
                                   int type, void* reqBuff, int reqSize,
                                   void* respBuff, int respSize) {
  void* opId = allocateOpId();
  
  // Call async
  NCCLCHECK(ncclProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize,
                               respSize, opId));
  
  // Spin until complete
  ncclResult_t ret;
  do {
    ret = ncclPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (ret == ncclInProgress);
  
  return ret;
}
```

**Use Case**: Simple operations where caller can wait

#### Async Call

**Function**: `ncclProxyCallAsync()`

**Location**: `src/proxy.cc`

**Implementation**:
```c
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm,
                                struct ncclProxyConnector* proxyConn,
                                int type, void* reqBuff, int reqSize,
                                int respSize, void* opId) {
  // Create async op
  ncclProxyAsyncOp* op;
  NCCLCHECK(ncclCalloc(&op, 1));
  op->type = type;
  op->reqSize = reqSize;
  op->respSize = respSize;
  op->opId = opId;
  
  // Copy request data
  if (reqSize > 0) {
    NCCLCHECK(ncclCalloc(&op->reqBuff, reqSize));
    memcpy(op->reqBuff, reqBuff, reqSize);
  }
  
  // Enqueue expected response
  NCCLCHECK(expectedProxyResponseEnqueue(proxyState, opId, respSize));
  
  // Send message header
  struct ncclIpcHdr hdr;
  hdr.type = type;
  hdr.rank = comm->rank;
  hdr.reqSize = reqSize;
  hdr.respSize = respSize;
  hdr.opId = opId;
  NCCLCHECK(ncclSocketSend(sock, &hdr, sizeof(hdr)));
  
  // Send request data
  if (reqSize > 0) {
    NCCLCHECK(ncclSocketSend(sock, reqBuff, reqSize));
  }
  
  return ncclSuccess;
}
```

**Use Case**: When caller needs to do other work while waiting

### Response Tracking

**Data Structure**: `ncclExpectedProxyResponse`

```c
struct ncclExpectedProxyResponse {
  void* opId;                               // Operation identifier
  int respSize;                             // Expected response size
  bool done;                                // Response received flag
  void* respBuff;                           // Pre-allocated response buffer
  ncclResult_t res;                         // Result code
  struct ncclExpectedProxyResponse* next;   // Next in queue
};
```

**Queue Management**:

```c
// Main thread side
proxyState->expectedResponses → [Op1] → [Op2] → [Op3] → NULL

// Enqueue when calling async
expectedProxyResponseEnqueue(state, opId, respSize);

// Service thread receives response
expectedProxyResponseStore(state, opId, respBuff, respSize, res);

// Main thread polls
expectedProxyResponseDequeue(state, opId, respBuff, &found);
```

## UDS Protocol

### Unix Domain Socket Communication

The UDS service thread handles cuMem-specific operations using Unix domain sockets with ancillary data for file descriptor passing.

### Message Format

```c
struct ncclIpcHdr {
  int type;                 // Message type (GetFd, QueryFd)
  int rank;                 // Sender rank
  int reqSize;              // Request size
  int respSize;             // Response size
  void *opId;               // Operation ID
  uint64_t data[16];        // Inline data (128 bytes)
};
```

### File Descriptor Passing

**Sending FD**:
```c
struct msghdr msg;
struct cmsghdr *cmsg;
char cmsgbuf[CMSG_SPACE(sizeof(int))];

msg.msg_control = cmsgbuf;
msg.msg_controllen = sizeof(cmsgbuf);

cmsg = CMSG_FIRSTHDR(&msg);
cmsg->cmsg_level = SOL_SOCKET;
cmsg->cmsg_type = SCM_RIGHTS;
cmsg->cmsg_len = CMSG_LEN(sizeof(int));
*(int*)CMSG_DATA(cmsg) = fd;

sendmsg(sock, &msg, 0);
```

**Receiving FD**:
```c
struct msghdr msg;
struct cmsghdr *cmsg;
char cmsgbuf[CMSG_SPACE(sizeof(int))];

msg.msg_control = cmsgbuf;
msg.msg_controllen = sizeof(cmsgbuf);

recvmsg(sock, &msg, 0);

cmsg = CMSG_FIRSTHDR(&msg);
if (cmsg && cmsg->cmsg_type == SCM_RIGHTS) {
  fd = *(int*)CMSG_DATA(cmsg);
}
```

### cuMem Operations

#### GetFd Operation

**Request**:
```c
struct {
  void* cuMemHandle;    // CUDA memory handle
};
```

**Response**:
```c
struct {
  int fd;               // File descriptor (via ancillary data)
};
```

**Flow**:
```
Main Thread              UDS Thread               CUDA Driver
    │                       │                         │
    │─── GetFd request ────▶│                         │
    │   (cuMemHandle)       │                         │
    │                       │─── cuMemExport ────────▶│
    │                       │◀───────────────────────┤
    │                       │   (fd)                  │
    │◀── Response ──────────┤                         │
    │   (fd via SCM_RIGHTS) │                         │
    │                       │                         │
```

#### QueryFd Operation

**Request**:
```c
struct {
  int localFd;          // Local file descriptor
};
```

**Response**:
```c
struct {
  int remoteFd;         // Corresponding remote FD
};
```

**Use Case**: Mapping file descriptors across processes

## Progress Protocol

### Transport Progress Function Interface

```c
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, 
                                            struct ncclProxyArgs*);
```

### Progress Function Contract

**Input**: `ncclProxyArgs` with operation parameters

**Output**: `ncclSuccess` or error

**Side Effects**:
- Update progress counters in `ncclProxySubArgs`
- Post/poll network operations
- Update `args->state` and `args->done`

**Return Conditions**:
- `ncclSuccess` + `args->state = ncclProxyOpProgress`: Continue progressing
- `ncclSuccess` + `args->state = ncclProxyOpNone`: Operation complete
- Error code: Operation failed

### Progress State Machine

```
State: ncclProxyOpReady
    │
    │ progress() called first time
    │ • Initialize sub-operations
    │ • Set up base counters
    ▼
State: ncclProxyOpProgress
    │
    │ progress() called repeatedly
    │ • Post network sends/receives
    │ • Poll for completions
    │ • Update counters
    │ • Check for completion
    ▼
    │ All sub-ops done?
    │   Yes ─▶ args->state = ncclProxyOpNone
    │   No ──▶ return ncclSuccess (stay in Progress)
    │
State: ncclProxyOpNone
    │
    │ removeOp() called
    │ • Remove from active list
    │ • Return to free pool
    ▼
Done
```

### Counter Protocol

**Counters** (per `ncclProxySubArgs`):

```c
uint64_t base;          // Base step number
uint64_t posted;        // Steps posted to network
uint64_t transmitted;   // Steps transmitted by GPU
uint64_t received;      // Steps received from network
uint64_t flushed;       // Steps flushed to memory
uint64_t done;          // Steps acknowledged complete
uint64_t end;           // End step number
```

**Counter Relationships**:

```
Send path:
  transmitted ≤ posted ≤ done ≤ end
  
Receive path:
  posted ≤ received ≤ flushed ≤ done ≤ end
```

**Update Protocol**:

1. **Proxy thread** updates:
   - `posted`: When network operation posted
   - `received`: When network data received
   - `flushed`: After memory flush (if GDR)

2. **GPU kernel** updates:
   - `transmitted`: When data written to buffer
   - `done`: When data consumed from buffer

3. **Synchronization**: All atomic operations

### Network Operation Posting

**Pattern**:
```c
// Check if can post more
if (sub->posted < sub->nsteps && 
    sub->posted < sub->done + NCCL_STEPS) {
  
  // Post operation
  size_t size = calculateSize(sub, sub->posted);
  void* buff = calculateBuffer(sub, sub->posted);
  
  NCCLCHECK(net->isend(sendComm, buff, size, 
                       &sub->requests[sub->posted % NCCL_STEPS]));
  
  // Update counter
  sub->posted++;
}

// Poll for completion
if (sub->done < sub->posted) {
  int done = 0;
  NCCLCHECK(net->test(sub->requests[sub->done % NCCL_STEPS], &done));
  if (done) {
    sub->done++;
  }
}
```

**Pipelining**: Use of `NCCL_STEPS` enables pipelining multiple operations

## Error Handling

### Error Propagation

1. **Service Thread Errors**:
   ```c
   // Store in response
   respHdr.res = ncclInternalError;
   send(sock, &respHdr, sizeof(respHdr));
   ```

2. **Progress Thread Errors**:
   ```c
   // Store in proxyState
   __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
   // Main thread will detect on next operation
   ```

3. **Transport Errors**:
   ```c
   // Returned from progress function
   return ncclRemoteError;
   // Operation removed, error propagated
   ```

### Abort Protocol

When abort flag set:

1. **Progress thread**:
   - Completes in-flight operations
   - Does not fetch new operations
   - Exits cleanly

2. **Service thread**:
   - Closes connections
   - Stops accepting new requests
   - Exits when all peers disconnect

3. **Main thread**:
   - No new operations posted
   - Waits for proxy threads to exit
   - Cleans up resources

## Summary

The RCCL proxy communication protocol provides:

1. **Multiple Channels**: Shared memory, sockets, UDS for different needs
2. **Asynchronous RPC**: Non-blocking communication with main threads
3. **Progress Protocol**: Efficient transport integration
4. **Counter-Based Sync**: Lock-free progress tracking
5. **Robust Error Handling**: Clear error propagation paths

Understanding these protocols is essential for:
- Debugging communication issues
- Adding new message types
- Optimizing protocol overhead
- Integrating new transports


