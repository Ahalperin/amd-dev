# Meta-TorchComm: Comprehensive Features Summary

**Document Version**: 1.0  
**Last Updated**: November 2, 2025  
**Repository**: [meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms)

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Backends](#supported-backends)
3. [Collective Operations](#collective-operations)
4. [CTRAN Library](#ctran-library)
5. [Advanced Features](#advanced-features)
6. [Data Types & Operations](#data-types--operations)
7. [Infrastructure Features](#infrastructure-features)
8. [API Reference](#api-reference)
9. [Installation & Build](#installation--build)
10. [Performance & Scalability](#performance--scalability)

---

## Overview

**TorchComm** is Meta's experimental communications library for PyTorch, providing a simplified, object-oriented API for distributed collective operations. It serves as the foundation for all of Meta's generative AI services and is designed to scale to **100,000+ GPUs**.

### Key Characteristics

- **Production-Ready**: Powers all Meta generative AI services
- **Multi-Backend**: Supports NVIDIA (NCCL/NCCLX), AMD (RCCL), and CPU (GLOO)
- **Scalable**: Tested at massive scale (100k+ GPUs)
- **PyTorch Native**: Seamless integration with PyTorch tensors and CUDA streams
- **Flexible**: Synchronous and asynchronous operation modes
- **Extensible**: Plugin-based backend architecture

### Repository Structure

```
torchcomms/
├── comms/
│   ├── torchcomms/          # Main Python API and backends
│   │   ├── nccl/            # NCCL backend
│   │   ├── ncclx/           # NCCLX backend (Meta)
│   │   ├── rccl/            # RCCL backend (AMD)
│   │   ├── gloo/            # GLOO backend (CPU)
│   │   └── transport/       # RDMA transport layer
│   ├── ctran/               # Core communication library
│   │   ├── algos/           # Algorithm implementations
│   │   └── backends/        # Network transport backends
│   ├── ncclx/               # NCCLX source (v2.27)
│   ├── utils/               # Utilities (logging, tracing, etc.)
│   └── common/              # Common algorithms
├── docs/                    # Documentation
└── scripts/                 # Build scripts
```

---

## Supported Backends

TorchComm provides a unified API across multiple backend implementations, allowing seamless switching based on hardware and requirements.

### 1. NCCLX (Meta's Production Backend)

**Status**: Production (Meta's Primary Backend)  
**Location**: `comms/ncclx/v2_27/`  
**Hardware**: NVIDIA GPUs

**Features**:
- Extended version of NCCL with Meta-specific optimizations
- Powers all Meta generative AI services (LLaMA, etc.)
- Production-tested at 100,000+ GPU scale
- Includes profiler extensions (`ext-profiler/`)
- Tuner support for performance optimization (`ext-tuner/`)
- Network plugin architecture (`ext-net/`)
- Header namespacing to coexist with standard NCCL

**Configuration**:
```bash
export USE_NCCLX=ON
./build_ncclx.sh
```

### 2. NCCL (Standard NVIDIA)

**Status**: Stable  
**Location**: `comms/torchcomms/nccl/`  
**Hardware**: NVIDIA GPUs

**Features**:
- Standard NVIDIA Collective Communications Library
- Uses PyTorch's built-in NCCL library (no separate build needed)
- Optimized for PCIe, NVLink, NVSwitch
- InfiniBand and TCP/IP support

**Configuration**:
```bash
export USE_NCCL=ON  # Default: ON
```

### 3. RCCL (AMD ROCm)

**Status**: Active Development  
**Location**: `comms/torchcomms/rccl/`  
**Hardware**: AMD GPUs (MI300, MI250, etc.)

**Features**:
- AMD ROCm Collective Communications Library
- Full support for AMD GPU architectures
- Compatible with Meta's training infrastructure
- Optimized for AMD Infinity Fabric

**Configuration**:
```bash
export ROCM_HOME=/opt/rocm
export RCCL_INCLUDE=$ROCM_HOME/include/rccl
export USE_RCCL=ON
./build_rccl.sh
```

**Prerequisites**:
```bash
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y
```

### 4. GLOO

**Status**: Stable  
**Location**: `comms/torchcomms/gloo/`  
**Hardware**: CPU

**Features**:
- CPU-based backend for CPU tensors
- Ideal for metadata transfer and control operations
- TCP socket-based communication
- No GPU required

**Configuration**:
```bash
export USE_GLOO=ON  # Default: ON
```

### Backend Selection

```python
import torch
from torchcomms import new_comm

# Select backend when creating communicator
device = torch.device("cuda")
comm = new_comm("ncclx", device, name="my_comm")  # or "nccl", "rccl", "gloo"
```

---

## Collective Operations

TorchComm provides a comprehensive set of collective operations, all supporting both synchronous and asynchronous execution modes.

### Basic Collectives

#### AllReduce
Perform reduction operation across all ranks, result available on all ranks.

**Signature**:
```python
all_reduce(tensor, op, async_op, hints=None, timeout=None) -> TorchWork
```

**Algorithms**:
- Ring
- ARG (AllReduce Group)
- Direct
- Shared memory optimized

**Example**:
```python
import torch
from torchcomms import new_comm, ReduceOp

comm = new_comm("ncclx", torch.device("cuda"))
tensor = torch.ones(1024, device="cuda") * comm.get_rank()

# Synchronous
comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

# Asynchronous
work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)
work.wait()
```

#### Broadcast
Broadcast tensor from root rank to all other ranks.

**Signature**:
```python
broadcast(tensor, root, async_op, hints=None, timeout=None) -> TorchWork
```

**Algorithms**:
- Binomial Tree
- Direct

#### Reduce
Reduce tensors from all ranks to a single root rank.

**Signature**:
```python
reduce(tensor, root, op, async_op, hints=None, timeout=None) -> TorchWork
```

#### Barrier
Synchronize all processes.

**Signature**:
```python
barrier(async_op, hints=None, timeout=None) -> TorchWork
```

### Gather/Scatter Operations

#### AllGather
Gather tensors from all ranks and distribute to all ranks.

**Signature**:
```python
all_gather(tensor_list, tensor, async_op, hints=None, timeout=None) -> TorchWork
all_gather_v(tensor_list, tensor, async_op, hints=None, timeout=None) -> TorchWork  # Variable sizes
all_gather_single(output, input, async_op, hints=None, timeout=None) -> TorchWork
```

**Algorithms**:
- Brucks Full-Feature
- Direct
- Ring
- Recursive Doubling
- Pipelined (AllGatherP)

**Example**:
```python
# AllGather with list of tensors
tensor = torch.ones(1024, device="cuda")
tensor_list = [torch.empty_like(tensor) for _ in range(comm.get_size())]
comm.all_gather(tensor_list, tensor, async_op=False)

# AllGather single tensor variant
output = torch.empty(comm.get_size() * 1024, device="cuda")
comm.all_gather_single(output, tensor, async_op=False)
```

#### ReduceScatter
Reduce tensors and scatter the results across ranks.

**Signature**:
```python
reduce_scatter(output, input_list, op, async_op, hints=None, timeout=None) -> TorchWork
reduce_scatter_v(output, input_list, op, async_op, hints=None, timeout=None) -> TorchWork
reduce_scatter_single(output, input, op, async_op, hints=None, timeout=None) -> TorchWork
```

**Algorithms**:
- Ring
- RHD (Ring Halving-Doubling)
- Direct

#### Scatter
Scatter tensors from root rank to all ranks.

**Signature**:
```python
scatter(output_tensor, input_tensor_list, root, async_op, hints=None, timeout=None) -> TorchWork
```

#### Gather
Gather tensors from all ranks to root rank.

**Signature**:
```python
gather(output_tensor_list, input_tensor, root, async_op, hints=None, timeout=None) -> TorchWork
```

### AllToAll Operations

#### AllToAll
Exchange data between all pairs of ranks.

**Signature**:
```python
all_to_all(output_tensor_list, input_tensor_list, async_op, hints=None, timeout=None) -> TorchWork
all_to_all_single(output, input, async_op, hints=None, timeout=None) -> TorchWork
all_to_all_v_single(output, input, output_split_sizes, input_split_sizes, async_op, hints=None, timeout=None) -> TorchWork
```

**Variants**:
- **Standard**: Fixed-size exchange
- **Deduplication**: Optimized for duplicate data (AllToAllDedup)
- **Dynamic**: Runtime-determined sizes (AllToAllvDynamic)
- **Variable-sized**: Different sizes per rank (AllToAllv)
- **Pipelined**: Overlapped communication (AllToAllP)
- **Split Non-Contiguous**: Non-contiguous memory layouts

**Example**:
```python
# Equal-sized all-to-all
input_tensor = torch.arange(1024 * comm.get_size(), device="cuda")
output_tensor = torch.empty_like(input_tensor)
comm.all_to_all_single(output_tensor, input_tensor, async_op=False)

# Variable-sized all-to-all
output_split_sizes = [100, 200, 300, 400]  # Different sizes per rank
input_split_sizes = [150, 250, 350, 250]
comm.all_to_all_v_single(output, input, output_split_sizes, input_split_sizes, async_op=False)
```

### Point-to-Point Operations

#### Send
Send a tensor to a destination rank.

**Signature**:
```python
send(tensor, dst, async_op, hints=None, timeout=None) -> TorchWork
```

#### Recv
Receive a tensor from a source rank.

**Signature**:
```python
recv(tensor, src, async_op, hints=None, timeout=None) -> TorchWork
```

#### Batch Send/Recv
Batch multiple point-to-point operations.

**Signature**:
```python
batch = comm.batch_op_create()
batch.send(tensor1, dst=1)
batch.recv(tensor2, src=2)
work = batch.issue(async_op=True)
work.wait()
```

**Example**:
```python
# Simple send/recv
if comm.get_rank() == 0:
    tensor = torch.ones(1024, device="cuda")
    comm.send(tensor, dst=1, async_op=False)
else:
    tensor = torch.zeros(1024, device="cuda")
    comm.recv(tensor, src=0, async_op=False)
```

### Operation Summary Table

| Operation | Input Ranks | Output Ranks | Variable Size | Description |
|-----------|-------------|--------------|---------------|-------------|
| AllReduce | All | All | No | Reduce across all ranks |
| Broadcast | Root | All | No | Root broadcasts to all |
| Reduce | All | Root | No | Reduce to single rank |
| AllGather | All | All | Yes (v) | Gather from all to all |
| ReduceScatter | All | All | Yes (v) | Reduce and scatter |
| AllToAll | All | All | Yes (v) | Exchange between all pairs |
| Scatter | Root | All | No | Distribute from root |
| Gather | All | Root | No | Collect to root |
| Barrier | All | All | N/A | Synchronization |
| Send | Src | Dst | N/A | Point-to-point send |
| Recv | Src | Dst | N/A | Point-to-point receive |

---

## CTRAN Library

**Location**: `comms/ctran/`

CTRAN (Communication Transport) is Meta's modular, self-contained collective communications library that emerged as a solution to challenges in NCCL. It provides a clean architecture supporting different GPU types (NVIDIA, AMD) and network topologies.

### Design Principles

1. **Independent Library**: No dependencies on NCCL, NCCLX, RCCLX, or MCCL
2. **Cross-Platform**: Works on both NVIDIA and AMD GPUs
3. **Modular**: Clean separation of algorithms and transport layers
4. **Minimal Dependencies**: Uses Folly and standard libraries
5. **Well-Tested**: Every component has comprehensive tests

### Algorithm Implementations

Located in `comms/ctran/algos/`, each collective has multiple algorithm implementations optimized for different scenarios.

#### AllReduce Algorithms

**Location**: `comms/ctran/algos/AllReduce/`

- **Ring**: Classic ring algorithm for large messages
  - Implementation: `AllReduceRing.cc`, `AllReduceRing.cu`
  - Best for: Large tensors, bandwidth-bound operations
  
- **ARG (AllReduce Group)**: Optimized group-based reduction
  - Implementation: `AllReduceARG.cc`, `AllReduceARG.cu`
  - Best for: Multi-GPU nodes with fast interconnects
  
- **Direct**: Direct GPU-to-GPU transfers
  - Implementation: `AllReduceDirect.cc`
  - Best for: Small tensors, low latency
  
- **Shared Memory**: Intra-node shared memory optimization
  - Implementation: `AllReduceShm.cu`
  - Best for: Single-node multi-GPU

#### AllGather Algorithms

**Location**: `comms/ctran/algos/AllGather/`

- **Brucks Full-Feature**: Brucks algorithm variant
  - Implementation: `AllGatherBrucksFF.cc`
  
- **Direct**: Direct gather approach
  - Implementation: `AllGatherDirect.cc`, `AllGatherDirect.cu`
  
- **Ring**: Ring-based gathering
  - Implementation: `AllGatherRing.cc`, `AllGatherRing.cu`
  
- **Recursive Doubling**: Logarithmic complexity algorithm
  - Implementation: `AllGatherRecDbl.cc`, `AllGatherRecDbl.cu`
  
- **Pipelined (AllGatherP)**: Pipelined variant for overlapped communication
  - Location: `comms/ctran/algos/AllGatherP/`
  - Implementations: Direct and Pipeline modes

#### AllToAll Algorithms

**Location**: `comms/ctran/algos/AllToAll/`

- **Standard AllToAll**: Basic all-to-all implementation
  - Files: `AllToAll.cc`, `AllToAll.cu`, `AllToAllImpl.cc`
  
- **AllToAllDedup**: Deduplication-optimized variant
  - Files: `AllToAllDedup.cc`, `AllToAllDedupImpl.cc`
  - Best for: Data with redundancy
  
- **AllToAllv**: Variable-sized all-to-all
  - Files: `AllToAllv.cc`, `AllToAllvImpl.h`
  
- **AllToAllvDynamic**: Runtime-determined sizes
  - Files: `AllToAllvDynamic.cc`, `AllToAllvDynamicCommon.cc`
  - Advanced hint system: `AllToAllvDynamicHintUtils.cc`
  
- **AllToAllP**: Pipelined variant
  - Files: `AllToAllP.cc`, `AllToAllPImpl.cc`
  
- **AllToAllvDynamicSplit**: Split algorithm for dynamic sizes
  - Files: `AllToAllvDynamicSplit.cc`

**Special Variant**: AllToAllvDedup (Location: `comms/ctran/algos/AllToAllvDedup/`)
- Combines variable sizes with deduplication
- Resource management: `ResourceImpl.cc`
- Forward group synchronization
- Worker synchronization primitives

#### ReduceScatter Algorithms

**Location**: `comms/ctran/algos/ReduceScatter/`

- **Ring**: Ring-based reduce-scatter
  - Files: `ReduceScatterRing.cc`, `ReduceScatterRing.cuh`
  
- **RHD (Ring Halving-Doubling)**: Optimized for power-of-2 ranks
  - Files: `ReduceScatterRHD.cc`, `ReduceScatterRHD.cuh`
  
- **Direct**: Direct approach
  - Files: `ReduceScatterDirect.cc`, `ReduceScatterDirect.cuh`

#### Broadcast Algorithms

**Location**: `comms/ctran/algos/Broadcast/`

- **Binomial Tree**: Logarithmic tree broadcast
  - Implementation: `BroadcastBinomialTree.cc`
  - Best for: Low latency, small messages
  
- **Direct**: Direct broadcast
  - Implementation: `BroadcastDirect.cc`

#### Send/Recv Operations

**Location**: `comms/ctran/algos/SendRecv/`

- Point-to-point communication primitives
- Files: `SendRecv.cc`, `SendRecv.cu`, `SendRecvImpl.h`

### Network Transport Backends

Located in `comms/ctran/backends/`, CTRAN supports multiple network transport layers.

#### InfiniBand (IB)

**Location**: `comms/ctran/backends/ib/`

**Features**:
- Full InfiniBand Verbs support
- Queue Pair (QP) management
- Connection management
- Memory registration
- Reliable Connection (RC) mode

**Key Components**:
- `CtranIb.cc`: Main IB transport implementation
- `CtranIbVc.cc`: Virtual connection management
- `CtranIbQpUtils.cc`: QP utilities
- `IbvWrap.cc`: InfiniBand verbs wrapper
- `ibutils.cc`: IB utility functions

**Tests**:
- Connection tests
- Memory registration tests
- Transport error handling
- HCA selection tests (exact match, prefix match, exclude)

#### NVLink (NVL)

**Location**: `comms/ctran/backends/nvl/`

**Features**:
- NVIDIA NVLink fabric support
- Direct GPU-to-GPU communication
- Optimized for NVSwitch systems

**Files**:
- `CtranNvl.cc`: NVLink transport implementation
- `CtranNvlImpl.cc`: Implementation details

#### Socket (TCP)

**Location**: `comms/ctran/backends/socket/`

**Features**:
- TCP socket-based communication
- Portable across all platforms
- Fallback transport

**Files**:
- `CtranSocket.cc`: Socket transport

#### TCP DevMem

**Location**: `comms/ctran/backends/tcpdevmem/`

**Features**:
- TCP with device memory support
- Zero-copy optimizations
- Singleton pattern for resource management

**Files**:
- `CtranTcpDm.cc`: TCP DevMem implementation
- `CtranTcpDmSingleton.cc`: Singleton manager

### Common Components

**Location**: `comms/ctran/algos/common/`

Shared components used across multiple algorithms:

- **BufManager**: Buffer management (`BufManager.cc`)
- **GpeKernel**: General Purpose Engine kernels
  - `GpeKernel.h`, `GpeKernelDev.cuh`
  - Synchronization: `GpeKernelSync.h`, `GpeKernelSyncDev.cuh`
- **MPSCTbSync**: Multi-Producer Single-Consumer Thread Block synchronization
  - `MPSCTbSync.h`, `MPSCTbSyncDev.cuh`

### Topology Support

**Location**: `comms/ctran/algos/topo/`

- **CtranRingBuilder**: Build ring topologies for ring-based algorithms
- Topology-aware algorithm selection

### Bootstrap & Initialization

**Location**: `comms/ctran/bootstrap/`

- Socket-based bootstrap for initial communication setup
- Files: `Socket.cc`, `Socket.h`

---

## Advanced Features

TorchComm provides several advanced features beyond basic collective operations.

### 1. Window Operations (Remote Memory Access)

**Location**: `comms/ctran/algos/RMA/`

Window operations provide one-sided communication primitives for Remote Memory Access (RMA), enabling direct read/write to remote GPU memory without active participation of the remote process.

#### Features

- **Put Operations**: Write data to remote memory
- **Get Operations**: Read data from remote memory
- **Signal/Wait Primitives**: Synchronization mechanisms
- **Window Allocation**: Allocate RMA-capable memory regions

#### API

```python
# Allocate a window
window = comm.window_allocate(
    window_size=1024*1024,    # Size in bytes
    cpu_buf=False,             # GPU memory
    signal_size=64             # Signal buffer size
)

# Get window properties
size = window.get_size()
device = window.get_device()

# Put operation (write to remote rank)
work = window.put(
    tensor=my_tensor,
    dst_rank=1,
    target_disp=0,             # Displacement in window
    async_op=True
)

# Get tensor view from window
remote_tensor = window.get_tensor(
    rank=0,
    sizes=[1024],
    dtype=torch.float32,
    offset=0
)

# Signal/Wait primitives
# Send signal to remote rank
window.signal(
    signal_disp=0,
    signal_val=1,
    dst_rank=1,
    async_op=False
)

# Wait for signal
window.wait_signal(
    signal_disp=0,
    signal_val=1,
    cmp_op=SignalCmpOp.GE,    # Greater or Equal
    async_op=False
)
```

#### Comparison Operations

```python
from torchcomms import SignalCmpOp

SignalCmpOp.EQ  # Equal
SignalCmpOp.GE  # Greater or Equal
SignalCmpOp.LE  # Less or Equal
```

#### Implementation

- `Get.cc`, `Get.cu`: Get operation implementation
- `PutSignal.cc`, `PutSignal.cu`: Put and signal operations
- `Types.h`: RMA type definitions

### 2. RDMA Transport

**Location**: `comms/torchcomms/transport/`

High-performance RDMA (Remote Direct Memory Access) communication library for zero-copy GPU-to-GPU data transfers.

#### Features

- **Zero-Copy Transfers**: Direct GPU-to-GPU memory access
- **Asynchronous Operations**: Future-based completion handling
- **Memory Management**: Automatic CUDA memory registration/deregistration
- **Event-Driven**: Integration with folly::EventBase
- **Thread-Safe**: Safe for multi-threaded applications
- **High Performance**: Up to 45 GB/s throughput (benchmarked on H100 + ConnectX-7)

#### API

**Transport Creation**:
```cpp
#include "comms/torchcomms/transport/RdmaTransport.h"
using namespace torch::comms;

// Check platform support
if (!RdmaTransport::supported()) {
    // Fallback to alternative transport
}

// Create transport for CUDA device
auto evbThread = std::make_unique<folly::ScopedEventBaseThread>();
auto transport = std::make_unique<RdmaTransport>(
    cudaDev,                      // CUDA device ID
    evbThread->getEventBase()     // Event base (optional)
);
```

**Connection Establishment**:
```cpp
// Bind and get URL
std::string myUrl = transport->bind();
// Share URL with peer via coordination mechanism

// Connect to peer
auto result = transport->connect(peerUrl);
if (result == commSuccess) {
    // Connection established
}

// Check connection status
bool isConnected = transport->connected();
```

**Memory Registration**:
```cpp
// Allocate CUDA memory
void* gpuBuffer = nullptr;
size_t bufferSize = 1024 * 1024;  // 1MB
cudaMalloc(&gpuBuffer, bufferSize);

// Register memory (requires > 4097 bytes)
RdmaMemory rdmaMemory(gpuBuffer, bufferSize, cudaDevice);

// Get local and remote keys
void* localKey = rdmaMemory.localKey();
std::string remoteKey = rdmaMemory.remoteKey();

// Create views into memory
auto view = rdmaMemory.createView(offset, length);
auto view2 = rdmaMemory.createView(ptr, length);

// Check if buffer is contained
bool contains = rdmaMemory.contains(ptr, len);
```

**Data Transfer**:
```cpp
// Prepare remote buffer info
RdmaRemoteBuffer remoteBuffer{
    .ptr = peerGpuPtr,
    .accessKey = peerRemoteKey
};

// Perform RDMA write
auto writeFuture = transport->write(
    localMemView,        // Local memory view
    remoteBuffer,        // Remote buffer info
    true                 // Notify receiver
);

// Wait for completion
auto result = std::move(writeFuture).get();
if (result == commSuccess) {
    // Transfer completed
}

// Receiver waits for data
auto waitFuture = transport->waitForWrite();
result = std::move(waitFuture).get();
```

**Batch Operations**:
```cpp
std::vector<folly::SemiFuture<commResult_t>> futures;

// Queue multiple writes
for (const auto& transfer : transfers) {
    auto view = memory.createView(transfer.src, transfer.size);
    futures.emplace_back(
        transport->write(view, transfer.dst, false)
    );
}

// Wait for all
auto results = folly::collectAll(std::move(futures)).get();
```

#### Performance Characteristics

Benchmarked on H100 + 400Gbps ConnectX-7:

| Message Size | Latency | Bandwidth |
|-------------|---------|-----------|
| 8 KB | 27 μs | 289 MB/s |
| 64 KB | 29 μs | 2.1 GB/s |
| 1 MB | 49 μs | 19.9 GB/s |
| 16 MB | 371 μs | 42.1 GB/s |
| 256 MB | 5.5 ms | 45.0 GB/s |

#### Key Classes

- `RdmaTransport`: Main transport class
- `RdmaMemory`: RAII memory registration wrapper
- `RdmaMemory::View`: Lightweight memory view
- `RdmaRemoteBuffer`: Remote buffer descriptor

#### Files

- `RdmaTransport.h`, `RdmaTransport.cc`: Core implementation
- `benchmarks/RdmaTransportBench.cc`: Performance benchmarks
- `tests/RdmaTransportTest.cc`: Unit tests

### 3. IBVerbX Library

**Location**: `comms/ctran/ibverbx/`

C++ wrapper library for InfiniBand Verbs with advanced abstractions for efficient RDMA operations.

#### Features

- **Virtual Queue Pairs**: Partition large messages across multiple QPs
- **Load Balancing**: DQPLB and Spray modes
- **Flexible Building**: Dynamic loading or direct linking
- **Type Safety**: ibverbx namespace to avoid conflicts
- **Virtual Completion Queues**: Unified completion tracking

#### Build Options

**Dynamic Loading** (Default):
```bash
buck build //comms/ctran/ibverbx:ibverbx
```
- Uses `dlopen`/`dlsym` for InfiniBand functions
- No InfiniBand libraries required at link time

**Direct Linking**:
```bash
buck build //comms/ctran/ibverbx:ibverbx-rdma-core
```
- Compiles with `-DIBVERBX_BUILD_RDMA_CORE`
- Direct function calls (better performance)
- Links against rdma-core library

#### Virtual Queue Pair (IbvVirtualQp)

Abstracts multiple data queue pairs for improved throughput and load balancing.

**Load Balancing Modes**:

1. **DQPLB Mode**: 
   - Uses `IBV_WR_RDMA_WRITE_WITH_IMM` on all QPs
   - Sequence numbers in immediate data
   - Per-message notification bit

2. **Spray Mode**:
   - Uses `IBV_WR_RDMA_WRITE` for data
   - Single zero-byte `IBV_WR_RDMA_WRITE_WITH_IMM` for notification
   - Lower latency for small messages

**Usage Example**:
```cpp
// Receiver: post receive
ibv_recv_wr recvWr = ...;
ibv_recv_wr* recvWrBad;
virtualQp.postRecv(&recvWr, &recvWrBad);

// Sender: post send
ibv_send_wr sendWr = ...;
ibv_send_wr* sendWrBad;
virtualQp.postSend(&sendWr, &sendWrBad);

// Poll for completion (sender)
while (!done) {
    auto wcVector = sendVirtualCq.pollCq(1);
    if (!wcVector.empty()) {
        // Process completion
        break;
    }
}

// Poll for completion (receiver)
while (!done) {
    auto wcVector = recvVirtualCq.pollCq(1);
    if (!wcVector.empty()) {
        // Process received data
        break;
    }
}
```

#### Key Components

- `Ibverbx.h`, `Ibverbx.cc`: Main wrapper implementation
- `Ibvcore.h`: Core type definitions
- `benchmarks/`: Performance benchmarks
- `tests/`: Comprehensive test suite

### 4. CUDA Graphs Support

All TorchComm operations are compatible with CUDA graph capture and replay for optimized performance.

#### Benefits

- **Reduced CPU Overhead**: Eliminate repeated kernel launches
- **Optimized Replay**: No CUDA event overhead during replay
- **Repeated Patterns**: Ideal for training loops

#### Usage

```python
import torch
from torchcomms import new_comm, ReduceOp

comm = new_comm("ncclx", torch.device("cuda"))
rank = comm.get_rank()

# Prepare tensor
tensor = torch.ones(1024, device="cuda") * rank

# Capture communication in CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    # Operations captured in graph mode
    work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)
    # Note: Don't call work.wait() inside capture

# Replay graph multiple times
for iteration in range(1000):
    graph.replay()
    torch.cuda.current_stream().synchronize()
    # tensor now contains sum of all ranks

comm.finalize()
```

#### Implementation Details

- Work objects have special handling for graph mode
- Operations execute on current CUDA stream
- Multiple replays supported
- Graph mode eliminates event overhead

### 5. Asynchronous Operations

All collective and point-to-point operations support asynchronous execution.

#### TorchWork Object

```python
class TorchWork:
    def is_completed(self) -> bool
    def wait(self) -> None
```

#### Synchronous vs Asynchronous

**Synchronous**:
```python
# Blocks until completion
comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
# tensor is ready to use
```

**Asynchronous**:
```python
# Returns immediately
work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

# Do other computation
compute_something_else()

# Check if completed
if work.is_completed():
    print("Operation finished")

# Wait for completion
work.wait()
# tensor is ready to use
```

#### Multiple Overlapped Operations

```python
# Launch multiple operations
work1 = comm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
work2 = comm.broadcast(tensor2, root=0, async_op=True)
work3 = comm.all_gather(output_list, tensor3, async_op=True)

# Do computation
compute()

# Wait for all
work1.wait()
work2.wait()
work3.wait()
```

### 6. Communicator Management

#### Split Communicators

Create subgroups from existing communicators.

```python
# Split into two groups
if comm.get_rank() < 4:
    ranks = [0, 1, 2, 3]
else:
    ranks = [4, 5, 6, 7]

sub_comm = comm.split(
    rank_groups=[ranks],
    name="sub_group",
    hints=None,
    timeout=None
)

# Use sub_comm for operations within the group
sub_comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
```

#### Communicator Properties

```python
# Get rank and size
rank = comm.get_rank()          # My rank
size = comm.get_size()          # World size
name = comm.get_name()          # Communicator name
backend = comm.get_backend()    # Backend name (e.g., "ncclx")
device = comm.get_device()      # Device (e.g., cuda:0)
options = comm.get_options()    # CommOptions object

# Cleanup
comm.finalize()
```

### 7. Configuration & Hints System

#### Hints

Pass backend-specific hints to operations:

```python
# NCCLX high-priority stream
hints = {
    "torchcomm::ncclx::high_priority_stream": "true"
}

comm.all_reduce(
    tensor, 
    ReduceOp.SUM, 
    async_op=False,
    hints=hints
)
```

#### Per-Operation Timeouts

```python
from datetime import timedelta

# Set timeout for specific operation
comm.all_reduce(
    tensor,
    ReduceOp.SUM,
    async_op=False,
    timeout=timedelta(seconds=60)
)
```

#### Communicator Options

```python
from torchcomms import CommOptions
from datetime import timedelta

options = CommOptions()
options.abort_process_on_timeout_or_error = False
options.timeout = timedelta(seconds=30)
options.name = "my_comm"
options.hints = {"key": "value"}

# Or pass directly to new_comm
comm = new_comm(
    "ncclx",
    device,
    abort_process_on_timeout_or_error=False,
    timeout=timedelta(seconds=30),
    name="my_comm",
    hints={"key": "value"}
)
```

#### Environment Variables

```bash
# Abort on error
export TORCHCOMM_ABORT_ON_ERROR=true  # Default: true

# Default timeout
export TORCHCOMM_TIMEOUT_SECONDS=30.0  # Default: 30.0
```

---

## Data Types & Operations

### Supported Reduction Operations

```python
from torchcomms import ReduceOp

ReduceOp.SUM         # Sum of elements
ReduceOp.PRODUCT     # Product of elements
ReduceOp.MIN         # Minimum element
ReduceOp.MAX         # Maximum element
ReduceOp.BAND        # Bitwise AND
ReduceOp.BOR         # Bitwise OR
ReduceOp.BXOR        # Bitwise XOR
ReduceOp.AVG         # Average of elements
ReduceOp.PREMUL_SUM(factor)  # Pre-multiplication sum
```

### Supported Tensor Types

#### CUDA Tensors (NVIDIA GPUs)

```python
# Via NCCL or NCCLX
tensor = torch.ones(1024, dtype=torch.float32, device="cuda")
comm = new_comm("ncclx", torch.device("cuda"))
```

**Supported dtypes**:
- `torch.float32`, `torch.float16`, `torch.bfloat16`
- `torch.float64`
- `torch.int32`, `torch.int64`
- `torch.uint8`, `torch.int8`

#### ROCm Tensors (AMD GPUs)

```python
# Via RCCL
tensor = torch.ones(1024, dtype=torch.float32, device="cuda")  # CUDA API on ROCm
comm = new_comm("rccl", torch.device("cuda"))
```

**Supported dtypes**: Same as CUDA tensors

#### CPU Tensors

```python
# Via GLOO
tensor = torch.ones(1024, dtype=torch.float32, device="cpu")
comm = new_comm("gloo", torch.device("cpu"))
```

**Supported dtypes**: All PyTorch dtypes

### Memory & Device Support

- **Multi-GPU**: Multiple GPUs per node
- **Multi-Node**: Distributed across multiple nodes
- **RDMA Memory**: Automatic registration for zero-copy transfers
- **Pinned Memory**: For efficient CPU-GPU transfers
- **Device Memory**: Direct GPU memory operations

---

## Infrastructure Features

### 1. Bootstrap Support

TorchComm supports multiple bootstrap mechanisms for initial process coordination.

#### TCPStore

```python
import torch
from torchcomms import new_comm

# Create TCP store
store = torch.distributed.TCPStore(
    host_name="master.node.com",
    port=12345,
    world_size=8,
    is_master=(rank == 0)
)

comm = new_comm("ncclx", device, store=store)
```

#### FileStore

```python
import torch
from torchcomms import new_comm

# Create file store
store = torch.distributed.FileStore("/tmp/torchcomm_test", world_size=8)

comm = new_comm("ncclx", device, store=store)
```

#### Auto-Detection (Torchrun)

```python
# No store needed when using torchrun
# TorchComm auto-detects environment
comm = new_comm("ncclx", device)
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=8 --nnodes=4 train.py
```

### 2. Profiling & Debugging

#### Collision Tracing

**Location**: `comms/utils/colltrace/`

Track and analyze collective operation patterns:
- Operation timing
- Data sizes
- Overlap analysis
- Performance metrics

#### NCCL Analyzer

**Location**: `comms/analyzer/`

Thrift-based analysis service for NCCL operations:
- `CommsTracingService.thrift`: Tracing service definition
- `NCCLAnalyzerState.thrift`: State management
- `NCCLAnalyzerVerdict.thrift`: Analysis results

#### Logger Utilities

**Location**: `comms/utils/logger/`

Configurable logging system:
- Multiple log levels
- Per-component logging
- Performance logging
- Debug output

**Usage**:
```cpp
#include "comms/utils/logger/Logger.h"

LOG(INFO) << "Starting communication";
LOG(DEBUG) << "Rank " << rank << " sending data";
LOG(ERROR) << "Communication failed";
```

#### Configuration Variables (CVars)

**Location**: `comms/utils/cvars/`

Runtime configuration system:
- Environment variable integration
- Dynamic configuration updates
- Type-safe configuration

### 3. Testing & Benchmarking

#### Unit Tests

Comprehensive test coverage throughout:
- Algorithm tests: `comms/ctran/algos/*/tests/`
- Backend tests: `comms/ctran/backends/*/tests/`
- Transport tests: `comms/torchcomms/transport/tests/`
- Python tests: `comms/torchcomms/tests/`

#### Distributed Tests

Multi-process testing support:
- `CtranIbDistUT.cc`: InfiniBand distributed tests
- `CtranNvlDistUT.cc`: NVLink distributed tests
- `CtranSocketDistUT.cc`: Socket distributed tests

#### Benchmarks

Performance benchmarking:
- RDMA Transport: `comms/torchcomms/transport/benchmarks/`
- IBVerbX: `comms/ctran/ibverbx/benchmarks/`
- Algorithm-specific benchmarks

**Running Benchmarks**:
```bash
# RDMA transport benchmarks
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench

# With filters
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench -- \
    --benchmark_filter="BM_RdmaTransport_Write"
```

### 4. Build System

#### CMake Configuration

**Main CMakeLists.txt**:
```cmake
option(USE_NCCL "Whether to build NCCL or not" ON)
option(USE_NCCLX "Whether to build NCCLX or not" ON)
option(USE_GLOO "Whether to build Gloo or not" ON)
option(USE_RCCL "Whether to build RCCL or not" OFF)
```

#### Build from Source

```bash
# Clone repository
git clone git@github.com:meta-pytorch/torchcomms.git
cd torchcomms

# Create conda environment
conda create -n torchcomms python=3.10
conda activate torchcomms

# Install PyTorch
pip install -r requirements.txt

# Configure backends
export USE_NCCL=ON
export USE_NCCLX=ON
export USE_GLOO=ON
export USE_RCCL=OFF

# Build and install
pip install --no-build-isolation -v .
```

#### Backend-Specific Builds

**NCCLX**:
```bash
# With system libraries
USE_SYSTEM_LIBS=1 ./build_ncclx.sh

# Build from source
./build_ncclx.sh
```

**RCCL**:
```bash
# Install prerequisites
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

# Set environment
export ROCM_HOME=/opt/rocm
export RCCL_INCLUDE=$ROCM_HOME/include/rccl

# Build
./build_rccl.sh
```

#### Python Wheel Generation

```bash
# Build wheel
python setup.py bdist_wheel

# Install wheel
pip install dist/torchcomms-*.whl
```

#### Dependencies

**Core**:
- Python 3.10+
- PyTorch 2.8+
- CMake 3.22+
- Ninja 1.10+

**Optional** (backend-specific):
- CUDA Toolkit (for NCCL/NCCLX)
- ROCm (for RCCL)
- InfiniBand libraries (for IB transport)
- Folly (included or system)
- glog, gflags, fmt (included or system)

---

## API Reference

### Python API

#### Creating Communicator

```python
def new_comm(
    backend: str,                              # "nccl", "ncclx", "rccl", or "gloo"
    device: torch.device,                      # Device to use
    abort_process_on_timeout_or_error: bool | None = None,
    timeout: timedelta | None = None,
    store: Any | None = None,                 # Optional store
    name: str | None = None,
    hints: Dict[str, str] | None = None
) -> TorchComm
```

#### TorchComm Methods

```python
class TorchComm:
    # Lifecycle
    def finalize(self) -> None
    
    # Properties
    def get_rank(self) -> int
    def get_size(self) -> int
    def get_name(self) -> str
    def get_device(self) -> torch.device
    def get_backend(self) -> str
    def get_options(self) -> CommOptions
    
    # Point-to-point
    def send(self, tensor, dst, async_op, hints=None, timeout=None) -> TorchWork
    def recv(self, tensor, src, async_op, hints=None, timeout=None) -> TorchWork
    
    # Collectives
    def broadcast(self, tensor, root, async_op, hints=None, timeout=None) -> TorchWork
    def all_reduce(self, tensor, op, async_op, hints=None, timeout=None) -> TorchWork
    def reduce(self, tensor, root, op, async_op, hints=None, timeout=None) -> TorchWork
    def all_gather(self, tensor_list, tensor, async_op, hints=None, timeout=None) -> TorchWork
    def all_gather_v(self, tensor_list, tensor, async_op, hints=None, timeout=None) -> TorchWork
    def all_gather_single(self, output, input, async_op, hints=None, timeout=None) -> TorchWork
    def reduce_scatter(self, output, input_list, op, async_op, hints=None, timeout=None) -> TorchWork
    def reduce_scatter_v(self, output, input_list, op, async_op, hints=None, timeout=None) -> TorchWork
    def reduce_scatter_single(self, output, input, op, async_op, hints=None, timeout=None) -> TorchWork
    def all_to_all(self, output_list, input_list, async_op, hints=None, timeout=None) -> TorchWork
    def all_to_all_single(self, output, input, async_op, hints=None, timeout=None) -> TorchWork
    def all_to_all_v_single(self, output, input, output_splits, input_splits, 
                           async_op, hints=None, timeout=None) -> TorchWork
    def scatter(self, output, input_list, root, async_op, hints=None, timeout=None) -> TorchWork
    def gather(self, output_list, input, root, async_op, hints=None, timeout=None) -> TorchWork
    def barrier(self, async_op, hints=None, timeout=None) -> TorchWork
    
    # Advanced
    def split(self, rank_groups, name, hints=None, timeout=None) -> TorchComm
    def batch_op_create(self) -> BatchSendRecv
    def window_allocate(self, window_size, cpu_buf=False, signal_size=None) -> TorchCommWindow
```

#### TorchWork Methods

```python
class TorchWork:
    def is_completed(self) -> bool
    def wait(self) -> None
```

#### BatchSendRecv

```python
class BatchSendRecv:
    def send(self, tensor, dst: int) -> None
    def recv(self, tensor, src: int) -> None
    def issue(self, async_op: bool, options: BatchP2POptions = None) -> TorchWork
```

#### TorchCommWindow (RMA)

```python
class TorchCommWindow:
    def get_size(self) -> int
    def get_device(self) -> torch.device
    def put(self, tensor, dst_rank, target_disp, async_op) -> TorchWork
    def signal(self, signal_disp, signal_val, dst_rank, async_op) -> None
    def wait_signal(self, signal_disp, signal_val, cmp_op, async_op) -> None
    def get_tensor(self, rank, sizes, dtype, offset) -> torch.Tensor
```

### C++ API

C++ backend interface defined in `comms/torchcomms/TorchCommBackend.hpp`:

```cpp
class TorchCommBackend {
public:
    // Lifecycle
    virtual void init(...) = 0;
    virtual void finalize() = 0;
    
    // Properties
    virtual int getRank() const = 0;
    virtual int getSize() const = 0;
    virtual std::string_view getCommName() const = 0;
    virtual std::string_view getBackendName() const = 0;
    virtual const CommOptions& getOptions() const = 0;
    virtual const at::Device& getDevice() const = 0;
    
    // Point-to-point
    virtual std::shared_ptr<TorchWork> send(...) = 0;
    virtual std::shared_ptr<TorchWork> recv(...) = 0;
    
    // Collectives
    virtual std::shared_ptr<TorchWork> broadcast(...) = 0;
    virtual std::shared_ptr<TorchWork> all_reduce(...) = 0;
    virtual std::shared_ptr<TorchWork> reduce(...) = 0;
    virtual std::shared_ptr<TorchWork> all_gather(...) = 0;
    virtual std::shared_ptr<TorchWork> reduce_scatter(...) = 0;
    virtual std::shared_ptr<TorchWork> all_to_all(...) = 0;
    virtual std::shared_ptr<TorchWork> barrier(...) = 0;
    virtual std::shared_ptr<TorchWork> scatter(...) = 0;
    virtual std::shared_ptr<TorchWork> gather(...) = 0;
    
    // Advanced
    virtual std::shared_ptr<TorchCommBackend> split(...) = 0;
    virtual std::shared_ptr<TorchCommWindow> window_allocate(...);
};
```

---

## Installation & Build

### Prerequisites

- **Python**: 3.10 or higher
- **PyTorch**: 2.8 or higher
- **CMake**: 3.22 or higher
- **Ninja**: 1.10 or higher
- **CUDA**: For NCCL/NCCLX backends
- **ROCm**: For RCCL backend

### Installation Options

#### Option 1: PyPI (Nightly Builds)

```bash
pip install --pre torch torchcomms --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### Option 2: Build from Source

```bash
# Clone repository
git clone git@github.com:meta-pytorch/torchcomms.git
cd torchcomms

# Create environment
conda create -n torchcomms python=3.10
conda activate torchcomms

# Install PyTorch
pip install -r requirements.txt

# Configure (optional)
export USE_NCCL=ON    # Default: ON
export USE_NCCLX=ON   # Default: ON
export USE_GLOO=ON    # Default: ON
export USE_RCCL=OFF   # Default: OFF

# Build and install
pip install --no-build-isolation -v .
```

#### Option 3: Backend-Specific Builds

**NCCLX Backend**:
```bash
# With system libraries (recommended)
USE_SYSTEM_LIBS=1 ./build_ncclx.sh

# Or build dependencies from source
./build_ncclx.sh

# Then install torchcomms
pip install --no-build-isolation -v .
```

**RCCL Backend** (AMD GPUs):
```bash
# Install prerequisites
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

# Set ROCm paths
export ROCM_HOME=/opt/rocm
export RCCL_INCLUDE=$ROCM_HOME/include/rccl

# Build RCCL components
./build_rccl.sh

# Enable RCCL backend
export USE_RCCL=ON

# Install torchcomms
pip install --no-build-isolation -v .
```

### Verifying Installation

```python
import torch
import torchcomms

# Check available backends
print(f"TorchComm version: {torchcomms.__version__}")

# Try creating communicator (single process)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "ncclx" if torch.cuda.is_available() else "gloo"

try:
    comm = torchcomms.new_comm(backend, device)
    print(f"Successfully created {backend} communicator")
    print(f"Rank: {comm.get_rank()}, Size: {comm.get_size()}")
    comm.finalize()
except Exception as e:
    print(f"Error: {e}")
```

### Running Examples

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=8 example.py

# Multi-node
# Node 0:
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 \
    --rdzv-endpoint="master:29500" train.py

# Node 1-3:
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=1 \
    --rdzv-endpoint="master:29500" train.py
```

---

## Performance & Scalability

### Design Goals

- **100,000+ GPU Support**: Designed and tested at massive scale
- **Low Latency**: Optimized algorithms for small messages
- **High Bandwidth**: Saturate network bandwidth for large messages
- **CPU Efficiency**: Minimal CPU overhead

### Performance Characteristics

#### RDMA Transport

Benchmarked on H100 + 400Gbps ConnectX-7:

| Message Size | Latency | Bandwidth | Efficiency |
|-------------|---------|-----------|------------|
| 8 KB | 27 μs | 289 MB/s | - |
| 64 KB | 29 μs | 2.1 GB/s | - |
| 256 KB | 33 μs | 7.4 GB/s | - |
| 1 MB | 49 μs | 19.9 GB/s | 40% |
| 16 MB | 371 μs | 42.1 GB/s | 84% |
| 256 MB | 5.5 ms | 45.0 GB/s | 90% |

#### Algorithm Selection

TorchComm automatically selects optimal algorithms based on:
- Message size
- Number of ranks
- Network topology
- Hardware capabilities

### Scalability Features

1. **Topology-Aware**: Algorithms adapt to network structure
2. **Hierarchical Collectives**: Optimize for multi-node deployments
3. **Overlap Computation**: Async operations hide communication latency
4. **CUDA Graphs**: Eliminate kernel launch overhead
5. **Zero-Copy**: RDMA for direct GPU-to-GPU transfers

### Production Deployment

TorchComm (NCCLX) is Meta's production backend for:
- **LLaMA Training**: Large language model training
- **Generative AI**: All Meta generative AI services
- **Large-Scale Training**: 10,000+ GPU clusters

### Best Practices

1. **Use NCCLX for NVIDIA**: Production-tested, Meta-optimized
2. **Use RCCL for AMD**: Native AMD GPU support
3. **Async Operations**: Overlap communication with computation
4. **CUDA Graphs**: For repeated patterns in training loops
5. **Batch Operations**: Group small transfers
6. **Proper Timeouts**: Set appropriate timeouts for large operations
7. **Monitor Performance**: Use profiling and tracing tools

---

## Key Repository Files

### Documentation

- **Main README**: `/README.md`
- **Getting Started**: `/docs/source/getting_started.md`
- **API Documentation**: `/comms/torchcomms/README.md`
- **CTRAN**: `/comms/ctran/README.md`
- **RDMA Transport**: `/comms/torchcomms/transport/README.md`
- **IBVerbX**: `/comms/ctran/ibverbx/README.md`

### Python API

- **Type Stubs**: `/comms/torchcomms/_comms.pyi`
- **Init**: `/comms/torchcomms/__init__.py`
- **Examples**: `/comms/torchcomms/examples/`

### Backend Implementations

- **NCCL**: `/comms/torchcomms/nccl/TorchCommNCCL.cpp`
- **NCCLX**: `/comms/torchcomms/ncclx/TorchCommNCCLX.cpp`
- **RCCL**: `/comms/torchcomms/rccl/TorchCommRCCL.cpp`
- **GLOO**: `/comms/torchcomms/gloo/TorchCommGloo.cpp`

### Build System

- **CMake**: `/CMakeLists.txt`
- **Setup**: `/setup.py`
- **NCCLX Build**: `/build_ncclx.sh`
- **RCCL Build**: `/build_rccl.sh`

### Testing

- **Python Tests**: `/comms/torchcomms/tests/`
- **C++ Tests**: Throughout `/comms/ctran/*/tests/`

---

## Summary

Meta-TorchComm is a comprehensive, production-ready communication library providing:

✅ **4 Backend Options**: NCCLX (Meta), NCCL, RCCL (AMD), GLOO (CPU)  
✅ **15+ Collective Operations**: AllReduce, AllGather, AllToAll, etc.  
✅ **Multiple Algorithm Variants**: Ring, Tree, Direct, Pipelined, etc.  
✅ **Advanced Features**: RDMA, RMA, CUDA Graphs, IBVerbX  
✅ **Production Scale**: 100,000+ GPUs  
✅ **Comprehensive API**: Python and C++  
✅ **Extensive Testing**: Unit, integration, and distributed tests  
✅ **Flexible Build**: Selective backend compilation  
✅ **High Performance**: Optimized for latency and bandwidth  
✅ **Active Development**: Continuously improved at Meta

**Use Cases**:
- Large-scale distributed training
- Multi-GPU model parallelism
- Data parallelism
- Parameter servers
- All-to-all communication patterns
- High-performance inference

**Repository**: [github.com/meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms)

---

*Document generated by analyzing the meta-pytorch/torchcomms repository structure and documentation.*

