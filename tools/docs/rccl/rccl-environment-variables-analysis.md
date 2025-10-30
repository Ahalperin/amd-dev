# RCCL Environment Variables Analysis

## Overview
This document provides a comprehensive analysis of all environment variables defined and used in the RCCL (ROCm Communication Collectives Library) codebase.

Environment variables in RCCL are defined using two primary macros:
- `NCCL_PARAM(name, env, defaultValue)` - Creates environment variable `NCCL_<env>`
- `RCCL_PARAM(name, env, defaultValue)` - Creates environment variable `RCCL_<env>`

## Table of Contents
1. [Configuration and Setup](#configuration-and-setup)
2. [Logging and Debugging](#logging-and-debugging)
3. [Algorithm and Protocol Control](#algorithm-and-protocol-control)
4. [Network and Transport](#network-and-transport)
5. [InfiniBand/RDMA Settings](#infiniband-rdma-settings)
6. [Performance Tuning](#performance-tuning)
7. [Topology and Hardware](#topology-and-hardware)
8. [Memory Management](#memory-management)
9. [MSCCL (Microsoft Collective Communication Library)](#msccl-settings)
10. [MSCCLPP Settings](#mscclpp-settings)
11. [RAS (Reliability, Availability, Serviceability)](#ras-settings)
12. [Profiling and Tracing](#profiling-and-tracing)
13. [Development and Testing](#development-and-testing)
14. [External Dependencies](#external-dependencies)

---

## Configuration and Setup

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_CONF_FILE` | `~/.rccl.conf` or `/etc/rccl.conf` | src/misc/param.cc | Path to RCCL configuration file |
| `NCCL_HOSTID` | - | src/misc/utils.cc | Host identifier for multi-node communication |
| `NCCL_LAUNCH_MODE` | - | src/init.cc | Launch mode: PARALLEL or GROUP |
| `NCCL_COMM_ID` | - | src/bootstrap.cc, src/misc/socket.cc | Communication ID for multi-process mode |

---

## Logging and Debugging

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_DEBUG` | - | src/debug.cc | Debug log level (VERSION, WARN, INFO, TRACE) |
| `NCCL_DEBUG_SUBSYS` | - | src/debug.cc | Debug subsystems filter (e.g., INIT,COLL,NET) |
| `NCCL_DEBUG_FILE` | - | src/debug.cc | Redirect debug output to file |
| `NCCL_DEBUG_TIMESTAMP_LEVELS` | - | src/debug.cc | Enable timestamps in debug logs |
| `NCCL_DEBUG_TIMESTAMP_FORMAT` | - | src/debug.cc | Timestamp format for debug logs |
| `NCCL_WARN_ENABLE_DEBUG_INFO` | - | src/debug.cc | Enable debug info in warnings |
| `NCCL_SET_THREAD_NAME` | 0 | src/debug.cc | Set thread names for debugging |
| `RCCL_LOG_LEVEL` | 1 | src/misc/recorder.cc | RCCL logging verbosity level |
| `RCCL_LOG_ROCTX` | 0 | src/misc/roctx.cc | Enable ROCTX logging |
| `RCCL_REPLAY_FILE` | - | src/misc/recorder.cc | File path for replay recording |

---

## Algorithm and Protocol Control

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_ALGO` | - | src/graph/tuning.cc, src/enqueue.cc | Force specific algorithm (TREE, RING, COLLNET_DIRECT, COLLNET_CHAIN, NVLS, NVLSTREE) |
| `NCCL_PROTO` | - | src/graph/tuning.cc, src/enqueue.cc, src/rccl_wrap.cc | Force specific protocol (LL, LL128, SIMPLE) |
| `RCCL_OVERRIDE_PROTO` | - | src/rccl_wrap.cc | Override protocol selection |
| `RCCL_OVERRIDE_ALGO` | - | src/rccl_wrap.cc | Override algorithm selection |
| `NCCL_THREAD_THRESHOLDS` | - | src/graph/tuning.cc, src/rccl_wrap.cc | Thread count thresholds for operations |
| `NCCL_NTHREADS` | -2 | src/graph/tuning.cc | Number of threads per block |
| `NCCL_LL128_NTHREADS` | -2 | src/graph/tuning.cc | Number of threads for LL128 protocol |
| `RCCL_CHANNEL_TUNING_ENABLE` | 1 | src/rccl_wrap.cc | Enable channel tuning |

---

## Network and Transport

### Socket Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_SOCKET_IFNAME` | - | src/misc/socket.cc | Network interface name(s) to use |
| `NCCL_SOCKET_FAMILY` | - | src/misc/socket.cc | Force IPv4/IPv6 (AF_INET, AF_INET6) |
| `NCCL_SOCKET_NTHREADS` | -2 | src/transport/net_socket.cc | Number of socket threads |
| `NCCL_NSOCKS_PERTHREAD` | -2 | src/transport/net_socket.cc | Sockets per thread |
| `NCCL_SOCKET_INLINE` | 128 | src/transport/net_socket.cc | Inline message size |
| `NCCL_SOCKET_MIN_TASKSIZE` | 65536 | src/transport/net_socket.cc | Minimum task size for socket |
| `NCCL_SOCKET_RETRY_CNT` | 34 | src/misc/socket.cc | Socket retry count |
| `NCCL_SOCKET_RETRY_SLEEP_MSEC` | 100 | src/misc/socket.cc | Socket retry timeout in ms |
| `NCCL_SOCKET_RCVBUF` | -1 | src/misc/socket.cc | Socket receive buffer size |
| `NCCL_SOCKET_SNDBUF` | -1 | src/misc/socket.cc | Socket send buffer size |
| `RCCL_SOCKET_REUSEADDR` | 0 | src/misc/socket.cc | Enable SO_REUSEADDR |
| `RCCL_SOCKET_LINGER` | -1 | src/misc/socket.cc | Socket linger timeout |

### Network General

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_NET` | - | src/init.cc | Network backend to use |
| `NCCL_NET_PLUGIN` | - | src/plugin/net.cc | Network plugin library path |
| `NCCL_NET_PLUGIN_REF_COUNT` | 1 | src/plugin/net.cc | Network plugin reference counting |
| `NCCL_NET_SHARED_BUFFERS` | -2 | src/transport/net.cc | Enable shared network buffers |
| `NCCL_NET_SHARED_COMMS` | 1 | src/transport/net.cc | Share network communicators |
| `RCCL_NET_CONTIGUOUS_MEM` | 0 | src/transport/net.cc | Use contiguous memory for network |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | 1 | src/transport/net.cc | Optional receive completion |
| `NCCL_NET_OVERHEAD` | -2 | src/graph/tuning.cc | Network overhead estimate |
| `NCCL_NET_MERGE_LEVEL` | PATH_PORT | src/graph/topo.cc | Network device merge level |
| `NCCL_NET_FORCE_MERGE` | - | src/graph/topo.cc | Force network device merging |
| `NCCL_NET_GDR_READ` | -2 | src/graph/paths.cc | Enable GPUDirect RDMA read |
| `NCCL_NET_GDR_C2C` | 1 | src/graph/paths.cc | Enable GDR for chip-to-chip |
| `NCCL_NET_GDR_LEVEL` | PATH_PHB | tools/topo_expl/model.cpp | GDR support level |
| `NCCL_NET_FORCE_FLUSH` | 0 | src/graph/paths.cc | Force network flush operations |
| `NCCL_NET_DISABLE_INTRA` | 1 | src/graph/paths.cc | Disable intra-node network |
| `RCCL_NET_HDP_FLUSH` | 0 | src/transport/net.cc | Enable HDP flush for network ops |
| `RCCL_ENABLE_INTRANET` | -2 | src/graph/paths.cc | Enable intra-node network |
| `RCCL_INTRANET_THRESHOLD` | 8388608 | src/enqueue.cc | Threshold for intra-node network |
| `NCCL_OOB_NET_IFNAME` | - | src/bootstrap.cc | Out-of-band network interface |
| `NCCL_OOB_NET_ENABLE` | 0 | src/bootstrap.cc | Enable out-of-band network |

### Topology

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_TOPO_FILE` | - | src/graph/topo.cc | XML topology file path |
| `NCCL_TOPO_DUMP_FILE` | - | src/init.cc, src/graph/topo.cc | Dump topology to file |
| `NCCL_TOPO_DUMP_FILE_RANK` | 0 | src/graph/topo.cc | Rank to dump topology from |
| `NCCL_GRAPH_FILE` | - | src/graph/search.cc | Graph file to load |
| `NCCL_GRAPH_DUMP_FILE` | - | src/graph/search.cc | Dump graph to file |
| `NCCL_GRAPH_DUMP_FILE_RANK` | 0 | src/init.cc | Rank to dump graph from |
| `NCCL_RINGS` | - | src/graph/search.cc | Custom ring topology |
| `RCCL_TREES` | - | src/graph/search.cc | Custom tree topology |
| `NCCL_RINGS_REMAP` | - | src/graph/search.cc | Ring remapping specification |
| `NCCL_CROSS_NIC` | 2 | src/graph/search.cc | Cross-NIC communication |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | src/graph/topo.cc | Ignore CPU affinity settings |
| `RCCL_DUMP_ROME_MODEL_FILE` | - | src/graph/rome_models.cc | Dump Rome model to file |
| `RCCL_MODEL_MATCHING_DISABLE` | 0 | src/graph/search.cc | Disable model matching |
| `RCCL_MODEL_REVERSAL_DISABLE` | 0 | src/graph/rome_models.cc | Disable model reversal |
| `RCCL_DISABLE_RAIL_TREES` | 0 | src/graph/rome_models.cc | Disable rail trees |
| `RCCL_OUTPUT_TREES` | 0 | src/graph/connect.cc | Output tree topology info |

---

## InfiniBand RDMA Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_IB_DISABLE` | 0 | src/transport/net_ib.cc | Disable InfiniBand transport |
| `NCCL_IB_HCA` | - | src/transport/net_ib.cc | InfiniBand device:port specification |
| `NCCL_IB_GID_INDEX` | -1 | src/transport/net_ib.cc | GID index for RoCE |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | 1 | src/transport/net_ib.cc | Routable FLID GID index |
| `NCCL_IB_ROCE_VERSION_NUM` | 2 | src/transport/net_ib.cc | RoCE version (1 or 2) |
| `NCCL_IB_TIMEOUT` | 20 | src/transport/net_ib.cc | IB timeout value |
| `NCCL_IB_RETRY_CNT` | 7 | src/transport/net_ib.cc | IB retry count |
| `NCCL_IB_PKEY` | 0 | src/transport/net_ib.cc | IB partition key |
| `NCCL_IB_USE_INLINE` | 0 | src/transport/net_ib.cc | Use inline data |
| `NCCL_IB_SL` | -1 | src/transport/net_ib.cc | Service level |
| `NCCL_IB_TC` | -1 | src/transport/net_ib.cc | Traffic class |
| `NCCL_IB_AR_THRESHOLD` | 8192 | src/transport/net_ib.cc | Adaptive routing threshold |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | src/transport/net_ib.cc | PCI relaxed ordering |
| `NCCL_IB_ADAPTIVE_ROUTING` | -2 | src/transport/net_ib.cc | Enable adaptive routing |
| `NCCL_IB_FIFO_TC` | -1 | src/transport/net_ib.cc | FIFO traffic class |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | 1 | src/transport/net_ib.cc | Return async events |
| `NCCL_IB_ECE_ENABLE` | 1 | src/transport/net_ib.cc | Enhanced connection establishment |
| `NCCL_IB_DATA_DIRECT` | 1 | src/transport/net_ib.cc | Direct data path |
| `NCCL_IB_MERGE_VFS` | 1 | src/transport/net_ib.cc | Merge virtual functions |
| `NCCL_IB_MERGE_NICS` | 1 | src/transport/net_ib.cc | Merge NICs |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | src/transport/net_ib.cc | Queue pairs per connection |
| `NCCL_IB_WARN_RAIL_LOCAL` | 0 | src/transport/net_ib.cc | Warn on rail locality issues |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | src/transport/net_ib.cc | Split data across QPs |
| `NCCL_IB_ADDR_FAMILY` | - | src/transport/net_ib.cc | Address family for IB |
| `NCCL_IB_ADDR_RANGE` | - | src/transport/net_ib.cc | Address range for IB |
| `NCCL_IB_MQP_RETRY_ALL` | 0 | src/misc/ibvwrap.cc | Retry all for multi-QP |
| `NCCL_IB_MQP_RETRY_CNT` | 34 | src/misc/ibvwrap.cc | Multi-QP retry count |
| `NCCL_IB_MQP_RETRY_SLEEP_MSEC` | 100 | src/misc/ibvwrap.cc | Multi-QP retry sleep time |
| `RCCL_FORCE_ENABLE_GDRDMA` | -1 | src/transport/net_ib.cc | Force enable GPUDirect RDMA |
| `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING` | 1 | src/transport/net_ib.cc | GDR flush without relaxed ordering |

---

## Performance Tuning

### Buffer and Memory Sizes

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_BUFFSIZE` | -2 | src/init.cc | Buffer size for collectives |
| `NCCL_LL_BUFFSIZE` | -2 | src/init.cc | Buffer size for LL protocol |
| `NCCL_LL128_BUFFSIZE` | -2 | src/init.cc | Buffer size for LL128 protocol |
| `NCCL_WORK_FIFO_BYTES` | (default) | src/init.cc | Work FIFO size in bytes |
| `NCCL_WORK_ARGS_BYTES` | INT64_MAX | src/init.cc | Work args buffer size |
| `NCCL_AGG_CHANNEL_SIZE` | -2 | src/init.cc | Aggregate channel size |
| `NCCL_CHUNK_SIZE` | 0 | src/enqueue.cc | Chunk size for operations |

### Channel and Connection Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_MIN_NCHANNELS` | -2 | src/graph/connect.cc, src/rccl_wrap.cc | Minimum number of channels |
| `NCCL_MAX_NCHANNELS` | -2 | src/graph/connect.cc, src/rccl_wrap.cc | Maximum number of channels |
| `NCCL_MIN_NRINGS` | -2 | src/graph/connect.cc | Minimum number of rings |
| `NCCL_MAX_NRINGS` | -2 | src/graph/connect.cc | Maximum number of rings |
| `NCCL_NCHANNELS_PER_NET_PEER` | -1 | src/graph/paths.cc | Channels per network peer |
| `NCCL_NCHANNELS_PER_PEER` | -2 | src/graph/paths.cc | Channels per peer |
| `NCCL_MIN_P2P_NCHANNELS` | 1 | src/graph/paths.cc | Minimum P2P channels |
| `NCCL_MAX_P2P_NCHANNELS` | MAXCHANNELS | src/graph/paths.cc | Maximum P2P channels |
| `NCCL_UNPACK_DOUBLE_NCHANNELS` | 1 | src/graph/connect.cc | Double channels for unpack |
| `NCCL_NVLS_NCHANNELS` | UNDEF | src/init.cc | NVLS channel count |

### P2P (Peer-to-Peer) Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_P2P_DISABLE` | - | src/graph/paths.cc | Disable P2P transport |
| `NCCL_P2P_LEVEL` | - | src/graph/paths.cc | P2P level (NVL, PIX, etc.) |
| `NCCL_P2P_READ_ENABLE` | -2 | src/transport/p2p.cc | Enable P2P read operations |
| `NCCL_P2P_DIRECT_DISABLE` | 0 | src/transport/p2p.cc | Disable P2P direct access |
| `NCCL_P2P_USE_CUDA_MEMCPY` | 0 | src/transport/p2p.cc | Use CUDA memcpy for P2P |
| `NCCL_P2P_LL_THRESHOLD` | 16384 | src/enqueue.cc | P2P LL protocol threshold |
| `NCCL_P2P_NET_CHUNKSIZE` | 131072 | src/init.cc, src/rccl_wrap.cc | P2P network chunk size |
| `NCCL_P2P_PCI_CHUNKSIZE` | 131072 | src/init.cc | P2P PCI chunk size |
| `NCCL_P2P_NVL_CHUNKSIZE` | 524288 | src/init.cc | P2P NVLink chunk size |
| `RCCL_P2P_NET_DISABLE` | 1 | src/init.cc | Disable P2P over network |
| `RCCL_P2P_NET_THRESHOLD` | 131072 | src/enqueue.cc | P2P network threshold |
| `RCCL_P2P_BATCH_ENABLE` | 0 | src/enqueue.cc | Enable P2P batching |
| `RCCL_P2P_BATCH_THRESHOLD` | 65536 | src/enqueue.cc | P2P batch threshold |
| `NCCL_P2P_PXN_LEVEL` | 2 | src/graph/search.cc | P2P PXN level |
| `NCCL_PXN_DISABLE` | 1 | src/graph/paths.cc, src/rccl_wrap.cc | Disable PXN |
| `NCCL_PXN_C2C` | 0 | src/graph/paths.cc | PXN chip-to-chip |
| `NCCL_IGNORE_DISABLED_P2P` | 0 | src/graph/paths.cc | Ignore disabled P2P paths |

### GPU and Kernel Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_MAX_CTAS` | UNDEF | src/init.cc | Maximum CTAs (thread blocks) |
| `NCCL_MIN_CTAS` | UNDEF | src/init.cc | Minimum CTAs |
| `NCCL_CGA_CLUSTER_SIZE` | UNDEF | src/init.cc | CGA cluster size |
| `NCCL_SYM_CTAS` | 0 | src/symmetric.cc | Symmetric CTAs |
| `NCCL_SYM_KERNEL` | - | src/symmetric.cc | Symmetric kernel name |
| `NCCL_L1_SHARED_MEMORY_CARVEOUT` | 0 | src/enqueue.cc | L1/shared memory carveout |
| `NCCL_SET_STACK_SIZE` | 0/1 | src/init.cc | Set kernel stack size |
| `RCCL_STACK_SIZE_OVERRIDE` | 0 | src/init.cc | Override stack size |

---

## Shared Memory and Local Transport

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_SHM_DISABLE` | 0 | src/transport/shm.cc | Disable shared memory transport |
| `NCCL_SHM_USE_CUDA_MEMCPY` | 0 | src/transport/shm.cc | Use CUDA memcpy for SHM |
| `NCCL_SHM_MEMCPY_MODE` | SEND_SIDE | src/transport/shm.cc | SHM memcpy mode |
| `NCCL_SHM_LOCALITY` | RECV_SIDE | src/transport/shm.cc | SHM locality mode |

---

## Memory Management

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_DMABUF_ENABLE` | 0 | src/misc/rocmwrap.cc | Enable DMA-BUF |
| `RCCL_FORCE_ENABLE_DMABUF` | 0 | src/misc/rocmwrap.cc | Force enable DMA-BUF |
| `NCCL_CUMEM_ENABLE` | 0 | src/misc/rocmwrap.cc | Enable CUDA memory pool |
| `NCCL_CUMEM_HOST_ENABLE` | -1 | src/misc/rocmwrap.cc | Enable CUDA host memory |
| `NCCL_LEGACY_CUDA_REGISTER` | 0 | src/transport/p2p.cc | Use legacy CUDA registration |
| `NCCL_LOCAL_REGISTER` | 0 | src/register/register.cc | Enable local registration |
| `NCCL_GRAPH_REGISTER` | 0 | src/enqueue.cc | Enable graph registration |

---

## GPUDirect and GDR Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_GDRCOPY_ENABLE` | 0 | src/init.cc | Enable GDRCopy |
| `NCCL_GDRCOPY_SYNC_ENABLE` | 1 | src/transport/net.cc | Enable GDRCopy sync |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | 0 | src/transport/net.cc | Enable GDRCopy flush |
| `NCCL_GDRCOPY_FIFO_ENABLE` | 1 | src/init.cc | Enable GDRCopy FIFO |
| `NCCL_GDR_FLUSH_DISABLE` | 0 | src/transport/net_ib.cc | Disable GDR flush |

---

## NVLS (NVLink Sharp) Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_NVLS_ENABLE` | 2 | src/transport/nvls.cc | Enable NVLS |
| `NCCL_NVLS_CHUNKSIZE` | 131072 | src/transport/nvls.cc | NVLS chunk size |
| `NCCL_NVLSTREE_MAX_CHUNKSIZE` | -2 | src/enqueue.cc | NVLS tree max chunk size |

---

## NVB (NVBridge) Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_NVB_DISABLE` | 0 | src/graph/paths.cc | Disable NVBridge |
| `NCCL_NVB_PRECONNECT` | 0 | src/init.cc | Pre-connect NVBridge |

---

## CollNet Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_COLLNET_ENABLE` | UNDEF | src/init.cc | Enable CollNet |
| `NCCL_COLLNET_NODE_THRESHOLD` | 2 | src/init.cc | CollNet node threshold |

---

## MSCCL Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `RCCL_MSCCL_ENABLE` | 1 | src/misc/msccl/msccl_lifecycle.cc | Enable MSCCL |
| `RCCL_MSCCL_FORCE_ENABLE` | 0 | src/misc/msccl/msccl_lifecycle.cc | Force enable MSCCL |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS` | 1 | src/misc/msccl/msccl_lifecycle.cc | Enable MSCCL for single process |
| `RCCL_MSCCL_WORK_FIFO_DEPTH` | 262144 | src/misc/msccl/msccl_setup.cc | MSCCL work FIFO depth |
| `RCCL_MSCCL_FORCE_FULLOPS` | 0 | src/misc/msccl/msccl_setup.cc | Force full operations |
| `MSCCL_ALGORITHM_DIR` | - | src/misc/msccl/msccl_lifecycle.cc | MSCCL algorithm directory |
| `MSCCL_SCHEDULER` | - | src/misc/msccl/msccl_lifecycle.cc | MSCCL scheduler path |

---

## MSCCLPP Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `RCCL_MSCCLPP_ENABLE` | (varies) | src/init.cc | Enable MSCCLPP |
| `RCCL_MSCCLPP_FORCE_ENABLE` | 0 | src/init.cc | Force enable MSCCLPP |
| `RCCL_MSCCLPP_THRESHOLD` | 16777216 | src/init.cc | MSCCLPP threshold |
| `MSCCLPP_DEBUG` | - | ext-src/mscclpp | MSCCLPP debug level |
| `MSCCLPP_READ_ALLRED` | - | ext-src patches | Enable read-based allreduce |
| `MSCCLPP_HIERARCHICAL_ALLRED` | - | ext-src patches | Enable hierarchical allreduce |
| `MSCCLPP_DISABLE_CHANNEL_CACHE` | - | ext-src patches | Disable channel cache |
| `MSCCLPP_DISABLE_REMOTE_UBR` | - | ext-src patches | Disable remote UBR |

---

## MNNVL Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_MNNVL_ENABLE` | 2 | src/init.cc | Enable MNNVL |
| `NCCL_MNNVL_UUID` | -1 | src/init.cc | MNNVL UUID |
| `NCCL_MNNVL_CLIQUE_ID` | -1 | src/init.cc | MNNVL clique ID |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | 1 | src/graph/search.cc | Enable scatter nets |

---

## RAS Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_RAS_ENABLE` | 1 | src/bootstrap.cc | Enable RAS |
| `NCCL_RAS_ADDR` | - | src/ras/client_support.cc | RAS server address |
| `NCCL_RAS_TIMEOUT_FACTOR` | 1 | src/ras/ras.cc | RAS timeout multiplier |
| `NCCL_UID_STAGGER_RATE` | 7000 | src/bootstrap.cc | UID stagger rate |
| `NCCL_UID_STAGGER_THRESHOLD` | 256 | src/bootstrap.cc | UID stagger threshold |

---

## Profiling and Tracing

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_PROFILER_PLUGIN` | - | src/plugin/profiler.cc | Profiler plugin library |
| `NCCL_TUNER_PLUGIN` | - | src/plugin/tuner.cc | Tuner plugin library |
| `NCCL_TUNER_CONFIG_FILE` | - | ext-tuner/example | Tuner config file |
| `RCCL_ENABLE_PROXY_TRACE` | 0 | src/init.cc | Enable proxy tracing |
| `RCCL_KERNEL_COLL_TRACE_ENABLE` | 0 | src/init.cc | Enable kernel collective tracing |
| `RCCL_KERNEL_COLL_TRACE_THREAD_ENABLE` | 0 | src/init.cc | Enable thread-level kernel tracing |
| `RCCL_ENABLE_CONTEXT_TRACKING` | 0 | src/init.cc | Enable context tracking |
| `RCCL_LATENCY_PROFILER` | - | src/misc/latency_profiler | Enable latency profiler |
| `NCCL_COLLTRACE_CHECK_INTERVAL_MS` | 10 | src/misc/latency_profiler | Collection trace check interval |
| `NCCL_COLLTRACE_RECORD_MAX` | 100 | src/misc/latency_profiler | Max collection trace records |
| `NCCL_COLLTRACE_MAX_DUMP_SIZE` | 20 | src/misc/latency_profiler | Max dump size for traces |
| `NCCL_COLLTRACE_DUMP_INTERVAL_SEC` | 300 | src/misc/latency_profiler | Dump interval for traces |
| `NPKIT_DUMP_DIR` | - | src/init.cc | NPKit dump directory |
| `NCCL_PROFILE_EVENT_MASK` | - | ext-profiler/example | Event mask for profiling |
| `NCCL_PROFILE_GROUP_POOL_SIZE` | - | ext-profiler/example | Group pool size |
| `NCCL_PROFILE_COLL_POOL_SIZE` | - | ext-profiler/example | Collective pool size |
| `NCCL_PROFILE_P2P_POOL_SIZE` | - | ext-profiler/example | P2P pool size |
| `NCCL_PROFILE_PROXY_CTRL_POOL_SIZE` | - | ext-profiler/example | Proxy control pool size |
| `NCCL_PROFILE_PROXY_DETACH_POOL_SIZE` | - | ext-profiler/example | Proxy detach pool size |
| `NCCL_PROFILE_DUMP_FILE` | - | ext-profiler/example | Profile dump file |

---

## Proxy and Progress Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_PROXY_APPEND_BATCH_SIZE` | 16 | src/proxy.cc | Proxy append batch size |
| `NCCL_PROXY_DUMP_SIGNAL` | -1 | src/proxy.cc | Signal for proxy dump |
| `NCCL_PROGRESS_APPENDOP_FREQ` | 8 | src/proxy.cc | Progress append operation frequency |
| `NCCL_CREATE_THREAD_CONTEXT` | 0 | src/proxy.cc | Create thread context |

---

## Group and Synchronization

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_GROUP_CUDA_STREAM` | (default) | src/init.cc | Group CUDA stream mode |
| `NCCL_WIN_ENABLE` | 1 | src/init.cc | Enable window operations |
| `NCCL_WIN_STRIDE` | -1 | src/group.cc | Window stride |
| `NCCL_RUNTIME_CONNECT` | 1 | src/init.cc | Runtime connection mode |
| `NCCL_COMM_BLOCKING` | UNDEF | src/init.cc | Blocking communication mode |

---

## Connection and Bootstrap

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_CONNECT_ROUND_MAX_PEERS` | 128 | src/transport.cc | Max peers per connection round |
| `NCCL_REPORT_CONNECT_PROGRESS` | 0 | src/transport.cc | Report connection progress |

---

## Collective-Specific Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `RCCL_PIVOT_ALLTOALL_ENABLE` | 1 | src/init.cc | Enable pivot alltoall |
| `RCCL_ALL_TO_ALL_PIVOT_ENABLE` | 0 | src/collectives.cc | Enable alltoall pivot |
| `RCCL_LL128_FORCE_ENABLE` | 0 | src/init.cc | Force enable LL128 |
| `NCCL_LL128_C2C` | 1 | src/graph/tuning.cc | LL128 chip-to-chip |
| `RCCL_DIRECT_ALLGATHER_THRESHOLD` | 75497472 | src/rccl_wrap.cc | Direct allgather threshold |
| `RCCL_PIPELINE_ALL_DATA_TYPES` | 0 | src/rccl_wrap.cc | Pipeline all data types |
| `RCCL_DISABLE_REDUCE_COPY_PIPELINING` | 0 | src/rccl_wrap.cc | Disable reduce-copy pipelining |
| `NCCL_ALLOC_P2P_NET_LL_BUFFERS` | 0 | src/init.cc | Allocate P2P network LL buffers |
| `NCCL_PAT_ENABLE` | 0 | src/graph/tuning.cc | Enable PAT (Performance Analysis Tool) |

---

## Launch and Execution

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_LAUNCH_ORDER_IMPLICIT` | 0 | src/enqueue.cc | Implicit launch order |
| `NCCL_LAUNCH_RACE_FATAL` | 1 | src/misc/strongstream.cc | Launch race detection is fatal |
| `NCCL_MEM_SYNC_DOMAIN` | Remote | src/enqueue.cc | Memory sync domain |
| `NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE` | 0 | src/enqueue.cc | NVLink utilization-centric scheduling |
| `NCCL_CTA_POLICY` | UNDEF | src/init.cc | CTA (thread block) policy |

---

## Graph and CUDA Graph Support

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_GRAPH_MIXING_SUPPORT` | 0 | src/misc/strongstream.cc | Graph mixing support |
| `NCCL_GRAPH_HELPER_DISABLE` | 0 | src/init.cc | Disable graph helper |

---

## Communication Split and Resource Sharing

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_COMM_SPLIT_SHARE_RESOURCES` | UNDEF | src/init.cc | Share resources on comm split |
| `NCCL_COMM_SHRINK_SHARE_RESOURCES` | UNDEF | src/init.cc | Share resources on comm shrink |

---

## Development and Testing

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `NCCL_CHECK_POINTERS` | 0 | src/init.cc | Enable pointer checking |
| `RCCL_INJECT_FAULTS` | 0 | src/init.cc | Inject faults for testing |
| `CUDA_LAUNCH_BLOCKING` | - | src/misc/rocmwrap.cc, src/misc/cudawrap.cc | CUDA blocking kernel launches |
| `VERBOSE` | - | tools/JitterBench | Verbose output |
| `LAUNCH_MODE` | - | tools/JitterBench | Launch mode for benchmarks |
| `RCCL_TEST_NETSOCKET_MAX_ATTEMPTS` | - | test/NetSocketTests.cpp | Max connection attempts |
| `RCCL_TEST_NETSOCKET_SLEEP_MS` | - | test/NetSocketTests.cpp | Sleep between attempts |
| `RCCL_ENABLE_SIGNALHANDLER` | 0 | src/misc/signals.cc | Enable custom signal handler |

---

## External Dependencies

### ROCm/HSA Settings

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `RCCL_ROCR_PATH` | - | src/misc/rocmwrap.cc | ROCr runtime path |
| `HSA_NO_SCRATCH_RECLAIM` | - | src/init.cc | HSA scratch memory reclaim |
| `HSA_FORCE_FINE_GRAIN_PCIE` | - | src/init.cc | Force fine-grain PCIe memory |
| `RCCL_GFX9_CHEAP_FENCE_OFF` | 0 | src/init.cc | Disable GFX9 cheap fence |
| `RCCL_USE_ROCM_SMI_LIB` | 0 | src/misc/rocm_smi_wrap.cc | Use ROCm SMI library |

### RCCL Path

| Environment Variable | Default | Source File | Description |
|---------------------|---------|-------------|-------------|
| `RCCL_PATH` | - | tools/scripts/rcclDiagnostics.py | RCCL installation path |

---

## Summary Statistics

### By Prefix

- **NCCL_** prefix: ~170 environment variables
- **RCCL_** prefix: ~45 environment variables
- **MSCCLPP_** prefix: ~5+ environment variables
- **HSA_** prefix: 2 environment variables
- **NPKIT_** prefix: 1 environment variable
- **Other**: ~5 environment variables

### By Category

1. Network and Transport: ~60 variables
2. InfiniBand/RDMA: ~30 variables
3. Performance Tuning: ~35 variables
4. Algorithm and Protocol: ~10 variables
5. Topology and Hardware: ~20 variables
6. Memory Management: ~15 variables
7. Profiling and Tracing: ~20 variables
8. Development and Testing: ~10 variables
9. MSCCL/MSCCLPP: ~15 variables
10. Miscellaneous: ~15 variables

---

## Key Files for Environment Variable Definitions

| File | Parameter Count | Description |
|------|----------------|-------------|
| src/init.cc | ~40 | Core initialization parameters |
| src/transport/net_ib.cc | ~25 | InfiniBand transport parameters |
| src/graph/paths.cc | ~13 | Topology path parameters |
| src/enqueue.cc | ~12 | Enqueue and execution parameters |
| src/transport/net_socket.cc | ~4 | Socket transport parameters |
| src/misc/socket.cc | ~8 | Socket utility parameters |
| src/graph/tuning.cc | ~5 | Tuning parameters |
| src/graph/topo.cc | ~3 | Topology parameters |
| src/graph/connect.cc | ~6 | Connection parameters |
| src/rccl_wrap.cc | ~4 | RCCL-specific wrappers |

---

## Notes

1. **Default Values**: Many parameters use special default values like `-2` (meaning auto-detect/dynamic) or `NCCL_CONFIG_UNDEF_INT` (undefined).

2. **Parameter Naming**: 
   - `NCCL_PARAM` creates environment variables with `NCCL_` prefix
   - `RCCL_PARAM` creates environment variables with `RCCL_` prefix

3. **Configuration Files**: RCCL supports reading environment variables from configuration files:
   - `~/.rccl.conf` (user-specific)
   - `/etc/rccl.conf` (system-wide)
   - Custom path via `NCCL_CONF_FILE`

4. **Compatibility**: Many `NCCL_*` variables are maintained for NVIDIA NCCL compatibility, while `RCCL_*` variables are AMD-specific extensions.

5. **Dynamic Parameters**: Some parameters marked with `-2` default are dynamically determined based on hardware capabilities and runtime conditions.

## Additional Environment Variables Found via Direct getenv() Calls

The following environment variables are accessed directly via `getenv()` rather than through the PARAM macros:

- `TEST_VAR` - Used in ParamTests.cpp for testing
- `TEST_VAR_WITH_NO_VALUE` - Used in ParamTests.cpp for testing
- Various MSCCLPP environment variables in the ext-src/mscclpp subdirectory

---

**Generated**: 2025-10-29  
**Source**: RCCL codebase at /Users/ahalperin/xai/amd-dev/amd/rccl  
**Analysis Method**: Comprehensive grep search for NCCL_PARAM, RCCL_PARAM macros and getenv() calls

