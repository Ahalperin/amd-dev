# NCCLX Device Networking Support

## Overview

Device networking support in NCCLX represents a significant architectural advancement over standard NCCL, enabling network operations to be directly invoked from GPU kernels. This capability allows for GPU-driven networking, reducing CPU involvement and latency in communication operations.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Device Networking Types](#device-networking-types)
3. [The UNPACK Device Type](#the-unpack-device-type)
4. [Device Handle Management](#device-handle-management)
5. [GPU-Side Implementation](#gpu-side-implementation)
6. [Memory Architecture](#memory-architecture)
7. [Integration with NCCL Kernels](#integration-with-nccl-kernels)
8. [Implementation Details](#implementation-details)
9. [Performance Implications](#performance-implications)
10. [Benefits and Use Cases](#benefits-and-use-cases)

---

## Architecture Overview

Device networking support allows network plugins to expose device-side functionality that can be invoked directly from CUDA kernels. This enables:

1. **GPU-Initiated Network Operations**: The GPU can directly control network send/receive operations without round-tripping through the CPU proxy
2. **Reduced Latency**: Eliminates CPU-GPU synchronization overhead for network operations
3. **Device-Side Data Processing**: Enables unpacking, format conversion, and other data transformations directly on the GPU
4. **Asynchronous Overlap**: Better overlap between compute, network, and data movement operations

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Kernel                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ncclNetDeviceUnpack (device function)                 │ │
│  │  • Invoked from collective primitives                  │ │
│  │  • Operates on bounce buffers                          │ │
│  │  • Uses device handles for metadata                    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Device Handle (ncclNetDeviceHandle_t)          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  • netDeviceType: UNPACK                               │ │
│  │  • netDeviceVersion: 0x7                               │ │
│  │  │  • handle: Pointer to unpackNetDeviceHandle         │ │
│  │  • size: Handle size                                   │ │
│  │  • needsProxyProgress: Requires proxy thread           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│        UNPACK-Specific Handle (unpackNetDeviceHandle)       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  • meta: Pointer to netUnpackMeta (mapped memory)      │ │
│  │  • bounce_buf: Staging buffer for network data         │ │
│  │  • head: Queue head pointer                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Metadata Structure (netUnpackMeta)                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  • mem[16][pages]: Load metadata for each request      │ │
│  │    - src_off, dst_off, len for each page              │ │
│  │  • cnt[16]: Page count per request                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Device Networking Types

### Type Enumeration

NCCLX defines two device networking types:

**File**: `src/include/net_device.h`

```c
typedef enum {
    NCCL_NET_DEVICE_HOST = 0,     // Standard host-driven networking
    NCCL_NET_DEVICE_UNPACK = 1    // Device-driven unpacking
} ncclNetDeviceType;
```

### Version Numbers

```c
#define NCCL_NET_DEVICE_INVALID_VERSION  0x0
#define NCCL_NET_DEVICE_UNPACK_VERSION   0x7  // Current version for UNPACK
```

The version number is an arbitrary identifier that must match exactly between NCCL and the network plugin. NCCL validates the version at initialization time.

### Device Handle Structure

**File**: `src/include/net_device.h`

```c
typedef struct {
    ncclNetDeviceType netDeviceType;  // Type: HOST or UNPACK
    int netDeviceVersion;              // Version identifier
    void* handle;                      // Pointer to type-specific handle
    size_t size;                       // Size of the handle data
    int needsProxyProgress;            // Whether proxy must make progress
} ncclNetDeviceHandle_t;
```

---

## The UNPACK Device Type

### Purpose

The UNPACK device type is designed to handle a specific optimization: unpacking non-contiguous network data directly on the GPU. When network data arrives in a scatter-gather (iovec) format, traditional approaches require the CPU to unpack this data. The UNPACK device type moves this work to the GPU.

### UNPACK Handle Structure

**File**: `src/device/network/unpack/unpack_defs.h`

```c
struct unpackNetDeviceHandle {
    struct netUnpackMeta *meta;  // Mapped metadata (GPU-accessible)
    void* bounce_buf;            // Staging buffer for network data
    uint64_t head;               // Current queue position
};
```

### Metadata Structure

The `netUnpackMeta` structure describes how to unpack data:

```c
#define NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH 16
#define NET_UNPACK_MAX_SLICE_SIZE 4194304  // 4MB per recv call
#define NET_UNPACK_MAX_SLICE_PAGES (NET_UNPACK_MAX_SLICE_SIZE / 4096 * 2)

struct netUnpackMeta {
    // For each queued request, array of page metadata
    loadMeta mem[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH][NET_UNPACK_MAX_SLICE_PAGES];
    // Count of pages for each request
    uint64_t cnt[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH];
};
```

### Load Metadata

Each page's metadata describes a single contiguous memory copy:

```c
union alignas(16) loadMeta {
    uint64_t r64[2];
    struct {
        uint32_t src_off;   // Source offset in bounce buffer
        uint32_t len;       // Number of bytes to copy
        uint64_t dst_off;   // Destination offset in user buffer
    };
};
```

The 16-byte alignment ensures efficient loading via `load128` instructions.

---

## Device Handle Management

### Handle Allocation

**File**: `src/transport/net.cc`

Device handles are allocated on-demand when a network plugin reports device networking support:

```c
static ncclResult_t ncclNetGetDeviceHandle(
    ncclNetDeviceType type, 
    int version, 
    bool isRecv, 
    ncclNetDeviceHandle_t** handle
) {
    bool needsDeviceHandle = false;
    
    // Currently, only UNPACK type for recv operations needs a device handle
    if (type == NCCL_NET_DEVICE_UNPACK) {
        if (version == NCCL_NET_DEVICE_UNPACK_VERSION && isRecv) {
            needsDeviceHandle = true;
        }
    }
    
    // Don't re-alloc netDeviceHandles
    if (needsDeviceHandle && (*handle == NULL)) {
        NCCLCHECK(ncclCalloc(handle, 1));
        (*handle)->netDeviceType = type;
        (*handle)->netDeviceVersion = version;
    }
    
    return ncclSuccess;
}
```

### Handle Lifecycle

1. **Discovery**: During `getProperties()`, the network plugin reports `netDeviceType` and `netDeviceVersion`
2. **Allocation**: NCCL allocates a generic `ncclNetDeviceHandle_t` structure
3. **Plugin Initialization**: The plugin's `connect()`/`accept()` fills in the handle details
4. **GPU Mapping**: The handle is passed to the GPU kernel via device memory
5. **Cleanup**: Handles are freed when connections are closed

### Resource Setup

Device handles are created during send/recv setup:

**For Send Operations**:
```c
// Line 713 in net.cc
NCCLCHECK(ncclNetGetDeviceHandle(
    resources->netDeviceType, 
    resources->netDeviceVersion, 
    false /*isRecv*/, 
    &resources->netDeviceHandle
));
```

**For Recv Operations**:
```c
// Line 882 in net.cc
NCCLCHECK(ncclNetGetDeviceHandle(
    resources->netDeviceType, 
    resources->netDeviceVersion, 
    true /*isRecv*/, 
    &resources->netDeviceHandle
));
```

---

## GPU-Side Implementation

### Shared Memory Structure

Device networking state is maintained in GPU shared memory for fast access:

**File**: `src/device/common.h`

```c
struct ncclShmemGroup {
    ncclConnInfo *recvConns[NCCL_MAX_ARITY];
    ncclConnInfo *sendConns[NCCL_MAX_ARITY];
    void* userInput;
    void* userOutput;
    void* srcs[NCCL_MAX_ARITY+1];
    void* dsts[NCCL_MAX_ARITY+1];
    
    // Device plugin state per group
    union {
        unpackGroupShmem unpack;
    } devicePlugin;
    
    int32_t dstSizes[NCCL_MAX_ARITY+1];
};

struct ncclShmemData {
    struct ncclDevKernelArgs args;
    int channelId;
    int aborted;
    struct ncclDevComm comm;
    struct ncclDevChannel channel;
    
    // ... other fields ...
    
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
    
    // Global device plugin state
    alignas(16) union {
        unpackShmem unpack;
    } devicePlugin;
};

// Shared memory accessible from all threads
extern __shared__ ncclShmemData ncclShmem;
```

### UNPACK Shared Memory State

**File**: `src/device/network/unpack/unpack_defs.h`

```c
#define NET_UNPACK_MAX_GROUPS 16
#define NET_UNPACK_MAX_NPEERS 2

// Per-warp scratch space
#define WARP_SHM_PAGE_CNT 4
#define WARP_SHM_SIZE (WARP_SHM_PAGE_CNT * sizeof(union loadMeta))

// Global unpack state
struct unpackShmem {
    void* bounce_buf;  // Shared across all groups
};

// Per-group unpack state
struct unpackGroupShmem {
    int unpackNetDeviceIndexMask;  // Bitmask of network recv peers
    uint64_t head[NET_UNPACK_MAX_NPEERS];  // Queue head per peer
    struct netUnpackMeta* g_meta[NET_UNPACK_MAX_NPEERS];  // Metadata pointers
};
```

### Device Function: Setup

**File**: `src/device/network/unpack/unpack.h`

```c
// Called once at initialization to map handle to group/peer
__device__ void ncclNetDeviceUnpackSetup(
    void* ohandle,      // Generic device handle
    const int group,    // Group index
    const int index     // Peer index
) {
    struct unpackNetDeviceHandle* handle = 
        (struct unpackNetDeviceHandle*) ohandle;
    
    // Store metadata pointer in shared memory for this group/peer
    ncclShmem.groups[group].devicePlugin.unpack.g_meta[index] = handle->meta;
    
    // Store bounce buffer (shared across all groups)
    ncclShmem.devicePlugin.unpack.bounce_buf = handle->bounce_buf;
    
    // Initialize queue head
    ncclShmem.groups[group].devicePlugin.unpack.head[index] = handle->head;
}
```

### Device Function: Unpack (Entry Point)

The main entry point for device-side unpacking:

```c
template <int Recv>
__device__ void ncclNetDeviceUnpack(
    const int tid,          // Thread ID
    const int tidInBlock,   // Thread ID in block
    const int nworkers,     // Number of worker threads
    const int group,        // Group index
    int mask,               // Bitmask of peers to unpack
    int Src,                // Source index
    int workSize            // Size of work to unpack
);

// Send specialization (no-op)
template <>
__device__ void ncclNetDeviceUnpack</*Recv=*/0>(
    const int tid, const int tidInBlock, const int nworkers, 
    const int group, int mask, int Src, int workSize
) {
    // Send unpack is empty - no unpacking needed for send
}

// Recv specialization (does the work)
template <>
__device__ void ncclNetDeviceUnpack</*Recv=*/1>(
    const int tid, const int tidInBlock, const int nworkers,
    const int group, int mask, int Src, int workSize
) {
    while (mask != 0) {
        int ix = __ffs(mask) - 1;  // Find first set bit (peer index)
        mask &= mask - 1;          // Clear that bit
        
        // Unpack data from this peer
        ncclNetDeviceUnpackInner(
            tid, tidInBlock, nworkers, group, ix,
            ncclShmem.groups[group].srcs[ix + Src],  // Destination buffer
            workSize,  // Size
            ncclShmem.groups[group].devicePlugin.unpack.head[ix]  // Queue position
        );
    }
}
```

### Device Function: Unpack Inner Logic

The core unpacking implementation that all threads participate in:

```c
__device__ void ncclNetDeviceUnpackInner(
    const int tid,
    const int tidInBlock,
    const int nworkers,
    const int group,
    const int index,
    void *src,              // User destination buffer
    const int nbytes,       // Total bytes to unpack
    const uint64_t step     // Current queue step
) {
    const int w = tid / WARP_SIZE;         // Warp number
    const int nw = nworkers / WARP_SIZE;   // Number of warps
    const int t = tid % WARP_SIZE;         // Lane within warp
    
    BytePack<16> reg;
    loadMeta meta;
    
    // Load handle information from shared memory
    uint64_t head = step % NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH;
    struct netUnpackMeta* g_meta_struct = 
        ncclShmem.groups[group].devicePlugin.unpack.g_meta[index];
    void* bounce_buf = ncclShmem.devicePlugin.unpack.bounce_buf;
    
    // Get metadata array for this request
    loadMeta* g_meta = g_meta_struct->mem[head];
    
    // Use per-warp scratch space in shared memory
    loadMeta* s_meta = (loadMeta*) ncclScratchForWarp(tidInBlock / WARP_SIZE);
    
    // Load the page count for this request (relaxed GPU load for polling)
    uint64_t meta_cnt;
    load64gpu(g_meta_struct->cnt + head, meta_cnt);
    
    // Calculate pages per warp for load balancing
    int PPW = ppw(nbytes, nw);
    
    // Each warp processes PPW pages at a time
    for (uint64_t meta_s = w * PPW; meta_s < meta_cnt; meta_s += nw * PPW) {
        uint64_t iter_meta_cnt = min(meta_cnt - meta_s, (uint64_t)PPW);
        
        // Step 1: Load metadata from global to shared memory
        if (t < PPW && t < iter_meta_cnt) {
            load128((const uint64_t*) (g_meta + (meta_s + t)), 
                    reg.u64[0], reg.u64[1]);
            storeShmem128(shmemCvtPtr((uint64_t *)(s_meta + (w * PPW + t))), 
                         reg.u64[0], reg.u64[1]);
        }
        __syncwarp();
        
        // Step 2: Process each page of this iteration
        for (int x = 0; x < iter_meta_cnt; x++) {
            int meta_idx = x + w * PPW;
            
            // Load page metadata from shared memory
            loadShmem128(shmemCvtPtr((uint64_t*) (s_meta + meta_idx)), 
                        meta.r64[0], meta.r64[1]);
            
            // Fast path: bulk copy if length >= 16 bytes
            if (meta.len >= 16) {
                // Determine alignment for optimal load size
                uint8_t align_off = (meta.src_off | meta.dst_off) % 16;
                align_off = align_off & -align_off;  // Keep lowest bit
                
                char* cpy_src = (char*) bounce_buf + meta.src_off;
                char* cpy_dst = (char*) src + meta.dst_off;
                
                if (align_off == 0) {
                    bulkLoad<16>(t, meta.len, cpy_src, cpy_dst, &reg, 
                                w, g_meta, s_meta, meta.src_off, meta.dst_off);
                } else if (align_off & 0x8) {
                    bulkLoad<8>(t, meta.len, cpy_src, cpy_dst, 
                               (BytePack<8>*) &reg, w, g_meta, s_meta, 
                               meta.src_off, meta.dst_off);
                } else if (align_off & 0x4) {
                    bulkLoad<4>(t, meta.len, cpy_src, cpy_dst, 
                               (BytePack<4>*) &reg, w, g_meta, s_meta, 
                               meta.src_off, meta.dst_off);
                } else if (align_off & 0x2) {
                    bulkLoad<2>(t, meta.len, cpy_src, cpy_dst, 
                               (BytePack<2>*) &reg, w, g_meta, s_meta, 
                               meta.src_off, meta.dst_off);
                } else {
                    bulkLoad<1>(t, meta.len, cpy_src, cpy_dst, 
                               (BytePack<1>*) &reg, w, g_meta, s_meta, 
                               meta.src_off, meta.dst_off);
                }
            }
            
            // Handle tail bytes (< 16 bytes or remainder)
            if (t < meta.len % 16) {
                uint64_t tail_offset = (meta.len / 16) * 16;
                volatile char* cpy_src = 
                    (char*) bounce_buf + meta.src_off + tail_offset + t;
                volatile char* cpy_dst = 
                    (char*) src + meta.dst_off + tail_offset + t;
                *cpy_dst = *cpy_src;
            }
        }
        __syncwarp();
    }
}
```

### Bulk Load Template

The bulk load function is templated on load size for maximum performance:

```c
// Example: 16-byte aligned loads (best case)
template <>
__device__ void bulkLoad<16>(
    const int t, 
    const uint32_t len, 
    char* cpy_src, 
    char* cpy_dst, 
    BytePack<16> reg[1], 
    const int w, 
    loadMeta* g_meta, 
    loadMeta* s_meta, 
    uint32_t src_off, 
    uint64_t dst_off
) {
    uint64_t data_s;
    for (data_s = t * 16; data_s + 15 < len; data_s += WARP_SIZE * 16) {
        // Load 16 bytes
        reg[0] = ld_volatile_global<16>((uintptr_t)(cpy_src + data_s));
        // Store 16 bytes
        st_global<16>((uintptr_t)(cpy_dst + data_s), reg[0]);
    }
}
```

Similar specializations exist for 8, 4, 2, and 1-byte loads to handle various alignments.

---

## Memory Architecture

### Three-Level Memory Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│  Host/Network                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Network Plugin fills:                                    │  │
│  │  • bounce_buf (network staging buffer)                    │  │
│  │  • netUnpackMeta.mem[] (page descriptors)                 │  │
│  │  • netUnpackMeta.cnt[] (page counts)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  GPU Global Memory                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Mapped Memory (GPU-accessible):                          │  │
│  │  • bounce_buf: Up to 4MB per request                      │  │
│  │  • netUnpackMeta: ~64KB (16 requests × 4KB)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  GPU Shared Memory (per block)                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Per-warp scratch (WARP_SHM_SIZE = 64 bytes):             │  │
│  │  • 4 × loadMeta (16 bytes each)                           │  │
│  │  • Used to cache page metadata                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ncclShmem.devicePlugin.unpack:                           │  │
│  │  • bounce_buf pointer                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ncclShmem.groups[].devicePlugin.unpack:                  │  │
│  │  • g_meta[2]: Metadata pointers                           │  │
│  │  • head[2]: Queue heads                                   │  │
│  │  • unpackNetDeviceIndexMask: Peer bitmask                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  GPU Registers (per thread)                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  BytePack<16> reg: Temporary data                         │  │
│  │  loadMeta meta: Current page metadata                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Memory Flow

1. **Network Receive**: Plugin receives data into `bounce_buf`
2. **Metadata Generation**: Plugin fills `netUnpackMeta` with page descriptors
3. **GPU Kernel Launch**: NCCL kernel reads handle from shared memory
4. **Metadata Staging**: Warps cooperatively load page metadata to shared memory
5. **Data Copy**: Threads use metadata to copy from bounce buffer to user destination
6. **Completion**: Queue head advances for next request

### Memory Access Patterns

**Coalesced Global Memory Access**:
- Threads in a warp access consecutive 16-byte chunks
- `bulkLoad` templates ensure proper alignment
- Fallback to smaller loads for unaligned data

**Efficient Shared Memory Usage**:
- Per-warp scratch space avoids bank conflicts
- Metadata cached in shared memory reduces global memory bandwidth
- `loadShmem128` and `storeShmem128` use PTX shared memory intrinsics

**Relaxed Consistency for Polling**:
```c
__device__ void load64gpu(const uint64_t* ptr, uint64_t &v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];"
                : "=l"(v) : "l"(ptr) : "memory");
#else
    asm volatile("ld.volatile.global.u64 {%0}, [%1];"
                : "=l"(v) : "l"(ptr) : "memory");
#endif
}
```

This relaxed load is used to poll `netUnpackMeta.cnt` without triggering excessive cache invalidations.

---

## Integration with NCCL Kernels

### Invocation from Primitives

Device unpacking is called from the NCCL collective primitives:

**File**: `src/device/prims_simple.h`

```c
// During receive operations, after network data arrives
ncclNetDeviceUnpack<Recv>(
    tid, 
    tidInBlock, 
    nworkers, 
    group, 
    ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask,
    Src, 
    workSize
);
```

### Setup During Initialization

**File**: `src/device/prims_simple.h` (approximately line 492)

```c
// Called once per recv connection during kernel initialization
if (netDeviceHandle) {
    ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
}
```

### Peer Mask Management

**File**: `src/device/prims_simple.h` (approximately line 656)

```c
// Set mask indicating which peers use device networking
ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
```

### Head Increment

After each successful unpack operation, the queue head advances:

**File**: `src/device/network/unpack/unpack.h`

```c
__device__ void ncclNetDeviceIncrementHead(const int group, const int index) {
    ncclShmem.groups[group].devicePlugin.unpack.head[index]++;
}
```

### Head Writeback

At the end of a kernel, the head is written back to the handle for the next kernel launch:

```c
__device__ void ncclNetDeviceSaveHead(
    void* ohandle, 
    const int group, 
    const int index
) {
    struct unpackNetDeviceHandle* handle = 
        (struct unpackNetDeviceHandle*) ohandle;
    handle->head = ncclShmem.groups[group].devicePlugin.unpack.head[index];
}
```

---

## Implementation Details

### Queue Management

The UNPACK device type uses a circular queue with a fixed depth:

```
Queue Structure:
┌─────────────────────────────────────────────────────┐
│  head % 16 → [Request 0]  ← Oldest request          │
│              [Request 1]                             │
│              [Request 2]                             │
│              ...                                     │
│  tail % 16 → [Request 15] ← Newest request          │
└─────────────────────────────────────────────────────┘
```

- **Depth**: 16 outstanding requests (`NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH`)
- **Wraparound**: `head %= NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH`
- **Synchronization**: GPU polls `cnt[head]` until non-zero (data ready)

### Page-Based Unpacking

Large transfers are divided into 4KB pages:

```c
#define SLICE_PAGE_SIZE 4096
#define NET_UNPACK_MAX_SLICE_SIZE 4194304  // 4MB
#define NET_UNPACK_MAX_SLICE_PAGES \
    (NET_UNPACK_MAX_SLICE_SIZE / SLICE_PAGE_SIZE * 2)  // 2048 pages
```

Each page has a `loadMeta` descriptor specifying:
- Source offset in bounce buffer
- Destination offset in user buffer
- Length to copy

### Warp-Level Parallelism

The unpack operation is parallelized across warps:

```c
int PPW = ppw(nbytes, nw);  // Pages per warp

for (uint64_t meta_s = w * PPW; meta_s < meta_cnt; meta_s += nw * PPW) {
    // Each warp processes PPW pages
    // ...
}
```

The `ppw()` function calculates pages per warp based on total pages and warp count, ensuring balanced work distribution.

### Alignment Optimization

The bulk load logic dynamically selects the largest possible load size based on alignment:

```c
uint8_t align_off = (meta.src_off | meta.dst_off) % 16;
align_off = align_off & -align_off;  // Isolate lowest set bit

if (align_off == 0) {
    // 16-byte aligned
    bulkLoad<16>(...);
} else if (align_off & 0x8) {
    // 8-byte aligned
    bulkLoad<8>(...);
} else if (align_off & 0x4) {
    // 4-byte aligned
    bulkLoad<4>(...);
} else if (align_off & 0x2) {
    // 2-byte aligned
    bulkLoad<2>(...);
} else {
    // 1-byte aligned (worst case)
    bulkLoad<1>(...);
}
```

This ensures maximum memory bandwidth utilization regardless of buffer alignment.

### Proxy Progress Requirement

Device handles can indicate they need proxy thread progress:

**File**: `src/include/net_device.h`

```c
typedef struct {
    // ...
    int needsProxyProgress;  // If true, proxy must make progress calls
} ncclNetDeviceHandle_t;
```

When `needsProxyProgress` is true, the CPU proxy thread continues to call the plugin's `test()` function to advance network operations, even though the GPU is doing the unpacking work.

---

## Performance Implications

### Advantages

1. **Reduced CPU Overhead**
   - No CPU-side memcpy or data unpacking
   - Proxy thread focuses on network progress, not data movement

2. **Lower Latency**
   - GPU doesn't wait for CPU to unpack data
   - Unpacking happens in parallel with network receive

3. **Better Overlap**
   - Compute kernels can overlap with unpack operations
   - Network and GPU work proceed concurrently

4. **Memory Bandwidth**
   - GPU memory subsystem is faster than PCIe transfers
   - Data stays on GPU, avoiding round-trip to host

5. **Scalability**
   - All GPU threads participate in unpacking
   - Workload distributes across warps automatically

### Potential Overheads

1. **Shared Memory Usage**
   - Per-warp scratch space (64 bytes/warp)
   - May limit occupancy for kernels with high shared memory usage

2. **Register Pressure**
   - `BytePack<16>` and metadata structures use registers
   - May spill to local memory if register count is high

3. **Complexity**
   - More complex error handling (device-side assertions)
   - Debugging is harder than host-side code

4. **Plugin Development**
   - Requires plugin to generate page metadata
   - Bounce buffer management adds complexity

### Performance Tuning

**Environment Variables** (potential - check NCCL documentation):
- `NCCL_NET_DEVICE_UNPACK_ENABLED`: Enable/disable device unpacking
- `NCCL_NET_DEVICE_UNPACK_THRESHOLD`: Minimum size to use device unpacking

**Load Balancing**:
The `ppw()` function adjusts pages per warp to balance work:

```c
inline __device__ int ppw(const int nbytes, int nw) {
    int v = DIVUP(nbytes, SLICE_PAGE_SIZE);  // Total pages
    v = DIVUP(v, nw);                         // Pages per warp
    while (v > WARP_SHM_PAGE_CNT) {
        v = DIVUP(v, 2);                      // Halve if too large
    }
    return v;
}
```

This ensures shared memory scratch space isn't exceeded.

---

## Benefits and Use Cases

### Ideal Scenarios

1. **Large Non-Contiguous Receives**
   - Scatter-gather I/O from network
   - InfiniBand receives with multiple SGEs
   - GPU can unpack faster than CPU memcpy

2. **High Message Rate**
   - Many small messages
   - CPU would become bottleneck for unpacking
   - GPU parallelism handles high rate efficiently

3. **Latency-Sensitive Collectives**
   - Every microsecond matters
   - Eliminating CPU synchronization saves time
   - Direct GPU-to-GPU path is fastest

4. **Limited CPU Resources**
   - CPU is busy with other tasks
   - Offloading unpack to GPU frees CPU cycles
   - Better overall system utilization

### Comparison with Traditional Approach

**Traditional (Host-Driven)**:
```
Network → Bounce Buffer (GPU memory)
       ↓
    CPU reads from GPU
       ↓
    CPU unpacks data
       ↓
    CPU writes to GPU
       ↓
User Buffer (GPU memory)
```

**Device Networking (GPU-Driven)**:
```
Network → Bounce Buffer (GPU memory)
       ↓
    GPU reads metadata
       ↓
    GPU unpacks directly to User Buffer
```

**Latency Savings**:
- Eliminates 2× PCIe transfers (GPU→CPU→GPU)
- Eliminates CPU processing time
- Eliminates CPU-GPU synchronization

### Real-World Applications

1. **Distributed Training (Large Models)**
   - AllReduce with large gradient tensors
   - Tensor parallelism communication
   - Pipeline parallelism stage transfers

2. **High-Frequency Trading Systems**
   - Ultra-low latency requirements
   - GPU-accelerated analytics
   - Direct GPU-to-network path

3. **Scientific Computing**
   - Multi-GPU simulations
   - Halo exchanges in domain decomposition
   - Large-scale parallel solvers

4. **Video/Image Processing Pipelines**
   - Distributed rendering
   - Real-time video analytics
   - Multi-GPU image stitching

---

## Plugin Implementation Guide

### Reporting Device Networking Support

In `getProperties()`:

```c
ncclResult_t myPlugin_getProperties(int dev, ncclNetProperties_t* props) {
    // ... set other properties ...
    
    // Enable device networking for this device
    props->netDeviceType = NCCL_NET_DEVICE_UNPACK;
    props->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
    
    return ncclSuccess;
}
```

### Allocating Device Handles

In `accept()` for recv:

```c
ncclResult_t myPlugin_accept(
    void* listenComm, 
    void** recvComm, 
    ncclNetDeviceHandle_t** recvDevComm
) {
    // ... establish connection ...
    
    if (recvDevComm != NULL) {
        // Allocate UNPACK handle
        struct unpackNetDeviceHandle* handle;
        NCCLCHECK(ncclCudaCalloc(&handle, 1));
        
        // Allocate metadata (GPU-accessible)
        NCCLCHECK(ncclCudaCalloc(&handle->meta, 1));
        
        // Allocate bounce buffer (up to 4MB)
        NCCLCHECK(ncclCudaMalloc(&handle->bounce_buf, 
                                  NET_UNPACK_MAX_SLICE_SIZE));
        
        handle->head = 0;
        
        // Fill in generic handle
        (*recvDevComm)->netDeviceType = NCCL_NET_DEVICE_UNPACK;
        (*recvDevComm)->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
        (*recvDevComm)->handle = handle;
        (*recvDevComm)->size = sizeof(*handle);
        (*recvDevComm)->needsProxyProgress = 1;  // Proxy must call test()
    }
    
    return ncclSuccess;
}
```

### Filling Metadata on Receive

In `irecv()`:

```c
ncclResult_t myPlugin_irecv(
    void* recvComm, 
    int n,           // Number of buffers (iovec count)
    void** data,     // Array of buffer pointers
    int* sizes,      // Array of buffer sizes
    int* tags, 
    void** mhandles, 
    void** request
) {
    // Post network receive into bounce_buf
    uint64_t tail = /* get current tail */;
    uint64_t slot = tail % NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH;
    
    struct unpackNetDeviceHandle* handle = get_handle(recvComm);
    
    // Fill metadata describing how to unpack
    uint64_t page_count = 0;
    uint32_t bounce_offset = 0;
    
    for (int i = 0; i < n; i++) {
        uint64_t dst_offset = (uint64_t)data[i];
        uint32_t remaining = sizes[i];
        
        while (remaining > 0) {
            uint32_t chunk = min(remaining, SLICE_PAGE_SIZE);
            
            handle->meta->mem[slot][page_count].src_off = bounce_offset;
            handle->meta->mem[slot][page_count].dst_off = dst_offset;
            handle->meta->mem[slot][page_count].len = chunk;
            
            page_count++;
            bounce_offset += chunk;
            dst_offset += chunk;
            remaining -= chunk;
        }
    }
    
    // Atomically publish page count (GPU polls this)
    __atomic_store_n(&handle->meta->cnt[slot], page_count, __ATOMIC_RELEASE);
    
    // Post actual network receive
    return backend_post_recv(recvComm, handle->bounce_buf, bounce_offset, request);
}
```

### Cleanup

In `closeRecv()`:

```c
ncclResult_t myPlugin_closeRecv(void* recvComm) {
    struct unpackNetDeviceHandle* handle = get_handle(recvComm);
    
    if (handle) {
        ncclCudaFree(handle->bounce_buf);
        ncclCudaFree(handle->meta);
        ncclCudaFree(handle);
    }
    
    // ... close connection ...
    return ncclSuccess;
}
```

---

## Future Directions

### Potential Enhancements

1. **Additional Device Types**
   - `NCCL_NET_DEVICE_PACK`: Device-driven packing for send
   - `NCCL_NET_DEVICE_RDMA`: Device-initiated RDMA operations
   - `NCCL_NET_DEVICE_COMPRESS`: On-the-fly compression/decompression

2. **Dynamic Page Sizes**
   - Adaptive page size based on message characteristics
   - Variable-sized pages for better utilization

3. **Zero-Copy Paths**
   - Direct network DMA to user buffers when possible
   - Bypass bounce buffer entirely for contiguous receives

4. **Multi-NIC Support**
   - Stripe large messages across multiple NICs
   - Load balance small messages

5. **Hardware Offload Integration**
   - BlueField DPU integration
   - SmartNIC acceleration
   - In-network compute for reductions

### Research Opportunities

1. **Workload-Aware Scheduling**
   - Profile and predict when device networking is beneficial
   - Dynamically switch between host and device modes

2. **Memory Pooling**
   - Reuse bounce buffers across connections
   - Reduce memory footprint

3. **Compression Integration**
   - GPU-side decompression after network receive
   - Reduce network bandwidth requirements

4. **Fault Tolerance**
   - Handle network errors on device side
   - Retry mechanisms without CPU involvement

---

## Summary

Device networking support in NCCLX represents a fundamental shift in how network communication is handled in GPU-accelerated systems:

- **GPU-Driven**: Network operations invoked directly from CUDA kernels
- **Low Latency**: Eliminates CPU-GPU synchronization overhead
- **High Bandwidth**: Leverages GPU memory subsystem for data movement
- **Scalable**: All GPU threads cooperate in data unpacking
- **Flexible**: Plugin API allows diverse implementations

The UNPACK device type is the first implementation of this architecture, enabling efficient handling of scatter-gather network receives directly on the GPU. This capability is particularly valuable for large-scale distributed training and HPC applications where communication latency is critical.

---

## References

### Key Files

| File | Purpose |
|------|---------|
| `src/include/net_device.h` | Device handle type definitions |
| `src/device/network/unpack/unpack_defs.h` | UNPACK structures |
| `src/device/network/unpack/unpack.h` | UNPACK device functions |
| `src/device/common.h` | Shared memory structures |
| `src/device/prims_simple.h` | Primitive integration |
| `src/transport/net.cc` | Host-side handle management |

### Related Documentation

- [NCCLX Network Plugin Extensions](./NCCLX_Network_Plugin_Extensions.md)
- [TorchComm Features Summary](./TorchComm_Features_Summary.md)
- [NCCLX Improvements Over NCCL](./NCCLX_Improvements_Over_NCCL.md)

### External Resources

- NCCL Plugin API Documentation
- CUDA Programming Guide (Device Functions, Shared Memory)
- InfiniBand Verbs Programming Guide
- NVIDIA GPUDirect RDMA Documentation

---

**Document Version**: 1.0  
**Last Updated**: November 2, 2025  
**Author**: Analysis of Meta TorchComm NCCLX Implementation






