# AMD-ANP (AINIC Network Plugin) - RCCL Plugin Interface Call Analysis

## Overview

AMD-ANP (AINIC Network Plugin) is a network plugin for RCCL that provides extended network transport support for AMD AINIC hardware. RCCL loads this plugin dynamically through the `NCCL_NET_PLUGIN` environment variable (default: `libnccl-net.so`).

## Plugin Interface Summary

The plugin implements the **ncclNet_v10_t** interface (backward compatible with v6-v9) which includes:
- Point-to-point communication (P2P)
- Collective communication (CollNet)
- Memory registration and management
- Device offload support

---

## RCCL Calls to AMD-ANP Plugin

### 1. **Plugin Initialization & Discovery**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`init()`](amd-dev/amd/rccl/src/plugin/net.cc#L143) | `src/plugin/net.cc` | [143](amd-dev/amd/rccl/src/plugin/net.cc#L143), [150](amd-dev/amd/rccl/src/plugin/net.cc#L150) | Initialize the network plugin with debug logger and profiler callback |
| [`devices()`](amd-dev/amd/rccl/src/plugin/net.cc#L144) | `src/plugin/net.cc` | [144](amd-dev/amd/rccl/src/plugin/net.cc#L144), [151](amd-dev/amd/rccl/src/plugin/net.cc#L151), [312](amd-dev/amd/rccl/src/plugin/net.cc#L312) | Query the number of available network devices |
| [`devices()`](amd-dev/amd/rccl/src/bootstrap.cc#L502) | `src/bootstrap.cc` | [502](amd-dev/amd/rccl/src/bootstrap.cc#L502) | Get network device count during bootstrap |

---

### 2. **Device Properties & Capabilities**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`getProperties()`](amd-dev/amd/rccl/src/plugin/net.cc#L119) | `src/plugin/net.cc` | [119](amd-dev/amd/rccl/src/plugin/net.cc#L119), [317](amd-dev/amd/rccl/src/plugin/net.cc#L317) | Get device properties (speed, GDR support, etc.) |
| [`getProperties()`](amd-dev/amd/rccl/src/transport/net.cc#L676) | `src/transport/net.cc` | [676](amd-dev/amd/rccl/src/transport/net.cc#L676), [716](amd-dev/amd/rccl/src/transport/net.cc#L716) | Get properties for NET transport setup |
| [`getProperties()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L334) | `src/transport/coll_net.cc` | [334](amd-dev/amd/rccl/src/transport/coll_net.cc#L334), [454](amd-dev/amd/rccl/src/transport/coll_net.cc#L454) | Get properties for CollNet transport |
| [`getProperties()`](amd-dev/amd/rccl/src/graph/paths.cc#L528) | `src/graph/paths.cc` | [528](amd-dev/amd/rccl/src/graph/paths.cc#L528) | Get properties for topology path calculation |
| [`getProperties()`](amd-dev/amd/rccl/src/bootstrap.cc#L506) | `src/bootstrap.cc` | [506](amd-dev/amd/rccl/src/bootstrap.cc#L506), [529](amd-dev/amd/rccl/src/bootstrap.cc#L529) | Get properties during bootstrap phase |

---

### 3. **Connection Establishment (P2P)**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`listen()`](amd-dev/amd/rccl/src/transport/net.cc#L731) | `src/transport/net.cc` | [731](amd-dev/amd/rccl/src/transport/net.cc#L731) | Create a listening endpoint for incoming connections |
| [`listen()`](amd-dev/amd/rccl/src/plugin/net.cc#L331) | `src/plugin/net.cc` | [331](amd-dev/amd/rccl/src/plugin/net.cc#L331) | Listen for GDR support testing |
| [`listen()`](amd-dev/amd/rccl/src/bootstrap.cc#L663) | `src/bootstrap.cc` | [663](amd-dev/amd/rccl/src/bootstrap.cc#L663), [810](amd-dev/amd/rccl/src/bootstrap.cc#L810) | Listen during bootstrap network setup |
| [`connect()`](amd-dev/amd/rccl/src/transport/net.cc#L794) | `src/transport/net.cc` | [794](amd-dev/amd/rccl/src/transport/net.cc#L794), [797](amd-dev/amd/rccl/src/transport/net.cc#L797), [806](amd-dev/amd/rccl/src/transport/net.cc#L806), [808](amd-dev/amd/rccl/src/transport/net.cc#L808), [815](amd-dev/amd/rccl/src/transport/net.cc#L815), [817](amd-dev/amd/rccl/src/transport/net.cc#L817) | Connect to remote peer for sending |
| [`connect()`](amd-dev/amd/rccl/src/plugin/net.cc#L343) | `src/plugin/net.cc` | [343](amd-dev/amd/rccl/src/plugin/net.cc#L343) | Connect during GDR capability test |
| [`connect()`](amd-dev/amd/rccl/src/bootstrap.cc#L547) | `src/bootstrap.cc` | [547](amd-dev/amd/rccl/src/bootstrap.cc#L547) | Connect during bootstrap |
| [`accept()`](amd-dev/amd/rccl/src/transport/net.cc#L997) | `src/transport/net.cc` | [997](amd-dev/amd/rccl/src/transport/net.cc#L997), [1000](amd-dev/amd/rccl/src/transport/net.cc#L1000), [1009](amd-dev/amd/rccl/src/transport/net.cc#L1009), [1011](amd-dev/amd/rccl/src/transport/net.cc#L1011), [1018](amd-dev/amd/rccl/src/transport/net.cc#L1018), [1020](amd-dev/amd/rccl/src/transport/net.cc#L1020) | Accept incoming connection from peer |
| [`accept()`](amd-dev/amd/rccl/src/plugin/net.cc#L346) | `src/plugin/net.cc` | [346](amd-dev/amd/rccl/src/plugin/net.cc#L346) | Accept connection for GDR test |
| [`accept()`](amd-dev/amd/rccl/src/bootstrap.cc#L549) | `src/bootstrap.cc` | [549](amd-dev/amd/rccl/src/bootstrap.cc#L549) | Accept connection during bootstrap |
| [`closeSend()`](amd-dev/amd/rccl/src/transport/net.cc#L1185) | `src/transport/net.cc` | [1185](amd-dev/amd/rccl/src/transport/net.cc#L1185), [1187](amd-dev/amd/rccl/src/transport/net.cc#L1187), [1190](amd-dev/amd/rccl/src/transport/net.cc#L1190) | Close send communicator |
| [`closeSend()`](amd-dev/amd/rccl/src/plugin/net.cc#L364) | `src/plugin/net.cc` | [364](amd-dev/amd/rccl/src/plugin/net.cc#L364) | Close send comm after GDR test |
| [`closeSend()`](amd-dev/amd/rccl/src/bootstrap.cc#L1166) | `src/bootstrap.cc` | [1166](amd-dev/amd/rccl/src/bootstrap.cc#L1166) | Close send comm during cleanup |
| [`closeRecv()`](amd-dev/amd/rccl/src/transport/net.cc#L1226) | `src/transport/net.cc` | [1226](amd-dev/amd/rccl/src/transport/net.cc#L1226), [1228](amd-dev/amd/rccl/src/transport/net.cc#L1228), [1231](amd-dev/amd/rccl/src/transport/net.cc#L1231) | Close receive communicator |
| [`closeRecv()`](amd-dev/amd/rccl/src/plugin/net.cc#L362) | `src/plugin/net.cc` | [362](amd-dev/amd/rccl/src/plugin/net.cc#L362) | Close recv comm after GDR test |
| [`closeRecv()`](amd-dev/amd/rccl/src/bootstrap.cc#L1167) | `src/bootstrap.cc` | [1167](amd-dev/amd/rccl/src/bootstrap.cc#L1167) | Close recv comm during cleanup |
| [`closeListen()`](amd-dev/amd/rccl/src/transport/net.cc#L1039) | `src/transport/net.cc` | [1039](amd-dev/amd/rccl/src/transport/net.cc#L1039) | Close listening communicator |
| [`closeListen()`](amd-dev/amd/rccl/src/plugin/net.cc#L365) | `src/plugin/net.cc` | [365](amd-dev/amd/rccl/src/plugin/net.cc#L365) | Close listen comm after GDR test |
| [`closeListen()`](amd-dev/amd/rccl/src/bootstrap.cc#L1168) | `src/bootstrap.cc` | [1168](amd-dev/amd/rccl/src/bootstrap.cc#L1168) | Close listen comm during cleanup |

---

### 4. **Memory Registration**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`regMr()`](amd-dev/amd/rccl/src/transport/net.cc#L945) | `src/transport/net.cc` | [945](amd-dev/amd/rccl/src/transport/net.cc#L945), [1138](amd-dev/amd/rccl/src/transport/net.cc#L1138), [1983](amd-dev/amd/rccl/src/transport/net.cc#L1983), [2017](amd-dev/amd/rccl/src/transport/net.cc#L2017) | Register memory region (host or device) |
| [`regMr()`](amd-dev/amd/rccl/src/plugin/net.cc#L352) | `src/plugin/net.cc` | [352](amd-dev/amd/rccl/src/plugin/net.cc#L352), [354](amd-dev/amd/rccl/src/plugin/net.cc#L354) | Register GPU memory for GDR test |
| [`regMr()`](amd-dev/amd/rccl/src/bootstrap.cc#L148) | `src/bootstrap.cc` | [148](amd-dev/amd/rccl/src/bootstrap.cc#L148) | Register memory for bootstrap communication |
| [`regMrDmaBuf()`](amd-dev/amd/rccl/src/transport/net.cc#L928) | `src/transport/net.cc` | [928](amd-dev/amd/rccl/src/transport/net.cc#L928), [938](amd-dev/amd/rccl/src/transport/net.cc#L938), [1121](amd-dev/amd/rccl/src/transport/net.cc#L1121), [1131](amd-dev/amd/rccl/src/transport/net.cc#L1131), [1976](amd-dev/amd/rccl/src/transport/net.cc#L1976), [2010](amd-dev/amd/rccl/src/transport/net.cc#L2010) | Register DMA-BUF memory (for GPU Direct) |
| [`deregMr()`](amd-dev/amd/rccl/src/transport/net.cc#L1163) | `src/transport/net.cc` | [1163](amd-dev/amd/rccl/src/transport/net.cc#L1163), [1208](amd-dev/amd/rccl/src/transport/net.cc#L1208), [2035](amd-dev/amd/rccl/src/transport/net.cc#L2035), [2046](amd-dev/amd/rccl/src/transport/net.cc#L2046) | Deregister memory region |
| [`deregMr()`](amd-dev/amd/rccl/src/plugin/net.cc#L353) | `src/plugin/net.cc` | [353](amd-dev/amd/rccl/src/plugin/net.cc#L353), [355](amd-dev/amd/rccl/src/plugin/net.cc#L355) | Deregister memory after GDR test |
| [`deregMr()`](amd-dev/amd/rccl/src/bootstrap.cc#L152) | `src/bootstrap.cc` | [152](amd-dev/amd/rccl/src/bootstrap.cc#L152) | Deregister memory during bootstrap cleanup |
| [`getDeviceMr()`](amd-dev/amd/rccl/src/transport/net.cc#L950) | `src/transport/net.cc` | [950](amd-dev/amd/rccl/src/transport/net.cc#L950), [1143](amd-dev/amd/rccl/src/transport/net.cc#L1143) | Get device-side memory handle for plugin |

---

### 5. **Data Transfer (Send/Recv)**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`isend()`](amd-dev/amd/rccl/src/transport/net.cc#L1381) | `src/transport/net.cc` | [1381](amd-dev/amd/rccl/src/transport/net.cc#L1381) | Asynchronous send operation |
| [`isend()`](amd-dev/amd/rccl/src/bootstrap.cc#L160) | `src/bootstrap.cc` | [160](amd-dev/amd/rccl/src/bootstrap.cc#L160) | Send data during bootstrap |
| [`irecv()`](amd-dev/amd/rccl/src/transport/net.cc#L1607) | `src/transport/net.cc` | [1607](amd-dev/amd/rccl/src/transport/net.cc#L1607) | Asynchronous receive operation (multi-buffer) |
| [`irecv()`](amd-dev/amd/rccl/src/bootstrap.cc#L175) | `src/bootstrap.cc` | [175](amd-dev/amd/rccl/src/bootstrap.cc#L175) | Receive data during bootstrap |
| [`irecvConsumed()`](amd-dev/amd/rccl/src/transport/net.cc#L1805) | `src/transport/net.cc` | [1805](amd-dev/amd/rccl/src/transport/net.cc#L1805) | Notify plugin that recv has been consumed by device |
| [`test()`](amd-dev/amd/rccl/src/transport/net.cc#L1419) | `src/transport/net.cc` | [1419](amd-dev/amd/rccl/src/transport/net.cc#L1419), [1651](amd-dev/amd/rccl/src/transport/net.cc#L1651), [1764](amd-dev/amd/rccl/src/transport/net.cc#L1764) | Test completion of send/recv request |
| [`test()`](amd-dev/amd/rccl/src/bootstrap.cc#L163) | `src/bootstrap.cc` | [163](amd-dev/amd/rccl/src/bootstrap.cc#L163), [178](amd-dev/amd/rccl/src/bootstrap.cc#L178) | Test completion during bootstrap |
| [`iflush()`](amd-dev/amd/rccl/src/transport/net.cc#L1738) | `src/transport/net.cc` | [1738](amd-dev/amd/rccl/src/transport/net.cc#L1738) | Flush GPU memory to ensure visibility |

---

### 6. **CollNet (Collective Network) Operations**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`listen()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L361) | `src/transport/coll_net.cc` | [361](amd-dev/amd/rccl/src/transport/coll_net.cc#L361) | Listen for collective network connections |
| [`connect()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L375) | `src/transport/coll_net.cc` | [375](amd-dev/amd/rccl/src/transport/coll_net.cc#L375) | Connect collective network group |
| [`closeColl()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L395) | `src/transport/coll_net.cc` | [395](amd-dev/amd/rccl/src/transport/coll_net.cc#L395) | Close collective communicator |
| [`closeListen()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L381) | `src/transport/coll_net.cc` | [381](amd-dev/amd/rccl/src/transport/coll_net.cc#L381) | Close collective listen comm |
| [`reduceSupport()`](amd-dev/amd/rccl/src/include/coll_net.h#L21) | `src/include/coll_net.h` | [21](amd-dev/amd/rccl/src/include/coll_net.h#L21) | Check if datatype/redop is supported |
| [`regMr()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L541) | `src/transport/coll_net.cc` | [541](amd-dev/amd/rccl/src/transport/coll_net.cc#L541), [620](amd-dev/amd/rccl/src/transport/coll_net.cc#L620) | Register memory for collective ops |
| [`regMrDmaBuf()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L533) | `src/transport/coll_net.cc` | [533](amd-dev/amd/rccl/src/transport/coll_net.cc#L533), [612](amd-dev/amd/rccl/src/transport/coll_net.cc#L612), [1317](amd-dev/amd/rccl/src/transport/coll_net.cc#L1317), [1353](amd-dev/amd/rccl/src/transport/coll_net.cc#L1353) | Register DMA-BUF for collective ops |
| [`deregMr()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L648) | `src/transport/coll_net.cc` | [648](amd-dev/amd/rccl/src/transport/coll_net.cc#L648), [669](amd-dev/amd/rccl/src/transport/coll_net.cc#L669), [1381](amd-dev/amd/rccl/src/transport/coll_net.cc#L1381), [1392](amd-dev/amd/rccl/src/transport/coll_net.cc#L1392) | Deregister collective memory |
| [`iallreduce()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L749) | `src/transport/coll_net.cc` | [749](amd-dev/amd/rccl/src/transport/coll_net.cc#L749), [769](amd-dev/amd/rccl/src/transport/coll_net.cc#L769) | Asynchronous all-reduce operation |
| [`iallgather()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L805) | `src/transport/coll_net.cc` | [805](amd-dev/amd/rccl/src/transport/coll_net.cc#L805), [828](amd-dev/amd/rccl/src/transport/coll_net.cc#L828) | Asynchronous all-gather operation |
| [`ireducescatter()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L861) | `src/transport/coll_net.cc` | [861](amd-dev/amd/rccl/src/transport/coll_net.cc#L861), [881](amd-dev/amd/rccl/src/transport/coll_net.cc#L881) | Asynchronous reduce-scatter operation |
| [`iflush()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L1059) | `src/transport/coll_net.cc` | [1059](amd-dev/amd/rccl/src/transport/coll_net.cc#L1059), [1066](amd-dev/amd/rccl/src/transport/coll_net.cc#L1066) | Flush collective network buffers |
| [`test()`](amd-dev/amd/rccl/src/transport/coll_net.cc#L1003) | `src/transport/coll_net.cc` | [1003](amd-dev/amd/rccl/src/transport/coll_net.cc#L1003), [1135](amd-dev/amd/rccl/src/transport/coll_net.cc#L1135) | Test collective operation completion |

---

### 7. **Virtual Device Management**

| Function | File | Line(s) | Purpose |
|----------|------|---------|---------|
| [`makeVDevice()`](amd-dev/amd/rccl/src/plugin/net/net_v9.cc#L34) | `src/plugin/net/net_v9.cc` | [34](amd-dev/amd/rccl/src/plugin/net/net_v9.cc#L34) | Create virtual NIC device |

---

## Plugin Loading Flow

```
1. ncclNetInit() [src/plugin/net.cc:254]
   └─> initPluginLibsOnceFunc() [src/plugin/net.cc:206]
       └─> Reads NCCL_NET_PLUGIN env var (default: "libnccl-net.so")
       └─> ncclNetPluginLoad() [src/plugin/net.cc:76]
           └─> ncclOpenNetPluginLib() - dlopen the plugin
           └─> getNcclNet_v10() - Get v10 interface (or fallback to v9-v6)
       └─> ncclNetPluginInit() [src/plugin/net.cc:140]
           └─> plugin->init(ncclDebugLog, ncclProfilerCallback)
           └─> plugin->devices(&ndev)
       └─> ncclNetPluginAssignToComm() [src/plugin/net.cc:164]
           └─> Assign plugin to communicator
```

---

## Key Files Using Plugin Interface

| File | Primary Purpose | Plugin Functions Used |
|------|----------------|----------------------|
| [src/plugin/net.cc](amd-dev/amd/rccl/src/plugin/net.cc) | Plugin loading & initialization | init, devices, getProperties, listen, connect, accept, regMr, deregMr, closeSend, closeRecv, closeListen |
| [src/transport/net.cc](amd-dev/amd/rccl/src/transport/net.cc) | P2P network transport layer | All P2P functions (2057 lines) |
| [src/transport/coll_net.cc](amd-dev/amd/rccl/src/transport/coll_net.cc) | Collective network transport | All CollNet functions |
| [src/bootstrap.cc](amd-dev/amd/rccl/src/bootstrap.cc) | Bootstrap communication | listen, connect, accept, regMr, deregMr, isend, irecv, test, close* |
| [src/graph/paths.cc](amd-dev/amd/rccl/src/graph/paths.cc) | Topology path calculation | getProperties |
| [src/proxy.cc](amd-dev/amd/rccl/src/proxy.cc) | Proxy service for offloading | Various through transport layers |

---

## Special AMD-ANP Features

The RCCL code specifically checks for AMD-ANP plugin:

```c
#define RCCL_ANP_PLUGIN_STR  "RCCL-ANP"  // src/transport/net.cc:32
```

### AMD-Specific Code Paths

1. **GDR Support**: AMD GPUs (gfx90a, gfx942, gfx950) have special handling for GPU Direct RDMA
2. **HDP Register Flushing**: AMD-specific HDP (Host Data Path) memory flush control
3. **Telemetry**: AMD-ANP supports optional telemetry when built with `ANP_TELEMETRY_ENABLED=1`

---

## Plugin Configuration

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `NCCL_NET_PLUGIN` | Plugin library path | `libnccl-net.so` |
| `RCCL_ANP_CONFIG_FILE` | ANP telemetry config | Not set |
| `NCCL_NET_SHARED_BUFFERS` | Share buffers across connections | -2 (auto) |
| `NCCL_NET_SHARED_COMMS` | Share communicators | 1 |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | Optional recv completion | 1 |
| `NCCL_GDRCOPY_SYNC_ENABLE` | GDR copy sync | 1 |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | GDR copy flush | 0 |

---

## Summary Statistics

| Category | Function Calls | Unique Functions |
|----------|----------------|------------------|
| Initialization | 5 | 2 (init, devices) |
| Properties | 8 | 1 (getProperties) |
| Connection Management | 18 | 6 (listen, connect, accept, closeSend, closeRecv, closeListen) |
| Memory Management | 24 | 4 (regMr, regMrDmaBuf, deregMr, getDeviceMr) |
| Data Transfer | 12 | 6 (isend, irecv, irecvConsumed, test, iflush) |
| CollNet Operations | 22 | 10 (init, devices, getProperties, listen, connect, closeColl, closeListen, iallreduce, iallgather, ireducescatter, iflush, test, regMr, deregMr, reduceSupport) |
| **TOTAL** | **~89 call sites** | **~23 unique API functions** |

---

## Version Compatibility

RCCL supports multiple plugin API versions through wrapper layers:
- **v10** (current): [src/plugin/net/net_v10.cc](amd-dev/amd/rccl/src/plugin/net/net_v10.cc) (full feature set)
- **v9**: [src/plugin/net/net_v9.cc](amd-dev/amd/rccl/src/plugin/net/net_v9.cc) (makeVDevice, iallgather, ireducescatter)
- **v8**: [src/plugin/net/net_v8.cc](amd-dev/amd/rccl/src/plugin/net/net_v8.cc) (32-bit size limitations)
- **v7**: [src/plugin/net/net_v7.cc](amd-dev/amd/rccl/src/plugin/net/net_v7.cc) (basic support)
- **v6**: [src/plugin/net/net_v6.cc](amd-dev/amd/rccl/src/plugin/net/net_v6.cc) (legacy)

AMD-ANP plugin uses **v10** interface for full functionality.

---

## Plugin Interface Definition

The complete plugin interface is defined in:
- [src/include/plugin/net/net_v10.h](amd-dev/amd/rccl/src/include/plugin/net/net_v10.h) - ncclNet_v10_t structure (100 functions)
- [src/include/plugin/nccl_net.h](amd-dev/amd/rccl/src/include/plugin/nccl_net.h) - Main plugin header

---

Generated: 2025-10-29
