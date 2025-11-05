# RCCL Proxy Thread Documentation

## Overview

The RCCL Proxy Thread mechanism is a critical component of the RCCL (ROCm Communication Collectives Library) architecture that enables asynchronous communication operations. This documentation provides comprehensive information about the design, implementation, and operation of the proxy thread system.

## Purpose

The proxy threads serve several essential functions in RCCL:

1. **Asynchronous Communication**: Enable GPU operations to proceed without blocking on network I/O
2. **Network Progress**: Actively progress network send/receive operations independently from GPU kernels
3. **Resource Management**: Handle setup, teardown, and management of communication resources
4. **Inter-Process Communication**: Facilitate communication between ranks that may be in different processes
5. **Protocol Adaptation**: Bridge between GPU-visible memory operations and network transport protocols

## Key Benefits

- **Performance**: Overlap GPU computation with network communication
- **Scalability**: Handle multiple concurrent operations across channels
- **Flexibility**: Support multiple transport types (network, shared memory, peer-to-peer)
- **Reliability**: Centralized error handling and resource cleanup

## Architecture Components

The proxy system consists of three main thread types:

1. **Progress Thread** (`ncclProxyProgress`)
   - Continuously progresses active communication operations
   - Polls for new operations from main threads
   - Manages operation lifecycle and completion

2. **Service Thread** (`ncclProxyService`)
   - Handles connection setup and management
   - Processes control messages from local ranks
   - Manages socket-based inter-process communication

3. **UDS Service Thread** (`ncclProxyServiceUDS`)
   - Handles Unix Domain Socket communication
   - Supports CUDA unified memory (cuMem) API operations
   - Manages file descriptor passing between processes

## Documentation Structure

This directory contains the following documentation:

- **[architecture.md](architecture.md)**: Detailed proxy architecture and design patterns
- **[threading-model.md](threading-model.md)**: Threading model and lifecycle management
- **[data-structures.md](data-structures.md)**: Key data structures and their relationships
- **[communication-protocol.md](communication-protocol.md)**: Communication patterns and message types
- **[transport-integration.md](transport-integration.md)**: How proxy threads integrate with different transports
- **[performance-tuning.md](performance-tuning.md)**: Performance considerations and tuning parameters

## Quick Reference

### Key Files

- **Header**: `src/include/proxy.h`
- **Implementation**: `src/proxy.cc`
- **Transport Integrations**:
  - Network: `src/transport/net.cc`
  - Shared Memory: `src/transport/shm.cc`
  - P2P: `src/transport/p2p.cc`
  - CollNet: `src/transport/coll_net.cc`

### Key Data Structures

- `ncclProxyState`: Main proxy state per communicator
- `ncclProxyProgressState`: Progress thread state
- `ncclProxyOp`: Individual operation descriptor
- `ncclProxyArgs`: Batched operation arguments for progress
- `ncclProxyConnection`: Connection state between ranks

### Environment Variables

Key environment variables affecting proxy behavior:

- `NCCL_PROGRESS_APPENDOP_FREQ`: Controls operation batching frequency (default: 8)
- `NCCL_PROXY_APPEND_BATCH_SIZE`: Maximum operations per batch (default: 16)
- `NCCL_PROXY_DUMP_SIGNAL`: Signal number for dumping proxy state (default: -1, disabled)
- `NCCL_CREATE_THREAD_CONTEXT`: Create dedicated CUDA context for proxy threads (default: 0)

## Related Documentation

- [RCCL Design Overview](../rccl-design-overview.md)
- [RCCL Technical Internals](../rccl-technical-internals.md)
- [RCCL Pipelining Architecture](../rccl-pipelining-architecture.md)
- [RCCL Host-Device Synchronization](../rccl-host-device-synchronization.md)

## Getting Started

For a comprehensive understanding of the proxy thread mechanism, we recommend reading the documentation in this order:

1. This README (you are here)
2. [Architecture](architecture.md) - Understand the overall design
3. [Threading Model](threading-model.md) - Learn about thread lifecycle
4. [Data Structures](data-structures.md) - Understand the data model
5. [Communication Protocol](communication-protocol.md) - Learn about message passing
6. [Transport Integration](transport-integration.md) - See how it works with transports

## Version Information

This documentation reflects the RCCL codebase as of the AMD modifications to NCCL, including ROCm-specific enhancements and the integration of advanced features like proxy tracing and profiler support.

## Contributing

When modifying proxy-related code, please ensure:

1. Update relevant documentation in this directory
2. Consider thread safety implications
3. Test with multiple transports and configurations
4. Profile performance impact on both latency and bandwidth


