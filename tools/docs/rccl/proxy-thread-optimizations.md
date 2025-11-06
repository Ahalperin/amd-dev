# RCCL Proxy Thread Performance Optimization Proposals

**Date:** November 4, 2025  
**Purpose:** Detailed analysis and code modifications to improve proxy thread performance  
**Related:** [RCCL Technical Internals](rccl-technical-internals.md)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Bottlenecks](#current-bottlenecks)
3. [Optimization Proposals](#optimization-proposals)
4. [Implementation Details](#implementation-details)
5. [Expected Performance Gains](#expected-performance-gains)

---

## Executive Summary

The RCCL proxy thread is responsible for handling network operations asynchronously to avoid blocking GPU kernels. Current analysis reveals several performance bottlenecks:

1. **Mutex contention** between main thread and proxy progress thread
2. **Polling overhead** with individual network completion tests
3. **Memory allocation** inefficiencies
4. **Linked list traversal** without cache optimization
5. **Suboptimal batching** of operations

**Expected Improvements:** 10-30% reduction in small message latency, 15-40% improvement in multi-operation throughput.

---

## Current Bottlenecks

### 1. Mutex Lock Contention (Critical Path)

**Location:** `src/proxy.cc:812-826`

```c
// Current implementation - blocks on mutex every iteration
if (state->active == NULL && (pool->nextOps == -1 || pthread_mutex_trylock(&pool->mutex) != 0))
    return ncclSuccess;

if (state->active == NULL) {
    pthread_mutex_lock(&pool->mutex);  // BLOCKING - can stall for microseconds
    if (pool->nextOps == -1 && !state->stop) {
        pthread_cond_wait(&pool->cond, &pool->mutex);  // BLOCKING - adds latency
    }
}
```

**Problem:** Mutex operations add 100-500ns latency on each call. For small messages, this is significant overhead.

### 2. Individual Network Polling

**Location:** `src/transport/net.cc:1411, 1648`

```c
// Send progress - polls each request individually
NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));

// Receive progress - same issue
NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
```

**Problem:** Each `test()` call may involve system calls or device memory access. Batching multiple tests would amortize overhead.

### 3. Linked List Traversal Without Prefetching

**Location:** `src/proxy.cc:778-796`

```c
static ncclResult_t progressOps(struct ncclProxyState* proxyState, 
                                struct ncclProxyProgressState* state, 
                                struct ncclProxyArgs* opStart, int* idle) {
    struct ncclProxyArgs* prevOp = NULL;
    struct ncclProxyArgs* op = opStart;
    while (op) {
        // No prefetch here!
        ncclResult_t ret = op->progress(proxyState, op);  // Cache miss likely
        // ...
        op = op->next;  // Another potential cache miss
    }
}
```

**Problem:** Cold cache misses on each operation node (40-100 cycles per miss).

### 4. Frequent sched_yield() Calls

**Location:** `src/proxy.cc:989`

```c
if (added == 0) {
    sched_yield(); // Expensive - yields CPU for 1-10 microseconds
}
```

**Problem:** `sched_yield()` is expensive when there are actually operations to process shortly.

### 5. Inefficient Operation Batching

**Location:** `src/proxy.cc:978`

```c
if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
    // Default frequency is 8 - may not be optimal
```

**Problem:** Fixed batching parameter doesn't adapt to workload characteristics.

---

## Optimization Proposals

### Optimization 1: Lock-Free Queue for Proxy Operations (High Priority)

Replace mutex-based queue with lock-free SPSC (Single Producer Single Consumer) ring buffer.

**Benefits:**
- Eliminate mutex overhead (~100-500ns per operation)
- Reduce contention between main and proxy threads
- Better cache behavior with ring buffer

**Implementation:**

```c
// Add to src/include/proxy.h
#define PROXY_RING_SIZE 1024  // Power of 2 for fast modulo

struct ncclProxyOpsRingBuffer {
    struct ncclProxyOp ops[PROXY_RING_SIZE];
    volatile uint64_t head __attribute__((aligned(64)));  // Producer writes
    volatile uint64_t tail __attribute__((aligned(64)));  // Consumer writes
    char padding[64 - sizeof(uint64_t)*2];  // Prevent false sharing
};

// Lock-free enqueue (called by main thread)
static inline int proxyRingPush(struct ncclProxyOpsRingBuffer* ring, 
                                struct ncclProxyOp* op) {
    uint64_t head = __atomic_load_n(&ring->head, __ATOMIC_RELAXED);
    uint64_t tail = __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE);
    
    // Check if full
    if (head - tail >= PROXY_RING_SIZE) {
        return 0;  // Full
    }
    
    // Copy operation
    memcpy(&ring->ops[head % PROXY_RING_SIZE], op, sizeof(struct ncclProxyOp));
    
    // Publish
    __atomic_store_n(&ring->head, head + 1, __ATOMIC_RELEASE);
    return 1;  // Success
}

// Lock-free dequeue (called by proxy thread)
static inline int proxyRingPop(struct ncclProxyOpsRingBuffer* ring, 
                               struct ncclProxyOp* op) {
    uint64_t tail = __atomic_load_n(&ring->tail, __ATOMIC_RELAXED);
    uint64_t head = __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE);
    
    // Check if empty
    if (tail >= head) {
        return 0;  // Empty
    }
    
    // Copy operation
    memcpy(op, &ring->ops[tail % PROXY_RING_SIZE], sizeof(struct ncclProxyOp));
    
    // Consume
    __atomic_store_n(&ring->tail, tail + 1, __ATOMIC_RELEASE);
    return 1;  // Success
}
```

**Modify `src/proxy.cc:ncclProxyGetPostedOps()`:**

```c
static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyState* proxyState, int* added) {
    struct ncclProxyProgressState* state = &proxyState->progressState;
    struct ncclProxyOpsRingBuffer* ring = state->opsRing;  // Use lock-free ring
    
    struct ncclProxyOp op;
    int count = 0;
    
    // Batch dequeue operations (no mutex needed!)
    while (count < ncclParamProxyAppendBatchSize() && proxyRingPop(ring, &op)) {
        NCCLCHECK(ProxyAppend(state, &op));
        (*added)++;
        count++;
    }
    
    return ncclSuccess;
}
```

### Optimization 2: Batched Network Polling (High Priority)

Batch multiple network test operations together to amortize syscall/device access overhead.

**Implementation:**

Modify `src/transport/net.cc:sendProxyProgress()`:

```c
// Add at file scope
#define MAX_BATCH_TEST 16

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, 
                                      struct ncclProxyArgs* args) {
    // ... existing code ...
    
    if (args->state == ncclProxyOpProgress) {
        // NEW: Batch collect requests to test
        void* batchRequests[MAX_BATCH_TEST];
        int* batchDone[MAX_BATCH_TEST];
        int* batchSizes[MAX_BATCH_TEST];
        int batchCount = 0;
        int doneFlags[MAX_BATCH_TEST];
        int doneSizes[MAX_BATCH_TEST];
        
        // Collect up to MAX_BATCH_TEST pending operations
        for (int s=0; s<args->nsubs && batchCount < MAX_BATCH_TEST; s++) {
            struct ncclProxySubArgs* sub = args->subs+s;
            if (sub->done < sub->transmitted) {
                int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
                batchRequests[batchCount] = sub->requests[buffSlot];
                batchDone[batchCount] = &doneFlags[batchCount];
                batchSizes[batchCount] = &doneSizes[batchCount];
                batchCount++;
            }
        }
        
        // Batch test all at once if supported
        if (batchCount > 1 && proxyState->ncclNet->testBatch) {
            // NEW: Batched test operation
            NCCLCHECK(proxyState->ncclNet->testBatch(batchCount, batchRequests, 
                                                     batchDone, batchSizes));
        } else {
            // Fallback to individual tests
            for (int i=0; i<batchCount; i++) {
                NCCLCHECK(proxyState->ncclNet->test(batchRequests[i], 
                                                    batchDone[i], batchSizes[i]));
            }
        }
        
        // Process completed operations
        int batchIdx = 0;
        for (int s=0; s<args->nsubs; s++) {
            struct ncclProxySubArgs* sub = args->subs+s;
            if (sub->done < sub->transmitted) {
                if (batchDone[batchIdx] && *batchDone[batchIdx]) {
                    // Process completion
                    // ... rest of existing completion code ...
                    sub->done += args->sliceSteps;
                    args->idle = 0;
                }
                batchIdx++;
            }
        }
    }
    return ncclSuccess;
}
```

### Optimization 3: Software Prefetching for Linked Lists (Medium Priority)

Add prefetch hints for next operations in the linked list.

**Modify `src/proxy.cc:progressOps()`:**

```c
static ncclResult_t progressOps(struct ncclProxyState* proxyState, 
                                struct ncclProxyProgressState* state, 
                                struct ncclProxyArgs* opStart, int* idle) {
    struct ncclProxyArgs* prevOp = NULL;
    struct ncclProxyArgs* op = opStart;
    
    while (op) {
        // Prefetch next operation to reduce cache misses
        if (op->next) {
            __builtin_prefetch(op->next, 0, 3);  // Read, high temporal locality
            __builtin_prefetch(&op->next->progress, 0, 3);  // Prefetch function pointer
        }
        
        op->retry_total++;
        if (op->state == ncclProxyOpNone) return ncclInternalError;
        
        TIME_START(0); TIME_START(1);
        ncclResult_t ret = op->progress(proxyState, op);
        if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } 
        else { TIME_CANCEL(1); TIME_STOP(0); }
        
        *idle &= op->idle;
        
        if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
            TIME_START(2);
            NCCLCHECK(removeOp(state, &op, &prevOp));
            TIME_STOP(2);
        } else {
            prevOp = op;
            op = op->next;
        }
    }
    return ncclSuccess;
}
```

### Optimization 4: Adaptive Spin-Wait Instead of sched_yield() (Medium Priority)

Replace `sched_yield()` with adaptive spinning that only yields after extended idle period.

**Modify `src/proxy.cc:ncclProxyProgress()`:**

```c
#define SPIN_COUNT_MAX 100
#define YIELD_THRESHOLD 1000  // Yield after 1000 consecutive idle cycles

void* ncclProxyProgress(void *proxyState_) {
    // ... existing setup code ...
    
    int lastIdle = 0;
    int proxyOpAppendCounter = 0;
    int consecutiveIdleCount = 0;  // NEW: Track consecutive idle loops
    
    do {
        int idle = 1;
        ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
        if (ret != ncclSuccess) {
            __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
            break;
        }
        
        if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
            int added = 0;
            proxyOpAppendCounter = 0;
            ret = ncclProxyGetPostedOps(proxyState, &added);
            if (ret != ncclSuccess) {
                __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
            }
            
            if (added == 0) {
                consecutiveIdleCount++;
                
                // Adaptive backoff strategy
                if (consecutiveIdleCount < SPIN_COUNT_MAX) {
                    // Tight spin with pause for a short time
                    for (int i = 0; i < 10; i++) {
                        __asm__ __volatile__("pause" ::: "memory");  // x86/x64
                        // For ARM: __asm__ __volatile__("yield" ::: "memory");
                    }
                } else if (consecutiveIdleCount < YIELD_THRESHOLD) {
                    // Medium wait - use nanosleep for short duration
                    struct timespec ts = {0, 100};  // 100 ns
                    nanosleep(&ts, NULL);
                } else {
                    // Long idle - yield CPU
                    sched_yield();
                }
            } else {
                consecutiveIdleCount = 0;  // Reset on activity
            }
        }
        lastIdle = idle;
    } while ((state->stop == 0 || (state->stop == 1 && state->active)) && 
             __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) == 0);
    
    return NULL;
}
```

### Optimization 5: Operation Pool with Better Cache Locality (Medium Priority)

Redesign operation pool to improve cache locality by allocating operations in contiguous chunks.

**Modify `src/proxy.cc`:**

```c
#define OPS_PER_CACHE_LINE (64 / sizeof(struct ncclProxyArgs*))
#define POOL_CHUNK_SIZE 64  // 64 operations per chunk

struct ncclProxyOpsChunk {
    struct ncclProxyArgs ops[POOL_CHUNK_SIZE] __attribute__((aligned(64)));
    struct ncclProxyOpsChunk* next;
};

struct ncclProxyArgsPoolV2 {
    struct ncclProxyOpsChunk* chunks;
    int chunkCount;
    struct ncclProxyArgs* freeList;
    
    // Pre-allocated hot path operations
    struct ncclProxyArgs hotOps[16] __attribute__((aligned(64)));
    int hotOpsUsed;
};

static ncclResult_t allocateArgsV2(struct ncclProxyProgressState* state, 
                                   struct ncclProxyArgs** argsptr) {
    struct ncclProxyArgsPoolV2* pool = state->poolV2;
    
    // Try hot path first (no allocation needed)
    if (pool->hotOpsUsed < 16) {
        *argsptr = &pool->hotOps[pool->hotOpsUsed++];
        memset(*argsptr, 0, sizeof(struct ncclProxyArgs));
        return ncclSuccess;
    }
    
    // Use free list
    if (pool->freeList) {
        *argsptr = pool->freeList;
        pool->freeList = pool->freeList->next;
        memset(*argsptr, 0, sizeof(struct ncclProxyArgs));
        return ncclSuccess;
    }
    
    // Allocate new chunk
    struct ncclProxyOpsChunk* chunk;
    NCCLCHECK(ncclCalloc(&chunk, 1));
    chunk->next = pool->chunks;
    pool->chunks = chunk;
    pool->chunkCount++;
    
    // Initialize free list from chunk
    for (int i = 0; i < POOL_CHUNK_SIZE - 1; i++) {
        chunk->ops[i].next = &chunk->ops[i + 1];
    }
    chunk->ops[POOL_CHUNK_SIZE - 1].next = NULL;
    
    *argsptr = &chunk->ops[0];
    pool->freeList = &chunk->ops[1];
    
    return ncclSuccess;
}
```

### Optimization 6: Dedicated Fast Path for Small Messages (High Priority)

Create specialized fast path for small messages (<4KB) that bypasses some bookkeeping.

**Add to `src/proxy.cc`:**

```c
#define SMALL_MSG_THRESHOLD 4096

// Fast path structure for small messages
struct ncclProxyFastOp {
    void* sendComm;
    void* recvComm;
    void* buffer;
    size_t size;
    int tag;
    void* request;
    uint8_t state;  // 0=posted, 1=testing, 2=done
} __attribute__((aligned(64)));

struct ncclProxyFastPath {
    struct ncclProxyFastOp ops[32];
    int count;
    uint64_t lastProcessedCycle;
};

static ncclResult_t processFastPath(struct ncclProxyState* proxyState, 
                                    struct ncclProxyFastPath* fastPath) {
    // Batch test all fast path operations
    int allDone = 1;
    
    for (int i = 0; i < fastPath->count; i++) {
        struct ncclProxyFastOp* op = &fastPath->ops[i];
        
        if (op->state == 0) {  // Posted
            op->state = 1;
        } else if (op->state == 1) {  // Testing
            int done = 0;
            NCCLCHECK(proxyState->ncclNet->test(op->request, &done, NULL));
            if (done) {
                op->state = 2;
            } else {
                allDone = 0;
            }
        }
    }
    
    // If all done, reset fast path
    if (allDone) {
        fastPath->count = 0;
    }
    
    return ncclSuccess;
}
```

### Optimization 7: Reduce Atomic Operations (Low Priority)

Replace some atomic operations with cheaper alternatives where sequencing is guaranteed.

**In `src/proxy.cc:ncclProxyProgress()`:**

```c
// Instead of:
__atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);

// Use regular store when only proxy thread writes:
proxyState->asyncResult = ret;
__atomic_thread_fence(__ATOMIC_RELEASE);  // Single fence cheaper than per-store atomic
```

---

## Implementation Priority

### Phase 1: High Impact, Low Risk (Week 1-2)
1. **Lock-Free Queue** (Optimization 1) - Biggest single improvement
2. **Adaptive Spin-Wait** (Optimization 4) - Easy to implement, significant gains
3. **Software Prefetching** (Optimization 3) - Low risk, measurable benefit

### Phase 2: High Impact, Medium Risk (Week 3-4)
4. **Batched Network Polling** (Optimization 2) - Requires network plugin support
5. **Fast Path for Small Messages** (Optimization 6) - Needs careful testing

### Phase 3: Medium Impact, Optimization (Week 5-6)
6. **Operation Pool Redesign** (Optimization 5) - Incremental improvement
7. **Reduce Atomic Operations** (Optimization 7) - Micro-optimization

---

## Expected Performance Gains

### Latency Improvements (Small Messages <4KB)
- **Lock-free queue:** -200 to -500 ns per operation
- **Adaptive spin-wait:** -50 to -200 ns per iteration
- **Fast path:** -100 to -300 ns for eligible operations
- **Total expected:** **15-25% reduction in small message latency**

### Throughput Improvements (Large Messages >1MB)
- **Batched polling:** +10-20% throughput (amortized syscall overhead)
- **Cache optimization:** +5-10% throughput (better memory bandwidth utilization)
- **Total expected:** **15-30% increase in sustained throughput**

### Multi-Operation Scenarios
- **Lock-free queue:** Eliminates contention, up to 50% improvement with high concurrency
- **Batching:** More effective with multiple simultaneous operations
- **Total expected:** **30-50% improvement in multi-stream scenarios**

---

## Testing and Validation

### Microbenchmarks
```bash
# Test small message latency
./build/test/single/sendrecv_test -b 1 -e 4K -f 2 -g 1 -n 1000

# Test large message bandwidth
./build/test/single/all_reduce_test -b 1M -e 128M -f 2 -g 1 -n 100

# Test multi-operation throughput
./build/test/single/sendrecv_test -b 64K -e 64K -f 1 -g 1 -n 100 -w 16
```

### Profiling Points
1. Proxy thread CPU utilization (should decrease with optimizations)
2. Mutex wait time (should be eliminated with lock-free queue)
3. Network test call frequency (should decrease with batching)
4. Cache miss rate (should improve with prefetching)

### Success Criteria
- No correctness regressions (all existing tests pass)
- ≥10% improvement in small message latency
- ≥15% improvement in multi-operation throughput
- Reduced CPU usage for proxy thread

---

## Additional Considerations

### AMD-Specific Optimizations

For AMD GPUs (MI200, MI300X series):

```c
// Use AMD-specific features
#if defined(__HIP_PLATFORM_AMD__)
    // MI300X has faster atomic operations
    #if defined(__gfx942__)
        #define USE_FINE_GRAIN_ATOMICS 1
    #endif
    
    // ROCm-specific memory fence
    #define PROXY_FENCE() __builtin_amdgcn_s_sleep(1)
#else
    #define PROXY_FENCE() __asm__ __volatile__("pause" ::: "memory")
#endif
```

### Environment Variables for Tuning

Add new tunable parameters:

```bash
export NCCL_PROXY_LOCKFREE=1          # Enable lock-free queue (default: 1)
export NCCL_PROXY_BATCH_TEST=16       # Batch size for network tests (default: 16)
export NCCL_PROXY_SPIN_COUNT=100      # Spin iterations before yield (default: 100)
export NCCL_PROXY_FAST_PATH_THRESH=4096  # Fast path threshold bytes (default: 4096)
```

---

## References

1. [Lock-Free Programming Patterns](https://www.1024cores.net/home/lock-free-algorithms/queues)
2. [Software Prefetching Techniques](https://lwn.net/Articles/255364/)
3. [Adaptive Spin-Waiting](https://www.kernel.org/doc/Documentation/locking/spinlocks.txt)
4. [AMD MI300X Architecture](https://www.amd.com/en/products/server-accelerators/instinct-mi300x)

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-04 | AI Assistant | Initial proxy optimization proposals |


