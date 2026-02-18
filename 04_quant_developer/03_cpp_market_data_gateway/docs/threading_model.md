# Threading Model

## Overview

The feed handler uses a pipeline architecture with dedicated threads for each stage, prioritizing deterministic latency and CPU isolation.

## Thread Roles

### IO Thread (Core 0)

**Responsibility**: Read from network sockets

**Operations**:
- epoll_wait() on TCP and UDP sockets
- Non-blocking recv() into pre-allocated buffer
- Write raw bytes to ring buffer
- No parsing, no business logic

**Why dedicated thread?**
- Network IO is latency-critical
- Isolated from parsing overhead
- Can use SCHED_FIFO for real-time priority
- Easy to profile network performance

**CPU Affinity**: Core 0
**Priority**: SCHED_FIFO, max priority

### Decoder Thread (Core 1)

**Responsibility**: Parse messages

**Operations**:
- Read from ring buffer
- Detect protocol (FIX vs binary)
- Parse message (zero-copy)
- Write normalized event to event queue

**Why dedicated thread?**
- Parsing is CPU-intensive
- Isolated from IO and dispatch
- Can optimize for cache locality
- Easy to profile parsing performance

**CPU Affinity**: Core 1
**Priority**: SCHED_FIFO, max priority

### Dispatcher Thread (Core 2)

**Responsibility**: Publish events to consumers

**Operations**:
- Read from event queue
- Invoke consumer callbacks
- Handle backpressure

**Why dedicated thread?**
- Consumer callbacks may be slow
- Isolated from IO and parsing
- Can implement batching
- Easy to profile consumer performance

**CPU Affinity**: Core 2
**Priority**: SCHED_FIFO, max priority

## Inter-Thread Communication

### Ring Buffer (IO → Decoder)

**Type**: SPSC (single-producer-single-consumer)
**Size**: 16MB
**Element**: Variable-length byte stream

**Producer**: IO thread
**Consumer**: Decoder thread

**Synchronization**:
- Atomic write_pos and read_pos
- Memory barriers (acquire/release)
- No locks, no mutexes

**Backpressure**:
- If buffer full, IO thread drops packets (UDP) or blocks (TCP)
- In production, implement flow control

### Event Queue (Decoder → Dispatcher)

**Type**: SPSC (single-producer-single-consumer)
**Size**: 64K events
**Element**: Fixed-size event struct (64 bytes)

**Producer**: Decoder thread
**Consumer**: Dispatcher thread

**Synchronization**:
- Atomic write_idx and read_idx
- Memory barriers (acquire/release)
- No locks, no mutexes

**Backpressure**:
- If queue full, decoder thread drops events
- In production, implement flow control or multiple queues

## Thread Affinity

### Why Pin Threads to Cores?

1. Cache locality: Thread stays on same core, hot data stays in L1/L2
2. No migration: Avoid context switch overhead
3. Predictable latency: No scheduler interference
4. NUMA awareness: Pin to cores on same NUMA node as NIC

### How to Set Affinity

```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(core_id, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
```

### Kernel Configuration

```bash
# Isolate cores 0-2 for feed handler
isolcpus=0-2 nohz_full=0-2 rcu_nocbs=0-2
```

## Thread Priority

### Why SCHED_FIFO?

- Real-time scheduling: Preempts normal processes
- Deterministic: No time-slicing, runs until blocked
- Low latency: Minimal scheduler overhead

### How to Set Priority

```cpp
sched_param param;
param.sched_priority = sched_get_priority_max(SCHED_FIFO);
pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
```

**Note**: Requires root or CAP_SYS_NICE capability

## Synchronization Primitives

### Atomic Operations

Used for:
- Ring buffer pointers
- Event queue indices
- Statistics counters

**Memory ordering**:
- `memory_order_relaxed`: For statistics (not critical)
- `memory_order_acquire`: For reading shared state
- `memory_order_release`: For publishing shared state
- `memory_order_seq_cst`: Not used (too expensive)

### No Locks

**Rationale for avoiding mutexes**:
- Unpredictable latency: Lock contention, priority inversion
- Cache coherency: Lock ping-pong between cores
- Complexity: Deadlock, livelock

**Alternative**: Lock-free data structures

## Scaling Considerations

### Multiple Decoder Threads

**Problem**: Single decoder thread may bottleneck

**Solution**: Shard by symbol or message type
- Decoder 1: Symbols A-M
- Decoder 2: Symbols N-Z

**Tradeoff**: More complex, requires MPSC queue or per-decoder queues

### Multiple Dispatcher Threads

**Problem**: Single dispatcher thread may bottleneck

**Solution**: Shard by consumer type
- Dispatcher 1: Strategy engines
- Dispatcher 2: Risk systems
- Dispatcher 3: Logging

**Tradeoff**: More complex, requires MPMC queue or per-dispatcher queues

## Debugging and Profiling

### Per-Thread Statistics

Each thread maintains:
- Messages processed
- Errors encountered
- Queue depths
- Latency histograms

### Profiling Tools

- perf: CPU profiling per thread
- flamegraph: Visualize hot paths
- bpftrace: Trace system calls
- Intel VTune: Advanced profiling

### Common Issues

1. Thread starvation: Check CPU affinity and priority
2. Cache misses: Check data alignment and access patterns
3. False sharing: Ensure cache-line separation
4. Queue overflow: Check backpressure handling
