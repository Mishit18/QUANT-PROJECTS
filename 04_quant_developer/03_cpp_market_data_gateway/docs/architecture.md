# Architecture

## Design Philosophy

This feed handler prioritizes deterministic latency through careful memory management, lock-free algorithms, and thread isolation.

## Core Principles

1. **Pre-allocated Memory**: All buffers allocated at startup
2. **Lock-Free Communication**: SPSC queues between pipeline stages
3. **Cache-Line Awareness**: Hot structures are 64-byte aligned
4. **Minimal Copies**: Pointer-based parsing where feasible
5. **Non-Blocking IO**: epoll-based event loop
6. **Thread Isolation**: Dedicated cores per thread

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Network Layer                            │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ UDP Listener │              │ TCP Listener │            │
│  │ (Mkt Data)   │              │ (Exec Rpts)  │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼──────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │        IO Thread              │
          │  - epoll event loop           │
          │  - Non-blocking recv          │
          │  - Write to ring buffer       │
          └──────────────┬───────────────┘
                         │ Ring Buffer (16MB)
                         │ SPSC, lock-free
          ┌──────────────▼───────────────┐
          │      Decoder Thread           │
          │  - Protocol detection         │
          │  - FIX/Binary parsing         │
          │  - Zero-copy extraction       │
          └──────────────┬───────────────┘
                         │ Event Queue (64K)
                         │ SPSC, lock-free
          ┌──────────────▼───────────────┐
          │    Dispatcher Thread          │
          │  - Normalize events           │
          │  - Publish to consumers       │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │        Consumers              │
          │  - Strategy engines           │
          │  - Risk systems               │
          │  - Logging                    │
          └───────────────────────────────┘
```

## Data Flow

### Inbound Path

1. **Network → IO Thread**
   - UDP datagrams arrive on multicast group
   - TCP connections from exchange gateways
   - epoll notifies IO thread of readable sockets
   - Non-blocking recv() into pre-allocated buffer

2. **IO Thread → Ring Buffer**
   - Reserve space in ring buffer (lock-free)
   - memcpy from recv buffer to ring buffer
   - Commit write (atomic pointer update)
   - No allocation, no locks

3. **Ring Buffer → Decoder Thread**
   - Peek at available data (lock-free)
   - Detect protocol (FIX vs binary)
   - Parse message using pointer traversal
   - Commit read (atomic pointer update)

4. **Decoder Thread → Event Queue**
   - Populate normalized event struct
   - Push to event queue (lock-free)
   - No allocation, fixed-size events

5. **Event Queue → Dispatcher**
   - Pop event from queue (lock-free)
   - Invoke consumer callbacks
   - Callbacks execute on dispatcher thread

### Outbound Path (Not Implemented)

For order submission, a similar pipeline would be used in reverse:
- Strategy → Order Queue → Encoder Thread → Network

## Why This Architecture?

### Thread Separation

**Rationale**: Separate IO, decoder, and dispatcher threads
- CPU isolation: Each thread has dedicated core
- Pipelining: Parallel processing of different stages
- Determinism: No contention for CPU resources
- Profiling: Easy to identify bottlenecks per stage

**Tradeoff**: More threads = more context switches
**Mitigation**: Pin threads to cores, use SCHED_FIFO

### Ring Buffer vs Queue

**Rationale for ring buffer between IO and decoder**:
- Variable-length messages: FIX messages vary in size
- Partial reads: TCP may deliver partial messages
- Minimal copies: Decoder reads directly from buffer
- Backpressure: Buffer full = slow down IO

**Rationale for event queue between decoder and dispatcher**:
- Fixed-size events: All events are 64 bytes
- Type safety: Strongly typed event structs
- Batching: Dispatcher can batch events

### Lock-Free Queues

**Rationale for SPSC (single-producer-single-consumer)**:
- No locks: Atomic operations only
- Cache-friendly: Producer and consumer on different cache lines
- Predictable: No lock contention, no priority inversion

**Limitation**: Only works for 1:1 communication
**Alternative**: For MPSC, use per-producer queues or lock-free MPSC queue

### Parsing Strategy

**Where minimal copies are possible**:
- Binary protocol: Struct overlay on buffer (with alignment checks)
- FIX protocol: StringView points into buffer
- No std::string: Avoid heap allocation

**Where copies are necessary**:
- Partial messages: Must copy to reassembly buffer
- Endianness conversion: Must swap bytes
- Normalization: Must copy to event struct

## Scalability

### Vertical Scaling (Single Machine)

- **More cores**: Add more decoder threads
- **Faster CPU**: Higher clock speed = lower latency
- **Faster NIC**: 10GbE → 25GbE → 100GbE
- **Kernel bypass**: DPDK, Solarflare OpenOnload

### Horizontal Scaling (Multiple Machines)

- **Symbol sharding**: Different machines handle different symbols
- **Feed redundancy**: Primary/backup feed handlers
- **Geographic distribution**: Co-located at multiple exchanges

## Performance Characteristics

### Latency Breakdown (Typical)

- Network (NIC → kernel): 5-10μs
- Kernel → userspace: 2-5μs
- IO thread (recv): 0.5-1μs
- Ring buffer write: 50-100ns
- Decoder (parse): 200-500ns
- Event queue write: 50-100ns
- Dispatcher (callback): 100-200ns

**Note**: These are estimates. Actual latency depends on hardware, kernel configuration, and load.

### Throughput

- **Single core**: 5M+ messages/sec
- **Bottleneck**: Usually parsing, not IO
- **Optimization**: SIMD parsing, lookup tables

## System Configuration

### CPU Isolation

```bash
# Isolate cores 0-3 for feed handler
isolcpus=0-3 nohz_full=0-3 rcu_nocbs=0-3
```

### IRQ Affinity

```bash
# Pin NIC interrupts to core 4
echo 10 > /proc/irq/<IRQ>/smp_affinity
```

### Huge Pages

```bash
# Enable 2MB huge pages
echo 1024 > /proc/sys/vm/nr_hugepages
```

### NUMA

- Pin threads and memory to same NUMA node
- Use `numactl` to control placement

### Monitoring

- Latency histograms: p50, p99, p99.9, p99.99
- Queue depths: Detect backpressure
- Drop rates: UDP packet loss
- CPU utilization: Per-core usage
