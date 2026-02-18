# Technical Design Notes

## Comparison to Production Systems

### CME Globex
- Multi-threaded architecture with lock-free data structures
- Separate threads for matching, market data, and risk management
- Reported median latency around 2-3μs (includes network stack)
- Uses custom memory allocators
- Core matching logic is conceptually similar to this implementation

### Nasdaq INET
- FPGA-based matching for sub-microsecond latency
- Order book state maintained in FPGA BRAM
- Software fallback for complex order types
- Hardware implementation trades flexibility for performance

### IEX
- Intentionally adds 350μs delay ("speed bump")
- Matching performance less critical than fairness mechanism
- Standard software implementation

**This Implementation**: Similar architecture to software-based exchanges, simplified to single-threaded single-symbol operation.

## Why Single-Writer Architecture

### Synchronization Costs

**Locks**: Uncontended mutex operations cost 20-50ns. Contended locks can exceed 1μs.

**Atomics**: Lock-free operations still incur memory fence overhead (10-20ns). CAS loops can spin under contention. Atomic operations constrain compiler optimizations.

**Single-Writer**: Eliminates synchronization overhead entirely. Provides deterministic execution and simpler correctness reasoning.

### Real-World Scaling

Production systems use **pipeline parallelism**:
```
Core 0: Network RX → Parse FIX/ITCH
Core 1: Matching Engine (single-writer to order book)
Core 2: Market Data Generation
Core 3: Network TX
```

Communication via lock-free SPSC queues (single-producer, single-consumer). Each stage is single-threaded. No shared mutable state between cores.

### NUMA Implications

Multi-socket systems have non-uniform memory access:
- Local memory: 80-100ns latency
- Remote socket: 120-150ns latency
- Cross-socket atomics: 200-300ns

**Solution**: Pin threads to cores on same socket. Allocate memory from local NUMA node. Use `numactl` to enforce placement.

## Cache-Line Analysis

### Order Structure (64 bytes)
```
OrderId:          8 bytes
Price:            8 bytes
Quantity:         4 bytes
Filled Quantity:  4 bytes
Timestamp:        8 bytes
Side:             1 byte
Status:           1 byte
Queue Position:   2 bytes
Padding:          4 bytes
Next Pointer:     8 bytes
Prev Pointer:     8 bytes
----------------------------
Total:           64 bytes (1 cache line)
```

**Why 64 bytes?** Modern CPUs fetch memory in 64-byte cache lines. Accessing any byte in a line loads entire line. Fitting order in one line minimizes cache traffic.

**Padding**: Explicit padding prevents false sharing. If two orders shared a cache line and were accessed by different cores, cache coherency protocol would ping-pong the line between cores.

### Price Level Structure

PriceLevel is ~40 bytes. Multiple levels can share cache lines. This is acceptable because we typically access one level at a time (best bid/ask).

### Memory Access Patterns

**Hot Path** (order submission):
1. Allocate from pool (1 cache line - free list head)
2. Initialize order (1 cache line - the order itself)
3. Match against best level (1-2 cache lines - level + first order)
4. Update book (1 cache line - best bid/ask pointers)

**Total**: 4-6 cache lines touched. At 4 cycles per L1 hit, ~20 cycles. At 3GHz, ~7ns. Actual latency higher due to logic, but cache efficiency is critical.

## Failure Modes

### Pool Exhaustion

**Symptom**: `allocate()` returns nullptr.

**Cause**: More than 1M live orders.

**Mitigation**: Monitor pool usage. Alert at 80% capacity. Increase pool size or reject orders gracefully.

**Production**: Dynamic pool expansion (allocate new slabs). Requires careful memory management to avoid fragmentation.

### Price Level Overflow

**Symptom**: More than 10K active price levels.

**Cause**: Wide price range or fragmented liquidity.

**Mitigation**: Increase MAX_PRICE_LEVELS or use dynamic structure (skip list, hash table).

**Production**: Skip list handles arbitrary price ranges with O(log n) lookup.

### Timestamp Wraparound

**Symptom**: uint64_t nanoseconds wraps after 584 years.

**Cause**: Using nanoseconds since epoch.

**Mitigation**: Use nanoseconds since process start. Or use 128-bit timestamps.

**Production**: Not a real concern. Process doesn't run for 584 years.

### Cache Thrashing

**Symptom**: Latency spikes when order book grows.

**Cause**: Working set exceeds L3 cache (typically 20-40MB).

**Mitigation**: Limit order book depth. Cancel stale orders. Use cache-oblivious data structures.

**Production**: Monitor cache miss rates (perf stat). Tune data structures to fit in cache.

## What Would Change in Production

### Multi-Symbol Support

**Current**: Single symbol only.

**Production**: Symbol directory with per-symbol order books. Options:
1. Separate matching thread per symbol (simple, scales to many cores)
2. Partition symbols across threads (load balancing)
3. Single thread handles multiple symbols (cache-friendly if symbols fit in cache)

**Tradeoff**: Separate threads scale better but increase context switch risk. Single thread maximizes cache utilization but limits throughput.

### Price Level Data Structure

**Current**: Linear search through flat array.

**Production**: Skip list or hash table.

**Skip List**: O(log n) lookup with better cache behavior than trees. Probabilistic balancing avoids rebalancing overhead. Used by Redis, LevelDB.

**Hash Table**: O(1) average case. Open addressing for cache-friendliness. Requires good hash function and load factor management.

**Tradeoff**: Skip list more predictable latency. Hash table higher throughput but worse tail latency.

### Risk Checks

**Current**: None.

**Production**: Pre-trade risk before matching:
- Position limits (net long/short per symbol)
- Credit limits (total exposure per account)
- Fat finger detection (price/quantity reasonableness)

**Challenge**: Risk checks add latency. Must be fast (<100ns). Typically use hash table lookup for account limits.

### Market Data Generation

**Current**: No market data output.

**Production**: Separate thread generates:
- Incremental updates (add/modify/delete)
- Periodic snapshots (full book state)
- Trade reports

**Protocol**: Binary format (custom or ITCH-like). Multicast to subscribers. Sequenced and gap-free.

**Latency**: Market data thread reads from lock-free queue. Matching thread writes updates. Queue must be SPSC for performance.

### Networking

**Current**: In-process only.

**Production**: Kernel bypass for low latency:
- Solarflare OpenOnload (user-space TCP/IP stack)
- Mellanox VMA (RDMA-based)
- DPDK (poll-mode drivers)

**Hardware Timestamping**: NIC timestamps packets on arrival. Eliminates kernel jitter.

**Tradeoff**: Kernel bypass requires dedicated NICs and careful tuning. Adds operational complexity.

### Persistence

**Current**: In-memory only. Crash loses state.

**Production**: Write-ahead log for recovery:
- Append-only log of all order events
- Replay log on restart to rebuild order book
- Periodic snapshots to bound replay time

**Persistent Memory**: Intel Optane provides DRAM-like latency with persistence. Can write directly to pmem without log.

**Tradeoff**: Persistence adds latency (100-500ns per write). Critical for exchange but not for market maker's internal book.

### Monitoring

**Current**: Basic counters (total orders, matches, cancels).

**Production**: Detailed metrics:
- Latency histograms (p50, p90, p99, p99.9, p99.99)
- Queue depths (orders per level, total book depth)
- Match rates (matches per second)
- Pool utilization (allocated vs capacity)
- Cache miss rates (perf counters)

**Export**: Prometheus metrics. Grafana dashboards. Alerting on anomalies.

## Design Tradeoffs

### Memory Pool vs Malloc

**Pool Advantages**:
- Deterministic O(1) latency
- No lock contention
- No syscalls
- Cache-friendly (sequential allocation)

**Pool Disadvantages**:
- Fixed capacity
- Memory not returned to OS
- Fragmentation if objects have different sizes

**When to Use**: Performance-critical paths where predictability matters more than flexibility.

### Intrusive vs Non-Intrusive Lists

**Intrusive Advantages**:
- One allocation per object (not two)
- Better cache locality
- No separate node overhead

**Intrusive Disadvantages**:
- Object can only be in one list
- Invasive (object must know about list)
- Harder to use with standard algorithms

**When to Use**: Performance-critical data structures where object is naturally in one container.

### Single-Threaded vs Multi-Threaded

**Single-Threaded Advantages**:
- No synchronization overhead
- Deterministic execution
- Easier to reason about correctness
- Better cache utilization

**Single-Threaded Disadvantages**:
- Doesn't scale to multiple cores
- Single point of failure
- Can't parallelize independent work

**When to Use**: Hot path where synchronization cost exceeds parallelism benefit. Use pipeline parallelism for scaling.

### Fixed-Point vs Floating-Point

**Fixed-Point Advantages**:
- Exact representation (no rounding errors)
- Deterministic across platforms
- Fast integer arithmetic

**Fixed-Point Disadvantages**:
- Limited range and precision
- Manual scaling required
- Overflow risk

**When to Use**: Financial calculations where exactness matters. Floating-point acceptable for analytics.

### Flat Array vs Tree

**Flat Array Advantages**:
- Cache-friendly sequential access
- Simple implementation
- Predictable performance

**Flat Array Disadvantages**:
- O(n) search
- Doesn't scale to large n

**When to Use**: Small to medium datasets (n < 500) where cache effects dominate algorithmic complexity.

## Implementation Rationale

### Systems-Level Considerations

**Cache Efficiency**: 64-byte aligned orders fit in single cache lines. Intrusive lists minimize pointer indirection. Price level access patterns favor sequential traversal.

**Memory Management**: Custom allocator provides deterministic behavior. Free-list allocation is O(1) without malloc overhead or lock contention.

**CPU Architecture**: Branch hints guide prediction. Memory fence costs inform synchronization decisions. Pipeline vs data parallelism tradeoffs are explicit.

### Operational Considerations

**Failure Modes**: Pool exhaustion, cache thrashing, and capacity limits are documented and detectable.

**Observability**: Design supports latency histograms, queue depth metrics, and pool utilization monitoring.

**Scalability Path**: Current limitations (single symbol, linear price search) have known solutions (symbol directory, skip lists).

### Engineering Tradeoffs

**Single vs Multi-threaded**: Single-threaded design optimizes hot path latency. Multi-threading adds value for I/O and market data generation where synchronization costs are amortized.

**Data Structure Selection**: Linear search vs tree structures depends on working set size and cache behavior, not just algorithmic complexity.

**Abstraction Level**: Minimal abstraction reduces indirection overhead. Each layer has measurable performance justification.

### Comparison to Alternatives

**vs std::map**: Tree traversal incurs multiple cache misses. Linear search through flat array is faster for typical order book depth.

**vs std::unordered_map**: Dynamic allocation and hash table resizing add latency variance. Fixed-capacity open-addressing provides deterministic behavior.

**vs malloc/new**: General-purpose allocators have unpredictable latency due to lock contention and system calls. Custom pools provide O(1) guarantees.

### Production Readiness

This implementation demonstrates core concepts but lacks production requirements:
- Multi-symbol support requires symbol directory and routing logic
- Risk management adds pre-trade validation overhead
- Market data generation needs separate thread and protocol handling
- Persistence requires write-ahead logging or persistent memory
- Network I/O needs kernel bypass and protocol parsers

Each addition has performance implications that must be measured and optimized independently.
