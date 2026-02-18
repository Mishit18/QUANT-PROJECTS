# Limit Order Book

Single-threaded matching engine with deterministic execution and allocation-free hot path.

## Overview

This is a limit order book implementation focused on predictable latency and deterministic behavior. It demonstrates core matching engine concepts in a simplified single-symbol context.

Key characteristics:
- Price-time priority (FIFO at each price level)
- Deterministic execution
- Pre-allocated memory pools
- Single-threaded design
- Fixed-capacity data structures

This is a reference implementation. Production systems require additional components: multi-symbol support, risk management, market data distribution, and network I/O handling.

## Matching Semantics

**Price-Time Priority**: Orders at better prices match first. Within a price level, earlier orders match first (FIFO).

**Aggressive vs Passive**: 
- Aggressive order crosses the spread (buy ≥ best ask, sell ≤ best bid)
- Matches immediately at passive order's price
- Unmatched quantity rests in book as passive order

**Partial Fills**: Order can match multiple times until fully filled or no more liquidity available.

**Order Lifecycle**:
```
NEW → PARTIAL_FILL → FULL_FILL
  ↓
CANCELLED
```

**Modify Semantics**: Only quantity increases allowed (maintains queue priority). Decreasing quantity requires cancel + new order.

## Design Decisions

**Single-Threaded**: Avoids synchronization overhead. Production systems typically use pipeline parallelism with separate threads for I/O, parsing, matching, and market data generation.

**Intrusive Lists**: Orders contain list pointers directly, avoiding separate node allocations and improving cache locality.

**Memory Pool**: Pre-allocated storage with free-list allocation. Provides deterministic O(1) allocation without malloc overhead.

**Open-Addressing Hash Table**: Custom implementation for order lookup. Fixed capacity, no dynamic resizing, deterministic probe sequence.

**Linear Search for Price Levels**: Efficient for typical order book depth (50-200 active levels). Production systems with wider price ranges use skip lists or hash tables.

**Fixed-Point Prices**: Integer representation with implicit decimal places. Avoids floating-point non-determinism and rounding issues.

**Static Polymorphism**: No virtual functions in performance-critical paths.

## Performance Considerations

- Pre-allocated memory pools eliminate runtime allocation
- 64-byte cache-aligned order structures
- Intrusive data structures reduce pointer indirection
- Branch prediction hints for common paths
- Compiled without exceptions or RTTI
- Optional CPU core pinning support
- Aggressive compiler optimizations (-O3, -march=native, LTO)

## Scope and Limitations

This implementation focuses on core matching logic. The following are explicitly out of scope:

- Multi-symbol support (single symbol only)
- Wide price ranges (linear search degrades beyond ~500 levels)
- Market orders (limit orders only)
- Time-in-force variants (IOC, FOK, GTD, etc.)
- Pre-trade risk checks
- Market data feed generation
- Persistence and crash recovery
- Network I/O and protocol handling

These components are necessary for production systems but are independent of the core matching algorithm.

## Build & Run

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Basic test:
```bash
./lob_engine
```

Benchmark (1M orders):
```bash
./lob_benchmark 1000000 ../benchmarks/results.csv
```

Full benchmark suite:
```bash
cd ..
chmod +x scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh
```

## Performance Characteristics

Typical latency on modern x86 CPU (pinned to isolated core):
- Mean: 200-400ns per order
- p50: 250-350ns
- p99: 500-800ns
- p99.9: 1-2μs
- Max: 3-5μs (cold cache)

Measurements include order submission, matching, and book updates. Network I/O and serialization are not included.

## Known Limitations

**Fixed Capacity**: Order pool (1M orders) and price level array (10K levels) have fixed sizes. Exhaustion results in order rejection.

**Price Range**: Linear search through active price levels is efficient for typical depth but degrades beyond ~500 levels.

**Single Symbol**: No symbol directory or multi-symbol support.

**No Persistence**: State is in-memory only. Crash recovery would require write-ahead logging or persistent memory.

**No Market Data**: Does not generate incremental updates or snapshots.

**No Risk Management**: No position limits, credit checks, or pre-trade validation.

## Production Considerations

Real-world systems typically add:
- Skip lists or hash tables for price levels (handles wider ranges)
- Lock-free queues for inter-thread communication
- Pipeline architecture (separate threads for I/O, parsing, matching, market data)
- NUMA-aware memory allocation
- Kernel bypass networking
- Hardware timestamping
- Protocol handlers (FIX, ITCH, etc.)
- Write-ahead logging or persistent memory

## Technical Notes

**Why single-threaded?** Avoids synchronization overhead. Production systems use pipeline parallelism with lock-free queues between stages.

**Why intrusive lists?** Eliminates separate node allocations and improves cache locality.

**Why linear search for price levels?** Cache-friendly for typical order book depth. Tree structures incur multiple cache misses per lookup.

**Why custom hash table?** Avoids std::unordered_map allocation overhead and provides deterministic behavior.

See `docs/design_notes.md` for detailed technical discussion.
