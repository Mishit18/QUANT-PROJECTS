# Market Data Feed Handler

Low-latency market data ingestion system with multi-threaded pipeline architecture.

## System Overview

Multi-threaded feed handler that ingests market data via UDP and execution reports via TCP, parses FIX and binary protocols, and distributes normalized events to downstream consumers.

**Design Principles:**
- Pre-allocated memory (no heap allocation on hot path)
- Lock-free SPSC queues for inter-thread communication
- Cache-line aligned data structures
- Non-blocking IO with epoll
- Pointer-based parsing to minimize copies
- Deterministic latency focus

## Architecture

```
┌─────────────┐     ┌─────────────┐
│ UDP Socket  │     │ TCP Socket  │
│ (Mkt Data)  │     │ (Exec Rpts) │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               │
        ┌──────▼──────┐
        │  IO Thread  │  (epoll, recv)
        └──────┬──────┘
               │ SPSC Queue
        ┌──────▼──────┐
        │   Decoder   │  (FIX/Binary parse)
        │   Thread    │
        └──────┬──────┘
               │ SPSC Queue
        ┌──────▼──────┐
        │ Dispatcher  │  (Normalize & publish)
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Consumers   │
        └─────────────┘
```

## Data Flow

1. **IO Thread**: Receives raw bytes from TCP/UDP sockets using epoll
2. **Ring Buffer**: Pre-allocated circular buffer for zero-copy message staging
3. **Decoder Thread**: Parses FIX or binary protocol without allocation
4. **Dispatcher**: Publishes normalized events to downstream consumers
5. **Consumers**: Strategy engines, risk systems, logging, etc.

## Threading Model

See [docs/threading_model.md](docs/threading_model.md)

- **IO Thread**: Pinned to core 0, handles socket reads only
- **Decoder Thread**: Pinned to core 1, parses messages
- **Dispatcher Thread**: Pinned to core 2, publishes events
- **SPSC Queues**: Lock-free single-producer-single-consumer between stages

## Memory Model

See [docs/memory_model.md](docs/memory_model.md)

- **Pre-allocated Buffers**: All memory allocated at startup
- **Ring Buffers**: Fixed-size circular buffers (16MB default)
- **Cache-Line Alignment**: 64-byte alignment for hot structures
- **Zero-Copy**: Pointer-based parsing, no string copies
- **Memory Pools**: Fixed-size object pools for events

## Protocol Support

### FIX Protocol
See [docs/fix_parser.md](docs/fix_parser.md)

Supports FIX 4.2/4.4 tag-value parsing:
- Zero-copy tag extraction
- Checksum validation
- Partial message reassembly
- No std::string allocation

### Binary Protocol
See [docs/binary_parser.md](docs/binary_parser.md)

Custom binary market data format:
- 16-byte header (type, length, sequence)
- Variable payload (price, size, side)
- Explicit endianness handling
- Struct overlay with alignment checks

## Performance Characteristics

**Micro-Benchmark Results** (parser components only, measured in isolation):
- Ring buffer: ~20ns per operation
- Event queue: ~25ns per operation
- Binary parser: ~70ns per message
- FIX parser: ~300ns per message

**Note**: These are component-level benchmarks. End-to-end latency includes network stack, kernel processing, and queuing delays which are not measured here.

**Throughput** (parser-limited):
- Binary protocol: 10-15M msg/sec
- FIX protocol: 3-5M msg/sec

**Test Environment**:
- Intel Xeon (Skylake or newer)
- Isolated CPU cores
- Raw POSIX sockets
- Single-threaded parsers

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Run

```bash
# Start feed handler
./feed_handler --udp-port 9000 --tcp-port 9001

# Run benchmark
./benchmark --duration 60 --rate 1000000
```

## Benchmark

```bash
./benchmark
```

Measures:
- Messages per second
- End-to-end latency (p50, p99, p99.9)
- Parser throughput
- Queue depth statistics

## Testing

```bash
./tests
```

Unit tests for:
- Ring buffer operations
- FIX parser correctness
- Binary parser correctness
- Message reassembly

## System Tuning Considerations

1. **CPU Isolation**: Use `isolcpus` kernel parameter
2. **IRQ Affinity**: Pin NIC interrupts away from worker cores
3. **Huge Pages**: Enable for large buffers
4. **NUMA**: Pin threads and memory to same NUMA node
5. **TSC**: Ensure constant_tsc and nonstop_tsc CPU flags

## License

Educational Project
