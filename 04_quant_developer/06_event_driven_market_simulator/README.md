# Event-Driven Market Simulator

A C++ implementation of an event-driven market simulator with a continuous limit order book and price-time priority matching engine.

## Overview

This simulator implements core exchange mechanics:
- Event-driven architecture with deterministic replay
- Continuous limit order book (CLOB) with price-time priority
- Microsecond-resolution logical clock
- Configurable latency models
- Performance benchmarking tools

## Architecture

### Core Components

**Event System** (`src/core/`)
- Event types: order submit, cancel, amend, fill, market data
- Priority queue for chronological event processing
- Logical clock decoupled from wall time

**Order Book** (`src/orderbook/`)
- Price-time priority matching (FIFO at each price level)
- Partial fill support
- Order lifecycle management (active, partially filled, filled, cancelled)

**Latency Modeling** (`src/latency/`)
- Fixed, uniform, normal, and queueing delay models
- Latency injection between order submission and matching

**Market Data** (`src/replay/`)
- Synthetic order generation with deterministic seeding

See `docs/architecture.md` for detailed design.

## Build

Requires C++17 or newer and CMake 3.15+.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Windows:
```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Run

```bash
# Main simulator
./build/market_simulator

# Throughput benchmark
./build/latency_benchmark

# Latency distribution
./build/latency_distribution
```

## Matching Engine

Implements strict price-time priority:
1. **Price priority**: Best bid (highest) matches with best ask (lowest)
2. **Time priority**: Orders at the same price match in FIFO order
3. **Partial fills**: Orders can be filled across multiple matches

Example:
```
Order Book:
BID: 100 @ 150.00 (t=1000)
ASK: 100 @ 150.50 (t=2000)

Incoming: SELL 50 @ 150.00
Result: Matches 50 @ 150.00 with bid (t=1000)
        Bid remaining: 50 @ 150.00
```

See `docs/matching_engine.md` for details.

## Latency Models

- **Fixed**: Constant delay (baseline testing)
- **Uniform**: Random delay in [min, max] range
- **Normal**: Gaussian distribution (mean, stddev)
- **Queueing**: Load-dependent delay (simulates congestion)

See `docs/latency_model.md` for implementation details.

## Performance

Typical throughput on modern hardware:
- 600K-800K orders/sec (single-threaded)
- Microsecond-level simulation time
- Nanosecond clock resolution

Latency distribution example (100K orders, burst traffic):
```
p50:  2 μs
p90:  73 μs
p99:  3.9 ms
p999: 27 ms
```

## Determinism

The simulator is deterministic:
- Same input → same output
- Fixed RNG seed for reproducibility
- Logical clock (no wall-clock dependency)
- Ordered containers for consistent iteration

## Limitations

This is a research-grade simulator, not a production exchange.

**Not included**:
- Network stack (latency is abstract)
- FIX protocol or market data feeds
- Risk management or position limits
- Persistence or audit trail
- Multi-threading or lock-free structures
- Advanced order types (market, stop, iceberg)
- Self-trade prevention

See `docs/limitations.md` for complete list.

## Use Cases

- Market microstructure research
- Strategy backtesting (execution simulation)
- Exchange mechanics education
- Performance benchmarking

## Documentation

- `docs/architecture.md` - System design and event flow
- `docs/matching_engine.md` - Matching logic and complexity
- `docs/latency_model.md` - Latency injection and measurement
- `docs/limitations.md` - Scope and non-goals

## Testing

```bash
./build/run_tests
```

Basic unit tests for order book and matching engine operations.

## License

MIT License - See LICENSE file.

## Disclaimer

This software is for educational and research purposes. It is not intended for production trading or financial decision-making.
