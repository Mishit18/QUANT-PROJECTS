# Limitations

## Scope

This is a research-grade simulator for studying market microstructure and testing execution strategies. It is not a production exchange.

## Infrastructure

### No Kernel Bypass
- Missing: DPDK, io_uring, AF_XDP
- Impact: Cannot achieve sub-microsecond network latency
- Reason: Platform-specific, requires privileged access

### No NUMA Optimization
- Missing: NUMA-aware memory allocation, thread pinning
- Impact: Performance degradation on multi-socket systems
- Reason: Single-threaded design, platform-specific

### No Hardware Timestamping
- Missing: NIC hardware timestamps
- Impact: Cannot measure true network latency
- Reason: Requires specialized NICs (Intel X710, Mellanox ConnectX)

## Network

### No Real Network Stack
- Missing: TCP/IP, UDP, multicast
- Impact: Latency is abstract, not measured from real network
- Reason: Focus on matching logic, not network simulation

### No FIX Protocol
- Missing: FIX 4.2, FIX 4.4, FAST encoding
- Impact: Cannot connect to real exchanges
- Reason: Protocol implementation is orthogonal to matching logic

### No Market Data Protocols
- Missing: ITCH, OUCH, BATS PITCH, CME MDP
- Impact: Cannot consume real market data feeds
- Reason: Focus on matching engine, not data normalization

## Matching Engine

### Single-Threaded
- Missing: Parallel matching across symbols
- Impact: Cannot scale beyond single-core performance
- Reason: Simplicity, determinism, no lock contention

### Limited Order Types
- Missing: Market orders, stop orders, iceberg, FOK, IOC, GTD
- Impact: Only limit orders supported
- Reason: Focus on core price-time priority logic

### No Self-Trade Prevention
- Missing: Reject orders that would match own orders
- Impact: Can trade with yourself
- Reason: Requires participant ID tracking

### No Order Routing
- Missing: Smart order routing, venue selection
- Impact: Single order book only
- Reason: Focus on single-venue matching

## Risk Management

### No Position Limits
- Missing: Max position size, concentration limits
- Impact: Can accumulate unlimited positions
- Reason: Risk management is separate concern

### No Credit Checks
- Missing: Pre-trade credit checks, margin requirements
- Impact: Can submit orders without capital
- Reason: Focus on matching, not risk

### No Circuit Breakers
- Missing: Trading halts, price bands, volatility interruptions
- Impact: No protection against flash crashes
- Reason: Market surveillance is separate system

## Data Management

### No Persistence
- Missing: Database storage, write-ahead log
- Impact: All state lost on restart
- Reason: In-memory simulation for speed

### No Audit Trail
- Missing: Immutable event log, regulatory reporting
- Impact: Cannot reconstruct historical state
- Reason: Compliance is separate concern

### No Replication
- Missing: Active-active, active-passive failover
- Impact: Single point of failure
- Reason: Single-process simulation

## Performance

### No Lock-Free Data Structures
- Missing: Lock-free order book, lock-free queue
- Impact: Cannot scale to multiple threads efficiently
- Reason: Single-threaded design, standard containers sufficient

### No Memory Pools
- Missing: Pre-allocated order objects, custom allocators
- Impact: Heap allocation overhead
- Reason: Standard allocator is fast enough for research

### No SIMD Optimization
- Missing: Vectorized price level search, batch processing
- Impact: Scalar operations only
- Reason: Compiler auto-vectorization is sufficient

## Monitoring

### No Real-Time Metrics
- Missing: Prometheus, Grafana, StatsD
- Impact: No live monitoring dashboard
- Reason: Simulation outputs results at end

### No Distributed Tracing
- Missing: OpenTelemetry, Jaeger, Zipkin
- Impact: Cannot trace requests across systems
- Reason: Single-process simulation

## Testing

### Limited Test Coverage
- Missing: Fuzz testing, property-based testing
- Impact: May have edge cases not covered
- Reason: Manual test cases cover core scenarios

### No Stress Testing
- Missing: Sustained high load, memory leak detection
- Impact: Unknown behavior under extreme load
- Reason: Benchmarks cover typical scenarios

## Compliance

### No Regulatory Reporting
- Missing: CAT, OATS, MiFID II reporting
- Impact: Cannot meet regulatory requirements
- Reason: Compliance is separate system

### No Best Execution
- Missing: NBBO checks, price improvement
- Impact: No guarantee of best price
- Reason: Single venue, no routing

## What This System Is

- Research-grade simulator
- Correct matching logic (price-time priority)
- Deterministic execution
- Measurable performance
- Educational tool

## What This System Is Not

- Production exchange
- Real-time trading system
- Regulatory compliant
- Kernel-bypass optimized
- Multi-threaded matching engine

## Path to Production

Estimated effort to production-ready exchange: 10-20 engineer-years

Required additions:
- Infrastructure: Kernel bypass, NUMA, CPU pinning
- Network: FIX protocol, market data feeds
- Risk: Position limits, credit checks, circuit breakers
- Persistence: Database, audit trail, replication
- Monitoring: Metrics, tracing, alerting
- Testing: Fuzz testing, stress testing, chaos engineering
- Compliance: Regulatory reporting, best execution
- Operations: Deployment automation, runbooks, on-call

## Use Cases

### Appropriate
- Market microstructure research
- Strategy backtesting (execution simulation)
- Exchange mechanics education
- Performance benchmarking

### Inappropriate
- Production trading
- Real money deployment
- Regulatory compliance
- High-frequency trading (without significant hardening)
