# Architecture

## Overview

Event-driven simulator with deterministic execution. All state changes are driven by events processed in chronological order.

## Core Components

### Event System

**Event** (`Event.h`)
- Base class for all event types
- Contains timestamp and type
- Ordering: timestamp first, then type (for determinism)

**EventQueue** (`EventQueue.h`)
- Priority queue (min-heap) ordered by timestamp
- O(log n) insertion and extraction
- Deterministic tie-breaking by event type

**SimulationClock** (`Clock.h`)
- Logical time counter (uint64_t nanoseconds)
- Advances only when processing events
- Decoupled from system clock

**Simulator** (`Simulator.h`)
- Main event loop
- Pops events, advances clock, dispatches to handlers
- Coordinates matching engine and latency model

### Order Book

**Order** (`Order.h`)
- Limit order representation
- States: ACTIVE, PARTIALLY_FILLED, FILLED, CANCELLED
- Tracks original and remaining quantity

**PriceLevel** (`PriceLevel.h`)
- FIFO queue of orders at a single price
- Implemented with std::deque
- Maintains total quantity at level

**OrderBook** (`OrderBook.h`)
- Bid and ask sides (std::map<double, PriceLevel>)
- Fast order lookup (std::unordered_map<id, Order>)
- O(log m) price level access, O(1) best bid/ask

**MatchingEngine** (`MatchingEngine.h`)
- Processes order submissions, cancels, amends
- Executes matches with price-time priority
- Generates fill events

### Latency Modeling

**LatencyModel** (`LatencyModel.h`)
- Abstract interface for latency injection
- Implementations: Fixed, Uniform, Normal, Queueing
- Applied between order submission and matching

### Market Data

**MarketDataReplayer** (`MarketDataReplayer.h`)
- Generates synthetic order flow
- Deterministic with fixed seed
- Configurable order rate and price distribution

## Event Flow

```
1. Event created and added to EventQueue
2. Simulator pops next event (lowest timestamp)
3. Clock advances to event timestamp
4. Event dispatched based on type:
   - ORDER_SUBMIT → handleOrderSubmit()
   - ORDER_CANCEL → handleOrderCancel()
   - ORDER_AMEND → handleOrderAmend()
5. Latency model adds delay
6. Order submitted to MatchingEngine
7. MatchingEngine attempts match:
   - Check opposite side of book
   - Match against best price (price priority)
   - Match against front of queue (time priority)
   - Execute partial or full fills
8. Generate OrderFillEvent for each fill
9. If order has remaining quantity, add to book
10. Return to step 2
```

## Logical Time

### Decoupling from Wall Clock

Simulation time is independent of system time:
- Clock starts at 0
- Advances only when processing events
- No relationship to std::chrono or system clock

### Benefits

- **Determinism**: No dependency on system clock jitter
- **Fast-forward**: Skip idle periods instantly
- **Reproducibility**: Same input → same timestamps → same output
- **Debugging**: Can pause and inspect at any logical time

### Example

```cpp
clock.now() == 0

// Event at T=1000ns
clock.advanceTo(1000)
clock.now() == 1000

// Event at T=5000ns
clock.advanceTo(5000)
clock.now() == 5000
```

## Determinism

### Enforcement Points

1. **Event Ordering**
   - Events sorted by timestamp, then type
   - Tie-breaking ensures consistent ordering
   - No dependency on insertion order

2. **Random Number Generation**
   - Fixed seed for reproducibility
   - Same seed → same random sequence

3. **Order Book Operations**
   - std::map for price levels (deterministic iteration)
   - std::deque for FIFO queues
   - No hash maps where iteration order matters

4. **Matching Logic**
   - Strict price-time priority (no randomness)
   - Deterministic traversal of price levels

5. **No External Dependencies**
   - No network I/O
   - No file I/O during simulation
   - No system calls depending on wall time

## Data Structures

### Event Queue
- **Type**: std::priority_queue
- **Complexity**: O(log n) insert/extract
- **Rationale**: Efficient chronological ordering

### Price Levels
- **Type**: std::map<double, PriceLevel>
- **Complexity**: O(log m) lookup
- **Rationale**: Sorted by price, deterministic iteration

### Orders at Level
- **Type**: std::deque<Order>
- **Complexity**: O(1) front/back access
- **Rationale**: FIFO queue for time priority

### Order Lookup
- **Type**: std::unordered_map<uint64_t, Order*>
- **Complexity**: O(1) average case
- **Rationale**: Fast cancel/amend by ID

## Latency Injection

### Injection Point

```cpp
void handleOrderSubmit(OrderSubmitEvent* event) {
    uint64_t latency_ns = latency_model_->getLatency();
    uint64_t execution_time = event->getTimestamp() + latency_ns;
    Order order(..., execution_time);
    matching_engine_.submitOrder(order);
}
```

### What Latency Represents

- Network propagation delay
- NIC processing time
- Application processing time
- OS scheduling delay
- Queueing delay (in QueueingLatencyModel)

### Measurement

End-to-end latency: `fill_time - submit_time`

## Concurrency

This is a single-threaded simulator:
- No locks or atomics
- No concurrent access to order book
- No parallel matching
- Standard containers (std::map, std::deque, std::priority_queue)

Parallelization would require:
- Lock-free event queue (multi-producer)
- Thread per symbol (independent order books)
- Synchronization for shared market data

## Performance Characteristics

### Time Complexity

- Event insertion: O(log n)
- Event extraction: O(log n)
- Order submission: O(log m + k) where m = price levels, k = matches
- Order cancellation: O(log m + p) where p = orders at level
- Best bid/ask: O(1)

### Space Complexity

- Event queue: O(n) events
- Order book: O(m × p) where m = levels, p = orders/level
- Order lookup: O(o) where o = total orders

### Bottlenecks

1. Order cancellation (linear scan at price level)
2. Price level lookup (logarithmic)
3. Event queue operations (logarithmic)

## Design Rationale

### Why Event-Driven?

- Efficient: Only process when state changes
- Deterministic: Events processed in strict order
- Scalable: Easy to parallelize across symbols
- Realistic: Matches real exchange architecture

### Why Single-Threaded?

- Simplicity: No lock contention
- Determinism: No race conditions
- Sufficient: 600K+ orders/sec on single core

### Why Logical Clock?

- Determinism: No wall-clock jitter
- Fast-forward: Skip idle periods
- Reproducibility: Critical for research

## Extensibility

### Adding Event Types

Inherit from Event base class:
```cpp
class CustomEvent : public Event {
    // Custom fields
};
```

### Adding Latency Models

Implement LatencyModel interface:
```cpp
class CustomLatencyModel : public LatencyModel {
    uint64_t getLatency() override {
        // Custom logic
    }
};
```

### Adding Order Types

Extend MatchingEngine to handle new order types in submitOrder().

## Comparison to Production Systems

### Similarities

- Event-driven architecture
- Price-time priority matching
- Deterministic replay capability
- Microsecond-level latency

### Differences

- No network layer (latency is abstract)
- No persistence (in-memory only)
- No redundancy (single process)
- No regulatory compliance (no audit trail)
- Simplified market data (no full depth)

## References

- Harris, L. (2003). *Trading and Exchanges*
- Hasbrouck, J. (2007). *Empirical Market Microstructure*
