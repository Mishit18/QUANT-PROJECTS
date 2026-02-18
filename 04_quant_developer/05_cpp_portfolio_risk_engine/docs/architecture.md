# System Architecture

## Threading Model

### Thread Pool Design

Custom thread pool implementation with:
- Worker threads created at initialization
- Lock-free task queue with condition variables
- Work stealing for load balancing
- Graceful shutdown with task completion

### Parallelization Strategy

Monte Carlo paths are embarrassingly parallel:
1. Partition paths across threads
2. Each thread generates independent paths with **deterministic RNG streams**
3. Aggregate results after completion
4. No synchronization during simulation

### RNG Stream Determinism

**Critical for Production**: Results must be bitwise reproducible regardless of thread count.

**Implementation**: SplitMix64 hash-based stream splitting
```cpp
// Each thread gets a unique, non-overlapping RNG stream
uint64_t stream_seed = hash(base_seed, stream_id);
```

**Guarantees**:
- Same seed → same results with 1, 8, or 16 threads
- No sequence overlap between threads
- Regulatory compliance (reproducible risk metrics)
- Debugging (deterministic behavior)

**Why Not seed + i?**
- MT19937 with seeds 42, 43, 44 produces overlapping sequences
- Results change with thread count
- Non-reproducible for compliance

### Cache Optimization

- Thread-local path storage to avoid false sharing
- Aligned allocations for optimal memory access
- Sequential memory access patterns

## Memory Layout

### Structure of Arrays (SoA)

Data layout optimized for cache efficiency:
```cpp
// Cache-friendly sequential access
double* paths = new double[M * N];
// Access: paths[asset * M * N + path * N + step]
```

Benefits:
- Sequential memory access patterns
- Better cache line utilization
- Reduced memory bandwidth
- Vectorization-ready (future optimization)

**Note**: Current implementation does not use SIMD vectorization. The SoA layout provides cache benefits and enables future vectorization when performance profiling identifies it as a bottleneck.

## Component Interaction

```
main()
  └─> VaRCalculator
       ├─> ThreadPool (parallel execution)
       ├─> PathGenerator (per thread, deterministic streams)
       │    └─> CorrelatedRNG (Cholesky + stream splitting)
       ├─> Portfolio (PnL computation)
       │    └─> OptionPricer (Greeks)
       └─> Risk Metrics (VaR, CVaR with confidence intervals)
```

## Variance Reduction

### Antithetic Variates

**Implementation**: For each random draw Z, also simulate -Z
```cpp
// Path pair with antithetic shocks
path1 = S * exp(drift + vol * Z)
path2 = S * exp(drift + vol * (-Z))
```

**Benefits**:
- ~2x variance reduction for symmetric payoffs
- Zero computational overhead
- Enabled by default (`use_antithetic_variates = true`)

**Theory**: Antithetic variates exploit negative correlation:
```
Var((X1 + X2)/2) = (Var(X1) + Var(X2) + 2*Cov(X1,X2)) / 4
```
When Cov(X1, X2) < 0, variance is reduced.

## Performance Considerations

### Hot Path Optimization

1. Path generation (70% of runtime)
2. PnL computation (20% of runtime)
3. Sorting for VaR (10% of runtime)

### Bottleneck Mitigation

- RNG: Use fast MT19937-64
- Exp/log: Standard library (compiler-optimized)
- Sorting: Partial sort for quantiles
- Memory: Sequential access patterns

### Scalability

Expected speedup with N threads:
```
Speedup ≈ N / (1 + overhead)
```

Overhead sources:
- Thread creation/destruction
- Task queue synchronization
- Result aggregation
- Cache coherency traffic
