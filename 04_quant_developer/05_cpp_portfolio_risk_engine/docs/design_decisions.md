# Design Decisions

## Language Choice: C++17

**Rationale**: Balance between modern features and compiler support
- Smart pointers and RAII for memory safety
- Template metaprogramming for zero-cost abstractions
- Standard library threading primitives
- Compatibility with production HFT environments

## Threading Model: Custom Thread Pool

**Why not std::async?**
- Better control over thread lifecycle
- Reduced overhead from repeated thread creation
- Work queue allows load balancing
- Explicit thread count management

**Why not OpenMP?**
- More fine-grained control
- Better integration with custom allocators
- Explicit task dependencies
- Production-grade error handling

## Memory Layout: Structure of Arrays

**AoS vs SoA Trade-off**:
```cpp
// AoS: Poor cache locality
struct Path { double prices[252]; };
Path paths[100000];

// SoA: Cache-friendly, SIMD-ready
double paths[100000 * 252];
```

**Benefits**:
- Sequential memory access
- Vectorization-friendly
- Reduced cache misses
- Better memory bandwidth utilization

**Drawbacks**:
- More complex indexing
- Harder to debug

**Decision**: SoA for hot paths, AoS for configuration

## RNG: MT19937-64

**Alternatives considered**:
- PCG: Faster but less established
- Xorshift: Fast but lower quality
- Hardware RNG: Not portable

**Decision**: MT19937-64
- Industry standard
- Good statistical properties
- Fast enough for our use case
- Widely validated

## Correlation: Cholesky Decomposition

**Alternatives**:
- Eigenvalue decomposition: More expensive
- Spectral decomposition: Overkill for our needs

**Decision**: Cholesky
- O(n³) one-time cost
- O(n²) per sample
- Numerically stable
- Standard in finance

## Greeks: Analytical vs Finite Difference

**Approach**: Analytical (Black-Scholes)

**Rationale**:
- Exact for European options
- No discretization error
- Faster than finite difference
- Standard in industry

**Limitation**: Only works for vanilla options
**Future**: Add finite difference for exotics

## VaR: Historical Simulation

**Alternatives**:
- Parametric VaR: Assumes normal distribution
- Variance-Covariance: Fast but inaccurate for options

**Decision**: Historical (Monte Carlo)
- Captures non-linear payoffs
- No distributional assumptions
- Handles fat tails
- Industry standard for options

## Parallelization Strategy

**Path-level vs Step-level**:
- Path-level: Embarrassingly parallel, chosen
- Step-level: Requires synchronization

**Work Distribution**:
- Static partitioning: Simple, predictable
- Dynamic work stealing: Complex, better load balance

**Decision**: Static partitioning
- Paths have uniform cost
- Minimal overhead
- Predictable performance

## Error Handling

**Strategy**: Exceptions for configuration, assertions for logic errors

```cpp
// Configuration errors: throw
if (correlation_matrix.empty()) {
    throw std::invalid_argument("Empty correlation matrix");
}

// Logic errors: assert
assert(index < size && "Index out of bounds");
```

**Rationale**:
- Configuration errors are recoverable
- Logic errors indicate bugs
- Assertions removed in release builds

## Testing Strategy

**Unit tests**: Core components (RNG, pricing, portfolio)
**Integration tests**: Full simulation pipeline
**Benchmarks**: Performance regression detection

**No mocking**: Direct testing of real implementations
**Rationale**: Performance-critical code, mocking adds overhead

## Build System: CMake

**Alternatives**: Make, Bazel, Meson

**Decision**: CMake
- Industry standard
- Cross-platform
- Good IDE integration
- Mature ecosystem

## Documentation Philosophy

**Principle**: Code should be self-documenting

**Comments only for**:
- Non-obvious algorithms (Cholesky)
- Performance optimizations (SIMD)
- Mathematical formulas
- API contracts

**No comments for**:
- Obvious code
- Implementation details
- Redundant information

## Performance vs Readability

**Guideline**: Readability first, optimize hot paths

**Hot paths** (optimized):
- Path generation
- PnL computation
- Sorting for VaR

**Cold paths** (readable):
- Configuration
- Initialization
- Reporting

**Measurement**: Profile before optimizing
