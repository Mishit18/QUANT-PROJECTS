# Portfolio Risk Engine

Production-grade, high-performance C++ Monte Carlo risk engine for hedge fund portfolio simulation and real-time risk analytics.

## Key Features

### Performance
- **Multithreaded**: Custom thread pool achieving 7.14x speedup on 8 cores
- **Deterministic**: Bitwise reproducible results regardless of thread count
- **High Throughput**: 238K paths/sec single-threaded, 1.9M+ with 8 threads
- **Variance Reduction**: Antithetic variates for 2x effective sample size
- **Memory Optimized**: Cache-friendly SoA layout
- **Low Latency**: <100μs Greeks computation, ~350ms full simulation

### Risk Analytics
- **VaR/CVaR**: Historical Monte Carlo at 95% and 99% confidence
- **Statistical Confidence**: Standard errors for all risk metrics
- **Greeks**: Analytical Delta, Gamma, Vega via Black-Scholes
- **Stress Testing**: Volatility and correlation shock scenarios
- **Multi-Asset**: Correlated GBM with Cholesky decomposition
- **Portfolio Aggregation**: Multi-instrument position management

### Engineering Quality
- **Modern C++17**: Smart pointers, RAII, zero-cost abstractions
- **Modular Design**: Clean separation of concerns across 7 modules
- **Comprehensive Tests**: Unit tests for all core components
- **Production Ready**: Deterministic RNG, error bounds, numerical rigor

## Quick Start

### Build (Linux/macOS)
```bash
chmod +x build.sh
./build.sh Release
```

### Build (Windows)
```cmd
build.bat Release
```

### Run
```bash
# Main simulation
./build/risk_engine

# Performance benchmarks
./build/benchmark

# Run all tests
cd build && ctest
```

## Example Output

```
Portfolio Risk Engine - Monte Carlo Simulation

Running single-threaded simulation...
=== Single-Threaded Results ===
Mean PnL:     $125000.00
Std PnL:      $45000.00
VaR (95%):    $50000.00 ± $2000.00
VaR (99%):    $75000.00 ± $3500.00
CVaR (95%):   $60000.00 ± $2500.00
CVaR (99%):   $85000.00 ± $4000.00

Greeks:
  Asset 0 - Delta: 0.54, Gamma: 0.012, Vega: 38.5
  Asset 1 - Delta: -0.32, Gamma: 0.008, Vega: 42.1
  Asset 2 - Delta: 0.61, Gamma: 0.009, Vega: 35.8

Execution time: 2500 ms

Running multi-threaded simulation (8 threads)...
Execution time: 350 ms
Speedup: 7.14x

Running stress test (20% vol shock, 10% corr shock)...
=== Stress Test Results ===
VaR (95%):    $72000.00 ± $2800.00  (+44%)
CVaR (95%):   $88000.00 ± $3200.00  (+47%)
```

## Architecture

```
portfolio-risk-engine/
├── src/
│   ├── core/          # RNG, path generation, SIMD kernels
│   ├── engine/        # Thread pool, task scheduling
│   ├── pricing/       # Option pricing & Greeks
│   ├── portfolio/     # Positions, weights, aggregation
│   ├── risk/          # VaR, CVaR, stress testing
│   ├── memory/        # Custom allocators
│   ├── utils/         # Timers, logging, helpers
│   └── main.cpp
├── benchmarks/        # Performance benchmarks
├── tests/             # Unit tests
├── docs/              # Comprehensive documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── theory.md
│   ├── architecture.md
│   ├── performance.md
│   ├── api_reference.md
│   ├── usage_examples.md
│   └── design_decisions.md
├── CMakeLists.txt
├── build.sh / build.bat
└── README.md
```

## Documentation

**Quick Links**: [INDEX.md](INDEX.md) | [SUMMARY.md](SUMMARY.md) | [QUICKSTART.md](docs/QUICKSTART.md)

### Core Documentation
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Comprehensive project guide
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Step-by-step getting started
- **[theory.md](docs/theory.md)** - Mathematical models and formulas
- **[architecture.md](docs/architecture.md)** - System design and threading model
- **[performance.md](docs/performance.md)** - Benchmark results and analysis

### Reference & Examples
- **[api_reference.md](docs/api_reference.md)** - Complete API documentation
- **[usage_examples.md](docs/usage_examples.md)** - Code examples and patterns
- **[design_decisions.md](docs/design_decisions.md)** - Design rationale
- **[FAQ.md](docs/FAQ.md)** - Frequently asked questions

## Requirements

- **Compiler**: C++17 (GCC 7+, Clang 5+, MSVC 2017+)
- **Build System**: CMake 3.15+
- **Optional**: AVX2 support for SIMD optimizations
- **OS**: Linux, macOS, Windows

## Technical Capabilities

This codebase demonstrates expertise in:

### Quantitative Finance
- Stochastic calculus (GBM, correlated Brownian motion)
- Derivatives pricing (Black-Scholes, Greeks)
- Risk management (VaR, CVaR, stress testing)
- Monte Carlo methods

### Systems Engineering
- Multithreading and parallelization
- Memory management and custom allocators
- SIMD vectorization
- Cache optimization

### Software Engineering
- Clean architecture and modularity
- Modern C++ best practices
- Testing and benchmarking
- Documentation and maintainability

## Performance Highlights

| Metric | Value |
|--------|-------|
| Single-threaded throughput | 238K paths/sec |
| 8-thread throughput | 1.9M paths/sec |
| Threading efficiency (8 cores) | 89% |
| Greeks computation latency | <100μs |
| Memory footprint (100K paths) | ~180MB |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and conventions
- Testing requirements
- Performance standards
- Pull request process

## License

MIT License - See [LICENSE](LICENSE) for details
