# Frequently Asked Questions

## General

### Q: What is this project for?
A: This is a production-grade Monte Carlo risk engine designed to demonstrate quantitative systems engineering skills for hedge fund interviews. It simulates correlated asset paths and computes portfolio risk metrics (VaR, CVaR, Greeks).

### Q: Is this production-ready?
A: Yes, with caveats. The core engine is production-quality with proper error handling, testing, and documentation. However, it would need additional features for real production use:
- More asset models (jump diffusion, stochastic volatility)
- More option types (Asian, Barrier, American)
- Real-time data integration
- Distributed computing support
- Regulatory reporting

### Q: What firms would find this relevant?
A: Quantitative trading firms (quantitative trading firms, quantitative trading firms, quantitative trading firms), HFT firms (high-frequency trading firms, trading firms, trading firms), and investment banks (financial institutions, financial institutions) for quant developer and risk systems roles.

## Technical

### Q: Why C++ instead of Python?
A: Performance. C++ provides:
- 20x faster execution than Python/NumPy
- Direct hardware control (SIMD, threading)
- Deterministic memory management
- Industry standard for HFT systems

### Q: Why custom thread pool instead of OpenMP?
A: Better control and production readiness:
- Explicit thread lifecycle management
- Custom work queue for load balancing
- Better error handling
- More suitable for integration with other systems

### Q: Why not use GPU acceleration?
A: GPUs are excellent for massive parallelism but have trade-offs:
- Data transfer overhead
- Higher latency for small problems
- More complex debugging
- Not always available in production

This engine is optimized for CPU and can be extended with GPU support.

### Q: What's the difference between VaR and CVaR?
A: 
- **VaR (Value-at-Risk)**: Maximum expected loss at a confidence level (e.g., 95%)
- **CVaR (Conditional VaR)**: Average loss beyond VaR, captures tail risk

CVaR is more conservative and preferred by regulators.

### Q: How accurate are the Greeks?
A: Very accurate for European options using analytical Black-Scholes formulas. For exotic options, you'd need finite difference or pathwise methods.

### Q: Can I add more asset models?
A: Yes! The architecture is modular. To add a new model:
1. Extend `PathGenerator` with new dynamics
2. Update `AssetConfig` with model parameters
3. Implement the SDE discretization
4. Add tests

## Performance

### Q: Why isn't the speedup exactly 8x with 8 threads?
A: Several factors:
- Amdahl's Law: Some code is sequential (initialization, aggregation)
- Cache effects: More threads = more cache contention
- Synchronization overhead: Thread pool management
- Memory bandwidth: Can become bottleneck

89% efficiency on 8 cores is excellent.

### Q: How can I improve performance further?
A: Several options:
1. **Algorithmic**: Variance reduction (antithetic variates, control variates)
2. **Hardware**: AVX-512, GPU acceleration
3. **Numerical**: Quasi-random numbers (Sobol sequences)
4. **Caching**: Precompute expensive operations

### Q: What's the memory bottleneck?
A: Path storage. For N paths, M steps, K assets:
```
Memory = N × M × K × sizeof(double) = N × M × K × 8 bytes
```

For 1M paths, 252 steps, 3 assets: ~6GB

Solution: Stream processing or reduce path storage.

### Q: Can this run on embedded systems?
A: Depends on constraints. The engine needs:
- ~100MB minimum memory
- C++17 compiler
- Threading support

Could work on high-end embedded systems (ARM Cortex-A series).

## Usage

### Q: How do I add a new option type?
A: 
1. Extend `OptionType` enum in `option.h`
2. Implement payoff in `OptionPricer::payoff()`
3. Add pricing formula (if analytical exists)
4. Update Greeks computation
5. Add tests

### Q: How do I change the correlation structure?
A: Update the correlation matrix in `SimulationConfig`:
```cpp
config.correlation_matrix = {
    {1.0, 0.6, 0.3},
    {0.6, 1.0, 0.4},
    {0.3, 0.4, 1.0}
};
```

Ensure it's symmetric and positive definite.

### Q: Can I use real market data?
A: Yes! Replace the simulated paths with historical data:
1. Load data from CSV/database
2. Compute returns
3. Estimate correlation matrix
4. Use for simulation or bootstrap

### Q: How do I calibrate volatility?
A: Several methods:
1. **Historical**: Compute standard deviation of returns
2. **Implied**: Back out from option prices
3. **GARCH**: Time-varying volatility model

This engine uses constant volatility (Black-Scholes assumption).

## Troubleshooting

### Q: Build fails with "C++17 not supported"
A: Update your compiler:
```bash
# Linux
sudo apt-get install g++-9
export CXX=g++-9

# macOS
brew install gcc@11
export CXX=g++-11
```

### Q: Runtime error: "Correlation matrix not positive definite"
A: Check your correlation matrix:
- Must be symmetric: `corr[i][j] == corr[j][i]`
- Diagonal must be 1.0: `corr[i][i] == 1.0`
- Eigenvalues must be positive

Use a correlation matrix validator or adjust values.

### Q: Simulation is very slow
A: Checklist:
1. Build in Release mode: `cmake -DCMAKE_BUILD_TYPE=Release`
2. Enable optimizations: `-O3 -march=native`
3. Use multithreading: `compute_risk_metrics_multithreaded()`
4. Reduce paths for testing: `config.num_paths = 10000`

### Q: Memory allocation error
A: Reduce memory usage:
```cpp
config.num_paths = 50000;  // Instead of 1000000
config.num_steps = 126;    // Instead of 252
```

Or use memory pool allocator.

## Extensions

### Q: Can I add American options?
A: Yes, but requires different approach:
- Least Squares Monte Carlo (Longstaff-Schwartz)
- Binomial tree
- Finite difference methods

American options need early exercise logic.

### Q: Can I add interest rate derivatives?
A: Yes! Implement interest rate models:
- Vasicek
- Cox-Ingersoll-Ross (CIR)
- Hull-White
- Heath-Jarrow-Morton (HJM)

Requires extending `PathGenerator` with new dynamics.

### Q: Can I add credit risk?
A: Yes! Add credit models:
- Merton structural model
- Reduced-form models
- CDS pricing

Requires default probability and recovery rate modeling.

### Q: Can I distribute across multiple machines?
A: Yes, with MPI (Message Passing Interface):
1. Partition paths across nodes
2. Each node runs local simulation
3. Aggregate results at master node

Requires MPI library and network infrastructure.

## Best Practices

### Q: How many paths should I use?
A: Depends on accuracy requirements:
- **Testing**: 10,000 paths
- **Development**: 50,000 paths
- **Production**: 100,000 - 1,000,000 paths

More paths = better accuracy but slower execution.

### Q: How often should I recompute risk?
A: Depends on use case:
- **Pre-trade**: On-demand
- **Intraday**: Every 15-30 minutes
- **End-of-day**: Once daily
- **Regulatory**: Daily/weekly

### Q: Should I use historical or implied volatility?
A: Depends on purpose:
- **Historical**: Backward-looking, actual realized volatility
- **Implied**: Forward-looking, market expectation

For risk management, implied volatility is often preferred.

### Q: How do I validate results?
A: Multiple approaches:
1. Compare with analytical solutions (when available)
2. Cross-check with commercial systems (financial technology firms, QuantLib)
3. Convergence testing (increase paths, check stability)
4. Backtesting (compare predictions with actual outcomes)

## Contributing

### Q: How can I contribute?
A: See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Areas needing work:
- Additional option types
- More asset models
- GPU acceleration
- Variance reduction techniques
- Documentation improvements

### Q: Can I use this in my project?
A: Yes! MIT License allows commercial and non-commercial use. Attribution appreciated but not required.

### Q: How do I report bugs?
A: Open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, compiler, etc.)
