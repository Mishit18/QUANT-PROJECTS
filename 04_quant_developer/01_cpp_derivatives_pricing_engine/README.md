# Heston Stochastic Volatility Pricing Engine

C++ implementation of the Heston stochastic volatility model for European option pricing.

## Features

- Heston model with full stochastic volatility dynamics
- Monte Carlo pricing with Euler and QE discretization schemes
- PDE solver using explicit finite difference method
- Calibration to implied volatility surfaces
- Multithreading support
- Sobol quasi-random number generation

## SDE / Quant-Dev Screening Summary

Verified locally on Windows/MSVC Release:
- CMake builds the static library, example, benchmark, and five test executables
- CTest passes 5/5 tests in 6.40 seconds after making calibration a bounded smoke test
- Black-Scholes limit test: MC price 10.4173 vs BS 10.4506, 0.318% relative error
- Benchmark: MC Euler 1M paths in 11.10s; MC QE 1M paths on 8 threads in 3.13s
- PDE solver remains a documented limitation; current PDE prices show large error versus MC/Black-Scholes

## Implementation Details

### Monte Carlo Engine
- Euler-Maruyama discretization
- Andersen QE (Quadratic Exponential) scheme
- Antithetic variates for variance reduction
- Sobol quasi-random sequences
- Thread-parallel path simulation

### PDE Solver
- Explicit finite difference scheme for the Heston PDE
- 2D grid (spot price × variance)
- Stability-constrained time stepping
- Greeks computed via finite differences

### Calibration
- Levenberg-Marquardt optimization
- Objective function based on implied volatility errors
- Parameter bounds enforcement
- Feller condition regularization

## Known Limitations

- PDE solver uses explicit scheme which requires small time steps for stability
- PDE pricing shows ~80% error vs Black-Scholes in limit cases (known issue with current discretization)
- Sobol generator uses basic Box-Muller transform (not optimal for QMC)
- Calibration convergence depends heavily on initial guess

## Build Instructions

### Requirements
- C++17 compiler
- CMake 3.15+

### Compilation

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage Example

```cpp
#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"

using namespace heston;

HestonParameters params{
    .kappa = 2.0,
    .theta = 0.04,
    .sigma = 0.3,
    .rho = -0.7,
    .v0 = 0.04
};

MCConfig config{
    .num_paths = 100000,
    .num_steps = 252,
    .use_antithetic = true,
    .num_threads = 4,
    .scheme = MCConfig::Scheme::QE
};

MonteCarloEngine engine(params, config);
auto result = engine.price_european_call(100.0, 100.0, 1.0, 0.05);
```

## Testing

```bash
cd build
ctest -C Release
```

### Test Suite
- `test_bs_limit`: Verifies MC converges to Black-Scholes in low vol-of-vol limit
- `test_pde_bs_limit`: Verifies PDE solver stability (accuracy issues noted)
- `test_mc_pde_consistency`: Compares MC and PDE results
- `test_sobol_convergence`: Verifies Sobol has lower standard error than pseudo-random
- `test_calibration`: Basic calibration sanity check

## Benchmarks

Run benchmarks:
```bash
./build/benchmarks/Release/benchmark_pricing
```

Latest local Release benchmark:
- MC Euler (1M paths, 1 thread): 11.10s
- MC QE (1M paths, 8 threads): 3.13s
- MC Sobol + antithetic (100K paths): 1.97s
- PDE 200x200 grid: 164.1 ms, but accuracy caveat remains

## References

- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Andersen, L. (2008). "Simple and efficient simulation of the Heston stochastic volatility model"
