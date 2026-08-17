# SDE / Quant-Dev Screening Pack

## Resume Positioning

- Built C++ Heston stochastic-volatility pricing engine with Monte Carlo Euler/QE schemes, Sobol sampling, antithetic variates, and calibration smoke tests.
- Verified Black-Scholes limit behavior with MC price 10.4173 vs BS 10.4506, or 0.318% relative error.
- Added CMake/CTest coverage for BS limit, MC/PDE consistency smoke test, calibration smoke test, PDE stability, and Sobol convergence.
- Benchmarked Release build: MC Euler 1M paths in 11.10s; MC QE 1M paths on 8 threads in 3.13s; Sobol + antithetic 100K paths in 1.97s.
- Kept PDE solver limitation explicit because current PDE output has large pricing error versus MC/Black-Scholes.

## Interview Defense

Use this as a supporting quant-dev project, not the SDE flagship. The strongest parts are:

- Heston process implementation and MC path simulation.
- QE discretization and antithetic variance reduction.
- Multithreaded path pricing with reproducible test cases.
- Explicit model-limit tests instead of pretending every numerical method is accurate.
- Honest documentation of PDE weakness, which is better than overclaiming a broken solver.

## Verified Local Results

Windows/MSVC Release build:

```powershell
cmake -S . -B build_codex
cmake --build build_codex --config Release
ctest --test-dir build_codex -C Release --output-on-failure
```

Latest verification:

- `BSLimit`: passed
- `MCPDEConsistency`: passed as smoke test, but reports large PDE error
- `CalibrationSanity`: passed as bounded smoke test
- `PDEBSLimit`: passed as stability test, but reports large PDE error
- `SobolConvergence`: passed
- Full CTest runtime: 6.40s

## Honest Scope

The Monte Carlo engine is the usable part for resume framing. The PDE solver is not accurate enough to claim production-grade pricing. Present it as a numerical-methods project with verified MC behavior, tests, benchmarks, and an explicitly documented PDE limitation.

## Next Hardening Ideas

- Replace the PDE solver with a stable ADI/Hundsdorfer-Verwer scheme and validate against analytic Heston prices.
- Add characteristic-function Heston semi-closed-form pricing as a reference benchmark.
- Add implied-vol calibration against deterministic synthetic surfaces instead of stochastic MC-generated prices.
- Add CLI/JSON configuration for paths, steps, seed, discretization scheme, and thread count.
- Add benchmark CSV export with compiler, CPU, and seed metadata.
