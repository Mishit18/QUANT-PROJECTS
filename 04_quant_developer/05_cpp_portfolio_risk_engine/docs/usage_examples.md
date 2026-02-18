# Usage Examples

## Basic Portfolio Simulation

```cpp
#include "core/path_generator.h"
#include "portfolio/position.h"
#include "risk/var_calculator.h"

using namespace risk_engine;

int main() {
    // Configure simulation
    core::SimulationConfig config;
    config.num_paths = 10000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    
    // Define assets
    config.assets = {
        {100.0, 0.05, 0.20},  // S0=100, drift=5%, vol=20%
        {150.0, 0.05, 0.25}   // S0=150, drift=5%, vol=25%
    };
    
    // Correlation matrix
    config.correlation_matrix = {
        {1.0, 0.6},
        {0.6, 1.0}
    };
    
    // Build portfolio
    portfolio::Portfolio port;
    port.add_position({
        "CALL_100", 0, 100.0,
        {pricing::OptionType::Call, 100.0, 1.0}
    });
    
    // Compute risk
    risk::VaRCalculator calc(port, config);
    auto metrics = calc.compute_risk_metrics_single_threaded();
    
    std::cout << "VaR (95%): " << metrics.var_95 << std::endl;
    std::cout << "CVaR (95%): " << metrics.cvar_95 << std::endl;
    
    return 0;
}
```

## Multi-Asset Portfolio

```cpp
portfolio::Portfolio port;

// Long call on asset 0
port.add_position({
    "LONG_CALL_0", 0, 100.0,
    {pricing::OptionType::Call, 100.0, 1.0}
});

// Short put on asset 1
port.add_position({
    "SHORT_PUT_1", 1, -50.0,
    {pricing::OptionType::Put, 150.0, 1.0}
});

// Long call on asset 2
port.add_position({
    "LONG_CALL_2", 2, 75.0,
    {pricing::OptionType::Call, 200.0, 0.5}
});
```

## Stress Testing

```cpp
risk::VaRCalculator calc(portfolio, config);

// 20% volatility shock, 10% correlation increase
auto stressed = calc.compute_stress_test(0.20, 0.10);

std::cout << "Base VaR: " << base_metrics.var_95 << std::endl;
std::cout << "Stressed VaR: " << stressed.var_95 << std::endl;
std::cout << "Increase: " << (stressed.var_95 - base_metrics.var_95) << std::endl;
```

## Greeks Computation

```cpp
auto metrics = calc.compute_risk_metrics_single_threaded();

for (size_t i = 0; i < metrics.portfolio_deltas.size(); ++i) {
    std::cout << "Asset " << i << ":" << std::endl;
    std::cout << "  Delta: " << metrics.portfolio_deltas[i] << std::endl;
    std::cout << "  Gamma: " << metrics.portfolio_gammas[i] << std::endl;
    std::cout << "  Vega: " << metrics.portfolio_vegas[i] << std::endl;
}
```

## Custom Correlation Structure

```cpp
// 3-asset portfolio with specific correlations
std::vector<std::vector<double>> corr = {
    {1.0, 0.7, 0.3},
    {0.7, 1.0, 0.5},
    {0.3, 0.5, 1.0}
};

config.correlation_matrix = corr;
```

## Performance Benchmarking

```cpp
#include "utils/timer.h"

utils::Timer timer;

// Single-threaded
auto metrics_st = calc.compute_risk_metrics_single_threaded();
double time_st = timer.elapsed_ms();

// Multi-threaded
timer.reset();
auto metrics_mt = calc.compute_risk_metrics_multithreaded(8);
double time_mt = timer.elapsed_ms();

std::cout << "Speedup: " << (time_st / time_mt) << "x" << std::endl;
```

## Memory Pool Usage

```cpp
#include "memory/pool_allocator.h"

// Create pool for 1000 blocks of 1KB each
memory::MemoryPool pool(1024, 1000);

// Use pool allocator
memory::PoolAllocator<double> alloc(pool);
std::vector<double, memory::PoolAllocator<double>> data(alloc);
```
