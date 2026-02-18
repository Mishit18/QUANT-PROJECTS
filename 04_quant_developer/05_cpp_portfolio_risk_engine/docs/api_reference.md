# API Reference

## Core Module

### RNG

```cpp
class RNG {
    explicit RNG(uint64_t seed = 42);
    double normal();
    void fill_normal(double* buffer, size_t count);
};
```

### CorrelatedRNG

```cpp
class CorrelatedRNG {
    CorrelatedRNG(const std::vector<std::vector<double>>& correlation_matrix, 
                  uint64_t seed = 42);
    void generate_correlated_normals(std::vector<double>& output);
    size_t dimension() const;
};
```

### PathGenerator

```cpp
struct AssetConfig {
    double spot;
    double drift;
    double volatility;
};

struct SimulationConfig {
    size_t num_paths;
    size_t num_steps;
    double dt;
    std::vector<AssetConfig> assets;
    std::vector<std::vector<double>> correlation_matrix;
};

class PathGenerator {
    explicit PathGenerator(const SimulationConfig& config, uint64_t seed = 42);
    void generate_paths(std::vector<std::vector<std::vector<double>>>& paths);
};
```

## Engine Module

### ThreadPool

```cpp
class ThreadPool {
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<...>;
    
    void wait_all();
    size_t num_threads() const;
};
```

## Pricing Module

### OptionPricer

```cpp
enum class OptionType { Call, Put };

struct OptionSpec {
    OptionType type;
    double strike;
    double maturity;
};

class OptionPricer {
    static double payoff(const OptionSpec& spec, double spot);
    static double black_scholes(const OptionSpec& spec, double spot, 
                                double rate, double vol);
    static double delta(const OptionSpec& spec, double spot, 
                       double rate, double vol);
    static double gamma(const OptionSpec& spec, double spot, 
                       double rate, double vol);
    static double vega(const OptionSpec& spec, double spot, 
                      double rate, double vol);
};
```

## Portfolio Module

### Portfolio

```cpp
struct Position {
    std::string instrument_id;
    size_t asset_index;
    double quantity;
    OptionSpec option_spec;
};

class Portfolio {
    void add_position(const Position& pos);
    const std::vector<Position>& positions() const;
    double compute_pnl(const std::vector<double>& asset_prices) const;
    double compute_delta(size_t asset_idx, const std::vector<double>& asset_prices,
                        double rate, const std::vector<double>& vols) const;
    double compute_gamma(size_t asset_idx, const std::vector<double>& asset_prices,
                        double rate, const std::vector<double>& vols) const;
    double compute_vega(size_t asset_idx, const std::vector<double>& asset_prices,
                       double rate, const std::vector<double>& vols) const;
};
```

## Risk Module

### VaRCalculator

```cpp
struct RiskMetrics {
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double mean_pnl;
    double std_pnl;
    std::vector<double> portfolio_deltas;
    std::vector<double> portfolio_gammas;
    std::vector<double> portfolio_vegas;
};

class VaRCalculator {
    VaRCalculator(const Portfolio& portfolio,
                  const SimulationConfig& sim_config,
                  double risk_free_rate = 0.05);
    
    RiskMetrics compute_risk_metrics_single_threaded();
    RiskMetrics compute_risk_metrics_multithreaded(size_t num_threads);
    RiskMetrics compute_stress_test(double vol_shock, double corr_shock);
};
```

## Memory Module

### MemoryPool

```cpp
class MemoryPool {
    explicit MemoryPool(size_t block_size, size_t num_blocks);
    void* allocate();
    void deallocate(void* ptr);
    size_t block_size() const;
    size_t num_blocks() const;
};

template<typename T>
class PoolAllocator {
    explicit PoolAllocator(MemoryPool& pool);
    T* allocate(size_t n);
    void deallocate(T* ptr, size_t n);
};
```

## Utils Module

### Timer

```cpp
class Timer {
    Timer();
    void reset();
    double elapsed_ms() const;
    double elapsed_us() const;
    double elapsed_s() const;
};

class ScopedTimer {
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();  // Prints elapsed time
};
```

### Logger

```cpp
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
    static Logger& instance();
    void set_level(LogLevel level);
    
    template<typename... Args>
    void debug(Args&&... args);
    
    template<typename... Args>
    void info(Args&&... args);
    
    template<typename... Args>
    void warning(Args&&... args);
    
    template<typename... Args>
    void error(Args&&... args);
};
```
