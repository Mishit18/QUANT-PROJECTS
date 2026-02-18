# Latency Modeling

## Overview

Latency modeling is critical for realistic market simulation. This document describes the latency models implemented and their use cases.

## Why Model Latency?

Real-world trading systems experience latency from multiple sources:
- Network propagation delay
- Switch/router queueing
- NIC processing
- Application processing
- OS scheduling

Accurate latency modeling enables:
- Realistic strategy backtesting
- Infrastructure capacity planning
- Performance regression detection
- Queueing theory validation

## Latency Model Interface

All latency models implement:

```cpp
class LatencyModel {
public:
    virtual uint64_t getLatency() = 0;  // Returns nanoseconds
};
```

## Implemented Models

### 1. Fixed Latency

**Use case**: Baseline testing, deterministic scenarios

```cpp
FixedLatencyModel(uint64_t latency_ns);
```

- Returns constant latency every call
- Simplest model, fully deterministic
- Useful for isolating matching engine performance

**Example**:
```cpp
auto model = std::make_shared<FixedLatencyModel>(1000);  // 1 microsecond
```

**Characteristics**:
- Mean = latency_ns
- Variance = 0
- p50 = p99 = p999 = latency_ns

### 2. Uniform Random Latency

**Use case**: Bounded jitter, simple randomness

```cpp
UniformLatencyModel(uint64_t min_ns, uint64_t max_ns);
```

- Returns random value in [min, max] range
- Uniform distribution (all values equally likely)
- Models simple network jitter

**Example**:
```cpp
auto model = std::make_shared<UniformLatencyModel>(500, 2000);  // 0.5-2 μs
```

**Characteristics**:
- Mean = (min + max) / 2
- Variance = (max - min)² / 12
- p50 ≈ (min + max) / 2

### 3. Normal Distribution Latency

**Use case**: Realistic statistical variation

```cpp
NormalLatencyModel(double mean_ns, double stddev_ns);
```

- Returns normally distributed latency
- Models aggregate effect of many independent sources
- Clipped at zero (no negative latency)

**Example**:
```cpp
auto model = std::make_shared<NormalLatencyModel>(1000, 300);  // 1μs ± 0.3μs
```

**Characteristics**:
- Mean = mean_ns
- Variance = stddev_ns²
- p50 ≈ mean_ns
- p99 ≈ mean_ns + 2.33 × stddev_ns

### 4. Queueing Latency

**Use case**: Burst traffic, load-dependent delay

```cpp
QueueingLatencyModel(uint64_t base_latency_ns, double load_factor);
```

- Models M/M/1 queue behavior (simplified)
- Latency increases with queue depth
- Simulates congestion under load

**Example**:
```cpp
auto model = std::make_shared<QueueingLatencyModel>(1000, 0.8);  // 80% load
```

**Characteristics**:
- Base latency + queueing delay
- Queueing delay ∝ queue_depth × load_factor
- Non-linear increase under high load

**Queueing Theory**:
- Load factor ρ = arrival_rate / service_rate
- Mean queue time = base_latency / (1 - ρ)
- At ρ = 0.8: ~5× base latency
- At ρ = 0.9: ~10× base latency

## Latency Injection Points

### Order Submission

```
User → [Latency] → Matching Engine
```

- Models network + application processing
- Applied before order enters matching engine
- Affects order arrival time

### Order Matching

```
Matching Engine → [Latency] → Fill Notification
```

- Models matching engine processing
- Applied after match decision
- Affects fill notification time

### Market Data

```
Exchange → [Latency] → Market Data Consumer
```

- Models market data dissemination
- Applied to price updates
- Affects strategy reaction time

## Measurement and Validation

### Latency Distribution

Measure and report:
- Mean (average)
- Median (p50)
- p90, p99, p999 (tail latency)
- Min/max (range)

### Validation Techniques

1. **Histogram**: Visual distribution check
2. **Q-Q Plot**: Compare to theoretical distribution
3. **Statistical Tests**: Kolmogorov-Smirnov, Anderson-Darling
4. **Autocorrelation**: Check for time-series dependencies

## Advanced Models (Future Work)

### Correlated Latency

Model geographic proximity:
```cpp
class CorrelatedLatencyModel : public LatencyModel {
    // Orders from same source have correlated latency
};
```

### Time-of-Day Effects

Model diurnal patterns:
```cpp
class TimeVaryingLatencyModel : public LatencyModel {
    // Higher latency during market open/close
};
```

### Network Topology

Model switch/router hops:
```cpp
class NetworkTopologyModel : public LatencyModel {
    // Latency based on network path
};
```

### Kernel Scheduling

Model OS scheduling delays:
```cpp
class SchedulingLatencyModel : public LatencyModel {
    // Occasional large delays (context switches)
};
```

## Calibration

### Data Sources

1. **Ping measurements**: Network RTT
2. **Timestamp analysis**: Exchange timestamps vs. local
3. **Vendor data**: Latency specs from providers
4. **Empirical testing**: Measure your own infrastructure

### Calibration Process

1. Collect latency samples (1M+ data points)
2. Fit distribution (normal, log-normal, Weibull)
3. Validate fit (Q-Q plot, statistical tests)
4. Implement model with fitted parameters
5. Verify simulation matches empirical data

### Example Calibration

```python
import numpy as np
from scipy import stats

# Collect samples
samples = measure_latency(n=1000000)

# Fit normal distribution
mean, std = stats.norm.fit(samples)

# Validate fit
ks_stat, p_value = stats.kstest(samples, 'norm', args=(mean, std))

# Use in simulator
model = NormalLatencyModel(mean, std)
```

## Performance Considerations

### RNG Performance

- `std::mt19937_64`: ~10ns per call
- `std::uniform_int_distribution`: ~20ns per call
- `std::normal_distribution`: ~50ns per call

For high-frequency simulation, consider:
- Pre-generate random numbers
- Use faster RNG (xorshift, PCG)
- Batch generation with SIMD

### Determinism vs. Realism

Trade-off:
- **Deterministic**: Fixed seed, reproducible
- **Realistic**: True randomness, non-reproducible

Recommendation:
- Use fixed seed for testing/debugging
- Use random seed for production-like scenarios
- Always log seed for reproducibility

## Benchmarking Latency Models

### Throughput Test

```cpp
auto model = std::make_shared<NormalLatencyModel>(1000, 300);
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < 1000000; ++i) {
    volatile uint64_t latency = model->getLatency();
}

auto end = std::chrono::high_resolution_clock::now();
// Report: calls per second
```

### Distribution Test

```cpp
std::vector<uint64_t> samples;
for (int i = 0; i < 100000; ++i) {
    samples.push_back(model->getLatency());
}

// Compute percentiles
std::sort(samples.begin(), samples.end());
double p50 = samples[50000];
double p99 = samples[99000];
double p999 = samples[99900];
```

## Real-World Latency Ranges

### Co-located Trading

- Median: 10-50 microseconds
- p99: 100-500 microseconds
- p999: 1-5 milliseconds

### Cross-venue Arbitrage

- Median: 100-500 microseconds
- p99: 1-5 milliseconds
- p999: 10-50 milliseconds

### Retail Trading

- Median: 10-100 milliseconds
- p99: 100-500 milliseconds
- p999: 1-5 seconds

## Conclusion

Latency modeling is essential for realistic simulation. Key principles:

1. **Choose appropriate model**: Fixed for testing, statistical for realism
2. **Calibrate from data**: Measure real-world latency
3. **Validate distributions**: Ensure model matches reality
4. **Report tail latency**: p99/p999 matter more than mean
5. **Consider load effects**: Queueing delays under burst traffic

The implemented models provide a foundation for research-grade simulation. Production systems require more sophisticated models calibrated to specific infrastructure.
