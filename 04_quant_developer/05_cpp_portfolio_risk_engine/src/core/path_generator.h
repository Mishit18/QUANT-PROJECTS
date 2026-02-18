#pragma once

#include "rng.h"
#include <vector>
#include <memory>

namespace risk_engine {
namespace core {

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
    bool use_antithetic_variates = true;  // Variance reduction enabled by default
};

class PathGenerator {
public:
    explicit PathGenerator(const SimulationConfig& config, uint64_t seed = 42, uint64_t stream_id = 0);
    
    void generate_paths(std::vector<std::vector<std::vector<double>>>& paths);
    
    const SimulationConfig& config() const { return config_; }
    
private:
    SimulationConfig config_;
    std::unique_ptr<CorrelatedRNG> corr_rng_;
};

} // namespace core
} // namespace risk_engine
