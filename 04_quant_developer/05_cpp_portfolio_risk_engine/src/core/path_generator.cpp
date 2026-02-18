#include "path_generator.h"
#include <cmath>

namespace risk_engine {
namespace core {

PathGenerator::PathGenerator(const SimulationConfig& config, uint64_t seed, uint64_t stream_id)
    : config_(config) {
    
    if (config_.assets.empty()) {
        throw std::invalid_argument("Must have at least one asset");
    }
    
    if (config_.correlation_matrix.size() != config_.assets.size()) {
        throw std::invalid_argument("Correlation matrix size must match number of assets");
    }
    
    corr_rng_ = std::make_unique<CorrelatedRNG>(config_.correlation_matrix, seed, stream_id);
}

void PathGenerator::generate_paths(std::vector<std::vector<std::vector<double>>>& paths) {
    const size_t num_assets = config_.assets.size();
    const size_t num_paths = config_.num_paths;
    const size_t num_steps = config_.num_steps;
    const bool use_antithetic = config_.use_antithetic_variates;
    
    // Ensure even number of paths for antithetic variates
    const size_t effective_paths = use_antithetic ? (num_paths / 2) * 2 : num_paths;
    
    paths.resize(effective_paths);
    for (auto& path : paths) {
        path.resize(num_assets);
        for (auto& asset_path : path) {
            asset_path.resize(num_steps + 1);
        }
    }
    
    std::vector<double> dW(num_assets);
    
    if (use_antithetic) {
        // Generate paths in pairs using antithetic variates
        for (size_t p = 0; p < effective_paths; p += 2) {
            // Initialize both paths
            for (size_t a = 0; a < num_assets; ++a) {
                paths[p][a][0] = config_.assets[a].spot;
                paths[p + 1][a][0] = config_.assets[a].spot;
            }
            
            // Generate path pair with antithetic normals
            for (size_t t = 0; t < num_steps; ++t) {
                corr_rng_->generate_correlated_normals(dW);
                
                for (size_t a = 0; a < num_assets; ++a) {
                    const auto& asset = config_.assets[a];
                    double drift_term = (asset.drift - 0.5 * asset.volatility * asset.volatility) * config_.dt;
                    double vol_sqrt_dt = asset.volatility * std::sqrt(config_.dt);
                    
                    // Path 1: positive shock
                    double S1 = paths[p][a][t];
                    double diffusion1 = vol_sqrt_dt * dW[a];
                    paths[p][a][t + 1] = S1 * std::exp(drift_term + diffusion1);
                    
                    // Path 2: antithetic (negative shock)
                    double S2 = paths[p + 1][a][t];
                    double diffusion2 = vol_sqrt_dt * (-dW[a]);
                    paths[p + 1][a][t + 1] = S2 * std::exp(drift_term + diffusion2);
                }
            }
        }
    } else {
        // Standard independent paths
        for (size_t p = 0; p < num_paths; ++p) {
            for (size_t a = 0; a < num_assets; ++a) {
                paths[p][a][0] = config_.assets[a].spot;
            }
            
            for (size_t t = 0; t < num_steps; ++t) {
                corr_rng_->generate_correlated_normals(dW);
                
                for (size_t a = 0; a < num_assets; ++a) {
                    const auto& asset = config_.assets[a];
                    double S = paths[p][a][t];
                    double drift_term = (asset.drift - 0.5 * asset.volatility * asset.volatility) * config_.dt;
                    double diffusion_term = asset.volatility * std::sqrt(config_.dt) * dW[a];
                    paths[p][a][t + 1] = S * std::exp(drift_term + diffusion_term);
                }
            }
        }
    }
}

} // namespace core
} // namespace risk_engine
