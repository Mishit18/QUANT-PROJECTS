#include "../src/core/path_generator.h"
#include "../src/portfolio/position.h"
#include "../src/pricing/option.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace risk_engine;

double compute_variance(const std::vector<double>& values) {
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    return sq_sum / values.size();
}

void test_antithetic_variance_reduction() {
    // Configure simulation
    core::SimulationConfig config;
    config.num_paths = 10000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.assets = {{100.0, 0.05, 0.20}};
    config.correlation_matrix = {{1.0}};
    
    // Test without antithetic variates
    config.use_antithetic_variates = false;
    core::PathGenerator gen_standard(config, 42);
    std::vector<std::vector<std::vector<double>>> paths_standard;
    gen_standard.generate_paths(paths_standard);
    
    // Compute terminal values
    std::vector<double> terminal_standard;
    for (const auto& path : paths_standard) {
        terminal_standard.push_back(path[0].back());
    }
    double var_standard = compute_variance(terminal_standard);
    
    // Test with antithetic variates
    config.use_antithetic_variates = true;
    core::PathGenerator gen_antithetic(config, 42);
    std::vector<std::vector<std::vector<double>>> paths_antithetic;
    gen_antithetic.generate_paths(paths_antithetic);
    
    // Compute terminal values
    std::vector<double> terminal_antithetic;
    for (const auto& path : paths_antithetic) {
        terminal_antithetic.push_back(path[0].back());
    }
    double var_antithetic = compute_variance(terminal_antithetic);
    
    // Antithetic variates should reduce variance
    double reduction_ratio = var_standard / var_antithetic;
    
    std::cout << "Variance (standard): " << var_standard << std::endl;
    std::cout << "Variance (antithetic): " << var_antithetic << std::endl;
    std::cout << "Reduction ratio: " << reduction_ratio << "x" << std::endl;
    
    // For European options, expect ~1.5-2x reduction
    assert(reduction_ratio > 1.3);
    assert(reduction_ratio < 2.5);
    
    std::cout << "Antithetic variance reduction test passed" << std::endl;
}

void test_antithetic_determinism() {
    // Test that antithetic variates produce deterministic results
    core::SimulationConfig config;
    config.num_paths = 1000;
    config.num_steps = 100;
    config.dt = 1.0 / 252.0;
    config.assets = {{100.0, 0.05, 0.20}};
    config.correlation_matrix = {{1.0}};
    config.use_antithetic_variates = true;
    
    // Generate paths twice with same seed
    core::PathGenerator gen1(config, 42);
    std::vector<std::vector<std::vector<double>>> paths1;
    gen1.generate_paths(paths1);
    
    core::PathGenerator gen2(config, 42);
    std::vector<std::vector<std::vector<double>>> paths2;
    gen2.generate_paths(paths2);
    
    // Verify bitwise identical
    assert(paths1.size() == paths2.size());
    for (size_t p = 0; p < paths1.size(); ++p) {
        for (size_t a = 0; a < paths1[p].size(); ++a) {
            for (size_t t = 0; t < paths1[p][a].size(); ++t) {
                assert(paths1[p][a][t] == paths2[p][a][t]);
            }
        }
    }
    
    std::cout << "Antithetic determinism test passed" << std::endl;
}

int main() {
    test_antithetic_variance_reduction();
    test_antithetic_determinism();
    std::cout << "All variance reduction tests passed!" << std::endl;
    return 0;
}
