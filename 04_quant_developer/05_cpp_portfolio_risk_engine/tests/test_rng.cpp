#include "../src/core/rng.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace risk_engine::core;

void test_basic_rng() {
    RNG rng(42);
    
    const size_t n = 10000;
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double x = rng.normal();
        sum += x;
        sum_sq += x * x;
    }
    
    double mean = sum / n;
    double variance = sum_sq / n - mean * mean;
    
    assert(std::abs(mean) < 0.05);
    assert(std::abs(variance - 1.0) < 0.05);
    
    std::cout << "Basic RNG test passed" << std::endl;
}

void test_stream_independence() {
    // Test that different streams produce different sequences
    RNG rng1(42, 0);
    RNG rng2(42, 1);
    
    double x1 = rng1.normal();
    double x2 = rng2.normal();
    
    // Different streams should produce different values
    assert(std::abs(x1 - x2) > 1e-10);
    
    std::cout << "Stream independence test passed" << std::endl;
}

void test_stream_determinism() {
    // Test that same stream produces same sequence
    RNG rng1a(42, 5);
    RNG rng1b(42, 5);
    
    for (int i = 0; i < 100; ++i) {
        double x1 = rng1a.normal();
        double x2 = rng1b.normal();
        assert(x1 == x2);  // Bitwise identical
    }
    
    std::cout << "Stream determinism test passed" << std::endl;
}

void test_correlated_rng() {
    std::vector<std::vector<double>> corr = {
        {1.0, 0.5},
        {0.5, 1.0}
    };
    
    CorrelatedRNG crng(corr, 42, 0);
    
    const size_t n = 10000;
    std::vector<double> sum(2, 0.0);
    double cov = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> samples;
        crng.generate_correlated_normals(samples);
        
        sum[0] += samples[0];
        sum[1] += samples[1];
        cov += samples[0] * samples[1];
    }
    
    double correlation = cov / n;
    
    assert(std::abs(correlation - 0.5) < 0.1);
    
    std::cout << "Correlated RNG test passed" << std::endl;
}

int main() {
    test_basic_rng();
    test_stream_independence();
    test_stream_determinism();
    test_correlated_rng();
    std::cout << "All RNG tests passed!" << std::endl;
    return 0;
}
