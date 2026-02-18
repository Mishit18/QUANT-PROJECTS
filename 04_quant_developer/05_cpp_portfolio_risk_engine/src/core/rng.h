#pragma once

#include <random>
#include <vector>
#include <cmath>

namespace risk_engine {
namespace core {

class RNG {
public:
    explicit RNG(uint64_t seed = 42, uint64_t stream_id = 0) 
        : gen_(compute_stream_seed(seed, stream_id)), dist_(0.0, 1.0) {}
    
    double normal() {
        return dist_(gen_);
    }
    
    void fill_normal(double* buffer, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = dist_(gen_);
        }
    }
    
private:
    static uint64_t compute_stream_seed(uint64_t base_seed, uint64_t stream_id) {
        // Use SplitMix64 hash to generate independent streams
        // Ensures no overlap between streams regardless of thread count
        uint64_t z = (base_seed + stream_id * 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    
    std::mt19937_64 gen_;
    std::normal_distribution<double> dist_;
};

class CorrelatedRNG {
public:
    CorrelatedRNG(const std::vector<std::vector<double>>& correlation_matrix, 
                  uint64_t seed = 42, uint64_t stream_id = 0);
    
    void generate_correlated_normals(std::vector<double>& output);
    
    size_t dimension() const { return dim_; }
    
private:
    void compute_cholesky(const std::vector<std::vector<double>>& corr);
    
    size_t dim_;
    std::vector<double> cholesky_lower_;
    RNG rng_;
};

} // namespace core
} // namespace risk_engine
