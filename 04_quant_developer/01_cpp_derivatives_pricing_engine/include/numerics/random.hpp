#pragma once

#include <random>
#include <vector>
#include <memory>
#include <cstdint>

namespace heston {

class RandomNumberGenerator {
public:
    virtual ~RandomNumberGenerator() = default;
    virtual double next() = 0;
    virtual void next_pair(double& z1, double& z2) = 0;
    virtual void reset() = 0;
    virtual std::unique_ptr<RandomNumberGenerator> clone() const = 0;
};

class MersenneTwisterRNG : public RandomNumberGenerator {
public:
    explicit MersenneTwisterRNG(uint64_t seed = 12345)
        : gen_(seed), dist_(0.0, 1.0) {}
    
    double next() override {
        return dist_(gen_);
    }
    
    void next_pair(double& z1, double& z2) override {
        z1 = dist_(gen_);
        z2 = dist_(gen_);
    }
    
    void reset() override {
        gen_.seed(12345);
    }
    
    std::unique_ptr<RandomNumberGenerator> clone() const override {
        return std::make_unique<MersenneTwisterRNG>(*this);
    }
    
private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> dist_;
};

class SobolRNG : public RandomNumberGenerator {
public:
    explicit SobolRNG(size_t dimension = 2);
    
    double next() override;
    void next_pair(double& z1, double& z2) override;
    void reset() override;
    std::unique_ptr<RandomNumberGenerator> clone() const override;
    
private:
    void generate_next_point();
    double box_muller_transform(double u) const;
    
    size_t dimension_;
    size_t index_;
    std::vector<uint32_t> direction_numbers_;
    std::vector<uint32_t> current_point_;
    std::vector<double> cached_normals_;
    size_t cache_index_;
};

} // namespace heston
