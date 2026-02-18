#pragma once

#include <cstdint>
#include <random>
#include <memory>

namespace simulator {

// Base latency model interface
class LatencyModel {
public:
    virtual ~LatencyModel() = default;
    virtual uint64_t getLatency() = 0;  // Returns latency in nanoseconds
};

// Fixed latency model
class FixedLatencyModel : public LatencyModel {
public:
    explicit FixedLatencyModel(uint64_t latency_ns)
        : latency_ns_(latency_ns) {}
    
    uint64_t getLatency() override {
        return latency_ns_;
    }

private:
    uint64_t latency_ns_;
};

// Uniform random latency model
class UniformLatencyModel : public LatencyModel {
public:
    UniformLatencyModel(uint64_t min_ns, uint64_t max_ns)
        : distribution_(min_ns, max_ns),
          generator_(std::random_device{}()) {}
    
    uint64_t getLatency() override {
        return distribution_(generator_);
    }

private:
    std::uniform_int_distribution<uint64_t> distribution_;
    std::mt19937_64 generator_;
};

// Normal distribution latency model
class NormalLatencyModel : public LatencyModel {
public:
    NormalLatencyModel(double mean_ns, double stddev_ns)
        : distribution_(mean_ns, stddev_ns),
          generator_(std::random_device{}()) {}
    
    uint64_t getLatency() override {
        double latency = distribution_(generator_);
        return static_cast<uint64_t>(std::max(0.0, latency));
    }

private:
    std::normal_distribution<double> distribution_;
    std::mt19937_64 generator_;
};


// Queueing delay model (simulates burst traffic effects)
class QueueingLatencyModel : public LatencyModel {
public:
    QueueingLatencyModel(uint64_t base_latency_ns, double load_factor)
        : base_latency_ns_(base_latency_ns),
          load_factor_(load_factor),
          queue_depth_(0) {}
    
    uint64_t getLatency() override {
        // Simple M/M/1 queue approximation
        // Latency increases with queue depth
        uint64_t queueing_delay = static_cast<uint64_t>(
            queue_depth_ * base_latency_ns_ * load_factor_);
        
        // Simulate queue dynamics (simplified)
        if (queue_depth_ > 0) {
            queue_depth_--;
        }
        if (load_factor_ > 0.5) {
            queue_depth_++;
        }
        
        return base_latency_ns_ + queueing_delay;
    }

private:
    uint64_t base_latency_ns_;
    double load_factor_;
    uint64_t queue_depth_;
};

} // namespace simulator
