#pragma once

#include <cstdint>
#include <chrono>

namespace simulator {

// Logical clock for deterministic simulation
// Decoupled from wall-clock time
class SimulationClock {
public:
    SimulationClock() : current_time_(0) {}
    
    // Get current logical time (nanoseconds)
    uint64_t now() const {
        return current_time_;
    }
    
    // Advance clock to specific time
    void advanceTo(uint64_t timestamp) {
        if (timestamp >= current_time_) {
            current_time_ = timestamp;
        }
    }
    
    // Advance clock by delta
    void advance(uint64_t delta_ns) {
        current_time_ += delta_ns;
    }
    
    // Reset clock
    void reset() {
        current_time_ = 0;
    }
    
    // Convert to microseconds for readability
    double toMicroseconds() const {
        return current_time_ / 1000.0;
    }
    
    // Convert to milliseconds
    double toMilliseconds() const {
        return current_time_ / 1000000.0;
    }

private:
    uint64_t current_time_;
};

// Wall-clock timer for benchmarking
class WallClock {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    static TimePoint now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    static uint64_t elapsedNanos(TimePoint start, TimePoint end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
    }
    
    static double elapsedMicros(TimePoint start, TimePoint end) {
        return elapsedNanos(start, end) / 1000.0;
    }
};

} // namespace simulator
