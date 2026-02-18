#include "../core/Simulator.h"
#include "../core/Clock.h"
#include "../latency/LatencyModel.h"
#include "../replay/MarketDataReplayer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <unordered_map>

using namespace simulator;

struct LatencyMeasurement {
    uint64_t order_id;
    uint64_t submit_time;
    uint64_t fill_time;
    uint64_t latency_ns;
};

class LatencyDistributionBenchmark {
public:
    void run() {
        std::cout << "=================================================\n";
        std::cout << "LATENCY DISTRIBUTION BENCHMARK\n";
        std::cout << "=================================================\n\n";
        
        runScenario("Normal Traffic (10K orders)", 10000, 1000000);
        runScenario("Burst Traffic (100K orders)", 100000, 1000);
    }

private:
    void runScenario(const std::string& name, size_t order_count, uint64_t time_delta) {
        std::cout << "Scenario: " << name << "\n";
        std::cout << "-------------------------------------------------\n";
        
        // Track submission times
        std::unordered_map<uint64_t, uint64_t> submit_times;
        std::vector<LatencyMeasurement> measurements;
        
        // Create simulator with fixed latency for baseline
        auto latency_model = std::make_shared<FixedLatencyModel>(1000);
        Simulator sim(latency_model);
        
        // Generate orders
        MarketDataReplayer replayer;
        replayer.generateSyntheticOrders(order_count, "AAPL", 150.0, time_delta);
        
        // Record submit times
        for (const auto& event : replayer.getEvents()) {
            if (event->getType() == EventType::ORDER_SUBMIT) {
                auto order_event = std::static_pointer_cast<OrderSubmitEvent>(event);
                submit_times[order_event->getOrderId()] = event->getTimestamp();
            }
        }
        
        // Register fill callback to measure latency
        sim.onFill([&](const OrderFillEvent& fill) {
            auto it = submit_times.find(fill.getOrderId());
            if (it != submit_times.end()) {
                LatencyMeasurement m;
                m.order_id = fill.getOrderId();
                m.submit_time = it->second;
                m.fill_time = fill.getTimestamp();
                m.latency_ns = m.fill_time - m.submit_time;
                measurements.push_back(m);
            }
        });
        
        // Schedule events
        for (const auto& event : replayer.getEvents()) {
            sim.scheduleEvent(event);
        }
        
        // Run simulation
        auto wall_start = WallClock::now();
        sim.run();
        auto wall_end = WallClock::now();
        
        // Compute statistics
        if (measurements.empty()) {
            std::cout << "No fills recorded\n\n";
            return;
        }
        
        std::vector<uint64_t> latencies;
        for (const auto& m : measurements) {
            latencies.push_back(m.latency_ns);
        }
        std::sort(latencies.begin(), latencies.end());
        
        // Print results
        std::cout << "Orders submitted: " << order_count << "\n";
        std::cout << "Fills recorded:   " << measurements.size() << "\n";
        std::cout << "Wall-clock time:  " << std::fixed << std::setprecision(2)
                  << WallClock::elapsedMicros(wall_start, wall_end) / 1000.0 << " ms\n\n";
        
        std::cout << "Latency Distribution (ns):\n";
        std::cout << "  p50   = " << percentile(latencies, 0.50) << "\n";
        std::cout << "  p90   = " << percentile(latencies, 0.90) << "\n";
        std::cout << "  p99   = " << percentile(latencies, 0.99) << "\n";
        std::cout << "  p999  = " << percentile(latencies, 0.999) << "\n";
        std::cout << "  max   = " << latencies.back() << "\n";
        std::cout << "\n";
    }
    
    uint64_t percentile(const std::vector<uint64_t>& sorted_data, double p) {
        if (sorted_data.empty()) return 0;
        size_t idx = static_cast<size_t>(p * sorted_data.size());
        if (idx >= sorted_data.size()) idx = sorted_data.size() - 1;
        return sorted_data[idx];
    }
};

int main() {
    LatencyDistributionBenchmark benchmark;
    benchmark.run();
    return 0;
}
