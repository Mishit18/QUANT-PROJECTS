#include "../core/Simulator.h"
#include "../core/Clock.h"
#include "../latency/LatencyModel.h"
#include "../replay/MarketDataReplayer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace simulator;

struct LatencyStats {
    std::vector<uint64_t> latencies;
    
    void record(uint64_t latency_ns) {
        latencies.push_back(latency_ns);
    }
    
    void compute() {
        if (latencies.empty()) return;
        std::sort(latencies.begin(), latencies.end());
    }
    
    double percentile(double p) const {
        if (latencies.empty()) return 0.0;
        size_t idx = static_cast<size_t>(p * latencies.size());
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        return latencies[idx] / 1000.0;  // Convert to microseconds
    }
    
    double mean() const {
        if (latencies.empty()) return 0.0;
        uint64_t sum = 0;
        for (auto l : latencies) sum += l;
        return (sum / latencies.size()) / 1000.0;
    }
};

void runBenchmark(const std::string& scenario, size_t num_orders,
                  std::shared_ptr<LatencyModel> latency_model) {
    std::cout << "\n=== " << scenario << " ===" << std::endl;
    
    Simulator sim(latency_model);
    LatencyStats stats;
    
    // Generate synthetic orders
    MarketDataReplayer replayer;
    replayer.generateSyntheticOrders(num_orders, "AAPL", 150.0, 1000);
    
    // Measure wall-clock time
    auto start = WallClock::now();
    
    // Schedule all events
    for (const auto& event : replayer.getEvents()) {
        sim.scheduleEvent(event);
    }
    
    
    // Run simulation
    sim.run();
    
    auto end = WallClock::now();
    double elapsed_ms = WallClock::elapsedMicros(start, end) / 1000.0;
    
    std::cout << "Orders processed: " << num_orders << std::endl;
    std::cout << "Wall-clock time: " << std::fixed << std::setprecision(2)
              << elapsed_ms << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0)
              << (num_orders / elapsed_ms * 1000.0) << " orders/sec" << std::endl;
}

void exportResults(const std::string& filename, const LatencyStats& stats) {
    std::ofstream file(filename);
    file << "metric,value_us\n";
    file << "mean," << stats.mean() << "\n";
    file << "p50," << stats.percentile(0.50) << "\n";
    file << "p90," << stats.percentile(0.90) << "\n";
    file << "p99," << stats.percentile(0.99) << "\n";
    file << "p999," << stats.percentile(0.999) << "\n";
    file.close();
}

int main() {
    std::cout << "Event-Driven Market Simulator - Latency Benchmark\n";
    std::cout << "==================================================\n";
    
    // Scenario 1: Fixed latency, normal load
    auto fixed_model = std::make_shared<FixedLatencyModel>(1000);  // 1 microsecond
    runBenchmark("Fixed Latency (1us) - 10K orders", 10000, fixed_model);
    
    // Scenario 2: Uniform random latency
    auto uniform_model = std::make_shared<UniformLatencyModel>(500, 2000);
    runBenchmark("Uniform Latency (0.5-2us) - 10K orders", 10000, uniform_model);
    
    // Scenario 3: Normal distribution latency
    auto normal_model = std::make_shared<NormalLatencyModel>(1000, 300);
    runBenchmark("Normal Latency (1us ± 0.3us) - 10K orders", 10000, normal_model);
    
    // Scenario 4: High load with queueing
    auto queue_model = std::make_shared<QueueingLatencyModel>(1000, 0.8);
    runBenchmark("Queueing Model (load=0.8) - 10K orders", 10000, queue_model);
    
    // Scenario 5: Burst traffic
    auto burst_model = std::make_shared<FixedLatencyModel>(1000);
    runBenchmark("Burst Traffic - 100K orders", 100000, burst_model);
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}
