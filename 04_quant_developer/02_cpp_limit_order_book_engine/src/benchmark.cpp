#include "matching_engine.h"
#include "cpu_affinity.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>

#undef max  // Windows.h defines max as a macro

using namespace lob;

struct LatencyStats {
    std::vector<uint64_t> latencies;
    
    void add(uint64_t latency_ns) {
        latencies.push_back(latency_ns);
    }
    
    void compute() {
        std::sort(latencies.begin(), latencies.end());
    }
    
    uint64_t mean() const {
        uint64_t sum = 0;
        for (auto l : latencies) sum += l;
        return sum / latencies.size();
    }
    
    uint64_t percentile(double p) const {
        size_t idx = static_cast<size_t>(latencies.size() * p);
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        return latencies[idx];
    }
    
    uint64_t max() const {
        return latencies.empty() ? 0 : latencies.back();
    }
};

void run_benchmark(size_t num_orders, const char* output_file) {
    if (!pin_to_core(0)) {
        std::cerr << "Warning: Failed to pin to core 0\n";
    }
    
    MatchingEngine engine;
    LatencyStats stats;
    
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<Price> price_dist(9000, 11000);
    std::uniform_int_distribution<Quantity> qty_dist(1, 1000);
    std::uniform_int_distribution<int> side_dist(0, 1);
    
    std::cout << "Warming up (10K orders)...\n";
    for (size_t i = 0; i < 10000; ++i) {
        Side side = side_dist(rng) == 0 ? Side::BUY : Side::SELL;
        Price price = price_dist(rng);
        Quantity qty = qty_dist(rng);
        engine.submit_order(i, side, price, qty);
    }
    
    engine.reset_stats();
    stats.latencies.reserve(num_orders);
    
    std::cout << "Running benchmark (" << num_orders << " orders)...\n";
    
    for (size_t i = 0; i < num_orders; ++i) {
        Side side = side_dist(rng) == 0 ? Side::BUY : Side::SELL;
        Price price = price_dist(rng);
        Quantity qty = qty_dist(rng);
        
        uint64_t start = get_timestamp_ns();
        engine.submit_order(10000 + i, side, price, qty);
        uint64_t end = get_timestamp_ns();
        
        stats.add(end - start);
    }
    
    stats.compute();
    
    std::cout << "\nResults:\n";
    std::cout << "  Mean:  " << stats.mean() << " ns\n";
    std::cout << "  p50:   " << stats.percentile(0.50) << " ns\n";
    std::cout << "  p90:   " << stats.percentile(0.90) << " ns\n";
    std::cout << "  p99:   " << stats.percentile(0.99) << " ns\n";
    std::cout << "  p99.9: " << stats.percentile(0.999) << " ns\n";
    std::cout << "  Max:   " << stats.max() << " ns\n";
    std::cout << "\nEngine stats:\n";
    std::cout << "  Orders:  " << engine.total_orders() << "\n";
    std::cout << "  Matches: " << engine.total_matches() << "\n";
    std::cout << "  Cancels: " << engine.total_cancels() << "\n";
    
    std::ofstream csv(output_file);
    csv << "metric,value_ns\n";
    csv << "mean," << stats.mean() << "\n";
    csv << "p50," << stats.percentile(0.50) << "\n";
    csv << "p90," << stats.percentile(0.90) << "\n";
    csv << "p99," << stats.percentile(0.99) << "\n";
    csv << "p99.9," << stats.percentile(0.999) << "\n";
    csv << "max," << stats.max() << "\n";
    csv.close();
    
    std::cout << "\nResults written to " << output_file << "\n";
}

int main(int argc, char** argv) {
    size_t num_orders = 1'000'000;
    const char* output = "benchmarks/sample_results.csv";
    
    if (argc > 1) {
        num_orders = std::stoull(argv[1]);
    }
    if (argc > 2) {
        output = argv[2];
    }
    
    run_benchmark(num_orders, output);
    return 0;
}
