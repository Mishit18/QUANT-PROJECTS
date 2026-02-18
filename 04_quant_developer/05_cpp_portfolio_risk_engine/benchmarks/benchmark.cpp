#include "../src/core/path_generator.h"
#include "../src/engine/thread_pool.h"
#include "../src/pricing/option.h"
#include "../src/portfolio/position.h"
#include "../src/risk/var_calculator.h"
#include "../src/utils/timer.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace risk_engine;

void benchmark_path_generation(size_t num_paths) {
    core::SimulationConfig config;
    config.num_paths = num_paths;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.assets = {{100.0, 0.05, 0.20}};
    config.correlation_matrix = {{1.0}};
    
    core::PathGenerator gen(config);
    std::vector<std::vector<std::vector<double>>> paths;
    
    utils::Timer timer;
    gen.generate_paths(paths);
    double elapsed = timer.elapsed_ms();
    
    std::cout << "Paths: " << num_paths << ", Time: " << elapsed << " ms, "
              << "Rate: " << (num_paths / elapsed * 1000.0) << " paths/sec" << std::endl;
}

void benchmark_threading_scalability() {
    core::SimulationConfig config;
    config.num_paths = 100000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.assets = {{100.0, 0.05, 0.20}, {150.0, 0.05, 0.25}};
    config.correlation_matrix = {{1.0, 0.5}, {0.5, 1.0}};
    
    portfolio::Portfolio portfolio;
    portfolio.add_position({"CALL", 0, 100.0, {pricing::OptionType::Call, 100.0, 1.0}});
    
    risk::VaRCalculator calc(portfolio, config);
    
    std::cout << "\nThreading Scalability:" << std::endl;
    std::cout << std::setw(10) << "Threads" << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup" << std::endl;
    
    utils::Timer timer;
    auto metrics_st = calc.compute_risk_metrics_single_threaded();
    double baseline = timer.elapsed_ms();
    
    std::cout << std::setw(10) << 1 << std::setw(15) << baseline 
              << std::setw(15) << "1.00x" << std::endl;
    
    for (size_t threads : {2, 4, 8, 16}) {
        if (threads > std::thread::hardware_concurrency()) break;
        
        timer.reset();
        auto metrics = calc.compute_risk_metrics_multithreaded(threads);
        double elapsed = timer.elapsed_ms();
        
        std::cout << std::setw(10) << threads << std::setw(15) << elapsed 
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << (baseline / elapsed) << "x" << std::endl;
    }
}

int main() {
    std::cout << "=== Portfolio Risk Engine Benchmarks ===" << std::endl;
    
    std::cout << "\nPath Generation Performance:" << std::endl;
    for (size_t n : {1000, 10000, 100000, 1000000}) {
        benchmark_path_generation(n);
    }
    
    benchmark_threading_scalability();
    
    return 0;
}
