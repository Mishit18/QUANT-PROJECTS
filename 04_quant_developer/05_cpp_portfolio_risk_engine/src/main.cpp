#include "core/path_generator.h"
#include "engine/thread_pool.h"
#include "pricing/option.h"
#include "portfolio/position.h"
#include "risk/var_calculator.h"
#include "utils/timer.h"
#include <iostream>
#include <iomanip>

using namespace risk_engine;

void print_risk_metrics(const risk::RiskMetrics& metrics, const std::string& label) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mean PnL:     $" << metrics.mean_pnl << std::endl;
    std::cout << "Std PnL:      $" << metrics.std_pnl << std::endl;
    std::cout << "VaR (95%):    $" << metrics.var_95 << " ± $" << metrics.var_95_stderr << std::endl;
    std::cout << "VaR (99%):    $" << metrics.var_99 << " ± $" << metrics.var_99_stderr << std::endl;
    std::cout << "CVaR (95%):   $" << metrics.cvar_95 << " ± $" << metrics.cvar_95_stderr << std::endl;
    std::cout << "CVaR (99%):   $" << metrics.cvar_99 << " ± $" << metrics.cvar_99_stderr << std::endl;
    
    std::cout << "\nGreeks:" << std::endl;
    for (size_t i = 0; i < metrics.portfolio_deltas.size(); ++i) {
        std::cout << "  Asset " << i << " - Delta: " << metrics.portfolio_deltas[i]
                  << ", Gamma: " << metrics.portfolio_gammas[i]
                  << ", Vega: " << metrics.portfolio_vegas[i] << std::endl;
    }
}

int main() {
    std::cout << "Portfolio Risk Engine - Monte Carlo Simulation\n" << std::endl;
    
    core::SimulationConfig sim_config;
    sim_config.num_paths = 100000;
    sim_config.num_steps = 252;
    sim_config.dt = 1.0 / 252.0;
    
    sim_config.assets = {
        {100.0, 0.05, 0.20},
        {150.0, 0.05, 0.25},
        {200.0, 0.05, 0.30}
    };
    
    sim_config.correlation_matrix = {
        {1.0, 0.5, 0.3},
        {0.5, 1.0, 0.4},
        {0.3, 0.4, 1.0}
    };
    
    portfolio::Portfolio portfolio;
    portfolio.add_position({
        "CALL_ASSET0_100", 0, 100.0,
        {pricing::OptionType::Call, 100.0, 1.0}
    });
    portfolio.add_position({
        "PUT_ASSET1_150", 1, -50.0,
        {pricing::OptionType::Put, 150.0, 1.0}
    });
    portfolio.add_position({
        "CALL_ASSET2_200", 2, 75.0,
        {pricing::OptionType::Call, 200.0, 1.0}
    });
    
    risk::VaRCalculator var_calc(portfolio, sim_config, 0.05);
    
    std::cout << "Running single-threaded simulation..." << std::endl;
    utils::Timer timer;
    auto metrics_st = var_calc.compute_risk_metrics_single_threaded();
    double time_st = timer.elapsed_ms();
    print_risk_metrics(metrics_st, "Single-Threaded Results");
    std::cout << "\nExecution time: " << time_st << " ms" << std::endl;
    
    size_t num_threads = std::thread::hardware_concurrency();
    std::cout << "\n\nRunning multi-threaded simulation (" << num_threads << " threads)..." << std::endl;
    timer.reset();
    auto metrics_mt = var_calc.compute_risk_metrics_multithreaded(num_threads);
    double time_mt = timer.elapsed_ms();
    print_risk_metrics(metrics_mt, "Multi-Threaded Results");
    std::cout << "\nExecution time: " << time_mt << " ms" << std::endl;
    std::cout << "Speedup: " << (time_st / time_mt) << "x" << std::endl;
    
    std::cout << "\n\nRunning stress test (20% vol shock, 10% corr shock)..." << std::endl;
    timer.reset();
    auto metrics_stress = var_calc.compute_stress_test(0.20, 0.10);
    double time_stress = timer.elapsed_ms();
    print_risk_metrics(metrics_stress, "Stress Test Results");
    std::cout << "\nExecution time: " << time_stress << " ms" << std::endl;
    
    return 0;
}
