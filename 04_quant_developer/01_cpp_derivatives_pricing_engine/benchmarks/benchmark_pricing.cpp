#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include "pde/pde_solver.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace heston {

template<typename Func>
double time_execution(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void benchmark_monte_carlo() {
    std::cout << "Monte Carlo Benchmarks\n";
    std::cout << "======================\n\n";
    
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.0;
    
    HestonParameters params;
    params.kappa = 2.0;
    params.theta = 0.04;
    params.sigma = 0.3;
    params.rho = -0.7;
    params.v0 = 0.04;
    
    std::cout << "Euler Scheme (1 thread, 1M paths):\n";
    {
        MCConfig config;
        config.num_paths = 1000000;
        config.num_steps = 252;
        config.use_antithetic = false;
        config.num_threads = 1;
        config.scheme = MCConfig::Scheme::Euler;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << "  Time:  " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
        std::cout << "  Price: " << std::setprecision(4) << result.price << "\n\n";
    }
    
    std::cout << "QE Scheme (8 threads, 1M paths):\n";
    {
        MCConfig config;
        config.num_paths = 1000000;
        config.num_steps = 252;
        config.use_antithetic = false;
        config.num_threads = 8;
        config.scheme = MCConfig::Scheme::QE;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << "  Time:  " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
        std::cout << "  Price: " << std::setprecision(4) << result.price << "\n\n";
    }
    
    std::cout << "Sobol + Antithetic (100K paths):\n";
    {
        MCConfig config;
        config.num_paths = 100000;
        config.num_steps = 252;
        config.use_antithetic = true;
        config.use_sobol = true;
        config.num_threads = 4;
        config.scheme = MCConfig::Scheme::QE;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << "  Time:  " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
        std::cout << "  Price: " << std::setprecision(4) << result.price << "\n\n";
    }
}

void benchmark_pde() {
    std::cout << "PDE Solver Benchmarks\n";
    std::cout << "=====================\n\n";
    
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.0;
    
    HestonParameters params;
    params.kappa = 2.0;
    params.theta = 0.04;
    params.sigma = 0.3;
    params.rho = -0.7;
    params.v0 = 0.04;
    
    std::cout << "Crank-Nicolson (200x200 grid):\n";
    {
        PDEConfig config;
        config.num_spot_steps = 200;
        config.num_var_steps = 200;
        config.theta = 0.5;
        
        CrankNicolsonSolver solver(params, config);
        PDEResult result;
        double elapsed = time_execution([&]() {
            result = solver.solve(S0, K, T, r, q, OptionType::Call);
        });
        
        std::cout << "  Time:  " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
        std::cout << "  Price: " << std::setprecision(4) << result.price << "\n";
        std::cout << "  Delta: " << std::setprecision(4) << result.delta << "\n";
        std::cout << "  Gamma: " << std::setprecision(4) << result.gamma << "\n";
        std::cout << "  Vega:  " << std::setprecision(4) << result.vega << "\n\n";
    }
}

void benchmark_comparison() {
    std::cout << "Method Comparison\n";
    std::cout << "=================\n\n";
    std::cout << std::left << std::setw(30) << "Method" 
              << std::setw(12) << "Paths/Grid" 
              << std::setw(12) << "Time (ms)" 
              << std::setw(12) << "Price" << "\n";
    std::cout << std::string(66, '-') << "\n";
    
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.0;
    
    HestonParameters params;
    params.kappa = 2.0;
    params.theta = 0.04;
    params.sigma = 0.3;
    params.rho = -0.7;
    params.v0 = 0.04;
    
    {
        MCConfig config;
        config.num_paths = 1000000;
        config.num_steps = 252;
        config.num_threads = 1;
        config.scheme = MCConfig::Scheme::Euler;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << std::left << std::setw(30) << "MC (Euler, 1 thread)" 
                  << std::setw(12) << "1M" 
                  << std::setw(12) << std::fixed << std::setprecision(0) << elapsed
                  << std::setw(12) << std::setprecision(4) << result.price << "\n";
    }
    
    {
        MCConfig config;
        config.num_paths = 1000000;
        config.num_steps = 252;
        config.num_threads = 8;
        config.scheme = MCConfig::Scheme::QE;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << std::left << std::setw(30) << "MC (QE, 8 threads)" 
                  << std::setw(12) << "1M" 
                  << std::setw(12) << std::fixed << std::setprecision(0) << elapsed
                  << std::setw(12) << std::setprecision(4) << result.price << "\n";
    }
    
    {
        MCConfig config;
        config.num_paths = 100000;
        config.num_steps = 252;
        config.use_antithetic = true;
        config.use_sobol = true;
        config.num_threads = 4;
        config.scheme = MCConfig::Scheme::QE;
        
        MonteCarloEngine engine(params, config);
        MCResult result;
        double elapsed = time_execution([&]() {
            result = engine.price_european_call(S0, K, T, r, q);
        });
        
        std::cout << std::left << std::setw(30) << "MC (Sobol + AV)" 
                  << std::setw(12) << "100K" 
                  << std::setw(12) << std::fixed << std::setprecision(0) << elapsed
                  << std::setw(12) << std::setprecision(4) << result.price << "\n";
    }
    
    {
        PDEConfig config;
        config.num_spot_steps = 200;
        config.num_var_steps = 200;
        config.theta = 0.5;
        
        CrankNicolsonSolver solver(params, config);
        PDEResult result;
        double elapsed = time_execution([&]() {
            result = solver.solve(S0, K, T, r, q, OptionType::Call);
        });
        
        std::cout << std::left << std::setw(30) << "PDE (CN)" 
                  << std::setw(12) << "200x200" 
                  << std::setw(12) << std::fixed << std::setprecision(0) << elapsed
                  << std::setw(12) << std::setprecision(4) << result.price << "\n";
    }
    
    std::cout << "\n";
}

} // namespace heston

int main() {
    heston::benchmark_monte_carlo();
    heston::benchmark_pde();
    heston::benchmark_comparison();
    return 0;
}
