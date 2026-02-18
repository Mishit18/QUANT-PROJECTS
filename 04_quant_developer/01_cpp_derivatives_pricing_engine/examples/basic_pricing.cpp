#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include "pde/pde_solver.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace heston;
    
    HestonParameters params;
    params.kappa = 2.0;
    params.theta = 0.04;
    params.sigma = 0.3;
    params.rho = -0.7;
    params.v0 = 0.04;
    
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.0;
    
    std::cout << "Heston Model Parameters:\n";
    std::cout << "  kappa (mean reversion): " << params.kappa << "\n";
    std::cout << "  theta (long-term var):  " << params.theta << "\n";
    std::cout << "  sigma (vol of vol):     " << params.sigma << "\n";
    std::cout << "  rho (correlation):      " << params.rho << "\n";
    std::cout << "  v0 (initial var):       " << params.v0 << "\n";
    std::cout << "  Feller condition:       " << (params.satisfies_feller() ? "Satisfied" : "Not satisfied") << "\n\n";
    
    std::cout << "Market Parameters:\n";
    std::cout << "  Spot:       " << S0 << "\n";
    std::cout << "  Strike:     " << K << "\n";
    std::cout << "  Maturity:   " << T << " years\n";
    std::cout << "  Rate:       " << r << "\n";
    std::cout << "  Dividend:   " << q << "\n\n";
    
    MCConfig mc_config;
    mc_config.num_paths = 100000;
    mc_config.num_steps = 252;
    mc_config.use_antithetic = true;
    mc_config.use_sobol = false;
    mc_config.num_threads = 4;
    mc_config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine mc_engine(params, mc_config);
    
    std::cout << "Monte Carlo Pricing (QE scheme):\n";
    auto call_result = mc_engine.price_european_call(S0, K, T, r, q);
    std::cout << "  Call Price: " << std::fixed << std::setprecision(4) 
              << call_result.price << " ± " << call_result.std_error << "\n";
    
    auto put_result = mc_engine.price_european_put(S0, K, T, r, q);
    std::cout << "  Put Price:  " << std::fixed << std::setprecision(4) 
              << put_result.price << " ± " << put_result.std_error << "\n\n";
    
    PDEConfig pde_config;
    pde_config.num_spot_steps = 150;
    pde_config.num_var_steps = 100;
    pde_config.theta = 0.5;
    
    CrankNicolsonSolver pde_solver(params, pde_config);
    
    std::cout << "PDE Pricing (Crank-Nicolson):\n";
    auto pde_call = pde_solver.solve(S0, K, T, r, q, OptionType::Call);
    std::cout << "  Call Price: " << std::fixed << std::setprecision(4) << pde_call.price << "\n";
    std::cout << "  Delta:      " << std::setprecision(4) << pde_call.delta << "\n";
    std::cout << "  Gamma:      " << std::setprecision(4) << pde_call.gamma << "\n";
    std::cout << "  Vega:       " << std::setprecision(4) << pde_call.vega << "\n\n";
    
    auto pde_put = pde_solver.solve(S0, K, T, r, q, OptionType::Put);
    std::cout << "  Put Price:  " << std::fixed << std::setprecision(4) << pde_put.price << "\n";
    std::cout << "  Delta:      " << std::setprecision(4) << pde_put.delta << "\n";
    std::cout << "  Gamma:      " << std::setprecision(4) << pde_put.gamma << "\n";
    std::cout << "  Vega:       " << std::setprecision(4) << pde_put.vega << "\n\n";
    
    double put_call_parity_lhs = call_result.price - put_result.price;
    double put_call_parity_rhs = S0 * std::exp(-q * T) - K * std::exp(-r * T);
    std::cout << "Put-Call Parity Check (MC):\n";
    std::cout << "  C - P = " << std::fixed << std::setprecision(4) << put_call_parity_lhs << "\n";
    std::cout << "  S - K*exp(-rT) = " << put_call_parity_rhs << "\n";
    std::cout << "  Difference: " << std::abs(put_call_parity_lhs - put_call_parity_rhs) << "\n";
    
    return 0;
}
