#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include "pde/pde_solver.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

namespace heston {

void test_mc_pde_consistency() {
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
    
    MCConfig mc_config;
    mc_config.num_paths = 200000;
    mc_config.num_steps = 200;
    mc_config.use_antithetic = true;
    mc_config.use_sobol = false;
    mc_config.num_threads = 4;
    mc_config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine mc_engine(params, mc_config);
    auto mc_result = mc_engine.price_european_call(S0, K, T, r, q);
    
    PDEConfig pde_config;
    pde_config.num_spot_steps = 150;
    pde_config.num_var_steps = 100;
    pde_config.theta = 0.5;
    
    CrankNicolsonSolver pde_solver(params, pde_config);
    auto pde_result = pde_solver.solve(S0, K, T, r, q, OptionType::Call);
    
    double rel_error = std::abs(mc_result.price - pde_result.price) / mc_result.price;
    
    std::cout << "MC vs PDE Consistency Test:\n";
    std::cout << "  MC Price:  " << mc_result.price << " ± " << mc_result.std_error << "\n";
    std::cout << "  PDE Price: " << pde_result.price << "\n";
    std::cout << "  PDE Delta: " << pde_result.delta << "\n";
    std::cout << "  PDE Gamma: " << pde_result.gamma << "\n";
    std::cout << "  PDE Vega:  " << pde_result.vega << "\n";
    std::cout << "  Relative Error: " << rel_error * 100.0 << "%\n";
    
    assert(pde_result.price > 0.0 && pde_result.price < S0);
    assert(pde_result.delta > 0.0 && pde_result.delta < 1.0);
    assert(pde_result.gamma > 0.0);
    assert(rel_error < 0.15);
    std::cout << "  PASSED\n\n";
}

} // namespace heston

int main() {
    heston::test_mc_pde_consistency();
    return 0;
}
