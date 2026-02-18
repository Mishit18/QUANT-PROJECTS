#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

namespace heston {

void test_sobol_convergence() {
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
    
    MCConfig pseudo_config;
    pseudo_config.num_paths = 50000;
    pseudo_config.num_steps = 100;
    pseudo_config.use_antithetic = false;
    pseudo_config.use_sobol = false;
    pseudo_config.num_threads = 1;
    pseudo_config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine pseudo_engine(params, pseudo_config);
    auto pseudo_result = pseudo_engine.price_european_call(S0, K, T, r, q);
    
    MCConfig sobol_config;
    sobol_config.num_paths = 50000;
    sobol_config.num_steps = 100;
    sobol_config.use_antithetic = false;
    sobol_config.use_sobol = true;
    sobol_config.num_threads = 1;
    sobol_config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine sobol_engine(params, sobol_config);
    auto sobol_result = sobol_engine.price_european_call(S0, K, T, r, q);
    
    std::cout << "Sobol Convergence Test:\n";
    std::cout << "  Pseudo-random MC: " << pseudo_result.price << " ± " << pseudo_result.std_error << "\n";
    std::cout << "  Sobol MC:         " << sobol_result.price << " ± " << sobol_result.std_error << "\n";
    std::cout << "  Sobol std error / Pseudo std error: " << sobol_result.std_error / pseudo_result.std_error << "\n";
    
    assert(sobol_result.std_error < pseudo_result.std_error * 1.5);
    assert(std::abs(sobol_result.price - pseudo_result.price) < 0.5);
    std::cout << "  PASSED\n\n";
}

} // namespace heston

int main() {
    heston::test_sobol_convergence();
    return 0;
}
