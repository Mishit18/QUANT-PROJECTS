#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

namespace heston {

double black_scholes_price(double S, double K, double T, double r, double q, double vol) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    double d2 = d1 - vol * std::sqrt(T);
    
    auto norm_cdf = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    
    return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

void test_bs_limit() {
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.0;
    double vol = 0.2;
    
    HestonParameters params;
    params.kappa = 10.0;
    params.theta = vol * vol;
    params.sigma = 0.01;
    params.rho = 0.0;
    params.v0 = vol * vol;
    
    MCConfig config;
    config.num_paths = 100000;
    config.num_steps = 100;
    config.use_antithetic = true;
    config.num_threads = 4;
    config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine engine(params, config);
    auto result = engine.price_european_call(S0, K, T, r, q);
    
    double bs_price = black_scholes_price(S0, K, T, r, q, vol);
    
    double rel_error = std::abs(result.price - bs_price) / bs_price;
    
    std::cout << "BS Limit Test:\n";
    std::cout << "  Heston MC Price: " << result.price << " ± " << result.std_error << "\n";
    std::cout << "  Black-Scholes:   " << bs_price << "\n";
    std::cout << "  Relative Error:  " << rel_error * 100.0 << "%\n";
    
    assert(rel_error < 0.02);
    std::cout << "  PASSED\n\n";
}

} // namespace heston

int main() {
    heston::test_bs_limit();
    return 0;
}
