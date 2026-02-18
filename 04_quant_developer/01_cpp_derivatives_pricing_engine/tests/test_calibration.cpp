#include "calibration/calibrator.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

namespace heston {

void test_calibration_sanity() {
    double S0 = 100.0;
    double r = 0.05;
    double q = 0.0;
    
    HestonParameters true_params;
    true_params.kappa = 2.0;
    true_params.theta = 0.04;
    true_params.sigma = 0.3;
    true_params.rho = -0.7;
    true_params.v0 = 0.04;
    
    MCConfig mc_config;
    mc_config.num_paths = 50000;
    mc_config.num_steps = 100;
    mc_config.num_threads = 4;
    mc_config.scheme = MCConfig::Scheme::QE;
    
    MonteCarloEngine engine(true_params, mc_config);
    
    std::vector<CalibrationPoint> market_data;
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    std::vector<double> maturities = {0.5, 1.0};
    
    for (double T : maturities) {
        for (double K : strikes) {
            auto result = engine.price_european_call(S0, K, T, r, q);
            
            auto bs_impl_vol = [&](double vol) {
                double d1 = (std::log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
                double d2 = d1 - vol * std::sqrt(T);
                auto norm_cdf = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
                return S0 * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
            };
            
            double implied_vol = 0.2;
            for (int iter = 0; iter < 20; ++iter) {
                double price = bs_impl_vol(implied_vol);
                double d1 = (std::log(S0 / K) + (r - q + 0.5 * implied_vol * implied_vol) * T) / (implied_vol * std::sqrt(T));
                double vega = S0 * std::exp(-q * T) * std::sqrt(T / (2.0 * 3.14159265358979323846)) * 
                             std::exp(-0.5 * d1 * d1);
                implied_vol -= (price - result.price) / (vega + 1e-10);
                implied_vol = std::max(0.01, std::min(2.0, implied_vol));
            }
            
            market_data.push_back({T, K, implied_vol, 1.0});
        }
    }
    
    HestonParameters initial_guess;
    initial_guess.kappa = 1.5;
    initial_guess.theta = 0.05;
    initial_guess.sigma = 0.4;
    initial_guess.rho = -0.5;
    initial_guess.v0 = 0.05;
    
    CalibrationConfig calib_config;
    calib_config.max_iterations = 20;
    calib_config.tolerance = 1e-4;
    calib_config.num_mc_paths = 20000;
    calib_config.num_threads = 4;
    
    HestonCalibrator calibrator(market_data, S0, r, q);
    auto calib_result = calibrator.calibrate(initial_guess, calib_config);
    
    std::cout << "Calibration Sanity Test:\n";
    std::cout << "  True Parameters:\n";
    std::cout << "    kappa: " << true_params.kappa << ", theta: " << true_params.theta 
              << ", sigma: " << true_params.sigma << ", rho: " << true_params.rho 
              << ", v0: " << true_params.v0 << "\n";
    std::cout << "  Calibrated Parameters:\n";
    std::cout << "    kappa: " << calib_result.parameters.kappa 
              << ", theta: " << calib_result.parameters.theta 
              << ", sigma: " << calib_result.parameters.sigma 
              << ", rho: " << calib_result.parameters.rho 
              << ", v0: " << calib_result.parameters.v0 << "\n";
    std::cout << "  Objective Value: " << calib_result.objective_value << "\n";
    std::cout << "  Iterations: " << calib_result.iterations << "\n";
    std::cout << "  Converged: " << (calib_result.converged ? "Yes" : "No") << "\n";
    
    assert(calib_result.objective_value < 0.1);
    std::cout << "  PASSED\n\n";
}

} // namespace heston

int main() {
    heston::test_calibration_sanity();
    return 0;
}
