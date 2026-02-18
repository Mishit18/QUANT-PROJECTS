#pragma once

#include "models/heston_model.hpp"
#include "monte_carlo/mc_engine.hpp"
#include <vector>
#include <functional>

namespace heston {

struct CalibrationPoint {
    double maturity;
    double strike;
    double implied_vol;
    double weight = 1.0;
};

struct CalibrationResult {
    HestonParameters parameters;
    double objective_value;
    size_t iterations;
    bool converged;
    std::vector<double> residuals;
};

struct CalibrationConfig {
    size_t max_iterations = 100;
    double tolerance = 1e-6;
    size_t num_mc_paths = 50000;
    size_t num_threads = 4;
    bool use_pde = false;
    
    struct Bounds {
        double kappa_min = 0.1, kappa_max = 10.0;
        double theta_min = 0.01, theta_max = 1.0;
        double sigma_min = 0.01, sigma_max = 2.0;
        double rho_min = -0.99, rho_max = 0.99;
        double v0_min = 0.01, v0_max = 1.0;
    } bounds;
};

class HestonCalibrator {
public:
    HestonCalibrator(const std::vector<CalibrationPoint>& market_data,
                     double spot, double rate, double dividend = 0.0);
    
    CalibrationResult calibrate(const HestonParameters& initial_guess,
                               const CalibrationConfig& config = CalibrationConfig{});
    
    double objective_function(const HestonParameters& params,
                             const CalibrationConfig& config) const;
    
private:
    double black_scholes_price(double S, double K, double T, double r, double q,
                              double vol, OptionType type) const;
    
    double implied_vol_to_price(const CalibrationPoint& point) const;
    
    std::vector<double> compute_gradient(const HestonParameters& params,
                                        const CalibrationConfig& config,
                                        double epsilon = 1e-5) const;
    
    HestonParameters enforce_bounds(const HestonParameters& params,
                                   const CalibrationConfig::Bounds& bounds) const;
    
    std::vector<CalibrationPoint> market_data_;
    double spot_;
    double rate_;
    double dividend_;
};

class LevenbergMarquardt {
public:
    using ObjectiveFunction = std::function<double(const std::vector<double>&)>;
    using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;
    
    LevenbergMarquardt(double lambda = 0.01, double lambda_factor = 10.0)
        : lambda_(lambda), lambda_factor_(lambda_factor) {}
    
    std::vector<double> optimize(const std::vector<double>& initial,
                                 ObjectiveFunction obj_func,
                                 GradientFunction grad_func,
                                 size_t max_iter, double tol);
    
private:
    double lambda_;
    double lambda_factor_;
};

} // namespace heston
