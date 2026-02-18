#include "calibration/calibrator.hpp"
#include "pde/pde_solver.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace heston {

namespace {
    double norm_cdf(double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
    
    double black_scholes_impl(double S, double K, double T, double r, double q, double vol, OptionType type) {
        if (vol <= 0.0 || T <= 0.0) return 0.0;
        
        double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double d2 = d1 - vol * std::sqrt(T);
        
        if (type == OptionType::Call) {
            return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        } else {
            return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
        }
    }
}

HestonCalibrator::HestonCalibrator(const std::vector<CalibrationPoint>& market_data,
                                   double spot, double rate, double dividend)
    : market_data_(market_data), spot_(spot), rate_(rate), dividend_(dividend) {
}

CalibrationResult HestonCalibrator::calibrate(const HestonParameters& initial_guess,
                                              const CalibrationConfig& config) {
    HestonParameters current = enforce_bounds(initial_guess, config.bounds);
    
    double prev_obj = std::numeric_limits<double>::max();
    bool converged = false;
    size_t iter = 0;
    
    std::vector<double> residuals(market_data_.size());
    
    for (iter = 0; iter < config.max_iterations; ++iter) {
        double obj = objective_function(current, config);
        
        if (std::abs(obj - prev_obj) < config.tolerance) {
            converged = true;
            break;
        }
        
        if (obj > prev_obj * 1.5) {
            break;
        }
        
        std::vector<double> grad = compute_gradient(current, config);
        
        double step_size = 0.01;
        HestonParameters next = current;
        next.kappa -= step_size * grad[0];
        next.theta -= step_size * grad[1];
        next.sigma -= step_size * grad[2];
        next.rho -= step_size * grad[3];
        next.v0 -= step_size * grad[4];
        
        next = enforce_bounds(next, config.bounds);
        
        double next_obj = objective_function(next, config);
        if (next_obj < obj) {
            current = next;
            prev_obj = obj;
        } else {
            step_size *= 0.5;
            if (step_size < 1e-8) break;
        }
    }
    
    double final_obj = objective_function(current, config);
    
    for (size_t i = 0; i < market_data_.size(); ++i) {
        MCConfig mc_config;
        mc_config.num_paths = config.num_mc_paths;
        mc_config.num_threads = config.num_threads;
        
        MonteCarloEngine engine(current, mc_config);
        double model_price = engine.price_european_call(
            spot_, market_data_[i].strike, market_data_[i].maturity, rate_, dividend_).price;
        
        double market_price = implied_vol_to_price(market_data_[i]);
        residuals[i] = (model_price - market_price) / market_price;
    }
    
    return CalibrationResult{current, final_obj, iter, converged, residuals};
}

double HestonCalibrator::objective_function(const HestonParameters& params,
                                           const CalibrationConfig& config) const {
    double sum_sq_error = 0.0;
    
    for (const auto& point : market_data_) {
        double market_price = implied_vol_to_price(point);
        
        double model_price = 0.0;
        if (config.use_pde) {
            PDEConfig pde_config;
            CrankNicolsonSolver solver(params, pde_config);
            model_price = solver.solve(spot_, point.strike, point.maturity, 
                                      rate_, dividend_, OptionType::Call).price;
        } else {
            MCConfig mc_config;
            mc_config.num_paths = config.num_mc_paths;
            mc_config.num_threads = config.num_threads;
            mc_config.scheme = MCConfig::Scheme::QE;
            
            MonteCarloEngine engine(params, mc_config);
            model_price = engine.price_european_call(
                spot_, point.strike, point.maturity, rate_, dividend_).price;
        }
        
        double error = (model_price - market_price) / market_price;
        sum_sq_error += point.weight * error * error;
    }
    
    double regularization = 0.0;
    if (!params.satisfies_feller()) {
        regularization += 10.0 * std::pow(1.0 - params.feller_ratio(), 2);
    }
    
    return sum_sq_error + regularization;
}

double HestonCalibrator::black_scholes_price(double S, double K, double T, double r, double q,
                                            double vol, OptionType type) const {
    return black_scholes_impl(S, K, T, r, q, vol, type);
}

double HestonCalibrator::implied_vol_to_price(const CalibrationPoint& point) const {
    return black_scholes_impl(spot_, point.strike, point.maturity, 
                             rate_, dividend_, point.implied_vol, OptionType::Call);
}

std::vector<double> HestonCalibrator::compute_gradient(const HestonParameters& params,
                                                      const CalibrationConfig& config,
                                                      double epsilon) const {
    std::vector<double> gradient(5);
    double f0 = objective_function(params, config);
    
    HestonParameters perturbed = params;
    
    perturbed.kappa = params.kappa + epsilon;
    gradient[0] = (objective_function(perturbed, config) - f0) / epsilon;
    perturbed.kappa = params.kappa;
    
    perturbed.theta = params.theta + epsilon;
    gradient[1] = (objective_function(perturbed, config) - f0) / epsilon;
    perturbed.theta = params.theta;
    
    perturbed.sigma = params.sigma + epsilon;
    gradient[2] = (objective_function(perturbed, config) - f0) / epsilon;
    perturbed.sigma = params.sigma;
    
    perturbed.rho = params.rho + epsilon;
    gradient[3] = (objective_function(perturbed, config) - f0) / epsilon;
    perturbed.rho = params.rho;
    
    perturbed.v0 = params.v0 + epsilon;
    gradient[4] = (objective_function(perturbed, config) - f0) / epsilon;
    
    return gradient;
}

HestonParameters HestonCalibrator::enforce_bounds(const HestonParameters& params,
                                                 const CalibrationConfig::Bounds& bounds) const {
    HestonParameters bounded = params;
    
    bounded.kappa = std::max(bounds.kappa_min, std::min(bounds.kappa_max, params.kappa));
    bounded.theta = std::max(bounds.theta_min, std::min(bounds.theta_max, params.theta));
    bounded.sigma = std::max(bounds.sigma_min, std::min(bounds.sigma_max, params.sigma));
    bounded.rho = std::max(bounds.rho_min, std::min(bounds.rho_max, params.rho));
    bounded.v0 = std::max(bounds.v0_min, std::min(bounds.v0_max, params.v0));
    
    return bounded;
}

std::vector<double> LevenbergMarquardt::optimize(const std::vector<double>& initial,
                                                ObjectiveFunction obj_func,
                                                GradientFunction grad_func,
                                                size_t max_iter, double tol) {
    std::vector<double> x = initial;
    double prev_obj = obj_func(x);
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<double> grad = grad_func(x);
        
        std::vector<double> x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - grad[i] / (lambda_ + 1.0);
        }
        
        double new_obj = obj_func(x_new);
        
        if (std::abs(new_obj - prev_obj) < tol) {
            return x_new;
        }
        
        if (new_obj < prev_obj) {
            x = x_new;
            prev_obj = new_obj;
            lambda_ /= lambda_factor_;
        } else {
            lambda_ *= lambda_factor_;
        }
    }
    
    return x;
}

} // namespace heston
