#pragma once

#include "models/heston_model.hpp"
#include <algorithm>

namespace heston {

struct PathState {
    double S;  // Asset price
    double v;  // Variance
};

class DiscretizationScheme {
public:
    virtual ~DiscretizationScheme() = default;
    
    virtual void step(PathState& state, double dt, double z1, double z2,
                     const HestonModel& model, double r, double q) const = 0;
};

class EulerScheme : public DiscretizationScheme {
public:
    void step(PathState& state, double dt, double z1, double z2,
             const HestonModel& model, double r, double q) const override {
        const auto& params = model.parameters();
        
        double sqrt_v = std::sqrt(std::max(state.v, 0.0));
        double sqrt_dt = std::sqrt(dt);
        
        // Correlated Brownian increments
        double dW1 = sqrt_dt * z1;
        double dW2 = sqrt_dt * (params.rho * z1 + std::sqrt(1.0 - params.rho * params.rho) * z2);
        
        // Update variance (with reflection at zero)
        double v_new = state.v + params.kappa * (params.theta - state.v) * dt 
                       + params.sigma * sqrt_v * dW2;
        state.v = std::abs(v_new);
        
        // Update asset price
        state.S *= std::exp((r - q - 0.5 * state.v) * dt + sqrt_v * dW1);
    }
};

class QEScheme : public DiscretizationScheme {
public:
    explicit QEScheme(double psi_c = 1.5) : psi_c_(psi_c) {}
    
    void step(PathState& state, double dt, double z1, double z2,
             const HestonModel& model, double r, double q) const override {
        const auto& params = model.parameters();
        
        double m = params.theta + (state.v - params.theta) * std::exp(-params.kappa * dt);
        double s2 = state.v * params.sigma * params.sigma * std::exp(-params.kappa * dt) 
                    * (1.0 - std::exp(-params.kappa * dt)) / params.kappa
                    + params.theta * params.sigma * params.sigma 
                    * (1.0 - std::exp(-params.kappa * dt)) * (1.0 - std::exp(-params.kappa * dt))
                    / (2.0 * params.kappa);
        
        double psi = s2 / (m * m);
        
        double v_new;
        if (psi <= psi_c_) {
            double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi) * std::sqrt(2.0 / psi - 1.0);
            double a = m / (1.0 + b2);
            v_new = a * (std::sqrt(b2) + z2) * (std::sqrt(b2) + z2);
        } else {
            double p = (psi - 1.0) / (psi + 1.0);
            double beta = (1.0 - p) / m;
            double u = std::erf(z2 / std::sqrt(2.0)) * 0.5 + 0.5;
            v_new = (u <= p) ? 0.0 : std::log((1.0 - p) / (1.0 - u)) / beta;
        }
        
        double K0 = -params.rho * params.kappa * params.theta * dt / params.sigma;
        double K1 = 0.5 * dt * (params.kappa * params.rho / params.sigma - 0.5) - params.rho / params.sigma;
        double K2 = 0.5 * dt * (params.kappa * params.rho / params.sigma - 0.5) + params.rho / params.sigma;
        double K3 = 0.5 * dt * (1.0 - params.rho * params.rho);
        
        state.S *= std::exp((r - q) * dt + K0 + K1 * state.v + K2 * v_new 
                           + std::sqrt(K3 * (state.v + v_new)) * z1);
        state.v = std::max(v_new, 0.0);
    }
    
private:
    double psi_c_;
};

} // namespace heston
