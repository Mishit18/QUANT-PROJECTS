#pragma once

#include <cmath>
#include <stdexcept>

namespace heston {

struct HestonParameters {
    double kappa;  // Mean reversion speed
    double theta;  // Long-term variance
    double sigma;  // Volatility of volatility
    double rho;    // Correlation between Brownian motions
    double v0;     // Initial variance
    
    void validate() const {
        if (kappa <= 0.0) throw std::invalid_argument("kappa must be positive");
        if (theta <= 0.0) throw std::invalid_argument("theta must be positive");
        if (sigma <= 0.0) throw std::invalid_argument("sigma must be positive");
        if (rho <= -1.0 || rho >= 1.0) throw std::invalid_argument("rho must be in (-1, 1)");
        if (v0 <= 0.0) throw std::invalid_argument("v0 must be positive");
    }
    
    bool satisfies_feller() const {
        return 2.0 * kappa * theta >= sigma * sigma;
    }
    
    double feller_ratio() const {
        return (2.0 * kappa * theta) / (sigma * sigma);
    }
};

struct MarketData {
    double spot;
    double strike;
    double maturity;
    double rate;
    double dividend;
    
    void validate() const {
        if (spot <= 0.0) throw std::invalid_argument("spot must be positive");
        if (strike <= 0.0) throw std::invalid_argument("strike must be positive");
        if (maturity <= 0.0) throw std::invalid_argument("maturity must be positive");
    }
};

enum class OptionType {
    Call,
    Put
};

class HestonModel {
public:
    explicit HestonModel(const HestonParameters& params) 
        : params_(params) {
        params_.validate();
    }
    
    const HestonParameters& parameters() const { return params_; }
    
    double drift_asset(double r, double q) const {
        return r - q;
    }
    
    double drift_variance(double v) const {
        return params_.kappa * (params_.theta - v);
    }
    
    double diffusion_asset(double v) const {
        return std::sqrt(std::max(v, 0.0));
    }
    
    double diffusion_variance(double v) const {
        return params_.sigma * std::sqrt(std::max(v, 0.0));
    }
    
    double correlation() const {
        return params_.rho;
    }
    
private:
    HestonParameters params_;
};

} // namespace heston
