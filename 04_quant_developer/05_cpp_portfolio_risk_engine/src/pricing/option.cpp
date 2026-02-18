#include "option.h"
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace risk_engine {
namespace pricing {

double OptionPricer::payoff(const OptionSpec& spec, double spot) {
    if (spec.type == OptionType::Call) {
        return std::max(spot - spec.strike, 0.0);
    } else {
        return std::max(spec.strike - spot, 0.0);
    }
}

double OptionPricer::normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

double OptionPricer::normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double OptionPricer::black_scholes(const OptionSpec& spec, double spot, double rate, double vol) {
    if (spec.maturity <= 0.0) {
        return payoff(spec, spot);
    }
    
    double d1 = (std::log(spot / spec.strike) + (rate + 0.5 * vol * vol) * spec.maturity) 
                / (vol * std::sqrt(spec.maturity));
    double d2 = d1 - vol * std::sqrt(spec.maturity);
    
    if (spec.type == OptionType::Call) {
        return spot * normal_cdf(d1) - spec.strike * std::exp(-rate * spec.maturity) * normal_cdf(d2);
    } else {
        return spec.strike * std::exp(-rate * spec.maturity) * normal_cdf(-d2) - spot * normal_cdf(-d1);
    }
}

double OptionPricer::delta(const OptionSpec& spec, double spot, double rate, double vol) {
    if (spec.maturity <= 0.0) {
        if (spec.type == OptionType::Call) {
            return spot > spec.strike ? 1.0 : 0.0;
        } else {
            return spot < spec.strike ? -1.0 : 0.0;
        }
    }
    
    double d1 = (std::log(spot / spec.strike) + (rate + 0.5 * vol * vol) * spec.maturity) 
                / (vol * std::sqrt(spec.maturity));
    
    if (spec.type == OptionType::Call) {
        return normal_cdf(d1);
    } else {
        return normal_cdf(d1) - 1.0;
    }
}

double OptionPricer::gamma(const OptionSpec& spec, double spot, double rate, double vol) {
    if (spec.maturity <= 0.0) {
        return 0.0;
    }
    
    double d1 = (std::log(spot / spec.strike) + (rate + 0.5 * vol * vol) * spec.maturity) 
                / (vol * std::sqrt(spec.maturity));
    
    return normal_pdf(d1) / (spot * vol * std::sqrt(spec.maturity));
}

double OptionPricer::vega(const OptionSpec& spec, double spot, double rate, double vol) {
    if (spec.maturity <= 0.0) {
        return 0.0;
    }
    
    double d1 = (std::log(spot / spec.strike) + (rate + 0.5 * vol * vol) * spec.maturity) 
                / (vol * std::sqrt(spec.maturity));
    
    return spot * normal_pdf(d1) * std::sqrt(spec.maturity);
}

} // namespace pricing
} // namespace risk_engine
