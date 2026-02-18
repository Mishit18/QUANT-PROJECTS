#pragma once

#include <vector>
#include <cmath>

namespace risk_engine {
namespace pricing {

enum class OptionType {
    Call,
    Put
};

struct OptionSpec {
    OptionType type;
    double strike;
    double maturity;
};

class OptionPricer {
public:
    static double payoff(const OptionSpec& spec, double spot);
    
    static double black_scholes(const OptionSpec& spec, double spot, double rate, double vol);
    
    static double delta(const OptionSpec& spec, double spot, double rate, double vol);
    
    static double gamma(const OptionSpec& spec, double spot, double rate, double vol);
    
    static double vega(const OptionSpec& spec, double spot, double rate, double vol);
    
private:
    static double normal_cdf(double x);
    static double normal_pdf(double x);
};

} // namespace pricing
} // namespace risk_engine
