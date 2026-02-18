#pragma once

#include "models/heston_model.hpp"
#include "monte_carlo/discretization.hpp"
#include "numerics/random.hpp"
#include "utils/memory.hpp"
#include <memory>
#include <vector>

namespace heston {

struct MCConfig {
    size_t num_paths = 100000;
    size_t num_steps = 252;
    bool use_antithetic = true;
    bool use_sobol = false;
    size_t num_threads = 1;
    uint64_t seed = 12345;
    
    enum class Scheme {
        Euler,
        QE
    } scheme = Scheme::QE;
};

struct MCResult {
    double price;
    double std_error;
    double variance;
    size_t paths_used;
};

class MonteCarloEngine {
public:
    MonteCarloEngine(const HestonParameters& params, const MCConfig& config);
    
    MCResult price_european_call(double S0, double K, double T, double r, double q = 0.0) const;
    MCResult price_european_put(double S0, double K, double T, double r, double q = 0.0) const;
    
    MCResult price_option(double S0, double K, double T, double r, double q,
                         OptionType type) const;
    
private:
    struct PathResult {
        double payoff;
        double control_variate;
    };
    
    PathResult simulate_path(RandomNumberGenerator& rng, double S0, double K,
                            double T, double r, double q, OptionType type) const;
    
    double black_scholes_price(double S0, double K, double T, double r, double q,
                              double vol, OptionType type) const;
    
    HestonModel model_;
    MCConfig config_;
    std::unique_ptr<DiscretizationScheme> scheme_;
    AlignedVector<double> payoffs_;
};

} // namespace heston
