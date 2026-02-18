#include "monte_carlo/mc_engine.hpp"
#include <thread>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace heston {

namespace {
    double norm_cdf(double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
    
    double black_scholes_impl(double S, double K, double T, double r, double q, double vol, OptionType type) {
        double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double d2 = d1 - vol * std::sqrt(T);
        
        if (type == OptionType::Call) {
            return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        } else {
            return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
        }
    }
}

MonteCarloEngine::MonteCarloEngine(const HestonParameters& params, const MCConfig& config)
    : model_(params), config_(config) {
    
    switch (config_.scheme) {
        case MCConfig::Scheme::Euler:
            scheme_ = std::make_unique<EulerScheme>();
            break;
        case MCConfig::Scheme::QE:
            scheme_ = std::make_unique<QEScheme>();
            break;
    }
}

MCResult MonteCarloEngine::price_european_call(double S0, double K, double T, double r, double q) const {
    return price_option(S0, K, T, r, q, OptionType::Call);
}

MCResult MonteCarloEngine::price_european_put(double S0, double K, double T, double r, double q) const {
    return price_option(S0, K, T, r, q, OptionType::Put);
}

MCResult MonteCarloEngine::price_option(double S0, double K, double T, double r, double q, OptionType type) const {
    const size_t paths_per_thread = config_.num_paths / config_.num_threads;
    const size_t remainder = config_.num_paths % config_.num_threads;
    
    std::vector<std::thread> threads;
    std::vector<double> thread_sums(config_.num_threads, 0.0);
    std::vector<double> thread_sum_sq(config_.num_threads, 0.0);
    
    auto worker = [&](size_t thread_id, size_t num_paths) {
        std::unique_ptr<RandomNumberGenerator> rng;
        if (config_.use_sobol) {
            rng = std::make_unique<SobolRNG>(2);
        } else {
            rng = std::make_unique<MersenneTwisterRNG>(config_.seed + thread_id);
        }
        
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (size_t i = 0; i < num_paths; ++i) {
            auto result = simulate_path(*rng, S0, K, T, r, q, type);
            double payoff = result.payoff;
            
            if (config_.use_antithetic) {
                auto result_anti = simulate_path(*rng, S0, K, T, r, q, type);
                payoff = 0.5 * (payoff + result_anti.payoff);
            }
            
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        
        thread_sums[thread_id] = sum;
        thread_sum_sq[thread_id] = sum_sq;
    };
    
    for (size_t i = 0; i < config_.num_threads; ++i) {
        size_t paths = paths_per_thread + (i < remainder ? 1 : 0);
        threads.emplace_back(worker, i, paths);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    double total_sum = std::accumulate(thread_sums.begin(), thread_sums.end(), 0.0);
    double total_sum_sq = std::accumulate(thread_sum_sq.begin(), thread_sum_sq.end(), 0.0);
    
    size_t effective_paths = config_.num_paths * (config_.use_antithetic ? 2 : 1);
    double mean = total_sum / config_.num_paths;
    double variance = (total_sum_sq / config_.num_paths - mean * mean);
    double std_error = std::sqrt(variance / config_.num_paths);
    
    double price = mean * std::exp(-r * T);
    double price_std_error = std_error * std::exp(-r * T);
    
    return MCResult{price, price_std_error, variance, effective_paths};
}

MonteCarloEngine::PathResult MonteCarloEngine::simulate_path(
    RandomNumberGenerator& rng, double S0, double K, double T, double r, double q, OptionType type) const {
    
    PathState state{S0, model_.parameters().v0};
    double dt = T / config_.num_steps;
    
    for (size_t step = 0; step < config_.num_steps; ++step) {
        double z1, z2;
        rng.next_pair(z1, z2);
        scheme_->step(state, dt, z1, z2, model_, r, q);
    }
    
    double payoff = 0.0;
    if (type == OptionType::Call) {
        payoff = std::max(state.S - K, 0.0);
    } else {
        payoff = std::max(K - state.S, 0.0);
    }
    
    return PathResult{payoff, 0.0};
}

double MonteCarloEngine::black_scholes_price(double S0, double K, double T, double r, double q,
                                            double vol, OptionType type) const {
    return black_scholes_impl(S0, K, T, r, q, vol, type);
}

} // namespace heston
#include "monte_carlo/mc_engine.hpp"
#include <thread>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace heston {

namespace {
    double norm_cdf(double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
    
    double black_scholes_impl(double S, double K, double T, double r, double q, double vol, OptionType type) {
        double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double d2 = d1 - vol * std::sqrt(T);
        
        if (type == OptionType::Call) {
            return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        } else {
            return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
        }
    }
}

MonteCarloEngine::MonteCarloEngine(const HestonParameters& params, const MCConfig& config)
    : model_(params), config_(config) {
    
    switch (config_.scheme) {
        case MCConfig::Scheme::Euler:
            scheme_ = std::make_unique<EulerScheme>();
            break;
        case MCConfig::Scheme::QE:
            scheme_ = std::make_unique<QEScheme>();
            break;
    }
}

MCResult MonteCarloEngine::price_european_call(double S0, double K, double T, double r, double q) const {
    return price_option(S0, K, T, r, q, OptionType::Call);
}

MCResult MonteCarloEngine::price_european_put(double S0, double K, double T, double r, double q) const {
    return price_option(S0, K, T, r, q, OptionType::Put);
}

MCResult MonteCarloEngine::price_option(double S0, double K, double T, double r, double q, OptionType type) const {
    const size_t paths_per_thread = config_.num_paths / config_.num_threads;
    const size_t remainder = config_.num_paths % config_.num_threads;
    
    std::vector<std::thread> threads;
    std::vector<double> thread_sums(config_.num_threads, 0.0);
    std::vector<double> thread_sum_sq(config_.num_threads, 0.0);
    
    auto worker = [&](size_t thread_id, size_t num_paths) {
        std::unique_ptr<RandomNumberGenerator> rng;
        if (config_.use_sobol) {
            rng = std::make_unique<SobolRNG>(2);
        } else {
            rng = std::make_unique<MersenneTwisterRNG>(config_.seed + thread_id);
        }
        
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (size_t i = 0; i < num_paths; ++i) {
            auto result = simulate_path(*rng, S0, K, T, r, q, type);
            double payoff = result.payoff;
            
            if (config_.use_antithetic) {
                auto result_anti = simulate_path(*rng, S0, K, T, r, q, type);
                payoff = 0.5 * (payoff + result_anti.payoff);
            }
            
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        
        thread_sums[thread_id] = sum;
        thread_sum_sq[thread_id] = sum_sq;
    };
    
    for (size_t i = 0; i < config_.num_threads; ++i) {
        size_t paths = paths_per_thread + (i < remainder ? 1 : 0);
        threads.emplace_back(worker, i, paths);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    double total_sum = std::accumulate(thread_sums.begin(), thread_sums.end(), 0.0);
    double total_sum_sq = std::accumulate(thread_sum_sq.begin(), thread_sum_sq.end(), 0.0);
    
    size_t effective_paths = config_.num_paths * (config_.use_antithetic ? 2 : 1);
    double mean = total_sum / config_.num_paths;
    double variance = (total_sum_sq / config_.num_paths - mean * mean);
    double std_error = std::sqrt(variance / config_.num_paths);
    
    double price = mean * std::exp(-r * T);
    double price_std_error = std_error * std::exp(-r * T);
    
    return MCResult{price, price_std_error, variance, effective_paths};
}

MonteCarloEngine::PathResult MonteCarloEngine::simulate_path(
    RandomNumberGenerator& rng, double S0, double K, double T, double r, double q, OptionType type) const {
    
    PathState state{S0, model_.parameters().v0};
    double dt = T / config_.num_steps;
    
    for (size_t step = 0; step < config_.num_steps; ++step) {
        double z1, z2;
        rng.next_pair(z1, z2);
        scheme_->step(state, dt, z1, z2, model_, r, q);
    }
    
    double payoff = 0.0;
    if (type == OptionType::Call) {
        payoff = std::max(state.S - K, 0.0);
    } else {
        payoff = std::max(K - state.S, 0.0);
    }
    
    return PathResult{payoff, 0.0};
}

double MonteCarloEngine::black_scholes_price(double S0, double K, double T, double r, double q,
                                            double vol, OptionType type) const {
    return black_scholes_impl(S0, K, T, r, q, vol, type);
}

} // namespace heston
