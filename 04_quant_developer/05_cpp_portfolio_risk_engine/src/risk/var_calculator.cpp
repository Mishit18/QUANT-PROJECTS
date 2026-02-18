#include "var_calculator.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace risk_engine {
namespace risk {

VaRCalculator::VaRCalculator(const portfolio::Portfolio& portfolio,
                             const core::SimulationConfig& sim_config,
                             double risk_free_rate)
    : portfolio_(portfolio), sim_config_(sim_config), risk_free_rate_(risk_free_rate) {}

RiskMetrics VaRCalculator::compute_risk_metrics_single_threaded() {
    core::PathGenerator path_gen(sim_config_);
    
    std::vector<std::vector<std::vector<double>>> paths;
    path_gen.generate_paths(paths);
    
    std::vector<double> pnl_dist;
    compute_pnl_distribution(paths, pnl_dist);
    
    // Sort once for all quantile-based metrics
    std::vector<double> sorted_pnl = pnl_dist;
    std::sort(sorted_pnl.begin(), sorted_pnl.end());
    
    RiskMetrics metrics;
    metrics.var_95 = compute_var(sorted_pnl, 0.95);
    metrics.var_99 = compute_var(sorted_pnl, 0.99);
    metrics.cvar_95 = compute_cvar(sorted_pnl, 0.95);
    metrics.cvar_99 = compute_cvar(sorted_pnl, 0.99);
    
    // Compute standard errors for risk metrics
    metrics.var_95_stderr = compute_var_stderr(sorted_pnl, 0.95);
    metrics.var_99_stderr = compute_var_stderr(sorted_pnl, 0.99);
    metrics.cvar_95_stderr = compute_cvar_stderr(sorted_pnl, 0.95);
    metrics.cvar_99_stderr = compute_cvar_stderr(sorted_pnl, 0.99);
    
    double sum = std::accumulate(pnl_dist.begin(), pnl_dist.end(), 0.0);
    metrics.mean_pnl = sum / pnl_dist.size();
    
    double sq_sum = 0.0;
    for (double pnl : pnl_dist) {
        sq_sum += (pnl - metrics.mean_pnl) * (pnl - metrics.mean_pnl);
    }
    metrics.std_pnl = std::sqrt(sq_sum / pnl_dist.size());
    
    compute_greeks(metrics);
    
    return metrics;
}

RiskMetrics VaRCalculator::compute_risk_metrics_multithreaded(size_t num_threads) {
    engine::ThreadPool pool(num_threads);
    
    size_t paths_per_thread = sim_config_.num_paths / num_threads;
    std::vector<std::future<std::vector<double>>> futures;
    
    for (size_t i = 0; i < num_threads; ++i) {
        futures.push_back(pool.enqueue([this, paths_per_thread, i]() {
            core::SimulationConfig local_config = sim_config_;
            local_config.num_paths = paths_per_thread;
            
            // Use stream_id to ensure deterministic, non-overlapping RNG streams
            core::PathGenerator path_gen(local_config, 42, i);
            std::vector<std::vector<std::vector<double>>> paths;
            path_gen.generate_paths(paths);
            
            std::vector<double> local_pnl;
            compute_pnl_distribution(paths, local_pnl);
            return local_pnl;
        }));
    }
    
    std::vector<double> pnl_dist;
    for (auto& fut : futures) {
        auto local_pnl = fut.get();
        pnl_dist.insert(pnl_dist.end(), local_pnl.begin(), local_pnl.end());
    }
    
    // Sort once for all quantile-based metrics
    std::vector<double> sorted_pnl = pnl_dist;
    std::sort(sorted_pnl.begin(), sorted_pnl.end());
    
    RiskMetrics metrics;
    metrics.var_95 = compute_var(sorted_pnl, 0.95);
    metrics.var_99 = compute_var(sorted_pnl, 0.99);
    metrics.cvar_95 = compute_cvar(sorted_pnl, 0.95);
    metrics.cvar_99 = compute_cvar(sorted_pnl, 0.99);
    
    // Compute standard errors for risk metrics
    metrics.var_95_stderr = compute_var_stderr(sorted_pnl, 0.95);
    metrics.var_99_stderr = compute_var_stderr(sorted_pnl, 0.99);
    metrics.cvar_95_stderr = compute_cvar_stderr(sorted_pnl, 0.95);
    metrics.cvar_99_stderr = compute_cvar_stderr(sorted_pnl, 0.99);
    
    double sum = std::accumulate(pnl_dist.begin(), pnl_dist.end(), 0.0);
    metrics.mean_pnl = sum / pnl_dist.size();
    
    double sq_sum = 0.0;
    for (double pnl : pnl_dist) {
        sq_sum += (pnl - metrics.mean_pnl) * (pnl - metrics.mean_pnl);
    }
    metrics.std_pnl = std::sqrt(sq_sum / pnl_dist.size());
    
    compute_greeks(metrics);
    
    return metrics;
}

RiskMetrics VaRCalculator::compute_stress_test(double vol_shock, double corr_shock) {
    core::SimulationConfig stressed_config = sim_config_;
    
    for (auto& asset : stressed_config.assets) {
        asset.volatility *= (1.0 + vol_shock);
    }
    
    for (auto& row : stressed_config.correlation_matrix) {
        for (auto& val : row) {
            if (val != 1.0) {
                val = std::min(1.0, std::max(-1.0, val * (1.0 + corr_shock)));
            }
        }
    }
    
    VaRCalculator stressed_calc(portfolio_, stressed_config, risk_free_rate_);
    return stressed_calc.compute_risk_metrics_single_threaded();
}

void VaRCalculator::compute_pnl_distribution(const std::vector<std::vector<std::vector<double>>>& paths,
                                             std::vector<double>& pnl_dist) {
    pnl_dist.resize(paths.size());
    
    for (size_t p = 0; p < paths.size(); ++p) {
        std::vector<double> terminal_prices(paths[p].size());
        for (size_t a = 0; a < paths[p].size(); ++a) {
            terminal_prices[a] = paths[p][a].back();
        }
        pnl_dist[p] = portfolio_.compute_pnl(terminal_prices);
    }
}

double VaRCalculator::compute_var(const std::vector<double>& sorted_pnl, double confidence_level) {
    // Assumes sorted_pnl is already sorted
    size_t idx = static_cast<size_t>((1.0 - confidence_level) * sorted_pnl.size());
    return -sorted_pnl[idx];
}

double VaRCalculator::compute_cvar(const std::vector<double>& sorted_pnl, double confidence_level) {
    // Assumes sorted_pnl is already sorted
    size_t idx = static_cast<size_t>((1.0 - confidence_level) * sorted_pnl.size());
    
    double sum = 0.0;
    for (size_t i = 0; i <= idx; ++i) {
        sum += sorted_pnl[i];
    }
    
    return -sum / (idx + 1);
}

double VaRCalculator::compute_var_stderr(const std::vector<double>& sorted_pnl, double confidence_level) {
    // Standard error for VaR using order statistics
    // SE(VaR) ≈ sqrt(α(1-α)/n) / f(VaR)
    // where f is the PDF at VaR, approximated by kernel density
    
    size_t n = sorted_pnl.size();
    double alpha = 1.0 - confidence_level;
    size_t idx = static_cast<size_t>(alpha * n);
    
    // Estimate PDF at VaR using local density (simple kernel)
    double h = 1.06 * std::sqrt(std::accumulate(sorted_pnl.begin(), sorted_pnl.end(), 0.0, 
        [mean = std::accumulate(sorted_pnl.begin(), sorted_pnl.end(), 0.0) / n](double acc, double x) {
            return acc + (x - mean) * (x - mean);
        }) / n) * std::pow(n, -0.2);  // Silverman's rule of thumb
    
    // Count points in bandwidth around VaR
    double var_value = sorted_pnl[idx];
    size_t count = 0;
    for (double pnl : sorted_pnl) {
        if (std::abs(pnl - var_value) <= h) count++;
    }
    
    double pdf_estimate = count / (n * 2.0 * h);
    if (pdf_estimate < 1e-10) pdf_estimate = 1e-10;  // Avoid division by zero
    
    return std::sqrt(alpha * (1.0 - alpha) / n) / pdf_estimate;
}

double VaRCalculator::compute_cvar_stderr(const std::vector<double>& sorted_pnl, double confidence_level) {
    // Standard error for CVaR (Expected Shortfall)
    // SE(CVaR) ≈ sqrt(Var(losses beyond VaR) / (n * α))
    
    size_t n = sorted_pnl.size();
    double alpha = 1.0 - confidence_level;
    size_t idx = static_cast<size_t>(alpha * n);
    
    // Compute variance of tail losses
    double tail_mean = 0.0;
    for (size_t i = 0; i <= idx; ++i) {
        tail_mean += sorted_pnl[i];
    }
    tail_mean /= (idx + 1);
    
    double tail_var = 0.0;
    for (size_t i = 0; i <= idx; ++i) {
        tail_var += (sorted_pnl[i] - tail_mean) * (sorted_pnl[i] - tail_mean);
    }
    tail_var /= (idx + 1);
    
    return std::sqrt(tail_var / (n * alpha));
}

void VaRCalculator::compute_greeks(RiskMetrics& metrics) {
    size_t num_assets = sim_config_.assets.size();
    
    std::vector<double> current_prices(num_assets);
    std::vector<double> vols(num_assets);
    
    for (size_t i = 0; i < num_assets; ++i) {
        current_prices[i] = sim_config_.assets[i].spot;
        vols[i] = sim_config_.assets[i].volatility;
    }
    
    metrics.portfolio_deltas.resize(num_assets);
    metrics.portfolio_gammas.resize(num_assets);
    metrics.portfolio_vegas.resize(num_assets);
    
    for (size_t i = 0; i < num_assets; ++i) {
        metrics.portfolio_deltas[i] = portfolio_.compute_delta(i, current_prices, risk_free_rate_, vols);
        metrics.portfolio_gammas[i] = portfolio_.compute_gamma(i, current_prices, risk_free_rate_, vols);
        metrics.portfolio_vegas[i] = portfolio_.compute_vega(i, current_prices, risk_free_rate_, vols);
    }
}

} // namespace risk
} // namespace risk_engine
