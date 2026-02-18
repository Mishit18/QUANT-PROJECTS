#pragma once

#include "../portfolio/position.h"
#include "../core/path_generator.h"
#include "../engine/thread_pool.h"
#include <vector>
#include <memory>

namespace risk_engine {
namespace risk {

struct RiskMetrics {
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double mean_pnl;
    double std_pnl;
    
    // Statistical confidence intervals (standard errors)
    double var_95_stderr;
    double var_99_stderr;
    double cvar_95_stderr;
    double cvar_99_stderr;
    
    std::vector<double> portfolio_deltas;
    std::vector<double> portfolio_gammas;
    std::vector<double> portfolio_vegas;
};

class VaRCalculator {
public:
    VaRCalculator(const portfolio::Portfolio& portfolio,
                  const core::SimulationConfig& sim_config,
                  double risk_free_rate = 0.05);
    
    RiskMetrics compute_risk_metrics_single_threaded();
    
    RiskMetrics compute_risk_metrics_multithreaded(size_t num_threads);
    
    RiskMetrics compute_stress_test(double vol_shock, double corr_shock);
    
private:
    void compute_pnl_distribution(const std::vector<std::vector<std::vector<double>>>& paths,
                                  std::vector<double>& pnl_dist);
    
    double compute_var(const std::vector<double>& pnl_dist, double confidence_level);
    
    double compute_cvar(const std::vector<double>& pnl_dist, double confidence_level);
    
    // Compute standard error for VaR using order statistics
    double compute_var_stderr(const std::vector<double>& sorted_pnl, double confidence_level);
    
    // Compute standard error for CVaR
    double compute_cvar_stderr(const std::vector<double>& sorted_pnl, double confidence_level);
    
    void compute_greeks(RiskMetrics& metrics);
    
    const portfolio::Portfolio& portfolio_;
    core::SimulationConfig sim_config_;
    double risk_free_rate_;
};

} // namespace risk
} // namespace risk_engine
