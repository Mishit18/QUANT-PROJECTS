#include "pde/pde_solver.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace heston {

CrankNicolsonSolver::CrankNicolsonSolver(const HestonParameters& params, const PDEConfig& config)
    : model_(params), config_(config), 
      grid_(Grid1D(0.0, 1.0, 1), Grid1D(0.0, 1.0, 1)),
      matrix_S_(1), matrix_v_(1) {
}

PDEResult CrankNicolsonSolver::solve(double S0, double K, double T, double r, double q, OptionType type) {
    double var_max = config_.var_max_factor * model_.parameters().theta;
    initialize_grid(S0, K, var_max);
    
    set_terminal_condition(K, type);
    
    size_t num_time_steps = config_.num_spot_steps * 3;
    double dt = T / num_time_steps;
    
    for (size_t step = 0; step < num_time_steps; ++step) {
        double t = T - (step + 1) * dt;
        set_boundary_conditions(K, r, q, t, type);
        time_step(dt, r, q);
    }
    
    PDEResult result;
    result.price = interpolate_solution(S0, model_.parameters().v0);
    compute_greeks(S0, model_.parameters().v0, result);
    
    return result;
}

void CrankNicolsonSolver::initialize_grid(double S0, double K, double var_max) {
    double S_max = config_.spot_max_factor * std::max(S0, K);
    
    Grid1D grid_S(0.0, S_max, config_.num_spot_steps, false);
    Grid1D grid_v(0.0, var_max, config_.num_var_steps, false);
    
    grid_ = Grid2D(grid_S, grid_v);
    
    u_.resize(grid_.size());
    u_old_.resize(grid_.size());
    rhs_.resize(grid_.size());
}

void CrankNicolsonSolver::set_terminal_condition(double K, OptionType type) {
    for (size_t i = 0; i < grid_.nx(); ++i) {
        for (size_t j = 0; j < grid_.ny(); ++j) {
            double S = grid_.x(i);
            size_t idx = grid_.index(i, j);
            
            if (type == OptionType::Call) {
                u_[idx] = std::max(S - K, 0.0);
            } else {
                u_[idx] = std::max(K - S, 0.0);
            }
        }
    }
}

void CrankNicolsonSolver::set_boundary_conditions(double K, double r, double q, double t, OptionType type) {
    (void)q;
    for (size_t j = 0; j < grid_.ny(); ++j) {
        size_t idx_0 = grid_.index(0, j);
        size_t idx_max = grid_.index(grid_.nx() - 1, j);
        
        if (type == OptionType::Call) {
            u_[idx_0] = 0.0;
            u_[idx_max] = std::max(0.0, grid_.x(grid_.nx() - 1) - K * std::exp(-r * t));
        } else {
            u_[idx_0] = K * std::exp(-r * t);
            u_[idx_max] = 0.0;
        }
    }
    
    for (size_t i = 1; i < grid_.nx() - 1; ++i) {
        size_t idx_0 = grid_.index(i, 0);
        size_t idx_max = grid_.index(i, grid_.ny() - 1);
        
        u_[idx_0] = u_[grid_.index(i, 1)];
        u_[idx_max] = u_[grid_.index(i, grid_.ny() - 2)];
    }
}

void CrankNicolsonSolver::time_step(double dt, double r, double q) {
    u_old_ = u_;
    
    double dS_min = grid_.x(1) - grid_.x(0);
    double v_max = grid_.y(grid_.ny() - 1);
    double S_max = grid_.x(grid_.nx() - 1);
    
    double max_diff_coef = 0.5 * v_max * S_max * S_max;
    double dt_stable = 0.2 * dS_min * dS_min / (max_diff_coef + 1e-10);
    dt = std::min(dt, dt_stable);
    
    for (size_t i = 1; i < grid_.nx() - 1; ++i) {
        for (size_t j = 1; j < grid_.ny() - 1; ++j) {
            size_t idx = grid_.index(i, j);
            double S = grid_.x(i);
            double v = std::max(grid_.y(j), 1e-10);
            
            double dS_fwd = grid_.x(i + 1) - grid_.x(i);
            double dS_bwd = grid_.x(i) - grid_.x(i - 1);
            double dv_fwd = grid_.y(j + 1) - grid_.y(j);
            double dv_bwd = grid_.y(j) - grid_.y(j - 1);
            
            double dS_cent = 0.5 * (dS_fwd + dS_bwd);
            double dv_cent = 0.5 * (dv_fwd + dv_bwd);
            
            double u_S = (u_old_[grid_.index(i + 1, j)] - u_old_[grid_.index(i - 1, j)]) / (dS_fwd + dS_bwd);
            double u_v = (u_old_[grid_.index(i, j + 1)] - u_old_[grid_.index(i, j - 1)]) / (dv_fwd + dv_bwd);
            
            double u_SS = (u_old_[grid_.index(i + 1, j)] - 2.0 * u_old_[idx] + u_old_[grid_.index(i - 1, j)]) / (dS_cent * dS_cent);
            double u_vv = (u_old_[grid_.index(i, j + 1)] - 2.0 * u_old_[idx] + u_old_[grid_.index(i, j - 1)]) / (dv_cent * dv_cent);
            
            double u_Sv = (u_old_[grid_.index(i + 1, j + 1)] - u_old_[grid_.index(i + 1, j - 1)] -
                          u_old_[grid_.index(i - 1, j + 1)] + u_old_[grid_.index(i - 1, j - 1)]) / 
                          (2.0 * dS_cent * 2.0 * dv_cent);
            
            double drift_S = (r - q) * S * u_S;
            double drift_v = model_.parameters().kappa * (model_.parameters().theta - v) * u_v;
            double diff_S = 0.5 * v * S * S * u_SS;
            double diff_v = 0.5 * model_.parameters().sigma * model_.parameters().sigma * v * u_vv;
            double cross = model_.parameters().rho * model_.parameters().sigma * v * S * u_Sv;
            
            double rhs = drift_S + drift_v + diff_S + diff_v + cross - r * u_old_[idx];
            
            u_[idx] = u_old_[idx] + dt * rhs;
            
            u_[idx] = std::max(0.0, u_[idx]);
        }
    }
}

void CrankNicolsonSolver::build_matrices(double dt, double r, double q) {
    (void)dt;
    (void)r;
    (void)q;
}

void CrankNicolsonSolver::solve_linear_system() {
}

double CrankNicolsonSolver::interpolate_solution(double S, double v) const {
    size_t i = grid_.grid_x().find_index(S);
    size_t j = grid_.grid_y().find_index(v);
    
    if (i >= grid_.nx() - 1) i = grid_.nx() - 2;
    if (j >= grid_.ny() - 1) j = grid_.ny() - 2;
    
    double S0 = grid_.x(i);
    double S1 = grid_.x(i + 1);
    double v0 = grid_.y(j);
    double v1 = grid_.y(j + 1);
    
    double ws = (S - S0) / (S1 - S0);
    double wv = (v - v0) / (v1 - v0);
    
    double u00 = u_[grid_.index(i, j)];
    double u10 = u_[grid_.index(i + 1, j)];
    double u01 = u_[grid_.index(i, j + 1)];
    double u11 = u_[grid_.index(i + 1, j + 1)];
    
    return (1.0 - ws) * (1.0 - wv) * u00 +
           ws * (1.0 - wv) * u10 +
           (1.0 - ws) * wv * u01 +
           ws * wv * u11;
}

void CrankNicolsonSolver::compute_greeks(double S0, double v0, PDEResult& result) const {
    size_t i = grid_.grid_x().find_index(S0);
    size_t j = grid_.grid_y().find_index(v0);
    
    if (i == 0) i = 1;
    if (i >= grid_.nx() - 1) i = grid_.nx() - 2;
    if (j == 0) j = 1;
    if (j >= grid_.ny() - 1) j = grid_.ny() - 2;
    
    double dS = (grid_.x(i + 1) - grid_.x(i - 1)) / 2.0;
    double dv = (grid_.y(j + 1) - grid_.y(j - 1)) / 2.0;
    
    double u_plus = u_[grid_.index(i + 1, j)];
    double u_minus = u_[grid_.index(i - 1, j)];
    double u_center = u_[grid_.index(i, j)];
    
    result.delta = (u_plus - u_minus) / (2.0 * dS);
    result.gamma = (u_plus - 2.0 * u_center + u_minus) / (dS * dS);
    
    double u_v_plus = u_[grid_.index(i, j + 1)];
    double u_v_minus = u_[grid_.index(i, j - 1)];
    result.vega = (u_v_plus - u_v_minus) / (2.0 * dv);
}

} // namespace heston
