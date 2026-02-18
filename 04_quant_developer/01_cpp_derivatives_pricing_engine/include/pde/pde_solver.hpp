#pragma once

#include "models/heston_model.hpp"
#include "pde/grid.hpp"
#include "numerics/sparse_matrix.hpp"
#include <vector>

namespace heston {

struct PDEConfig {
    size_t num_spot_steps = 200;
    size_t num_var_steps = 100;
    double spot_max_factor = 3.0;
    double var_max_factor = 5.0;
    double theta = 0.5;  // 0.5 for Crank-Nicolson
};

struct PDEResult {
    double price;
    double delta;
    double gamma;
    double vega;
};

class CrankNicolsonSolver {
public:
    CrankNicolsonSolver(const HestonParameters& params, const PDEConfig& config);
    
    PDEResult solve(double S0, double K, double T, double r, double q, OptionType type);
    
    const Grid2D& grid() const { return grid_; }
    const std::vector<double>& solution() const { return u_; }
    
private:
    void initialize_grid(double S0, double K, double var_max);
    void set_terminal_condition(double K, OptionType type);
    void set_boundary_conditions(double K, double r, double q, double t, OptionType type);
    void time_step(double dt, double r, double q);
    
    void build_matrices(double dt, double r, double q);
    void solve_linear_system();
    
    double interpolate_solution(double S, double v) const;
    void compute_greeks(double S0, double v0, PDEResult& result) const;
    
    HestonModel model_;
    PDEConfig config_;
    Grid2D grid_;
    
    std::vector<double> u_;      // Solution at current time
    std::vector<double> u_old_;  // Solution at previous time
    std::vector<double> rhs_;    // Right-hand side
    
    TridiagonalMatrix matrix_S_;
    TridiagonalMatrix matrix_v_;
};

} // namespace heston
