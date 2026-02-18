# Implementation Plan: Research-Grade Market Impact Framework

## Overview

This implementation plan follows a phased rebuild approach with aggressive cleanup first. Each phase must be validated before proceeding to the next. The focus is on theoretical correctness, explicit failure analysis, and ensuring the project doesn't feel "patched together."

## Phases

1. **Audit & Deletion**: Clean up invalid outputs, unused files, empty directories
2. **Data Generation**: Rebuild synthetic market with validation
3. **Model-by-Model Reconstruction**: Kyle → OW → Bouchaud (each with assumptions, calibration, validation, failure flags)
4. **Execution + Frontier**: Non-degenerate frontier or explicit explanation why impossible
5. **Documentation Pass**: README, research report, comments cleanup

## Tasks

### Phase 1: Audit & Deletion

- [ ] 1. Audit current project structure
  - Read and document current folder structure
  - Identify invalid outputs (scattered results, duplicate figures, inconsistent naming)
  - Identify unused files (old experiments, deprecated modules, test artifacts)
  - Identify empty directories
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 2. Delete aggressively
  - Remove all invalid outputs
  - Remove all unused files
  - Remove all empty directories
  - Remove all scattered results not in `results/` or `reports/figures/`
  - _Requirements: 8.2, 8.3, 8.4_

- [ ] 3. Present final cleaned structure
  - Document the cleaned directory structure
  - Verify only essential files remain
  - Ensure structure matches spec: `data/`, `models/`, `execution/`, `analysis/`, `results/`, `reports/figures/`
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 4. Checkpoint - User approval of cleaned structure
  - Present cleaned structure to user
  - Ensure all tests pass, ask the user if questions arise.

### Phase 2: Data Generation (Foundation)

- [ ] 5. Implement fractional Brownian motion data generator
  - [ ] 5.1 Implement `DataGenerator` class with fBm generation
    - Use Davies-Harte algorithm for exact fBm with controllable Hurst exponent H ∈ [0.5, 0.9]
    - Implement `generate_fractional_brownian_motion(n_steps, H)` method
    - _Requirements: 1.1_
  
  - [ ] 5.2 Implement order flow decomposition
    - Implement `generate_order_flow(n_steps, regime)` with informed/noise trader separation
    - Ensure informed traders correlate with future price moves
    - Ensure noise traders are zero-mean and uncorrelated
    - _Requirements: 1.3, 1.4_
  
  - [ ] 5.3 Implement Kyle's price impact equation
    - Implement `generate_price_process(order_flow, lambda_kyle, volatility)` using dP = λ × Q × dt + σ × dW
    - Enforce λ > 0 by construction
    - Enforce permanent impact fraction ∈ [0, 1]
    - _Requirements: 1.2, 1.5, 1.6_
  
  - [ ] 5.4 Implement regime-switching data generation
    - Implement `generate_regime_data(n_days, regime_type)` for low/medium/high liquidity regimes
    - Configure regime-specific parameters (λ, depth, volatility)
    - _Requirements: 1.7_

- [ ] 6. Validate data generation
  - [ ] 6.1 Validate sign of Kyle's lambda
    - Verify λ > 0 for all generated regimes
    - Plot λ by regime with confidence intervals
    - _Requirements: 1.2_
  
  - [ ] 6.2 Validate stationarity
    - Test order flow stationarity using ADF test
    - Test price returns stationarity
    - _Requirements: 1.4, 1.5_
  
  - [ ] 6.3 Validate regime separation
    - Verify regimes have distinct λ values
    - Verify regimes have distinct volatility
    - Plot regime characteristics side-by-side
    - _Requirements: 1.7_
  
  - [ ] 6.4 Validate fBm autocorrelation structure
    - Compute empirical autocorrelation
    - Compare to theoretical ρ(k) ∝ k^(2H-2)
    - Plot empirical vs theoretical ACF
    - _Requirements: 1.1, 1.4_

- [ ] 7. Produce diagnostic plots for data generation
  - Create `reports/figures/data_generation_diagnostics.png` with:
    - Kyle's lambda by regime (with error bars)
    - Autocorrelation function (empirical vs theoretical)
    - Regime separation visualization
    - Order flow time series by regime
  - _Requirements: 8.2_

- [ ] 8. Checkpoint - Data generation validation
  - Verify all validation tests pass
  - Review diagnostic plots
  - If data generation is wrong, everything else is wrong
  - Ensure all tests pass, ask the user if questions arise.

### Phase 3: Model-by-Model Reconstruction

#### Kyle's Lambda Model

- [ ] 9. Implement Kyle model with assumptions
#### Kyle's Lambda Model

- [ ] 9. Implement Kyle model with assumptions
  - [ ] 9.1 Document Kyle model assumptions explicitly
    - Linearity: Impact proportional to order size
    - Homoskedasticity: Constant error variance
    - No autocorrelation in residuals
    - Normality of errors (for inference)
    - _Requirements: 6.1, 10.3_
  
  - [ ] 9.2 Implement `KyleModel` class with OLS calibration
    - Implement `calibrate(order_flow, price_changes)` using OLS regression
    - Compute λ̂, standard errors, t-statistics, p-values, R², adjusted R²
    - Compute 95% confidence intervals: λ̂ ± 1.96 × SE(λ̂)
    - _Requirements: 2.1, 2.4, 2.7, 9.1_

- [ ] 10. Implement Kyle model validation
  - [ ] 10.1 Implement linearity validation
    - Implement `validate_linearity(order_flow, price_changes)` using RESET test
    - Implement quantile regression comparison
    - Identify breakdown points where linearity fails
    - _Requirements: 2.5, 2.6, 2.8_
  
  - [ ] 10.2 Implement assumption validation
    - Implement `validate_assumptions(residuals)` with Breusch-Pagan, Durbin-Watson, Jarque-Bera tests
    - Test homoskedasticity, autocorrelation, normality
    - _Requirements: 2.5, 9.4, 9.5, 9.6_
  
  - [ ] 10.3 Implement regime-specific calibration
    - Implement `regime_calibration(data, regime_labels)`
    - Calibrate separately for low/medium/high liquidity
    - _Requirements: 2.3_

- [ ] 11. Implement Kyle model failure flags
  - [ ] 11.1 Detect non-linearity failures
    - Flag when RESET p-value < 0.05
    - Flag when quantile λ range > 2× median λ
    - _Requirements: 6.2, 6.5_
  
  - [ ] 11.2 Detect heteroskedasticity failures
    - Flag when BP p-value < 0.05
    - _Requirements: 6.2, 6.5_
  
  - [ ] 11.3 Detect autocorrelation failures
    - Flag when DW statistic < 1.5 or > 2.5
    - _Requirements: 6.2, 6.5_
  
  - [ ] 11.4 Detect regime instability failures
    - Flag when CV(λ) across regimes > 0.5
    - _Requirements: 6.2, 6.6_
  
  - [ ] 11.5 Generate Kyle failure report
    - Document all detected failures with economic explanations
    - _Requirements: 6.3, 6.7, 6.8_

- [ ] 12. Checkpoint - Kyle model validation
  - Review calibration results
  - Review validation metrics
  - Review failure flags
  - Ensure all tests pass, ask the user if questions arise.

#### Obizhaeva-Wang Model

- [ ] 13. Implement OW model with assumptions
  - [ ] 13.1 Document OW model assumptions explicitly
    - Exponential decay of transient impact
    - Permanent fraction ∈ [0, 1]
    - Positive resilience rate ρ > 0
    - Finite half-life
    - _Requirements: 6.1, 10.3_
  
  - [ ] 13.2 Implement `ObizhaevaWangModel` class with constrained optimization
    - Implement `calibrate(order_sizes, impacts, times)` with constraints: 0 ≤ γ ≤ 1, ρ > 0
    - Use scipy.optimize.minimize with bounds
    - Compute bootstrap confidence intervals for γ and ρ
    - Compute half-life = ln(2) / ρ
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 9.2_
  
  - [ ] 13.3 Implement impact decomposition
    - Implement `decompose_impact(order_size, time_grid)` computing I_perm, I_temp(t), I_total(t)
    - Use formulas: I_perm = γ × σ × √(Q/V), I_temp(t) = (1-γ) × σ × √(Q/V) × exp(-ρt)
    - _Requirements: 3.1, 3.2_

- [ ] 14. Implement OW model validation
  - [ ] 14.1 Implement exponential decay validation
    - Implement `validate_exponential_decay(impacts, times)` using log-linear regression
    - Compare empirical decay rate to theoretical ρ
    - Compute R² of log-linear fit
    - _Requirements: 3.5, 3.8_
  
  - [ ] 14.2 Implement constraint validation
    - Implement `validate_constraints(gamma, rho)` checking γ ∈ [0,1], ρ > 0, finite half-life
    - Quantify violations explicitly (no NaN values)
    - _Requirements: 3.3, 3.6, 3.7_
  
  - [ ] 14.3 Implement regime-specific calibration
    - Calibrate OW separately for each regime
    - _Requirements: 2.3_

- [ ] 15. Implement OW model failure flags
  - [ ] 15.1 Detect constraint violation failures
    - Flag when γ < 0 or γ > 1
    - Flag when ρ ≤ 0
    - _Requirements: 3.7, 6.2, 6.5_
  
  - [ ] 15.2 Detect non-exponential decay failures
    - Flag when R² of log-linear fit < 0.7
    - _Requirements: 3.8, 6.2, 6.5_
  
  - [ ] 15.3 Detect infinite half-life failures
    - Flag when half-life > 60 minutes
    - _Requirements: 3.4, 6.2, 6.5_
  
  - [ ] 15.4 Detect regime dependence failures
    - Flag when CV(γ) > 0.3 or CV(ρ) > 0.5
    - _Requirements: 6.2, 6.6_
  
  - [ ] 15.5 Generate OW failure report
    - Document all detected failures with economic explanations
    - _Requirements: 6.3, 6.7, 6.8_

- [ ] 16. Checkpoint - OW model validation
  - Review calibration results
  - Review validation metrics
  - Review failure flags
  - Ensure all tests pass, ask the user if questions arise.

#### Bouchaud Propagator Model

- [ ] 17. Implement Bouchaud model with assumptions
  - [ ] 17.1 Document Bouchaud model assumptions explicitly
    - Power-law decay: G(τ) ∝ τ^(-β)
    - Long memory: β < 1
    - Finite total impact (requires cutoff)
    - β ∈ [0.3, 0.8]
    - _Requirements: 6.1, 10.3_
  
  - [ ] 17.2 Implement `BouchaudModel` class with power-law kernel
    - Implement kernel: G(τ) = A / (τ + τ₀)^β with regularization cutoff τ₀
    - Implement `compute_kernel(max_lag)` generating kernel values
    - _Requirements: 4.1, 4.4_
  
  - [ ] 17.3 Implement kernel calibration
    - Implement `calibrate(order_flow, price_changes)` using log-log regression
    - Fit log(G(τ)) = log(A) - β × log(τ + τ₀)
    - Apply constrained optimization: 0.3 ≤ β ≤ 0.8
    - Compute confidence intervals for A and β
    - _Requirements: 4.3, 4.4, 4.7, 9.3_
  
  - [ ] 17.4 Implement impact computation with truncated memory
    - Implement `compute_impact(order_flow)` using convolution with truncated memory horizon
    - Only integrate over [t - T_mem, t]
    - _Requirements: 4.2_

- [ ] 18. Implement Bouchaud model validation
  - [ ] 18.1 Implement power-law validation
    - Implement `validate_power_law(empirical_response)` comparing empirical vs theoretical decay
    - Use log-log regression to test power-law fit
    - Compute R² of log-log fit
    - _Requirements: 4.5, 4.7_
  
  - [ ] 18.2 Implement memory breakdown demonstration
    - Implement `demonstrate_memory_breakdown(order_flow, horizons)` showing when long-memory breaks
    - Compute impact for increasing execution horizons
    - _Requirements: 4.6, 4.8_
  
  - [ ] 18.3 Implement regime-specific calibration
    - Calibrate Bouchaud separately for each regime
    - _Requirements: 2.3_

- [ ] 19. Implement Bouchaud model failure flags
  - [ ] 19.1 Detect parameter divergence failures
    - Flag when β < 0.3 or β > 0.8
    - _Requirements: 4.7, 6.2, 6.5_
  
  - [ ] 19.2 Detect non-power-law failures
    - Flag when R² of log-log fit < 0.6
    - _Requirements: 4.5, 6.2, 6.5_
  
  - [ ] 19.3 Detect memory horizon exceeded failures
    - Flag when execution_time > memory_horizon
    - _Requirements: 4.8, 6.2, 6.5_
  
  - [ ] 19.4 Detect regularization failures
    - Flag when max(impact) > 10 × median(impact)
    - _Requirements: 4.2, 6.2, 6.5_
  
  - [ ] 19.5 Generate Bouchaud failure report
    - Document all detected failures with economic explanations
    - _Requirements: 6.3, 6.7, 6.8_

- [ ] 20. Checkpoint - Bouchaud model validation
  - Review calibration results
  - Review validation metrics
  - Review failure flags
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4: Execution + Frontier

- [ ] 21. Implement execution strategies
### Phase 4: Execution + Frontier

- [ ] 21. Implement execution strategies
  - [ ] 21.1 Implement `ExecutionStrategy` class
    - Implement `generate_schedule()` for TWAP, front-loaded, back-loaded strategies
    - TWAP: uniform slicing
    - Front-loaded: exponentially decreasing slices
    - Back-loaded: exponentially increasing slices
    - _Requirements: 5.1_
  
  - [ ] 21.2 Implement execution cost computation
    - Implement `compute_expected_cost(model, volatility)` computing impact cost and timing risk
    - Timing risk = σ × √(T_exec / 252)
    - _Requirements: 5.2, 5.3_
  
  - [ ] 21.3 Implement execution time optimization
    - Implement `optimize_execution_time(model, volatility, risk_aversion)` finding optimal T_exec
    - Minimize: Impact_Cost + risk_aversion × Timing_Risk
    - _Requirements: 5.5_

- [ ] 22. Implement efficient frontier
  - [ ] 22.1 Implement `EfficientFrontier` class
    - Implement `construct_frontier(order_size, risk_aversions)` computing Pareto frontier
    - For each risk aversion: optimize T_exec, compute costs
    - _Requirements: 5.4, 5.8_
  
  - [ ] 22.2 Implement frontier validation
    - Implement `validate_non_degeneracy()` checking strict convexity and no dominated strategies
    - _Requirements: 5.4, 5.6_
  
  - [ ] 22.3 Implement horizon shift demonstration
    - Implement `demonstrate_horizon_shift(risk_aversion_range)` showing T_exec vs risk aversion
    - _Requirements: 5.5, 5.7_

- [ ] 23. Validate frontier non-degeneracy
  - [ ] 23.1 Test frontier convexity
    - Verify all points lie on convex hull
    - Compute convexity score
    - _Requirements: 5.6_
  
  - [ ] 23.2 Test for dominated strategies
    - Verify no strategy is strictly worse than another
    - _Requirements: 5.6_
  
  - [ ] 23.3 Test risk aversion sensitivity
    - Verify correlation(risk_aversion, T_exec) > 0.5
    - _Requirements: 5.7_
  
  - [ ] 23.4 If frontier is degenerate, explain why
    - Document economic reasons for degeneracy
    - Identify which model assumptions are violated
    - _Requirements: 6.3, 6.8_

- [ ] 24. Checkpoint - Execution and frontier validation
  - Review frontier construction
  - Review non-degeneracy tests
  - If frontier is impossible, ensure explanation is clear
  - Ensure all tests pass, ask the user if questions arise.

### Phase 5: Documentation Pass

- [ ] 25. Write comprehensive README.md
  - [ ] 25.1 Problem statement with mathematical precision
    - Define market impact problem
    - State research objectives
    - _Requirements: 10.1_
  
  - [ ] 25.2 Mathematical intuition for each model
    - Kyle: Linear adverse selection
    - OW: Permanent/transient decomposition
    - Bouchaud: Long-memory propagator
    - _Requirements: 10.2_
  
  - [ ] 25.3 Explicit list of model assumptions
    - Document all assumptions for each model
    - _Requirements: 10.3_
  
  - [ ] 25.4 Empirical findings with statistical significance
    - Present calibration results with confidence intervals
    - Present validation metrics
    - _Requirements: 10.4_
  
  - [ ] 25.5 Failure analysis with economic reasoning
    - Summarize failure modes for each model
    - Explain economic reasons for failures
    - _Requirements: 10.5_
  
  - [ ] 25.6 Interview talking points
    - Key insights for quant interviews
    - When models work and when they break
    - _Requirements: 10.6_
  
  - [ ] 25.7 How to run instructions
    - Installation steps
    - Running the analysis
    - Interpreting results
    - _Requirements: 10.7_
  
  - [ ] 25.8 Results interpretation guide
    - How to read calibration tables
    - How to read validation metrics
    - How to read failure reports
    - _Requirements: 10.8_

- [ ] 26. Write research report
  - [ ] 26.1 Executive summary
    - Key findings
    - Model performance summary
    - Failure mode summary
    - _Requirements: 10.1, 10.4, 10.5_
  
  - [ ] 26.2 Detailed model analysis
    - Calibration results by model and regime
    - Validation results
    - Failure analysis
    - _Requirements: 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 26.3 Efficient frontier analysis
    - Frontier construction results
    - Non-degeneracy validation
    - Economic interpretation
    - _Requirements: 5.4, 5.6, 10.4_

- [ ] 27. Comments cleanup
  - [ ] 27.1 Add comprehensive docstrings
    - Every function: mathematical notation in docstring
    - Every class: purpose and theoretical foundation
    - Every parameter: economic interpretation
    - Every output: units and interpretation
    - _Requirements: 8.7_
  
  - [ ] 27.2 Remove obsolete comments
    - Remove TODO comments
    - Remove debugging comments
    - Remove commented-out code
    - _Requirements: 8.5_
  
  - [ ] 27.3 Ensure consistent style
    - Follow PEP 8
    - Consistent naming conventions
    - _Requirements: 8.5_

- [ ] 28. Final checkpoint - Complete validation
  - Run full analysis pipeline
  - Verify all outputs are research-grade
  - Verify no NaN values anywhere
  - Verify all failure modes are explicitly documented
  - Review README and research report
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Each phase must be completed and validated before proceeding to the next
- Phase 1 ensures clean slate - no "patched together" feel
- Phase 2 is foundation - if data generation is wrong, everything else is wrong
- Phase 3 builds models incrementally with full validation and failure analysis
- Phase 4 only proceeds after models are validated
- Phase 5 ensures research-grade documentation
- All code must enforce theoretical constraints (γ ∈ [0,1], β ∈ [0.3, 0.8], λ > 0)
- No ML or hyperparameter tuning allowed
- Cross-regime validation only (no overfitting)
- Explicit failure detection and economic explanation required for all models
