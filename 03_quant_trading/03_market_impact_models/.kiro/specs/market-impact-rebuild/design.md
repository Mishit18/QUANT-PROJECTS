# Design Document: Research-Grade Market Impact Framework

## Executive Summary

This document specifies the technical design for rebuilding the market impact research framework to advanced quantitative finance standards. The design prioritizes theoretical correctness, explicit failure analysis, and minimal structure suitable for quant interviews.

## Design Principles

1. **Theoretical Validity Over Completeness**: Every component must be mathematically rigorous
2. **Explicit Failure Modes**: Never hide model breakdowns behind aggregate metrics
3. **Minimal Structure**: No unnecessary abstractions or layers
4. **Economic Interpretability**: All parameters must have clear economic meaning
5. **No Overfitting**: Cross-regime validation only, no hyperparameter tuning

## System Architecture

### Directory Structure

```
market_impact/
├── data/
│   └── generate_data.py          # Fractional Brownian motion data generation
├── models/
│   ├── kyle.py                   # Kyle's Lambda with confidence intervals
│   ├── obizhaeva_wang.py         # OW with constraint enforcement
│   └── bouchaud.py               # Bouchaud with regularization
├── execution/
│   └── strategies.py             # TWAP, front/back-loaded strategies
├── analysis/
│   ├── calibration.py            # Model calibration with statistical rigor
│   ├── validation.py             # Cross-regime validation
│   └── failure_modes.py          # Explicit failure analysis
├── results/                      # All CSV outputs
├── reports/
│   └── figures/                  # All visualizations
├── run_analysis.py               # Main execution script
└── README.md                     # Research-grade documentation
```

## Component Design

### 1. Data Generation Module (`data/generate_data.py`)

#### Purpose
Generate synthetic market data with controllable statistical properties using fractional Brownian motion (fBm).

#### Mathematical Foundation

**Fractional Brownian Motion:**
- B_H(t) with Hurst exponent H ∈ [0.5, 0.9]
- H = 0.5: Standard Brownian motion (no memory)
- H > 0.5: Persistent (long memory)
- Autocorrelation: ρ(k) ∝ k^(2H-2)

**Kyle's Price Impact Equation:**
```
dP_t = λ × Q_t × dt + σ × dW_t
```
where:
- P_t: Mid-price
- Q_t: Signed order flow (informed + noise)
- λ: Kyle's lambda (adverse selection coefficient)
- σ: Volatility
- W_t: Standard Brownian motion

**Order Flow Decomposition:**
```
Q_t = Q_informed(t) + Q_noise(t)
Q_informed(t) ~ N(μ_info, σ_info²)  # Correlated with future returns
Q_noise(t) ~ N(0, σ_noise²)         # Uncorrelated
```

#### Implementation Design

**Class: `DataGenerator`**

```python
class DataGenerator:
    def __init__(self, hurst: float = 0.7, lambda_kyle: float = 0.001,
                 volatility: float = 0.02, informed_fraction: float = 0.3):
        """
        Parameters:
        - hurst: Hurst exponent for fBm (0.5 = random walk, >0.5 = persistent)
        - lambda_kyle: Kyle's lambda (must be positive)
        - volatility: Daily volatility
        - informed_fraction: Fraction of order flow that is informed
        """
```

**Key Methods:**

1. `generate_fractional_brownian_motion(n_steps, H)` → np.ndarray
   - Uses Davies-Harte algorithm for exact fBm generation
   - Returns: fBm path with specified Hurst exponent
   
2. `generate_order_flow(n_steps, regime)` → pd.DataFrame
   - Decomposes into informed and noise traders
   - Informed traders: correlated with future price moves
   - Noise traders: zero-mean, uncorrelated
   - Returns: DataFrame with columns [timestamp, Q_informed, Q_noise, Q_total]

3. `generate_price_process(order_flow, lambda_kyle, volatility)` → pd.DataFrame
   - Implements Kyle's price impact equation
   - Ensures λ > 0 by construction
   - Returns: DataFrame with columns [timestamp, price, mid_price, returns]

4. `generate_regime_data(n_days, regime_type)` → pd.DataFrame
   - regime_type ∈ {'low_liquidity', 'medium_liquidity', 'high_liquidity'}
   - Low liquidity: High λ, low depth
   - Medium liquidity: Moderate λ, moderate depth
   - High liquidity: Low λ, high depth
   - Returns: Complete dataset with regime labels

5. `compute_theoretical_statistics()` → Dict
   - Returns theoretical values for validation:
     - Expected λ
     - Expected autocorrelation structure
     - Expected permanent impact fraction
     - Confidence intervals for all parameters

#### Validation Checks

- Assert H ∈ [0.5, 0.9]
- Assert λ > 0
- Assert permanent_fraction ∈ [0, 1]
- Verify autocorrelation matches theoretical fBm
- Verify price process is martingale + drift



---

### 2. Kyle's Lambda Model (`models/kyle.py`)

#### Purpose
Estimate Kyle's lambda with statistical rigor and validate linearity assumption.

#### Mathematical Foundation

**Linear Impact Model:**
```
ΔP_t = α + λ × Q_t + ε_t
```

**Assumptions:**
1. Linearity: Impact proportional to order size
2. Homoskedasticity: Constant error variance
3. No autocorrelation in residuals
4. Normality of errors (for inference)

**Statistical Inference:**
- OLS estimator: λ̂ = (Q'Q)^(-1) Q'ΔP
- Standard error: SE(λ̂) = σ̂ / √(Q'Q)
- 95% CI: λ̂ ± 1.96 × SE(λ̂)
- t-statistic: t = λ̂ / SE(λ̂)

#### Implementation Design

**Class: `KyleModel`**

```python
class KyleModel:
    def __init__(self, confidence_level: float = 0.95):
        """
        Parameters:
        - confidence_level: Confidence level for intervals (default 95%)
        """
```

**Key Methods:**

1. `calibrate(order_flow, price_changes)` → Dict
   - Performs OLS regression
   - Returns: {
       'lambda': float,
       'lambda_ci_lower': float,
       'lambda_ci_upper': float,
       'std_error': float,
       't_statistic': float,
       'p_value': float,
       'r_squared': float,
       'adj_r_squared': float
     }

2. `validate_linearity(order_flow, price_changes)` → Dict
   - Tests linearity using:
     - Residual plots
     - RESET test (Ramsey)
     - Quantile regression comparison
   - Returns: {
       'is_linear': bool,
       'reset_statistic': float,
       'reset_p_value': float,
       'quantile_deviations': np.ndarray
     }

3. `validate_assumptions(residuals)` → Dict
   - Homoskedasticity: Breusch-Pagan test
   - Autocorrelation: Durbin-Watson statistic
   - Normality: Jarque-Bera test
   - Returns: {
       'homoskedastic': bool,
       'bp_statistic': float,
       'bp_p_value': float,
       'dw_statistic': float,
       'autocorrelation_detected': bool,
       'jb_statistic': float,
       'jb_p_value': float,
       'normal_residuals': bool
     }

4. `identify_breakdown_points(order_flow, price_changes, quantiles)` → Dict
   - Splits data by order size quantiles
   - Estimates λ for each quantile
   - Identifies where linearity breaks
   - Returns: {
       'quantile_lambdas': pd.DataFrame,
       'breakdown_threshold': float,
       'linearity_range': Tuple[float, float]
     }

5. `rolling_calibration(data, window_size)` → pd.DataFrame
   - Computes rolling λ estimates
   - Returns: DataFrame with columns [timestamp, lambda, ci_lower, ci_upper, r_squared]

6. `regime_calibration(data, regime_labels)` → pd.DataFrame
   - Calibrates λ separately for each regime
   - Returns: DataFrame with regime-specific estimates and confidence intervals

#### Failure Mode Analysis

**Failure Modes to Detect:**
1. **Non-linearity**: λ varies significantly across order size quantiles
2. **Heteroskedasticity**: Variance increases with order size
3. **Autocorrelation**: Residuals are serially correlated
4. **Regime instability**: λ changes dramatically across regimes

**Detection Criteria:**
- Non-linearity: RESET p-value < 0.05 OR quantile λ range > 2× median λ
- Heteroskedasticity: BP p-value < 0.05
- Autocorrelation: DW statistic < 1.5 or > 2.5
- Regime instability: Coefficient of variation across regimes > 0.5



---

### 3. Obizhaeva-Wang Model (`models/obizhaeva_wang.py`)

#### Purpose
Decompose impact into permanent and transient components with theoretical constraints.

#### Mathematical Foundation

**Total Impact:**
```
I(t) = I_perm + I_temp(t)
```

**Permanent Impact (Information Revelation):**
```
I_perm = γ × σ × √(Q / V)
```
where:
- γ: Permanent impact coefficient
- σ: Volatility
- Q: Order size
- V: Daily volume

**Transient Impact (Liquidity Pressure):**
```
I_temp(t) = (1-γ) × σ × √(Q / V) × exp(-ρt)
```
where:
- ρ: Resilience rate (order book recovery speed)
- t: Time since execution

**Theoretical Constraints:**
1. γ ∈ [0, 1] (permanent fraction must be valid probability)
2. ρ > 0 (decay must be positive)
3. Half-life = ln(2) / ρ must be finite
4. Total impact at t=0: I(0) = σ × √(Q / V)

#### Implementation Design

**Class: `ObizhaevaWangModel`**

```python
class ObizhaevaWangModel:
    def __init__(self, enforce_constraints: bool = True):
        """
        Parameters:
        - enforce_constraints: If True, enforce γ ∈ [0,1] as hard constraint
        """
```

**Key Methods:**

1. `calibrate(order_sizes, impacts, times)` → Dict
   - Uses constrained optimization
   - Minimizes: MSE(I_predicted, I_observed)
   - Subject to: 0 ≤ γ ≤ 1, ρ > 0
   - Returns: {
       'gamma': float,  # Permanent fraction
       'rho': float,    # Resilience rate
       'gamma_ci': Tuple[float, float],
       'rho_ci': Tuple[float, float],
       'half_life': float,
       'mse': float
     }

2. `decompose_impact(order_size, time_grid)` → pd.DataFrame
   - Computes I_perm, I_temp(t), I_total(t) over time grid
   - Returns: DataFrame with columns [time, I_perm, I_temp, I_total]

3. `validate_exponential_decay(impacts, times)` → Dict
   - Tests if decay is truly exponential
   - Fits log(I_temp) vs t
   - Returns: {
       'is_exponential': bool,
       'decay_rate_empirical': float,
       'decay_rate_theoretical': float,
       'deviation': float,
       'r_squared': float
     }

4. `validate_constraints(gamma, rho)` → Dict
   - Checks constraint violations
   - Returns: {
       'gamma_valid': bool,
       'gamma_violation': float,  # 0 if valid, else distance from [0,1]
       'rho_valid': bool,
       'half_life_finite': bool
     }

5. `compute_execution_cost(total_size, execution_time, strategy)` → Dict
   - strategy ∈ {'TWAP', 'front_loaded', 'back_loaded'}
   - Computes cost breakdown
   - Returns: {
       'total_cost': float,
       'permanent_cost': float,
       'transient_cost': float,
       'timing_risk': float
     }

#### Failure Mode Analysis

**Failure Modes to Detect:**
1. **Constraint Violation**: γ < 0 or γ > 1
2. **Non-Exponential Decay**: Empirical decay deviates from exp(-ρt)
3. **Infinite Half-Life**: ρ → 0
4. **Regime Dependence**: γ or ρ vary significantly across regimes

**Detection Criteria:**
- Constraint violation: γ ∉ [0, 1] OR ρ ≤ 0
- Non-exponential: R² of log-linear fit < 0.7
- Infinite half-life: half-life > 60 minutes
- Regime dependence: CV(γ) > 0.3 OR CV(ρ) > 0.5



---

### 4. Bouchaud Propagator Model (`models/bouchaud.py`)

#### Purpose
Model long-memory impact effects with power-law kernel and regularization.

#### Mathematical Foundation

**Propagator Kernel:**
```
G(τ) = A / (τ + τ₀)^β
```
where:
- A: Amplitude
- β: Power-law exponent (0.3 ≤ β ≤ 0.8)
- τ₀: Regularization cutoff (prevents divergence at τ=0)

**Impact via Convolution:**
```
I(t) = ∫₀^t G(t-s) × Q(s) ds
```

**Theoretical Properties:**
1. Long memory: β < 1 implies slow decay
2. Finite total impact: Requires exponential cutoff or finite horizon
3. Memory horizon: T_mem = time when G(T_mem) ≈ 0

**Regularization:**
- Truncated memory: Only integrate over [t - T_mem, t]
- Exponential cutoff: G(τ) = A / τ^β × exp(-τ / T_cutoff)

#### Implementation Design

**Class: `BouchaudModel`**

```python
class BouchaudModel:
    def __init__(self, memory_horizon: int = 120, 
                 regularization_cutoff: float = 1.0):
        """
        Parameters:
        - memory_horizon: Maximum memory length (minutes)
        - regularization_cutoff: τ₀ in kernel to prevent divergence
        """
```

**Key Methods:**

1. `calibrate(order_flow, price_changes)` → Dict
   - Estimates A and β using log-log regression
   - log(G(τ)) = log(A) - β × log(τ + τ₀)
   - Uses constrained optimization: 0.3 ≤ β ≤ 0.8
   - Returns: {
       'amplitude': float,
       'beta': float,
       'amplitude_ci': Tuple[float, float],
       'beta_ci': Tuple[float, float],
       'r_squared': float
     }

2. `compute_kernel(max_lag)` → np.ndarray
   - Generates kernel values G(1), G(2), ..., G(max_lag)
   - Applies regularization
   - Returns: Kernel array

3. `compute_impact(order_flow)` → np.ndarray
   - Convolves order flow with kernel
   - Uses truncated memory horizon
   - Returns: Impact time series

4. `validate_power_law(empirical_response)` → Dict
   - Tests if response follows power law
   - Compares empirical vs theoretical decay
   - Returns: {
       'is_power_law': bool,
       'beta_empirical': float,
       'beta_theoretical': float,
       'deviation': float,
       'r_squared': float
     }

5. `demonstrate_memory_breakdown(order_flow, horizons)` → pd.DataFrame
   - Shows when long-memory assumption breaks
   - Computes impact for increasing execution horizons
   - Returns: DataFrame showing breakdown point

6. `compute_memory_diagnostics(order_flow)` → Dict
   - Autocorrelation function
   - Memory half-life
   - Hurst exponent estimate
   - Returns: {
       'acf': np.ndarray,
       'half_life': float,
       'hurst_estimate': float,
       'long_memory_detected': bool
     }

#### Failure Mode Analysis

**Failure Modes to Detect:**
1. **Parameter Divergence**: β → 0 or β → 1
2. **Non-Power-Law**: Empirical response doesn't follow power law
3. **Memory Horizon Exceeded**: Execution time > memory horizon
4. **Regularization Failure**: Impact diverges despite cutoff

**Detection Criteria:**
- Parameter divergence: β < 0.3 OR β > 0.8
- Non-power-law: R² of log-log fit < 0.6
- Horizon exceeded: execution_time > memory_horizon
- Regularization failure: max(impact) > 10 × median(impact)



---

### 5. Execution Strategies Module (`execution/strategies.py`)

#### Purpose
Implement execution strategies and construct non-degenerate efficient frontier.

#### Mathematical Foundation

**Execution Cost:**
```
Cost = Impact_Cost + Risk_Aversion × Timing_Risk
```

**Impact Cost:**
- Depends on execution speed and strategy
- Faster execution → higher impact
- Slower execution → lower impact

**Timing Risk:**
```
Timing_Risk = σ × √(T_exec / 252)
```
where T_exec is execution time in days

**Efficient Frontier:**
- Pareto-optimal trade-off between cost and risk
- Non-degenerate: Must show clear convex curve
- Risk aversion parameter λ_risk ∈ [0.1, 10]

#### Implementation Design

**Class: `ExecutionStrategy`**

```python
class ExecutionStrategy:
    def __init__(self, strategy_type: str, total_size: float, 
                 execution_time: float):
        """
        Parameters:
        - strategy_type: 'TWAP', 'VWAP', 'front_loaded', 'back_loaded'
        - total_size: Total order size
        - execution_time: Total execution time (minutes)
        """
```

**Key Methods:**

1. `generate_schedule()` → np.ndarray
   - TWAP: Uniform slicing
   - Front-loaded: Exponentially decreasing slices
   - Back-loaded: Exponentially increasing slices
   - Returns: Array of slice sizes over time

2. `compute_expected_cost(model, volatility)` → Dict
   - Uses specified impact model (Kyle, OW, or Bouchaud)
   - Computes impact cost for schedule
   - Computes timing risk
   - Returns: {
       'impact_cost': float,
       'timing_risk': float,
       'total_cost': float
     }

3. `optimize_execution_time(model, volatility, risk_aversion)` → float
   - Finds optimal T_exec minimizing total cost
   - Returns: Optimal execution time

**Class: `EfficientFrontier`**

```python
class EfficientFrontier:
    def __init__(self, model, volatility: float):
        """
        Parameters:
        - model: Impact model (Kyle, OW, or Bouchaud)
        - volatility: Asset volatility
        """
```

**Key Methods:**

1. `construct_frontier(order_size, risk_aversions)` → pd.DataFrame
   - For each risk aversion level:
     - Optimize execution time
     - Compute impact cost and timing risk
   - Returns: DataFrame with columns [risk_aversion, T_exec, impact_cost, timing_risk, total_cost]

2. `validate_non_degeneracy()` → Dict
   - Checks frontier is strictly convex
   - Verifies no dominated strategies
   - Returns: {
       'is_convex': bool,
       'has_dominated_points': bool,
       'frontier_quality': float  # 0-1 score
     }

3. `demonstrate_horizon_shift(risk_aversion_range)` → pd.DataFrame
   - Shows how optimal horizon changes with risk aversion
   - Returns: DataFrame showing relationship

#### Failure Mode Analysis

**Failure Modes to Detect:**
1. **Degenerate Frontier**: All strategies have same cost
2. **Non-Convex**: Frontier is not convex
3. **Dominated Strategies**: Some strategies are strictly worse
4. **Insensitive to Risk Aversion**: Optimal horizon doesn't change

**Detection Criteria:**
- Degenerate: CV(total_cost) < 0.05
- Non-convex: Convexity test fails
- Dominated: Any point inside frontier
- Insensitive: Correlation(risk_aversion, T_exec) < 0.5



---

### 6. Calibration Module (`analysis/calibration.py`)

#### Purpose
Centralize model calibration with statistical rigor.

#### Implementation Design

**Functions:**

1. `calibrate_all_models(data, regimes)` → Dict
   - Calibrates Kyle, OW, and Bouchaud on each regime
   - Computes confidence intervals
   - Returns: Nested dict with all calibration results

2. `bootstrap_confidence_intervals(data, model, n_bootstrap)` → Tuple
   - Generates bootstrap samples
   - Re-calibrates model on each sample
   - Returns: (parameter_mean, ci_lower, ci_upper)

3. `compute_standard_errors(data, model)` → Dict
   - Analytical standard errors where available
   - Returns: Dict of standard errors for each parameter

---

### 7. Validation Module (`analysis/validation.py`)

#### Purpose
Cross-regime validation without overfitting.

#### Implementation Design

**Functions:**

1. `cross_regime_validation(models, data_by_regime)` → pd.DataFrame
   - Calibrate on regime A, validate on regime B
   - Compute out-of-sample errors
   - Returns: DataFrame with validation metrics

2. `parameter_stability_analysis(calibration_results)` → Dict
   - Computes coefficient of variation across regimes
   - Tests parameter stability
   - Returns: {
       'stable_parameters': List[str],
       'unstable_parameters': List[str],
       'cv_by_parameter': Dict
     }

3. `goodness_of_fit_tests(model, data)` → Dict
   - R², adjusted R², AIC, BIC
   - Residual diagnostics
   - Returns: Dict of test statistics

---

### 8. Failure Modes Module (`analysis/failure_modes.py`)

#### Purpose
Explicit failure analysis for all models.

#### Implementation Design

**Class: `FailureModeAnalyzer`**

```python
class FailureModeAnalyzer:
    def __init__(self, model_name: str):
        """
        Parameters:
        - model_name: 'kyle', 'obizhaeva_wang', or 'bouchaud'
        """
```

**Key Methods:**

1. `state_assumptions()` → List[str]
   - Returns list of mathematical assumptions for model

2. `test_assumptions(data, calibration_results)` → Dict
   - Tests each assumption empirically
   - Returns: {
       'assumption': str,
       'test_statistic': float,
       'p_value': float,
       'violated': bool,
       'economic_explanation': str
     }

3. `identify_failure_regimes(data_by_regime, calibration_results)` → pd.DataFrame
   - Identifies which regimes cause model failure
   - Returns: DataFrame with failure indicators by regime

4. `quantify_violations(data, calibration_results)` → Dict
   - Quantifies magnitude of assumption violations
   - Returns: Dict with violation metrics (no NaN values)

5. `generate_failure_report()` → str
   - Generates research-grade failure analysis
   - Returns: Markdown formatted report

---

### 9. Main Analysis Script (`run_analysis.py`)

#### Purpose
Orchestrate complete analysis pipeline.

#### Implementation Design

```python
def main():
    # 1. Generate data
    generator = DataGenerator(hurst=0.7, lambda_kyle=0.001)
    data_low = generator.generate_regime_data(n_days=100, regime='low_liquidity')
    data_med = generator.generate_regime_data(n_days=100, regime='medium_liquidity')
    data_high = generator.generate_regime_data(n_days=100, regime='high_liquidity')
    
    # 2. Calibrate models
    kyle = KyleModel()
    ow = ObizhaevaWangModel(enforce_constraints=True)
    bouchaud = BouchaudModel(memory_horizon=120)
    
    calibration_results = calibrate_all_models(
        {'low': data_low, 'med': data_med, 'high': data_high},
        [kyle, ow, bouchaud]
    )
    
    # 3. Validate models
    validation_results = cross_regime_validation(
        [kyle, ow, bouchaud],
        {'low': data_low, 'med': data_med, 'high': data_high}
    )
    
    # 4. Analyze failure modes
    for model_name in ['kyle', 'obizhaeva_wang', 'bouchaud']:
        analyzer = FailureModeAnalyzer(model_name)
        failure_report = analyzer.generate_failure_report()
        save_report(failure_report, f'reports/failure_modes_{model_name}.md')
    
    # 5. Construct efficient frontier
    frontier = EfficientFrontier(ow, volatility=0.02)
    frontier_data = frontier.construct_frontier(
        order_size=100000,
        risk_aversions=np.logspace(-1, 1, 20)
    )
    
    # 6. Generate visualizations
    plot_calibration_results(calibration_results)
    plot_validation_results(validation_results)
    plot_efficient_frontier(frontier_data)
    plot_failure_modes(failure_reports)
    
    # 7. Save results
    save_results(calibration_results, 'results/calibration.csv')
    save_results(validation_results, 'results/validation.csv')
    save_results(frontier_data, 'results/efficient_frontier.csv')
```



---

## Data Flow

```
DataGenerator
    ↓
[Low/Med/High Liquidity Data]
    ↓
Calibration Module
    ↓
[Kyle, OW, Bouchaud Models]
    ↓
Validation Module → Failure Mode Analyzer
    ↓                      ↓
[Validation Metrics]  [Failure Reports]
    ↓
Execution Strategies
    ↓
[Efficient Frontier]
    ↓
Visualization & Reporting
```

---

## Output Specifications

### Results Files (CSV format in `results/`)

1. `calibration_kyle.csv`: Kyle's lambda by regime with confidence intervals
2. `calibration_ow.csv`: OW parameters (γ, ρ) by regime with confidence intervals
3. `calibration_bouchaud.csv`: Bouchaud parameters (A, β) by regime with confidence intervals
4. `validation_metrics.csv`: Out-of-sample errors by model and regime
5. `efficient_frontier.csv`: Cost-risk trade-off points
6. `failure_modes_summary.csv`: Assumption violations by model and regime

### Figures (PNG format in `reports/figures/`)

1. `kyle_lambda_by_regime.png`: Bar chart with error bars
2. `kyle_linearity_test.png`: Residual plots and quantile comparison
3. `ow_decay_curves.png`: Empirical vs theoretical decay
4. `ow_constraint_validation.png`: γ estimates with [0,1] bounds highlighted
5. `bouchaud_kernel.png`: Log-log plot of kernel
6. `bouchaud_memory_breakdown.png`: Impact vs execution horizon
7. `efficient_frontier.png`: Cost vs risk with Pareto curve
8. `failure_modes_heatmap.png`: Assumption violations by model × regime

---

## Testing Strategy

### Unit Tests

1. Data generation:
   - Test H ∈ [0.5, 0.9] enforcement
   - Test λ > 0 enforcement
   - Test autocorrelation matches fBm theory

2. Kyle model:
   - Test confidence interval coverage
   - Test linearity detection
   - Test breakdown point identification

3. OW model:
   - Test constraint enforcement (γ ∈ [0,1])
   - Test exponential decay validation
   - Test half-life computation

4. Bouchaud model:
   - Test regularization prevents divergence
   - Test power-law validation
   - Test memory horizon enforcement

5. Execution strategies:
   - Test schedule generation
   - Test frontier non-degeneracy
   - Test convexity validation

### Integration Tests

1. End-to-end pipeline:
   - Generate data → Calibrate → Validate → Report
   - Verify all outputs are created
   - Verify no NaN values in results

2. Cross-regime consistency:
   - Verify parameter stability tests work
   - Verify validation metrics are computed correctly

---

## Performance Requirements

- Data generation: < 5 seconds for 100 days
- Model calibration: < 10 seconds per model per regime
- Validation: < 30 seconds for all cross-regime tests
- Visualization: < 20 seconds for all plots
- Total runtime: < 2 minutes for complete analysis

---

## Documentation Requirements

### README.md Structure

1. Problem Statement (mathematical precision)
2. Mathematical Intuition (each model)
3. Model Assumptions (explicit list)
4. Empirical Findings (with statistical significance)
5. Failure Analysis (economic reasoning)
6. Interview Talking Points
7. How to Run
8. Results Interpretation

### Code Documentation

- Every function: Docstring with mathematical notation
- Every class: Purpose and theoretical foundation
- Every parameter: Economic interpretation
- Every output: Units and interpretation

---

## Design Validation Checklist

- [ ] All parameters have economic meaning
- [ ] No ML or hyperparameter tuning
- [ ] Confidence intervals for all estimates
- [ ] Explicit failure mode detection
- [ ] Cross-regime validation only
- [ ] Constraint enforcement (γ ∈ [0,1], β ∈ [0.3, 0.8])
- [ ] Non-degenerate efficient frontier
- [ ] No NaN values in outputs
- [ ] Minimal structure (no unnecessary abstractions)
- [ ] Research-grade documentation

---

## Key Design Decisions and Rationale

### 1. Why Fractional Brownian Motion?
- Captures realistic autocorrelation in order flow
- Controllable Hurst exponent allows testing across memory regimes
- Theoretically grounded (used in academic literature)

### 2. Why Hard Constraints on OW Parameters?
- γ ∈ [0,1] is a theoretical requirement (permanent fraction is a probability)
- Violations indicate model breakdown, not estimation error
- Forces explicit failure detection rather than hiding behind unconstrained estimates

### 3. Why Regularization in Bouchaud Kernel?
- Pure power-law diverges at τ=0
- Exponential cutoff is theoretically justified (finite market memory)
- Prevents numerical instability

### 4. Why Cross-Regime Validation Only?
- Avoids overfitting to specific market conditions
- Tests model robustness, not just fit quality
- Aligns with how models are used in practice (calibrate on one period, apply to another)

### 5. Why Explicit Failure Mode Analysis?
- Demonstrates deep understanding of model limitations
- Critical for quant interviews (interviewers will ask "when does this break?")
- Prevents misuse of models outside validity range

---

**Design Status:** DRAFT - Awaiting User Review

**Next Step:** Review design and approve before proceeding to task breakdown.
