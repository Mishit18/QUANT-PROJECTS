# Requirements Document: Research-Grade Market Impact Framework Rebuild

## Introduction

This specification defines the requirements for rebuilding the market impact research framework to meet advanced quantitative finance interview standards. The focus is on theoretical correctness, explicit failure analysis, and clean minimal structure without ML overfitting.

## Glossary

- **Kyle's Lambda (λ)**: Linear price impact coefficient from Kyle (1985)
- **Obizhaeva-Wang (OW)**: Transient impact model with permanent/temporary decomposition
- **Bouchaud Propagator**: Long-memory impact kernel with power-law decay
- **Hurst Exponent (H)**: Measure of long-range dependence in time series (0.5 = random walk, >0.5 = persistent)
- **Metaorder**: Large parent order split into smaller child orders
- **Permanent Impact**: Information revelation component that persists
- **Transient Impact**: Temporary liquidity pressure that decays
- **Efficient Frontier**: Pareto-optimal trade-off between execution cost and timing risk

## Requirements

### Requirement 1: Structured Data Generation Process

**User Story:** As a quantitative researcher, I want realistic synthetic data with controllable statistical properties, so that I can validate models under known conditions.

#### Acceptance Criteria

1. THE Data_Generator SHALL produce order flow with controllable Hurst exponent H ∈ [0.5, 0.9]
2. THE Data_Generator SHALL ensure Kyle's lambda is positive by construction
3. THE Data_Generator SHALL decompose order flow into informed traders and noise traders
4. THE Data_Generator SHALL produce signed volume with realistic autocorrelation decay
5. THE Data_Generator SHALL generate price process consistent with Kyle and Bouchaud assumptions
6. THE Data_Generator SHALL enforce permanent impact fraction ∈ [0, 1]
7. THE Data_Generator SHALL support regime switching (low/medium/high liquidity)
8. THE Data_Generator SHALL produce statistically valid confidence intervals for all parameters

### Requirement 2: Kyle's Lambda Model with Validation

**User Story:** As a quantitative researcher, I want Kyle's lambda estimated with statistical rigor, so that I can understand when the linear assumption holds.

#### Acceptance Criteria

1. THE Kyle_Model SHALL estimate lambda using signed volume regression with OLS
2. THE Kyle_Model SHALL compute rolling window estimates with configurable window size
3. THE Kyle_Model SHALL provide regime-conditional calibration
4. THE Kyle_Model SHALL compute 95% confidence intervals for lambda estimates
5. THE Kyle_Model SHALL validate linearity assumption using residual analysis
6. THE Kyle_Model SHALL identify breakdown points for large metaorder sizes
7. THE Kyle_Model SHALL report R-squared and standard errors
8. WHEN order size exceeds linearity range THEN THE Kyle_Model SHALL flag the violation

### Requirement 3: Obizhaeva-Wang Model with Theory Alignment

**User Story:** As a quantitative researcher, I want OW model implementation that respects theoretical constraints, so that I can trust the permanent/transient decomposition.

#### Acceptance Criteria

1. THE OW_Model SHALL implement transient impact with exponential decay kernel
2. THE OW_Model SHALL implement permanent impact as information revelation
3. THE OW_Model SHALL enforce permanent fraction ∈ [0, 1] as hard constraint
4. THE OW_Model SHALL compute finite decay half-life
5. THE OW_Model SHALL validate theoretical decay against empirical decay
6. THE OW_Model SHALL quantify assumption violations explicitly (no NaN values)
7. WHEN permanent fraction > 1 THEN THE OW_Model SHALL report constraint violation
8. WHEN decay is non-exponential THEN THE OW_Model SHALL quantify deviation from theory

### Requirement 4: Bouchaud Propagator with Controlled Memory

**User Story:** As a quantitative researcher, I want Bouchaud kernel with regularization, so that I can demonstrate when long-memory holds and when it breaks.

#### Acceptance Criteria

1. THE Bouchaud_Model SHALL implement power-law kernel with exponential cutoff
2. THE Bouchaud_Model SHALL use truncated memory horizon to prevent divergence
3. THE Bouchaud_Model SHALL calibrate kernel using log-log regression
4. THE Bouchaud_Model SHALL apply regularization to prevent parameter divergence
5. THE Bouchaud_Model SHALL demonstrate when long-memory assumption holds
6. THE Bouchaud_Model SHALL demonstrate when long-memory breaks under extended execution
7. THE Bouchaud_Model SHALL validate power-law exponent β ∈ [0.3, 0.8]
8. WHEN execution horizon exceeds memory horizon THEN THE Bouchaud_Model SHALL flag breakdown

### Requirement 5: Execution Strategies with Non-Degenerate Frontier

**User Story:** As a quantitative researcher, I want execution strategies with proper cost-risk trade-offs, so that I can demonstrate optimal execution selection.

#### Acceptance Criteria

1. THE Execution_Module SHALL implement TWAP, VWAP, front-loaded, and back-loaded strategies
2. THE Execution_Module SHALL compute expected execution cost for each strategy
3. THE Execution_Module SHALL compute timing risk as price variance during execution
4. THE Execution_Module SHALL construct non-degenerate efficient frontier
5. THE Execution_Module SHALL show horizon shifts with varying risk aversion
6. THE Execution_Module SHALL validate that frontier is strictly convex
7. WHEN risk aversion increases THEN THE Execution_Module SHALL select longer horizons
8. THE Execution_Module SHALL demonstrate trade-off between impact cost and timing risk

### Requirement 6: Failure Mode Analysis for All Models

**User Story:** As a quantitative researcher, I want explicit failure analysis for each model, so that I can explain when and why models break in interviews.

#### Acceptance Criteria

1. FOR EACH model THE Failure_Analyzer SHALL state mathematical assumptions explicitly
2. FOR EACH model THE Failure_Analyzer SHALL demonstrate empirical violations
3. FOR EACH model THE Failure_Analyzer SHALL explain economic reasons for failure
4. THE Failure_Analyzer SHALL never hide failures behind aggregate metrics
5. THE Failure_Analyzer SHALL quantify assumption violations with specific metrics
6. THE Failure_Analyzer SHALL provide regime-specific failure analysis
7. THE Failure_Analyzer SHALL generate failure mode visualizations
8. THE Failure_Analyzer SHALL document failure modes in research-grade language

### Requirement 7: Cross-Regime Validation Without Overfitting

**User Story:** As a quantitative researcher, I want validation that proves model robustness, so that I can demonstrate understanding without overfitting.

#### Acceptance Criteria

1. THE Validation_Module SHALL use cross-regime validation only (no hyperparameter tuning)
2. THE Validation_Module SHALL ensure all parameters have economic meaning
3. THE Validation_Module SHALL prohibit machine learning methods
4. THE Validation_Module SHALL prohibit parameter tuning to match specific outputs
5. THE Validation_Module SHALL validate models on out-of-sample regimes
6. THE Validation_Module SHALL report validation metrics separately from calibration metrics
7. THE Validation_Module SHALL demonstrate parameter stability across regimes
8. WHEN parameters lack economic interpretation THEN THE Validation_Module SHALL reject them

### Requirement 8: Simplified Project Structure

**User Story:** As a quantitative researcher, I want clean minimal structure, so that the code is production-ready and easy to understand.

#### Acceptance Criteria

1. THE Project SHALL use only the specified directory structure (data/, models/, execution/, analysis/)
2. THE Project SHALL save all figures to reports/figures/ only
3. THE Project SHALL save all tables to results/ only
4. THE Project SHALL have no scattered outputs across multiple directories
5. THE Project SHALL have no unnecessary abstraction layers
6. THE Project SHALL have clear module boundaries
7. THE Project SHALL have comprehensive docstrings with mathematical notation
8. THE Project SHALL have a research-grade README with problem statement, assumptions, findings, and failure analysis

### Requirement 9: Statistical Rigor and Confidence Intervals

**User Story:** As a quantitative researcher, I want statistical confidence intervals for all estimates, so that I can quantify estimation uncertainty.

#### Acceptance Criteria

1. THE Statistical_Module SHALL compute 95% confidence intervals for Kyle's lambda
2. THE Statistical_Module SHALL compute bootstrap confidence intervals for OW parameters
3. THE Statistical_Module SHALL compute standard errors for Bouchaud kernel parameters
4. THE Statistical_Module SHALL perform residual diagnostics for all regressions
5. THE Statistical_Module SHALL test for heteroskedasticity
6. THE Statistical_Module SHALL test for autocorrelation in residuals
7. THE Statistical_Module SHALL report goodness-of-fit metrics (R², adjusted R², AIC, BIC)
8. THE Statistical_Module SHALL validate normality assumptions where applicable

### Requirement 10: Research-Grade Documentation

**User Story:** As a quantitative researcher, I want documentation that demonstrates deep understanding, so that interviewers recognize this as serious research.

#### Acceptance Criteria

1. THE README SHALL state the problem with mathematical precision
2. THE README SHALL explain mathematical intuition for each model
3. THE README SHALL list model assumptions explicitly
4. THE README SHALL present empirical findings with statistical significance
5. THE README SHALL explain why models fail with economic reasoning
6. THE README SHALL provide interview talking points
7. THE README SHALL avoid toy explanations and academic fluff
8. THE README SHALL demonstrate microstructure theory understanding

## Special Requirements Guidance

### Data Generation Requirements

The data generation process is CRITICAL. It must:
- Use fractional Brownian motion for order flow with controllable H
- Implement Kyle's price impact equation: dP = λ × Q × dt + σ × dW
- Separate informed (adverse selection) from noise traders
- Ensure autocorrelation structure matches empirical findings
- Generate regime-switching liquidity with Markov transitions

### Model Validation Requirements

Each model must include:
- Assumption testing (linearity, exponential decay, power-law)
- Residual diagnostics (normality, homoskedasticity, independence)
- Out-of-sample validation on different regimes
- Explicit failure identification with economic explanation

### No Overfitting Rules

Strictly enforce:
- No machine learning methods
- No hyperparameter tuning to match outputs
- All parameters must have economic interpretation
- Use theory-driven parameter bounds only
- Cross-regime validation without calibration

## Iteration and Feedback Rules

- The model MUST ask for explicit approval after requirements review
- The model MUST make modifications if the user requests changes
- The model MUST continue the feedback-revision cycle until explicit approval
- The model MUST NOT proceed to design until requirements are approved
- The model MUST incorporate all user feedback before proceeding

---

**Requirements Status:** DRAFT - Awaiting User Review

**Next Step:** Review requirements and approve before proceeding to design phase.
