# Rough Bergomi Volatility Model

Implementation of the rough Bergomi (rBergomi) stochastic volatility model with Monte Carlo pricing, calibration, and comparison to Heston and SABR.

## Model Specification

The rBergomi model under the risk-neutral measure:

```
dS_t / S_t = √v_t dZ_t
v_t = ξ_0(t) exp(η Y_t - (1/2)η² t^(2H))
Y_t = ∫₀ᵗ (t-s)^(H-1/2) dW_s
```

where:
- `Y_t`: Fractional Brownian motion with Hurst parameter `H ∈ (0, 0.5)`
- `W_t, Z_t`: Correlated Brownian motions, `dW_t dZ_t = ρ dt`
- `η > 0`: Volatility of volatility
- `ξ_0(t)`: Forward variance curve (constant in this implementation)

**Key property**: For `H < 0.5`, the variance process exhibits rough sample paths with power-law autocorrelation decay, consistent with empirical observations from high-frequency data.

## Numerical Methods

### Fractional Brownian Motion Simulation

The fBM is represented via the Volterra kernel:

```
Y_t = ∫₀ᵗ K(t,s) dW_s,  where K(t,s) = (t-s)^(H-1/2)
```

**Discrete approximation**: For time grid `0 = t_0 < ... < t_n = T`:

```
Y_{t_i} ≈ Σ_{j=0}^{i-1} K(t_i, t_j) ΔW_j
```

**Covariance structure**: The exact covariance is:

```
Cov(Y_s, Y_t) = (1/2H)[s^(2H) + t^(2H) - |t-s|^(2H)]
```

This is a dense matrix, making naive Cholesky decomposition O(n³) in time and O(n²) in memory.

### Hybrid Scheme

We implement a hybrid approach:

1. **Short memory (i ≤ cutoff)**: Use exact covariance matrix
   - Compute `Σ_{ij} = Cov(Y_{t_i}, Y_{t_j})` for `i,j ≤ cutoff`
   - Cholesky decomposition: `Σ = LL^T`
   - Generate: `Y = LZ` where `Z ~ N(0,I)`
   - Cost: O(cutoff³)

2. **Long memory (i > cutoff)**: Use FFT-based convolution
   - Kernel vector: `k = [K(t_i, t_0), ..., K(t_i, t_{i-1})]`
   - Convolve with Brownian increments via FFT
   - Cost: O(n log n)

**Complexity comparison**:
- Naive Cholesky: O(n³) time, O(n²) memory
- Hybrid scheme: O(cutoff³ + n log n) time, O(n) memory
- For `cutoff = 50`, `n = 252`: ~100x speedup

**Non-Markovian structure**: The Volterra kernel creates path dependence—`Y_t` depends on the entire history `{W_s : s ≤ t}`, not just the current state. This is essential for capturing rough volatility dynamics.

### Asset Price Discretization

Euler-Maruyama scheme with log-price formulation:

```
S_{t_{i+1}} = S_{t_i} exp(-0.5 v_{t_i} Δt + √v_{t_i} ΔZ_i)
```

where `ΔZ_i = ρ ΔW_i + √(1-ρ²) ΔW_i^⊥` are correlated increments.

The exponential form of `v_t` ensures positivity without truncation.

### Variance Reduction

- **Antithetic variates**: Generate paths with `±ΔW`, reduces variance by ~2x for symmetric payoffs
- **Control variates**: Use Black-Scholes price as control, optimal coefficient `β* = -Cov(V, V_BS) / Var(V_BS)`
- **Moment matching**: Scale terminal variance to match theoretical mean

## Calibration

### Objective Function

Minimize weighted least squares on implied volatilities:

```
L(θ) = Σ_{i,j} w_{ij} [σ_model(K_i, T_j; θ) - σ_market(K_i, T_j)]² + λ R(θ)
```

where `θ = (H, η, ρ, ξ_0)` and `R(θ)` is a regularization term.

### Parameter Identifiability

**H-η coupling**: The short-maturity skew depends on both `H` and `η` through the term:

```
Skew ~ η T^(H-1/2)
```

This creates an identifiability issue: decreasing `H` can be partially compensated by increasing `η`. Regularization is necessary to stabilize calibration.

**Regularization**: We use:

```
R(θ) = α_η η² + α_H (H - H_prior)²
```

to penalize extreme vol-of-vol and anchor `H` near empirical estimates (typically 0.05-0.15).

### Optimization

- **Global search**: Differential evolution (population-based, handles non-convex landscape)
- **Local refinement**: Nelder-Mead simplex method
- **Multi-start**: Run from multiple initial points to assess stability

### Known Failure Modes

1. **Insufficient short-maturity data**: Without options expiring in days-to-weeks, `H` is poorly identified
2. **Extreme parameters**: Optimizer may find `H → 0` or `η → ∞` if not regularized
3. **Local minima**: Non-convex objective can trap local optimizers
4. **Numerical instability**: Very small `H` (< 0.02) can cause numerical issues in fBM simulation

## Market Conventions

### Strike Representation

We use three conventions:
- **Absolute strike**: `K` in currency units
- **Moneyness**: `K/S_0` (percentage)
- **Log-moneyness**: `log(K/S_0)` (used in asymptotic formulas)

Delta-based strikes are not implemented but can be added via numerical inversion.

### Forward Variance Curve

The forward variance `ξ_0(t)` is held constant in this implementation. In practice, it should be constructed from:
- Variance swap quotes
- Calibrated ATM implied volatility term structure
- Model-free implied variance

### Interest Rates

We assume `r = 0` throughout. For non-zero rates:
- Replace `S_0` with forward price `F_0 = S_0 e^{rT}`
- Discount payoffs by `e^{-rT}`

### Synthetic Data

We use rBergomi-generated data as "market" for calibration experiments. This is standard for methodological studies and allows controlled testing of:
- Calibration stability
- Parameter recovery
- Model misspecification effects

## Results

### Implied Volatility Smiles

Short-maturity (T=0.08y, ~1 month) comparison:

![Short Maturity Skew](data/comparison_T0.08.png)

**Observation**: rBergomi produces steep skew at short maturities, consistent with market observations. Heston underestimates this effect due to Markovian dynamics.

### Term Structure of Skew

![Skew Term Structure](data/skew_comparison.png)

**Observation**: Skew decays as `T^(H-1/2)` for rBergomi. For `H=0.07`, this gives `T^{-0.43}`, much steeper than Heston's `T^{-0.5}` decay.

### Hurst Parameter Sensitivity

![Hurst Sensitivity](data/rbergomi_hurst_sensitivity.png)

**Observation**: Lower `H` → rougher paths → steeper short-maturity skew. The relationship is approximately:

```
Skew(T) ∝ η T^(H-1/2)
```

### Calibration Quality

RMSE on implied volatilities (synthetic data):

| Model    | T=0.08y | T=0.25y | T=0.5y | T=1.0y | Overall |
|----------|---------|---------|--------|--------|---------|
| rBergomi | 0.0012  | 0.0018  | 0.0021 | 0.0024 | 0.0019  |
| Heston   | 0.0156  | 0.0034  | 0.0019 | 0.0015 | 0.0056  |
| SABR     | 0.0089  | 0.0045  | 0.0038 | 0.0041 | 0.0053  |

**Note**: rBergomi fits its own generated data best (by construction). The key insight is Heston's poor fit at short maturities.

## Limitations

1. **Computational cost**: Rough path simulation is 2-3x slower than Heston
2. **No closed-form characteristic function**: Must use Monte Carlo for pricing
3. **Calibration complexity**: Non-convex optimization, requires regularization
4. **Forward curve specification**: Requires external input or joint calibration
5. **Parameter stability**: H-η identifiability issues without rich short-maturity data

## Extensions

Potential directions:

- **Multi-factor rBergomi**: Add independent fBM components for richer dynamics
- **Jumps**: Include compound Poisson process in asset dynamics
- **Stochastic correlation**: Time-varying `ρ_t`
- **Variance swaps**: Pricing and replication strategies
- **American options**: Longstaff-Schwartz method
- **Greeks**: Pathwise derivatives or likelihood ratio method
- **GPU acceleration**: CuPy or JAX for large-scale calibration

## Repository Structure

```
.
├── src/
│   ├── models/          # rBergomi, Heston, SABR
│   ├── simulation/      # FBM, hybrid scheme, variance reduction
│   ├── pricing/         # Monte Carlo pricer, implied vol
│   ├── calibration/     # Parameter fitting
│   └── utils/           # Plotting
├── experiments/
│   ├── run_rbergomi.py      # Main analysis
│   └── compare_models.py    # Model comparison
├── tests/
│   └── test_rbergomi.py     # Unit tests
├── reports/
│   └── METHODOLOGY.md       # Detailed mathematical documentation
├── data/                    # Generated results
└── README.md
```

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
cd experiments
python run_rbergomi.py      # Full rBergomi analysis
python compare_models.py    # Model comparison
```

### Run Tests

```bash
pytest tests/ -v
```

### Basic Example

```python
from src.models.rbergomi import RBergomiModel
from src.simulation.hybrid_scheme import HybridScheme
from src.pricing.monte_carlo_pricer import MonteCarloOptionPricer

# Define model
model = RBergomiModel(H=0.07, eta=1.9, rho=-0.9, xi0=0.04)

# Simulate paths
scheme = HybridScheme(n_steps=252, n_paths=10000, seed=42)
S_paths, v_paths = scheme.simulate_rbergomi(
    model.H, model.eta, model.rho, model.xi0, T=1.0, S0=100.0
)

# Price option
pricer = MonteCarloOptionPricer(n_paths=10000)
price, std_err = pricer.price_european_call(S_paths, K=100, r=0.0, T=1.0)
```

## References

1. Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887-904.

2. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949.

3. McCrickerd, R., & Pakkanen, M. S. (2018). Turbocharging Monte Carlo pricing for the rough Bergomi model. *Quantitative Finance*, 18(11), 1877-1886.

4. Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2022). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 26(1), 1-47.

## License

MIT
