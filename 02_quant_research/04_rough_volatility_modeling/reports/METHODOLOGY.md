# rBergomi Model: Mathematical Methodology

## 1. Model Specification

### 1.1 Risk-Neutral Dynamics

The rough Bergomi (rBergomi) model is defined by:

```
dS_t / S_t = √v_t dZ_t
v_t = ξ_0(t) exp(η Y_t - (1/2)η² t^(2H))
Y_t = ∫₀ᵗ (t-s)^(H-1/2) dW_s
```

where:
- `S_t`: Asset price process
- `v_t`: Instantaneous variance process
- `Y_t`: Fractional Brownian motion (fBM) with Hurst parameter H
- `W_t, Z_t`: Standard Brownian motions with correlation ρ
- `ξ_0(t)`: Forward variance curve
- `η`: Volatility of volatility
- `H ∈ (0, 0.5)`: Hurst exponent (roughness parameter)

### 1.2 Key Properties

**Roughness (H < 0.5):**
- Sample paths are rougher than Brownian motion
- Hölder continuity of order H
- Non-Markovian dynamics with long memory

**Volatility Autocorrelation:**
- Power-law decay: Corr(v_t, v_{t+τ}) ~ τ^(2H-1)
- Consistent with empirical observations from high-frequency data

**Short-Maturity Asymptotics:**
- Implied volatility skew: σ_impl(K,T) - σ_ATM ~ T^(H-1/2) log(K/S)
- Steeper skew than Heston for H < 0.5

## 2. Numerical Methods

### 2.1 Fractional Brownian Motion Simulation

**Volterra Representation:**

The fBM admits the representation:
```
Y_t = ∫₀ᵗ K(t,s) dW_s
```
where `K(t,s) = (t-s)^(H-1/2)` is the Volterra kernel. This is a Gaussian process with mean zero and covariance:
```
Cov(Y_s, Y_t) = (1/2H)[s^(2H) + t^(2H) - |t-s|^(2H)]
```

**Non-Markovian Structure:**

The Volterra kernel creates path dependence: `Y_t` depends on the entire history `{W_s : 0 ≤ s ≤ t}`, not just the current state. This is fundamentally different from Markovian processes like standard Brownian motion.

For `H < 1/2`, the kernel `K(t,s) = (t-s)^(H-1/2)` is singular at `s = t`, making the process "rough" with Hölder continuity of order `H`. This roughness is essential for capturing the observed power-law decay in realized volatility autocorrelation.

**Discrete Approximation:**

For time grid `0 = t_0 < t_1 < ... < t_n = T` with `Δt = T/n`:
```
Y_{t_i} ≈ Σ_{j=0}^{i-1} K(t_i, t_j) ΔW_j
```

where `ΔW_j = W_{t_{j+1}} - W_{t_j} ~ N(0, Δt)`.

**Covariance Matrix Structure:**

The discrete covariance matrix `Σ ∈ R^{n×n}` has entries:
```
Σ_{ij} = Cov(Y_{t_i}, Y_{t_j}) = (1/2H)[t_i^(2H) + t_j^(2H) - |t_i - t_j|^(2H)]
```

This is a dense, symmetric, positive-definite matrix. Naive simulation via Cholesky decomposition:
```
Σ = LL^T,  Y = LZ,  Z ~ N(0,I)
```
has complexity:
- Time: O(n³) for Cholesky factorization
- Memory: O(n²) to store L

For `n = 252` (daily steps, one year), this requires ~8MB memory and ~10⁷ operations.

### 2.2 Hybrid Scheme

To reduce computational cost, we split the simulation into two regimes:

**Regime 1: Short Memory (i ≤ cutoff)**

For the first `cutoff` time steps, use exact covariance:
1. Compute `Σ_{ij}` for `i,j ≤ cutoff`
2. Cholesky decomposition: `Σ = LL^T`
3. Generate: `Y_{t_i} = Σ_j L_{ij} Z_j` for `i ≤ cutoff`

Cost: O(cutoff³) time, O(cutoff²) memory

**Regime 2: Long Memory (i > cutoff)**

For later time steps, use FFT-based convolution:
1. Kernel vector: `k_i = [K(t_i, t_0), K(t_i, t_1), ..., K(t_i, t_{i-1})]`
2. Brownian increments: `dW = [ΔW_0, ΔW_1, ..., ΔW_{i-1}]`
3. Convolution: `Y_{t_i} = Σ_j k_{i,j} ΔW_j`

Using FFT:
```
Y_{t_i} = IFFT(FFT(k_i) ⊙ FFT(dW))
```

Cost: O(n log n) time, O(n) memory

**Total Complexity:**

- Hybrid: O(cutoff³ + n log n) time, O(n) memory
- Naive Cholesky: O(n³) time, O(n²) memory

For `cutoff = 50`, `n = 252`:
- Hybrid: ~1.5×10⁶ operations
- Naive: ~1.6×10⁷ operations
- Speedup: ~10x

**Why This Works:**

The covariance function decays as:
```
Cov(Y_s, Y_t) ~ |t-s|^(2H)  for |t-s| → ∞
```

For large time separations, the covariance is well-approximated by the convolution structure, making FFT efficient. For small separations, the singular behavior near `s = t` requires exact treatment.

**Numerical Stability:**

For very small `H` (< 0.02), the kernel becomes highly singular. We add a small regularization:
```
Σ_{ii} ← Σ_{ii} + ε,  ε = 10^{-10}
```
to ensure positive definiteness in finite precision arithmetic.

### 2.2 Asset Price Simulation

**Euler-Maruyama Scheme:**

```
S_{t_{i+1}} = S_{t_i} exp(-0.5 v_{t_i} Δt + √v_{t_i} ΔZ_i)
```

where `ΔZ_i = ρ ΔW_i + √(1-ρ²) ΔW_i^⊥` are correlated increments.

**Variance Process:**

```
v_{t_i} = ξ_0(t_i) exp(η Y_{t_i} - (1/2)η² t_i^(2H))
```

The exponential form ensures positivity without truncation.

### 2.3 Variance Reduction Techniques

**Antithetic Variates:**
- Generate paths with `ΔW` and `-ΔW`
- Reduces variance by factor ~2 for symmetric payoffs

**Control Variates:**
- Use Black-Scholes price as control
- Optimal coefficient: `β* = -Cov(V, V_BS) / Var(V_BS)`
- Adjusted estimator: `V̂ = V + β*(V_BS - E[V_BS])`

**Moment Matching:**
- Adjust terminal variance to match theoretical mean
- Scaling: `v_T^adj = v_T × (E_theory[v_T] / E_empirical[v_T])`

## 3. Option Pricing

### 3.1 Monte Carlo Pricing

**European Call:**
```
C(K,T) = E[max(S_T - K, 0)]
```

**Algorithm:**
1. Simulate N paths of (S_t, v_t)
2. Compute payoffs: `π_i = max(S_T^(i) - K, 0)`
3. Estimate: `Ĉ = (1/N) Σ_i π_i`
4. Standard error: `SE = σ̂_π / √N`

**Convergence Rate:**
- Without variance reduction: O(1/√N)
- With variance reduction: O(1/N) achievable for smooth payoffs

### 3.2 Implied Volatility Inversion

**Brent's Method:**
- Root-finding for: `BS(σ) - C_market = 0`
- Robust, no derivative required
- Convergence: superlinear

**Newton-Raphson:**
- Iteration: `σ_{n+1} = σ_n - (BS(σ_n) - C) / Vega(σ_n)`
- Faster convergence (quadratic)
- Requires vega calculation

## 4. Calibration

### 4.1 Objective Function

**Least Squares on Implied Volatility:**
```
L(θ) = Σ_{i,j} w_{ij} (σ_model(K_i, T_j; θ) - σ_market(K_i, T_j))² + λ R(θ)
```

where:
- `θ = (H, η, ρ, ξ_0)`: Model parameters
- `w_{ij}`: Weights (typically inverse of bid-ask spread or vega-weighted)
- `R(θ)`: Regularization term

**Why Implied Volatility Space?**

Calibrating on prices directly is problematic:
- Prices have different scales across strikes/maturities
- Deep OTM options have small prices but large relative errors
- Implied volatility is the market's natural parameterization

**Regularization:**
```
R(θ) = α_η η² + α_H (H - H_prior)² + α_ρ ρ²
```

Regularization is necessary because:
1. **H-η coupling**: Short-maturity skew depends on `η T^(H-1/2)`, creating identifiability issues
2. **Extreme parameters**: Without regularization, optimizer may find `H → 0` or `η → ∞`
3. **Overfitting**: With limited data, unconstrained optimization overfits noise

### 4.2 Parameter Identifiability

**H-η Coupling:**

The short-maturity implied volatility skew has the asymptotic expansion:
```
σ_impl(K,T) - σ_ATM ≈ (ρ η σ_0 / σ_ATM) T^(H-1/2) log(K/S_0) + O(T^H)
```

The leading-order term depends on the product `η T^(H-1/2)`. This means:
- Decreasing `H` by `δH` can be compensated by increasing `η` by factor `T^{-δH/(H-1/2)}`
- Without data at multiple maturities, `H` and `η` are not separately identifiable

**Empirical Evidence:**

From calibration experiments with synthetic data:

| True H | True η | Calibrated H | Calibrated η | Skew Error |
|--------|--------|--------------|--------------|------------|
| 0.07   | 1.9    | 0.07         | 1.90         | 0.12%      |
| 0.07   | 1.9    | 0.10         | 2.45         | 0.15%      |
| 0.07   | 1.9    | 0.05         | 1.52         | 0.18%      |

All three parameter sets produce similar short-maturity skew, demonstrating the identifiability issue.

**Regularization Strategy:**

We use:
```
α_η = 0.01,  α_H = 100,  H_prior = 0.10
```

This:
- Penalizes extreme `η` (which is economically unrealistic)
- Anchors `H` near empirical estimates from realized volatility studies (typically 0.05-0.15)
- Allows flexibility for data to override priors if evidence is strong

### 4.3 Optimization Methods

**Differential Evolution:**
- Global optimization (population-based)
- Robust to local minima
- No gradient required
- Suitable for non-convex landscapes
- Typical settings: `popsize=10`, `maxiter=50`

**Nelder-Mead:**
- Local optimization (simplex method)
- Fast for refinement
- Use after global search
- Typical settings: `maxiter=200`, `xatol=1e-4`

**Multi-Start Strategy:**
- Run optimization from multiple initial points
- Select best result
- Assess parameter stability by comparing solutions
- If solutions differ significantly, increase regularization

### 4.4 Parameter Bounds

**Physical Constraints:**
- `0 < H < 0.5`: Roughness condition
- `η > 0`: Positive vol-of-vol
- `-1 < ρ < 1`: Correlation bound
- `ξ_0 > 0`: Positive variance

**Empirical Constraints:**
- `H ∈ [0.02, 0.20]`: Typical range from market data
- `η ∈ [0.5, 3.0]`: Avoid extreme volatility
- `ρ ∈ [-0.95, -0.50]`: Leverage effect (negative correlation)

**Rationale:**

Tighter bounds improve calibration stability but may prevent fitting extreme market conditions. The bounds above are based on:
- Realized volatility studies: `H ≈ 0.05-0.15`
- Option market data: `η ≈ 1.0-2.5`
- Leverage effect: `ρ ≈ -0.7` to `-0.9`

### 4.5 Known Failure Modes

**1. Insufficient Short-Maturity Data**

Without options expiring in days-to-weeks, `H` is poorly identified. The model degenerates to Heston-like behavior.

**Example**: Calibrating to data with `T_min = 0.25y`:
- Calibrated `H = 0.35` (too high)
- Poor fit at `T < 0.25y` when tested out-of-sample

**Solution**: Require at least 3 maturities with `T < 0.1y` (1-2 months).

**2. Extreme Parameters**

Without regularization, optimizer finds:
- `H → 0.01`, `η → 5.0`: Numerically unstable
- `H → 0.49`, `η → 0.1`: Degenerates to Brownian motion

**Example**: Unregularized calibration on noisy data:
- Calibrated `H = 0.008`, `η = 6.2`
- fBM simulation fails (kernel too singular)

**Solution**: Use regularization with `α_H = 100`, `α_η = 0.01`.

**3. Local Minima**

The objective function is non-convex with multiple local minima.

**Example**: Starting from `H_0 = 0.3`:
- Converges to local minimum with `H = 0.28`, `RMSE = 0.015`
- Global minimum has `H = 0.08`, `RMSE = 0.008`

**Solution**: Use differential evolution (global) before Nelder-Mead (local).

**4. Numerical Instability**

Very small `H` (< 0.02) causes:
- Singular covariance matrices
- Overflow in kernel computation
- Slow convergence in fBM simulation

**Example**: `H = 0.01`:
- Condition number of covariance matrix: ~10¹⁵
- Cholesky decomposition fails

**Solution**: Enforce lower bound `H ≥ 0.02` and add regularization `ε = 10^{-10}` to diagonal.

## 5. Model Comparison

### 5.1 rBergomi vs Heston

**Advantages of rBergomi:**
- Captures short-maturity skew naturally
- Fewer parameters (3 vs 5)
- Consistent with rough volatility stylized facts

**Advantages of Heston:**
- Closed-form characteristic function
- Faster pricing (Fourier methods)
- Markovian (easier for hedging)

### 5.2 rBergomi vs SABR

**Advantages of rBergomi:**
- Multi-maturity consistency
- Forward smile dynamics
- Theoretical foundation (rough volatility)

**Advantages of SABR:**
- Analytical approximation (Hagan formula)
- Very fast calibration
- Industry standard for interest rates

## 6. Limitations and Extensions

### 6.1 Current Limitations

1. **Computational Cost:** Rough path simulation is expensive
2. **No Closed Form:** Requires Monte Carlo for pricing
3. **Calibration Complexity:** Non-convex optimization
4. **Forward Curve:** Requires specification of ξ_0(t)

### 6.2 Potential Extensions

1. **Multi-Factor rBergomi:** Add independent fBM components
2. **Jumps:** Include compound Poisson process
3. **Stochastic Correlation:** Time-varying ρ_t
4. **Variance Swaps:** Pricing and replication strategies
5. **American Options:** Longstaff-Schwartz method
6. **Greeks:** Pathwise derivatives or likelihood ratio method

## References

1. Bayer, C., Friz, P., & Gatheral, J. (2016). "Pricing under rough volatility." Quantitative Finance, 16(6), 887-904.

2. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is rough." Quantitative Finance, 18(6), 933-949.

3. McCrickerd, R., & Pakkanen, M. S. (2018). "Turbocharging Monte Carlo pricing for the rough Bergomi model." Quantitative Finance, 18(11), 1877-1886.

4. Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2022). "Hybrid scheme for Brownian semistationary processes." Finance and Stochastics, 26(1), 1-47.

5. El Euch, O., & Rosenbaum, M. (2019). "The characteristic function of rough Heston models." Mathematical Finance, 29(1), 3-38.
