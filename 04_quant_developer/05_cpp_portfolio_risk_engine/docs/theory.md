# Mathematical Theory

## Asset Dynamics

### Geometric Brownian Motion

Asset prices follow the stochastic differential equation:

```
dS_t = μ S_t dt + σ S_t dW_t
```

where:
- `S_t` is the asset price at time t
- `μ` is the drift (expected return)
- `σ` is the volatility
- `W_t` is a Wiener process (Brownian motion)

### Discretization

Using Euler-Maruyama discretization:

```
S_{t+Δt} = S_t exp((μ - σ²/2)Δt + σ√Δt Z)
```

where `Z ~ N(0,1)` is a standard normal random variable.

## Correlation Structure

### Cholesky Decomposition

For n correlated assets with correlation matrix Ρ:

```
Ρ = L L^T
```

where L is the lower triangular Cholesky factor.

Correlated normal samples:
```
Z_corr = L Z_indep
```

where `Z_indep` are independent standard normals.

## Risk Metrics

### Value-at-Risk (VaR)

VaR at confidence level α is the loss threshold such that:

```
P(Loss > VaR_α) = 1 - α
```

Computed as the (1-α) quantile of the PnL distribution.

**Standard Error**: VaR is a random variable. Standard error estimated using order statistics:
```
SE(VaR) ≈ sqrt(α(1-α)/n) / f(VaR)
```
where f is the PDF at VaR, estimated via kernel density.

### Conditional Value-at-Risk (CVaR)

CVaR (Expected Shortfall) is the expected loss given that loss exceeds VaR:

```
CVaR_α = E[Loss | Loss > VaR_α]
```

More conservative than VaR, captures tail risk.

**Standard Error**: Estimated from tail variance:
```
SE(CVaR) ≈ sqrt(Var(tail losses) / (n * α))
```

## Variance Reduction

### Antithetic Variates

**Principle**: For each random draw Z ~ N(0,1), also simulate -Z.

**Implementation**:
```
Path 1: S_t = S_0 * exp((μ - σ²/2)t + σ√t * Z)
Path 2: S_t = S_0 * exp((μ - σ²/2)t + σ√t * (-Z))
```

**Variance Reduction**:
```
Var((X1 + X2)/2) = (Var(X1) + Var(X2) + 2*Cov(X1,X2)) / 4
```

For symmetric payoffs (e.g., European options), Cov(X1, X2) < 0, yielding ~2x variance reduction.

**Computational Cost**: Zero overhead (same number of RNG calls, same path generation).

**When Effective**:
- European options (symmetric payoffs)
- Portfolio PnL (linear combinations)
- Path-dependent options with symmetric features

**When Less Effective**:
- Highly asymmetric payoffs
- Digital options
- Deep out-of-the-money options

## Greeks

### Delta

Sensitivity to underlying price:
```
Δ = ∂V/∂S
```

For Black-Scholes call: `Δ = N(d₁)`

### Gamma

Convexity (second derivative):
```
Γ = ∂²V/∂S²
```

For Black-Scholes: `Γ = N'(d₁)/(S σ √T)`

### Vega

Sensitivity to volatility:
```
ν = ∂V/∂σ
```

For Black-Scholes: `ν = S N'(d₁) √T`

## Black-Scholes Formula

Call option price:
```
C = S N(d₁) - K e^(-rT) N(d₂)
```

where:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

Put-call parity:
```
P = C - S + K e^(-rT)
```
