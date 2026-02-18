"""
Data Generation Module for Market Impact Research

Generates synthetic trade data with:
- Signed volume (buy/sell imbalance)
- Informed vs noise trader separation
- Stationary price increments
- Kyle's Lambda positive by construction
- Explicit regime switching

Mathematical Foundation:
    Price dynamics: dP_t = λ × Q_t × dt + σ × dW_t
    Order flow: Q_t = Q_informed(t) + Q_noise(t)
    
    Where:
    - λ (Kyle's lambda): Adverse selection coefficient (must be > 0)
    - Q_t: Signed order flow (positive = buy, negative = sell)
    - σ: Volatility
    - W_t: Standard Brownian motion

Assumptions:
1. Order flow has long-memory (Hurst > 0.5) for informed traders
2. Noise traders have no memory (Hurst = 0.5)
3. Price increments are stationary (no drift in returns)
4. Linear price impact (Kyle's model)
5. Regime parameters are fixed within regime (no time-varying)

Known Limitations:
- Fractional Brownian motion is Gaussian (no fat tails)
- Linear impact assumption breaks for large orders
- No intraday patterns or microstructure noise
- Regime switches are abrupt (no transition dynamics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from pathlib import Path


def fractional_brownian_motion(n: int, hurst: float, dt: float = 1.0) -> np.ndarray:
    """
    Generate fractional Brownian motion using Davies-Harte algorithm.
    
    Mathematical Properties:
    - H = 0.5: Standard Brownian motion (no memory)
    - H > 0.5: Persistent (positive autocorrelation)
    - Autocorrelation: ρ(k) ∝ k^(2H-2)
    
    Parameters:
        n: Number of steps
        hurst: Hurst exponent ∈ (0.5, 0.9)
        dt: Time step (default 1.0)
    
    Returns:
        fBm path of length n
        
    Raises:
        ValueError: If hurst not in valid range
    """
    if not (0.5 <= hurst <= 0.9):
        raise ValueError(f"Hurst must be in [0.5, 0.9], got {hurst}")
    
    # Special case: standard Brownian motion
    if hurst == 0.5:
        return np.cumsum(np.random.randn(n)) * np.sqrt(dt)
    
    # Autocovariance function for fBm
    def gamma(k):
        return 0.5 * (np.abs(k - 1)**(2*hurst) - 2*np.abs(k)**(2*hurst) + np.abs(k + 1)**(2*hurst))
    
    # Build circulant covariance matrix eigenvalues
    g = np.array([gamma(i) for i in range(n)])
    g = np.concatenate([g, g[-2:0:-1]])
    
    # FFT to get eigenvalues
    eigenvalues = np.fft.fft(g).real
    
    # Ensure non-negative eigenvalues (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Generate complex Gaussian noise
    z1 = np.random.randn(len(eigenvalues))
    z2 = np.random.randn(len(eigenvalues))
    
    # Construct fBm via FFT
    w = np.sqrt(eigenvalues / (2 * len(eigenvalues))) * (z1 + 1j * z2)
    w[0] = np.sqrt(eigenvalues[0] / len(eigenvalues)) * z1[0]
    
    fbm = np.fft.fft(w).real[:n]
    fbm = np.cumsum(fbm) * np.sqrt(dt)
    
    return fbm


def generate_market_data(
    n_periods: int = 10000,
    hurst: float = 0.7,
    lambda_kyle: float = 0.001,
    volatility: float = 0.02,
    informed_fraction: float = 0.3,
    regime: str = 'medium',
    random_seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic market data with Kyle price impact.
    
    Price Dynamics:
        dP_t = λ × Q_t × dt + σ × dW_t
        
    Order Flow Decomposition:
        Q_t = Q_informed(t) + Q_noise(t)
        Q_informed ~ fBm(H) with long memory
        Q_noise ~ N(0, σ²) with no memory
    
    Parameters:
        n_periods: Number of time periods
        hurst: Hurst exponent for informed order flow ∈ [0.5, 0.9]
        lambda_kyle: Kyle's lambda (must be > 0)
        volatility: Price volatility (daily)
        informed_fraction: Fraction of order flow that is informed ∈ [0, 1]
        regime: Liquidity regime ('low', 'medium', 'high')
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns:
        - time: Time index
        - price: Price level
        - returns: Price returns (basis points)
        - order_flow: Total signed volume
        - informed_flow: Informed trader component
        - noise_flow: Noise trader component
        - regime: Regime label
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if lambda_kyle <= 0:
        raise ValueError(f"Kyle's lambda must be positive, got {lambda_kyle}")
    if not (0 <= informed_fraction <= 1):
        raise ValueError(f"Informed fraction must be in [0, 1], got {informed_fraction}")
    if regime not in ['low', 'medium', 'high']:
        raise ValueError(f"Regime must be 'low', 'medium', or 'high', got {regime}")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Regime-dependent parameters
    # Low liquidity: Higher impact, higher volatility
    # High liquidity: Lower impact, lower volatility
    regime_params = {
        'low': {'depth_multiplier': 0.5, 'vol_multiplier': 1.5},
        'medium': {'depth_multiplier': 1.0, 'vol_multiplier': 1.0},
        'high': {'depth_multiplier': 2.0, 'vol_multiplier': 0.7}
    }
    
    params = regime_params[regime]
    adj_volatility = volatility * params['vol_multiplier']
    adj_lambda = lambda_kyle / params['depth_multiplier']
    
    # Generate informed order flow with long memory (fBm with H > 0.5)
    informed_flow = fractional_brownian_motion(n_periods, hurst)
    informed_flow = informed_flow - np.mean(informed_flow)  # Zero mean
    informed_flow = informed_flow / np.std(informed_flow) * informed_fraction  # Scale
    
    # Generate noise order flow (standard Brownian motion, H = 0.5)
    noise_flow = np.random.randn(n_periods) * (1 - informed_fraction)
    
    # Total signed order flow
    order_flow = informed_flow + noise_flow
    
    # Price process: dP = λ × Q + σ × dW
    # Ensure stationarity: no drift term
    price_innovations = np.random.randn(n_periods) * adj_volatility
    price_impact = adj_lambda * order_flow
    
    # Price changes (stationary increments)
    price_changes = price_impact + price_innovations
    
    # Cumulative price (starting at 100)
    price = 100 + np.cumsum(price_changes)
    
    # Returns (basis points)
    returns = price_changes / 100 * 10000
    
    # Construct DataFrame
    df = pd.DataFrame({
        'time': np.arange(n_periods),
        'price': price,
        'returns': returns,
        'order_flow': order_flow,
        'informed_flow': informed_flow,
        'noise_flow': noise_flow,
        'regime': regime
    })
    
    return df


def generate_regime_switching_data(
    n_periods_per_regime: int = 3000,
    hurst: float = 0.7,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate data for multiple liquidity regimes.
    
    Regime Characteristics:
    - Low liquidity: λ = 0.002, high volatility
    - Medium liquidity: λ = 0.001, medium volatility
    - High liquidity: λ = 0.0005, low volatility
    
    Parameters:
        n_periods_per_regime: Number of periods per regime
        hurst: Hurst exponent for order flow
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys 'low', 'medium', 'high' containing regime DataFrames
    """
    regimes = {}
    
    # Regime-specific Kyle's lambda values
    # Low liquidity → higher impact
    # High liquidity → lower impact
    lambda_values = {
        'low': 0.002,
        'medium': 0.001,
        'high': 0.0005
    }
    
    for i, regime in enumerate(['low', 'medium', 'high']):
        seed = random_seed + i if random_seed is not None else None
        
        regimes[regime] = generate_market_data(
            n_periods=n_periods_per_regime,
            hurst=hurst,
            lambda_kyle=lambda_values[regime],
            regime=regime,
            random_seed=seed
        )
    
    return regimes


def compute_diagnostics(data: pd.DataFrame, max_lag: int = 20) -> Dict[str, float]:
    """
    Compute diagnostic statistics for generated data.
    
    Checks:
    1. Return autocorrelation (should be ~0 for stationarity)
    2. Signed-volume autocorrelation (should be positive for H > 0.5)
    3. Impact linearity (correlation between order flow and returns)
    4. Kyle's lambda positivity (implicit in correlation sign)
    
    Parameters:
        data: DataFrame from generate_market_data()
        max_lag: Maximum lag for autocorrelation
    
    Returns:
        Dictionary of diagnostic metrics
    """
    order_flow = data['order_flow'].values
    returns = data['returns'].values
    
    # 1. Return autocorrelation (should be near zero)
    return_acf = []
    for lag in range(1, max_lag + 1):
        if len(returns) > lag:
            corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            return_acf.append(corr)
    
    # 2. Signed-volume autocorrelation (should be positive)
    volume_acf = []
    for lag in range(1, max_lag + 1):
        if len(order_flow) > lag:
            corr = np.corrcoef(order_flow[:-lag], order_flow[lag:])[0, 1]
            volume_acf.append(corr)
    
    # 3. Impact linearity check (order flow vs returns)
    impact_correlation = np.corrcoef(order_flow, returns)[0, 1]
    
    # 4. Basic statistics
    diagnostics = {
        'mean_returns': np.mean(returns),
        'std_returns': np.std(returns),
        'mean_order_flow': np.mean(order_flow),
        'std_order_flow': np.std(order_flow),
        'return_acf_lag1': return_acf[0] if return_acf else np.nan,
        'return_acf_mean': np.mean(return_acf) if return_acf else np.nan,
        'volume_acf_lag1': volume_acf[0] if volume_acf else np.nan,
        'volume_acf_mean': np.mean(volume_acf) if volume_acf else np.nan,
        'impact_correlation': impact_correlation,
        'lambda_positive': impact_correlation > 0  # Sanity check
    }
    
    return diagnostics


def plot_diagnostics(regimes: Dict[str, pd.DataFrame], save_path: str = 'results/figures/'):
    """
    Generate diagnostic plots for data validation.
    
    Plots:
    1. Return autocorrelation by regime
    2. Signed-volume autocorrelation by regime
    3. Impact linearity (order flow vs returns)
    4. Order flow time series by regime
    
    Parameters:
        regimes: Dictionary of regime DataFrames
        save_path: Directory to save figures
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Generation Diagnostics', fontsize=16, fontweight='bold')
    
    colors = {'low': '#d62728', 'medium': '#ff7f0e', 'high': '#2ca02c'}
    max_lag = 20
    
    # Plot 1: Return autocorrelation
    ax = axes[0, 0]
    for regime_name, data in regimes.items():
        returns = data['returns'].values
        acf = [np.corrcoef(returns[:-lag], returns[lag:])[0, 1] 
               for lag in range(1, max_lag + 1)]
        ax.plot(range(1, max_lag + 1), acf, 'o-', label=regime_name.capitalize(), 
                color=colors[regime_name], alpha=0.7)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Return Autocorrelation\n(Should be ~0 for stationarity)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Signed-volume autocorrelation
    ax = axes[0, 1]
    for regime_name, data in regimes.items():
        order_flow = data['order_flow'].values
        acf = [np.corrcoef(order_flow[:-lag], order_flow[lag:])[0, 1] 
               for lag in range(1, max_lag + 1)]
        ax.plot(range(1, max_lag + 1), acf, 'o-', label=regime_name.capitalize(), 
                color=colors[regime_name], alpha=0.7)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Signed-Volume Autocorrelation\n(Should be >0 for H>0.5)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Impact linearity
    ax = axes[1, 0]
    for regime_name, data in regimes.items():
        # Subsample for visibility
        sample_idx = np.random.choice(len(data), size=min(500, len(data)), replace=False)
        ax.scatter(data['order_flow'].iloc[sample_idx], 
                  data['returns'].iloc[sample_idx],
                  alpha=0.3, s=10, label=regime_name.capitalize(),
                  color=colors[regime_name])
    
    ax.set_xlabel('Order Flow (Signed Volume)', fontsize=11)
    ax.set_ylabel('Returns (bps)', fontsize=11)
    ax.set_title('Impact Linearity Check\n(Positive correlation confirms λ>0)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Order flow time series
    ax = axes[1, 1]
    for regime_name, data in regimes.items():
        # Plot first 500 points for visibility
        n_plot = min(500, len(data))
        ax.plot(data['time'].iloc[:n_plot], 
               data['order_flow'].iloc[:n_plot],
               alpha=0.7, linewidth=0.8, label=regime_name.capitalize(),
               color=colors[regime_name])
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Order Flow', fontsize=11)
    ax.set_title('Order Flow Time Series\n(First 500 periods)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/data_generation_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Diagnostic plots saved to {save_path}/data_generation_diagnostics.png")


def print_diagnostic_summary(regimes: Dict[str, pd.DataFrame]):
    """
    Print summary of diagnostic checks.
    
    Parameters:
        regimes: Dictionary of regime DataFrames
    """
    print("\n" + "="*70)
    print("DATA GENERATION DIAGNOSTIC SUMMARY")
    print("="*70)
    
    for regime_name, data in regimes.items():
        diagnostics = compute_diagnostics(data)
        
        print(f"\n{regime_name.upper()} LIQUIDITY REGIME:")
        print(f"  Returns:")
        print(f"    Mean: {diagnostics['mean_returns']:.6f} bps (should be ~0)")
        print(f"    Std:  {diagnostics['std_returns']:.4f} bps")
        print(f"    ACF(1): {diagnostics['return_acf_lag1']:.4f} (should be ~0)")
        
        print(f"  Order Flow:")
        print(f"    Mean: {diagnostics['mean_order_flow']:.6f} (should be ~0)")
        print(f"    Std:  {diagnostics['std_order_flow']:.4f}")
        print(f"    ACF(1): {diagnostics['volume_acf_lag1']:.4f} (should be >0)")
        
        print(f"  Impact:")
        print(f"    Correlation(Q, ΔP): {diagnostics['impact_correlation']:.4f}")
        print(f"    λ > 0: {'✓ PASS' if diagnostics['lambda_positive'] else '✗ FAIL'}")
    
    print("\n" + "="*70)
    print("VALIDATION CHECKS:")
    print("="*70)
    
    all_pass = True
    for regime_name, data in regimes.items():
        diagnostics = compute_diagnostics(data)
        
        # Check 1: Returns are stationary (low autocorrelation)
        returns_stationary = abs(diagnostics['return_acf_lag1']) < 0.1
        print(f"  {regime_name.capitalize()}: Returns stationary: {'✓' if returns_stationary else '✗'}")
        all_pass = all_pass and returns_stationary
        
        # Check 2: Order flow has memory (positive autocorrelation)
        volume_memory = diagnostics['volume_acf_lag1'] > 0.1
        print(f"  {regime_name.capitalize()}: Volume has memory: {'✓' if volume_memory else '✗'}")
        all_pass = all_pass and volume_memory
        
        # Check 3: Positive impact (λ > 0)
        lambda_positive = diagnostics['lambda_positive']
        print(f"  {regime_name.capitalize()}: λ > 0: {'✓' if lambda_positive else '✗'}")
        all_pass = all_pass and lambda_positive
    
    print("="*70)
    print(f"OVERALL: {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    """
    Standalone execution for Phase 2 validation.
    """
    print("\n" + "="*70)
    print("PHASE 2: DATA GENERATION")
    print("="*70 + "\n")
    
    # Generate regime-switching data
    print("Generating synthetic market data...")
    regimes = generate_regime_switching_data(
        n_periods_per_regime=3000,
        hurst=0.7,
        random_seed=42
    )
    print(f"✓ Generated {len(regimes)} regimes with {3000} periods each\n")
    
    # Compute and print diagnostics
    print_diagnostic_summary(regimes)
    
    # Generate diagnostic plots
    print("Generating diagnostic plots...")
    plot_diagnostics(regimes)
    
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print("\nNext: Review diagnostics and approve before Phase 3 (Model Implementation)")
