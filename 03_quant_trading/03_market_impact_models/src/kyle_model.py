"""
Kyle's Lambda Model: Linear Price Impact with Statistical Rigor

Mathematical Model:
    ΔP_t = α + λ × Q_t + ε_t
    
    Where:
    - ΔP_t: Price change (returns)
    - Q_t: Signed order flow (buy/sell imbalance)
    - λ: Kyle's lambda (adverse selection coefficient)
    - α: Intercept (should be ~0 for no drift)
    - ε_t: Error term

Assumptions (CRITICAL for interviews):
1. Linearity: Impact proportional to order size (λ constant)
2. Homoskedasticity: Constant error variance (no size-dependent volatility)
3. No autocorrelation: Errors are independent (no serial correlation)
4. Normality: Errors are Gaussian (for inference)

When Kyle Fails (Economic Intuition):
1. Large orders: Super-linear impact (√Q or Q^α with α > 1)
2. Thin markets: Heteroskedastic errors (volatility increases with size)
3. Informed trading: Autocorrelated errors (persistent impact)
4. Regime changes: Time-varying λ (crisis vs normal)

Interview Defense:
- "Kyle works for small-to-medium orders in liquid markets"
- "Breaks down for large metaorders due to non-linear impact"
- "Assumes no strategic behavior (traders don't game the model)"
- "Best for adverse selection, not liquidity pressure"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple
from pathlib import Path


class KyleModel:
    """
    Kyle's Lambda estimator with comprehensive statistical validation.
    
    Estimates linear price impact using OLS regression and validates
    all model assumptions with diagnostic tests.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Kyle model.
        
        Parameters:
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.confidence_level = confidence_level
        self.lambda_estimate = None
        self.std_error = None
        self.confidence_interval = None
        self.r_squared = None
        self.adj_r_squared = None
        self.residuals = None
        self.fitted_values = None
        self.n_obs = None
        
    def calibrate(self, order_flow: np.ndarray, returns: np.ndarray) -> Dict:
        """
        Estimate Kyle's lambda using OLS regression.
        
        Model: Returns = α + λ × OrderFlow + ε
        
        OLS Estimator:
            λ̂ = Cov(Q, ΔP) / Var(Q)
            SE(λ̂) = σ̂ / √(Σ(Q_i - Q̄)²)
            95% CI: λ̂ ± t_0.975 × SE(λ̂)
        
        Parameters:
            order_flow: Signed order flow (Q_t)
            returns: Price returns (ΔP_t)
        
        Returns:
            Dictionary with:
            - lambda: Point estimate
            - std_error: Standard error
            - ci_lower, ci_upper: Confidence interval bounds
            - t_statistic: t-statistic for H0: λ = 0
            - p_value: p-value for significance test
            - r_squared: Coefficient of determination
            - adj_r_squared: Adjusted R²
            - n_obs: Number of observations
        """
        n = len(order_flow)
        self.n_obs = n
        
        if n < 10:
            raise ValueError(f"Insufficient data: need at least 10 observations, got {n}")
        
        # OLS estimation: Y = Xβ + ε
        X = np.column_stack([np.ones(n), order_flow])
        
        # Solve normal equations: β = (X'X)^(-1) X'Y
        XtX = X.T @ X
        XtY = X.T @ returns
        
        try:
            coeffs = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix: order flow has no variation")
        
        alpha, lambda_est = coeffs
        self.lambda_estimate = lambda_est
        
        # Fitted values and residuals
        self.fitted_values = X @ coeffs
        self.residuals = returns - self.fitted_values
        
        # Residual sum of squares
        rss = np.sum(self.residuals**2)
        
        # Standard error of λ
        sigma_squared = rss / (n - 2)  # Unbiased estimator
        var_lambda = sigma_squared / np.sum((order_flow - np.mean(order_flow))**2)
        self.std_error = np.sqrt(var_lambda)
        
        # Confidence interval
        alpha_level = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha_level/2, n - 2)
        margin = t_critical * self.std_error
        self.confidence_interval = (lambda_est - margin, lambda_est + margin)
        
        # R-squared
        tss = np.sum((returns - np.mean(returns))**2)
        self.r_squared = 1 - rss / tss if tss > 0 else 0
        
        # Adjusted R-squared
        self.adj_r_squared = 1 - (rss / (n - 2)) / (tss / (n - 1)) if tss > 0 else 0
        
        # t-statistic and p-value for H0: λ = 0
        t_stat = lambda_est / self.std_error if self.std_error > 0 else np.inf
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        
        return {
            'lambda': lambda_est,
            'std_error': self.std_error,
            'ci_lower': self.confidence_interval[0],
            'ci_upper': self.confidence_interval[1],
            't_statistic': t_stat,
            'p_value': p_value,
            'r_squared': self.r_squared,
            'adj_r_squared': self.adj_r_squared,
            'n_obs': n
        }
    
    def validate_assumptions(self) -> Dict:
        """
        Test all regression assumptions with formal statistical tests.
        
        Tests:
        1. Normality: Jarque-Bera test (H0: errors are Gaussian)
        2. Autocorrelation: Durbin-Watson statistic (should be ~2)
        3. Homoskedasticity: Breusch-Pagan test (H0: constant variance)
        
        Returns:
            Dictionary with test statistics, p-values, and pass/fail flags
        """
        if self.residuals is None:
            raise ValueError("Must calibrate model first")
        
        residuals = self.residuals
        n = len(residuals)
        
        # 1. Normality: Jarque-Bera test
        # H0: Residuals are normally distributed
        # Reject if p < 0.05
        jb_stat, jb_pval = stats.jarque_bera(residuals)
        normal_residuals = jb_pval > 0.05
        
        # 2. Autocorrelation: Durbin-Watson statistic
        # DW ≈ 2 implies no autocorrelation
        # DW < 1.5 or DW > 2.5 suggests autocorrelation
        dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        no_autocorrelation = 1.5 < dw_stat < 2.5
        
        # 3. Homoskedasticity: Breusch-Pagan test
        # H0: Constant variance
        # Regress squared residuals on fitted values
        residuals_sq = residuals**2
        fitted = self.fitted_values
        
        # Auxiliary regression: ε² = γ₀ + γ₁ × ŷ + u
        X_aux = np.column_stack([np.ones(n), fitted])
        try:
            gamma = np.linalg.solve(X_aux.T @ X_aux, X_aux.T @ residuals_sq)
            fitted_aux = X_aux @ gamma
            ssr_aux = np.sum((fitted_aux - np.mean(residuals_sq))**2)
            
            # BP test statistic: n × R²_aux ~ χ²(1)
            tss_aux = np.sum((residuals_sq - np.mean(residuals_sq))**2)
            r_sq_aux = ssr_aux / tss_aux if tss_aux > 0 else 0
            bp_stat = n * r_sq_aux
            bp_pval = 1 - stats.chi2.cdf(bp_stat, 1)
            homoskedastic = bp_pval > 0.05
        except:
            bp_stat = np.nan
            bp_pval = np.nan
            homoskedastic = False
        
        return {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'normal_residuals': normal_residuals,
            'durbin_watson': dw_stat,
            'no_autocorrelation': no_autocorrelation,
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_pval': bp_pval,
            'homoskedastic': homoskedastic,
            'all_assumptions_hold': normal_residuals and no_autocorrelation and homoskedastic
        }
    
    def test_linearity(self, order_flow: np.ndarray, returns: np.ndarray, 
                       n_quantiles: int = 5) -> Dict:
        """
        Test linearity assumption by estimating λ across order size quantiles.
        
        Intuition:
        - If impact is linear, λ should be constant across order sizes
        - If λ increases with size → super-linear impact (model fails)
        - If λ decreases with size → market depth effects
        
        Method:
        - Split data into quantiles by |order_flow|
        - Estimate λ separately for each quantile
        - Check if λ varies significantly (CV > 0.3 suggests non-linearity)
        
        Parameters:
            order_flow: Signed order flow
            returns: Price returns
            n_quantiles: Number of quantiles (default 5)
        
        Returns:
            Dictionary with:
            - quantile_lambdas: DataFrame of λ by quantile
            - lambda_cv: Coefficient of variation of λ
            - linearity_holds: Boolean flag
        """
        abs_flow = np.abs(order_flow)
        
        try:
            quantiles = pd.qcut(abs_flow, q=n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            # If not enough unique values, use fewer quantiles
            quantiles = pd.qcut(abs_flow, q=3, labels=False, duplicates='drop')
            n_quantiles = 3
        
        quantile_results = []
        for q in range(n_quantiles):
            mask = quantiles == q
            n_q = np.sum(mask)
            
            if n_q < 10:  # Skip if too few observations
                continue
            
            q_flow = order_flow[mask]
            q_returns = returns[mask]
            
            # Estimate λ for this quantile
            X = np.column_stack([np.ones(n_q), q_flow])
            try:
                coeffs = np.linalg.solve(X.T @ X, X.T @ q_returns)
                lambda_q = coeffs[1]
            except:
                lambda_q = np.nan
            
            quantile_results.append({
                'quantile': q + 1,
                'lambda': lambda_q,
                'n_obs': n_q,
                'mean_size': np.mean(abs_flow[mask]),
                'median_size': np.median(abs_flow[mask])
            })
        
        df_quantiles = pd.DataFrame(quantile_results)
        
        # Check if λ varies significantly
        if len(df_quantiles) > 1 and not df_quantiles['lambda'].isna().all():
            lambda_mean = df_quantiles['lambda'].mean()
            lambda_std = df_quantiles['lambda'].std()
            lambda_cv = lambda_std / np.abs(lambda_mean) if lambda_mean != 0 else np.inf
            
            # Linearity holds if CV < 0.3 (30% variation)
            linearity_holds = lambda_cv < 0.3
        else:
            linearity_holds = True
            lambda_cv = 0.0
        
        return {
            'quantile_lambdas': df_quantiles,
            'lambda_cv': lambda_cv,
            'linearity_holds': linearity_holds
        }
    
    def identify_breakdown_point(self, order_flow: np.ndarray, returns: np.ndarray,
                                  percentiles: np.ndarray = None) -> pd.DataFrame:
        """
        Identify order size threshold where linear model breaks down.
        
        Method:
        - Estimate λ using data up to each percentile threshold
        - Compare to full-sample λ
        - Breakdown occurs when λ changes significantly
        
        Parameters:
            order_flow: Signed order flow
            returns: Price returns
            percentiles: Percentile thresholds to test (default [50, 75, 90, 95, 99])
        
        Returns:
            DataFrame with λ estimates by percentile threshold
        """
        if percentiles is None:
            percentiles = np.array([50, 75, 90, 95, 99])
        
        abs_flow = np.abs(order_flow)
        thresholds = np.percentile(abs_flow, percentiles)
        
        breakdown_results = []
        for pct, threshold in zip(percentiles, thresholds):
            mask = abs_flow <= threshold
            n_below = np.sum(mask)
            
            if n_below < 20:
                continue
            
            # Estimate λ below threshold
            X = np.column_stack([np.ones(n_below), order_flow[mask]])
            try:
                coeffs = np.linalg.solve(X.T @ X, X.T @ returns[mask])
                lambda_below = coeffs[1]
                
                # Compute R² for this subset
                fitted = X @ coeffs
                residuals = returns[mask] - fitted
                rss = np.sum(residuals**2)
                tss = np.sum((returns[mask] - np.mean(returns[mask]))**2)
                r_sq = 1 - rss / tss if tss > 0 else 0
            except:
                lambda_below = np.nan
                r_sq = np.nan
            
            breakdown_results.append({
                'percentile': pct,
                'threshold': threshold,
                'lambda': lambda_below,
                'r_squared': r_sq,
                'n_obs': n_below,
                'pct_data': n_below / len(order_flow) * 100
            })
        
        return pd.DataFrame(breakdown_results)
    
    def compare_to_true_lambda(self, true_lambda: float) -> Dict:
        """
        Compare estimated λ to true λ used in data generation.
        
        Metrics:
        - Absolute error
        - Relative error (%)
        - Coverage: Does CI contain true λ?
        
        Parameters:
            true_lambda: True λ from data generation
        
        Returns:
            Dictionary with comparison metrics
        """
        if self.lambda_estimate is None:
            raise ValueError("Must calibrate model first")
        
        abs_error = self.lambda_estimate - true_lambda
        rel_error = (abs_error / true_lambda) * 100 if true_lambda != 0 else np.inf
        
        ci_lower, ci_upper = self.confidence_interval
        ci_covers_true = ci_lower <= true_lambda <= ci_upper
        
        return {
            'true_lambda': true_lambda,
            'estimated_lambda': self.lambda_estimate,
            'absolute_error': abs_error,
            'relative_error_pct': rel_error,
            'ci_covers_true': ci_covers_true,
            'bias': abs_error,
            'bias_in_std_errors': abs_error / self.std_error if self.std_error > 0 else np.inf
        }


def calibrate_kyle_by_regime(regimes: Dict[str, pd.DataFrame], 
                              true_lambdas: Dict[str, float]) -> pd.DataFrame:
    """
    Calibrate Kyle model for each regime and compare to true λ.
    
    Parameters:
        regimes: Dictionary of regime DataFrames from data generation
        true_lambdas: Dictionary of true λ values by regime
    
    Returns:
        DataFrame with calibration results by regime
    """
    results = []
    
    for regime_name, data in regimes.items():
        model = KyleModel()
        
        # Calibrate
        calib = model.calibrate(
            order_flow=data['order_flow'].values,
            returns=data['returns'].values
        )
        
        # Validate assumptions
        assumptions = model.validate_assumptions()
        
        # Test linearity
        linearity = model.test_linearity(
            order_flow=data['order_flow'].values,
            returns=data['returns'].values
        )
        
        # Compare to true λ
        true_lambda = true_lambdas[regime_name]
        comparison = model.compare_to_true_lambda(true_lambda)
        
        results.append({
            'regime': regime_name,
            'true_lambda': true_lambda,
            'estimated_lambda': calib['lambda'],
            'std_error': calib['std_error'],
            'ci_lower': calib['ci_lower'],
            'ci_upper': calib['ci_upper'],
            'r_squared': calib['r_squared'],
            'p_value': calib['p_value'],
            'relative_error_pct': comparison['relative_error_pct'],
            'ci_covers_true': comparison['ci_covers_true'],
            'normal_residuals': assumptions['normal_residuals'],
            'no_autocorrelation': assumptions['no_autocorrelation'],
            'homoskedastic': assumptions['homoskedastic'],
            'linearity_holds': linearity['linearity_holds'],
            'lambda_cv': linearity['lambda_cv'],
            'n_obs': calib['n_obs']
        })
    
    return pd.DataFrame(results)


def plot_kyle_diagnostics(regimes: Dict[str, pd.DataFrame], 
                          calibration_results: pd.DataFrame,
                          save_path: str = 'results/figures/'):
    """
    Generate comprehensive diagnostic plots for Kyle model.
    
    Plots:
    1. λ estimates by regime (with CI and true values)
    2. Residual diagnostics (QQ plot, residuals vs fitted)
    3. Linearity check (λ by order size quantile)
    4. Breakdown analysis (λ by percentile threshold)
    
    Parameters:
        regimes: Dictionary of regime DataFrames
        calibration_results: DataFrame from calibrate_kyle_by_regime()
        save_path: Directory to save figures
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'low': '#d62728', 'medium': '#ff7f0e', 'high': '#2ca02c'}
    
    # Plot 1: λ estimates by regime
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(calibration_results))
    
    ax1.errorbar(x_pos, calibration_results['estimated_lambda'],
                yerr=1.96 * calibration_results['std_error'],
                fmt='o', markersize=10, capsize=5, capthick=2,
                label='Estimated λ (95% CI)', color='#1f77b4', linewidth=2)
    
    ax1.scatter(x_pos, calibration_results['true_lambda'],
               marker='x', s=200, linewidths=3,
               label='True λ', color='red', zorder=10)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(calibration_results['regime'].str.capitalize())
    ax1.set_ylabel("Kyle's Lambda", fontsize=12)
    ax1.set_title("Kyle's Lambda Estimates by Regime\n(Error bars = 95% CI)", 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add R² annotations
    for i, row in calibration_results.iterrows():
        ax1.text(i, row['estimated_lambda'] + 0.0003, 
                f"R²={row['r_squared']:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Plots 2-4: Residual diagnostics for each regime
    for idx, (regime_name, data) in enumerate(regimes.items()):
        model = KyleModel()
        model.calibrate(data['order_flow'].values, data['returns'].values)
        
        # QQ plot
        ax = fig.add_subplot(gs[1, idx])
        stats.probplot(model.residuals, dist="norm", plot=ax)
        ax.set_title(f'{regime_name.capitalize()}: QQ Plot', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Residuals vs fitted
        ax = fig.add_subplot(gs[2, idx])
        ax.scatter(model.fitted_values, model.residuals, 
                  alpha=0.3, s=10, color=colors[regime_name])
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Fitted Values', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.set_title(f'{regime_name.capitalize()}: Residuals vs Fitted', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{save_path}/kyle_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Kyle diagnostic plots saved to {save_path}/kyle_diagnostics.png")


def print_kyle_summary(calibration_results: pd.DataFrame):
    """
    Print comprehensive summary of Kyle model calibration.
    
    Parameters:
        calibration_results: DataFrame from calibrate_kyle_by_regime()
    """
    print("\n" + "="*80)
    print("KYLE'S LAMBDA MODEL: CALIBRATION SUMMARY")
    print("="*80)
    
    for _, row in calibration_results.iterrows():
        regime = row['regime'].upper()
        print(f"\n{regime} LIQUIDITY REGIME:")
        print(f"  True λ:      {row['true_lambda']:.6f}")
        print(f"  Estimated λ: {row['estimated_lambda']:.6f}")
        print(f"  Std Error:   {row['std_error']:.6f}")
        print(f"  95% CI:      [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]")
        print(f"  R²:          {row['r_squared']:.4f}")
        print(f"  p-value:     {row['p_value']:.4e}")
        print(f"  Relative Error: {row['relative_error_pct']:.2f}%")
        print(f"  CI Covers True: {'✓' if row['ci_covers_true'] else '✗'}")
        
        print(f"\n  Assumption Validation:")
        print(f"    Normal residuals:    {'✓' if row['normal_residuals'] else '✗'}")
        print(f"    No autocorrelation:  {'✓' if row['no_autocorrelation'] else '✗'}")
        print(f"    Homoskedastic:       {'✓' if row['homoskedastic'] else '✗'}")
        print(f"    Linearity holds:     {'✓' if row['linearity_holds'] else '✗'}")
        print(f"    Lambda CV:           {row['lambda_cv']:.4f}")
    
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT:")
    print("="*80)
    
    all_ci_cover = calibration_results['ci_covers_true'].all()
    all_assumptions = (calibration_results['normal_residuals'] & 
                      calibration_results['no_autocorrelation'] & 
                      calibration_results['homoskedastic'] &
                      calibration_results['linearity_holds']).all()
    
    print(f"  All CIs cover true λ:      {'✓ PASS' if all_ci_cover else '✗ FAIL'}")
    print(f"  All assumptions hold:      {'✓ PASS' if all_assumptions else '✗ FAIL'}")
    print(f"  Mean absolute error:       {calibration_results['relative_error_pct'].abs().mean():.2f}%")
    print(f"  Mean R²:                   {calibration_results['r_squared'].mean():.4f}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    """
    Standalone execution for Phase 3A validation.
    """
    import sys
    sys.path.append('.')
    from src.data_generation import generate_regime_switching_data
    
    print("\n" + "="*80)
    print("PHASE 3A: KYLE MODEL IMPLEMENTATION")
    print("="*80 + "\n")
    
    # Generate data
    print("Generating synthetic market data...")
    regimes = generate_regime_switching_data(
        n_periods_per_regime=3000,
        hurst=0.7,
        random_seed=42
    )
    
    # True λ values from data generation (converted to bps)
    # Data generation uses λ in price units, but returns are in bps
    # Conversion: λ_bps = λ_price / 100 * 10000 = λ_price * 100
    true_lambdas = {
        'low': 0.002 * 100,      # 0.2 bps per unit order flow
        'medium': 0.001 * 100,   # 0.1 bps per unit order flow
        'high': 0.0005 * 100     # 0.05 bps per unit order flow
    }
    
    print("✓ Data generated\n")
    
    # Calibrate Kyle model
    print("Calibrating Kyle model for each regime...")
    calibration_results = calibrate_kyle_by_regime(regimes, true_lambdas)
    print("✓ Calibration complete\n")
    
    # Print summary
    print_kyle_summary(calibration_results)
    
    # Save results
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    calibration_results.to_csv('results/tables/kyle_calibration.csv', index=False)
    print("✓ Results saved to results/tables/kyle_calibration.csv\n")
    
    # Generate diagnostic plots
    print("Generating diagnostic plots...")
    plot_kyle_diagnostics(regimes, calibration_results)
    
    print("\n" + "="*80)
    print("PHASE 3A COMPLETE")
    print("="*80)
    print("\nNext: Review calibration results and approve before Phase 3B (OW Model)")
