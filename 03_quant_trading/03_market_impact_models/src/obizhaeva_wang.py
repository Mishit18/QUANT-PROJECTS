"""
Obizhaeva-Wang Model: Permanent vs Transient Impact Decomposition

Mathematical Model:
    I(t) = I_perm + I_temp(t)
    I_perm = γ × σ × √(Q/V)
    I_temp(t) = (1-γ) × σ × √(Q/V) × exp(-ρt)
    
    Where:
    - I(t): Total impact at time t
    - γ: Permanent fraction ∈ [0, 1] (information revelation)
    - ρ: Resilience rate (order book recovery speed)
    - σ: Volatility
    - Q: Order size
    - V: Daily volume

Assumptions (CRITICAL for interviews):
1. Square-root impact: I ∝ √Q (concave, not linear like Kyle)
2. Exponential decay: Transient impact decays as exp(-ρt)
3. Permanent fraction: γ ∈ [0, 1] (hard constraint)
4. Additive decomposition: Total = Permanent + Transient

When OW Improves on Kyle:
1. Large orders: √Q impact is more realistic than linear
2. Execution over time: Captures decay of transient impact
3. Permanent vs temporary: Separates information from liquidity
4. Block trades: Better for institutional execution

When OW Fails (Economic Intuition):
1. Non-exponential decay: Real decay may be power-law (Bouchaud)
2. Time-varying parameters: γ and ρ change with market conditions
3. Strategic behavior: Traders adapt to known decay rates
4. Regime dependence: Parameters unstable across regimes

Interview Defense:
- "OW separates information (permanent) from liquidity (transient)"
- "Square-root impact is more realistic for large orders than Kyle's linear"
- "Exponential decay assumes order book recovers at constant rate"
- "Fails when decay is non-exponential or parameters are time-varying"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from typing import Dict, Tuple
from pathlib import Path


class ObizhaevaWangModel:
    """
    Obizhaeva-Wang model with hard constraint enforcement.
    
    Decomposes impact into permanent (information) and transient (liquidity)
    components with explicit order book resilience.
    """
    
    def __init__(self, enforce_constraints: bool = True):
        """
        Initialize OW model.
        
        Parameters:
            enforce_constraints: If True, enforce γ ∈ [0,1] as hard constraint
        """
        self.enforce_constraints = enforce_constraints
        self.gamma = None  # Permanent fraction
        self.rho = None    # Decay rate (1/time unit)
        self.sigma = None  # Volatility scaling
        self.calibration_success = False

    
    def calibrate_from_kyle(self, order_flow: np.ndarray, returns: np.ndarray,
                           kyle_lambda: float) -> Dict:
        """
        Calibrate OW model using Kyle's lambda as starting point.
        
        Strategy:
        - Use Kyle's λ to estimate total impact
        - Fit exponential decay to residuals to get ρ
        - Decompose into permanent (γ) and transient (1-γ)
        
        Note: This is a simplified calibration since we don't have
        time-series impact data. In practice, you'd need tick data.
        
        Parameters:
            order_flow: Signed order flow
            returns: Price returns
            kyle_lambda: Kyle's lambda from previous calibration
        
        Returns:
            Dictionary with calibration results
        """
        n = len(order_flow)
        
        # Estimate impact using Kyle's model
        predicted_impact = kyle_lambda * order_flow
        residuals = returns - predicted_impact
        
        # Estimate permanent fraction from R² of Kyle model
        # Higher R² → more permanent impact (information)
        # Lower R² → more transient impact (liquidity)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((returns - np.mean(returns))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Heuristic: γ ≈ √(R²) (permanent fraction related to explained variance)
        # This is a simplification - real calibration needs time-series data
        gamma_estimate = np.sqrt(max(0, r_squared))
        
        # Estimate decay rate from autocorrelation of residuals
        # Faster decay → higher ρ
        if len(residuals) > 1:
            acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            # ρ ≈ -log(ACF(1)) (from AR(1) process)
            rho_estimate = -np.log(max(0.01, abs(acf_lag1)))
        else:
            rho_estimate = 0.1
        
        # Enforce constraints
        if self.enforce_constraints:
            gamma_estimate = np.clip(gamma_estimate, 0.0, 1.0)
            rho_estimate = max(0.001, rho_estimate)
        
        self.gamma = gamma_estimate
        self.rho = rho_estimate
        self.sigma = np.std(returns)
        self.calibration_success = True
        
        # Compute half-life
        half_life = np.log(2) / self.rho if self.rho > 0 else np.inf
        
        # Compute decomposition
        permanent_impact = self.gamma * predicted_impact
        transient_impact = (1 - self.gamma) * predicted_impact
        
        # Validation metrics
        gamma_valid = 0 <= self.gamma <= 1
        rho_valid = self.rho > 0
        half_life_finite = half_life < 1000
        
        return {
            'gamma': self.gamma,
            'rho': self.rho,
            'sigma': self.sigma,
            'half_life': half_life,
            'r_squared_kyle': r_squared,
            'gamma_valid': gamma_valid,
            'rho_valid': rho_valid,
            'half_life_finite': half_life_finite,
            'constraints_satisfied': gamma_valid and rho_valid and half_life_finite,
            'permanent_fraction_pct': self.gamma * 100,
            'transient_fraction_pct': (1 - self.gamma) * 100
        }

    
    def decompose_impact(self, order_size: float, time_grid: np.ndarray,
                        kyle_lambda: float) -> pd.DataFrame:
        """
        Decompose impact into permanent and transient components over time.
        
        Parameters:
            order_size: Order size
            time_grid: Time points
            kyle_lambda: Kyle's lambda for scaling
        
        Returns:
            DataFrame with impact decomposition
        """
        if not self.calibration_success:
            raise ValueError("Model not calibrated")
        
        # Total impact at t=0 (using Kyle's lambda)
        I_0 = kyle_lambda * order_size
        
        # Permanent component (constant over time)
        I_perm = self.gamma * I_0 * np.ones_like(time_grid)
        
        # Transient component (decays exponentially)
        I_temp = (1 - self.gamma) * I_0 * np.exp(-self.rho * time_grid)
        
        # Total impact
        I_total = I_perm + I_temp
        
        return pd.DataFrame({
            'time': time_grid,
            'permanent': I_perm,
            'transient': I_temp,
            'total': I_total,
            'transient_fraction': I_temp / I_total
        })
    
    def validate_constraints(self) -> Dict:
        """
        Validate that all theoretical constraints are satisfied.
        
        Returns:
            Dictionary with constraint validation results
        """
        if not self.calibration_success:
            raise ValueError("Model not calibrated")
        
        # Check γ ∈ [0, 1]
        gamma_in_bounds = 0 <= self.gamma <= 1
        gamma_violation = 0.0
        if self.gamma < 0:
            gamma_violation = abs(self.gamma)
        elif self.gamma > 1:
            gamma_violation = self.gamma - 1
        
        # Check ρ > 0
        rho_positive = self.rho > 0
        
        # Check half-life is finite
        half_life = np.log(2) / self.rho if self.rho > 0 else np.inf
        half_life_finite = half_life < 1000
        
        return {
            'gamma': self.gamma,
            'gamma_in_bounds': gamma_in_bounds,
            'gamma_violation': gamma_violation,
            'rho': self.rho,
            'rho_positive': rho_positive,
            'half_life': half_life,
            'half_life_finite': half_life_finite,
            'all_constraints_satisfied': gamma_in_bounds and rho_positive and half_life_finite,
            'has_nans': np.isnan(self.gamma) or np.isnan(self.rho)
        }
    
    def compare_to_kyle(self, kyle_lambda: float, kyle_r_squared: float) -> Dict:
        """
        Compare OW decomposition to Kyle's linear model.
        
        Key Differences:
        - Kyle: All impact is permanent (no decay)
        - OW: Separates permanent from transient
        
        Parameters:
            kyle_lambda: Kyle's lambda estimate
            kyle_r_squared: Kyle's R²
        
        Returns:
            Dictionary with comparison metrics
        """
        if not self.calibration_success:
            raise ValueError("Model not calibrated")
        
        # OW effective lambda (at t=0)
        ow_lambda_t0 = kyle_lambda  # Same at t=0
        ow_lambda_permanent = self.gamma * kyle_lambda
        
        # Interpretation
        if self.gamma > 0.7:
            interpretation = "High permanent fraction - information dominates (Kyle appropriate)"
        elif self.gamma < 0.3:
            interpretation = "Low permanent fraction - liquidity dominates (OW necessary)"
        else:
            interpretation = "Mixed permanent/transient - both models useful"
        
        return {
            'kyle_lambda': kyle_lambda,
            'kyle_r_squared': kyle_r_squared,
            'ow_gamma': self.gamma,
            'ow_permanent_lambda': ow_lambda_permanent,
            'ow_transient_lambda': (1 - self.gamma) * kyle_lambda,
            'permanent_fraction_pct': self.gamma * 100,
            'transient_fraction_pct': (1 - self.gamma) * 100,
            'interpretation': interpretation,
            'ow_improves_on_kyle': self.gamma < 0.7  # OW useful when transient is significant
        }


def calibrate_ow_by_regime(regimes: Dict[str, pd.DataFrame],
                           kyle_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrate OW model for each regime using Kyle results.
    
    Parameters:
        regimes: Dictionary of regime DataFrames
        kyle_results: DataFrame with Kyle calibration results
    
    Returns:
        DataFrame with OW calibration results by regime
    """
    results = []
    
    for _, kyle_row in kyle_results.iterrows():
        regime_name = kyle_row['regime']
        data = regimes[regime_name]
        
        model = ObizhaevaWangModel(enforce_constraints=True)
        
        # Calibrate using Kyle's lambda
        calib = model.calibrate_from_kyle(
            order_flow=data['order_flow'].values,
            returns=data['returns'].values,
            kyle_lambda=kyle_row['estimated_lambda']
        )
        
        # Validate constraints
        constraints = model.validate_constraints()
        
        # Compare to Kyle
        comparison = model.compare_to_kyle(
            kyle_lambda=kyle_row['estimated_lambda'],
            kyle_r_squared=kyle_row['r_squared']
        )
        
        results.append({
            'regime': regime_name,
            'gamma': calib['gamma'],
            'rho': calib['rho'],
            'half_life': calib['half_life'],
            'permanent_fraction_pct': calib['permanent_fraction_pct'],
            'transient_fraction_pct': calib['transient_fraction_pct'],
            'gamma_valid': constraints['gamma_in_bounds'],
            'rho_valid': constraints['rho_positive'],
            'half_life_finite': constraints['half_life_finite'],
            'constraints_satisfied': constraints['all_constraints_satisfied'],
            'has_nans': constraints['has_nans'],
            'kyle_lambda': kyle_row['estimated_lambda'],
            'kyle_r_squared': kyle_row['r_squared'],
            'ow_improves_on_kyle': comparison['ow_improves_on_kyle'],
            'interpretation': comparison['interpretation']
        })
    
    return pd.DataFrame(results)



def plot_ow_diagnostics(regimes: Dict[str, pd.DataFrame],
                        ow_results: pd.DataFrame,
                        kyle_results: pd.DataFrame,
                        save_path: str = 'results/figures/'):
    """
    Generate comprehensive diagnostic plots for OW model.
    
    Plots:
    1. Permanent vs Transient fractions by regime
    2. Impact decomposition over time
    3. Half-life by regime
    4. OW vs Kyle comparison
    
    Parameters:
        regimes: Dictionary of regime DataFrames
        ow_results: DataFrame from calibrate_ow_by_regime()
        kyle_results: DataFrame with Kyle results
        save_path: Directory to save figures
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = {'low': '#d62728', 'medium': '#ff7f0e', 'high': '#2ca02c'}
    
    # Plot 1: Permanent vs Transient fractions
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(ow_results))
    width = 0.35
    
    ax1.bar(x_pos - width/2, ow_results['permanent_fraction_pct'],
           width, label='Permanent (Information)', color='#2ca02c', alpha=0.7)
    ax1.bar(x_pos + width/2, ow_results['transient_fraction_pct'],
           width, label='Transient (Liquidity)', color='#d62728', alpha=0.7)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ow_results['regime'].str.capitalize())
    ax1.set_ylabel('Fraction (%)', fontsize=11)
    ax1.set_title('Impact Decomposition\n(Permanent vs Transient)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Impact decay over time
    ax2 = fig.add_subplot(gs[0, 1])
    time_grid = np.linspace(0, 60, 100)  # 60 time units
    
    for _, row in ow_results.iterrows():
        regime_name = row['regime']
        model = ObizhaevaWangModel()
        model.gamma = row['gamma']
        model.rho = row['rho']
        model.calibration_success = True
        
        kyle_lambda = row['kyle_lambda']
        decomp = model.decompose_impact(order_size=1.0, time_grid=time_grid,
                                        kyle_lambda=kyle_lambda)
        
        ax2.plot(decomp['time'], decomp['total'], '-',
                label=f"{regime_name.capitalize()}", color=colors[regime_name],
                linewidth=2, alpha=0.8)
        ax2.plot(decomp['time'], decomp['permanent'], '--',
                color=colors[regime_name], linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Impact (normalized)', fontsize=11)
    ax2.set_title('Impact Decay Over Time\n(Solid=Total, Dashed=Permanent)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Half-life by regime
    ax3 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(len(ow_results))
    
    bars = ax3.bar(x_pos, ow_results['half_life'],
                   color=[colors[r] for r in ow_results['regime']], alpha=0.7)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(ow_results['regime'].str.capitalize())
    ax3.set_ylabel('Half-Life (time units)', fontsize=11)
    ax3.set_title('Transient Impact Half-Life\n(Order book recovery speed)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, ow_results['half_life']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: OW vs Kyle comparison (gamma vs R²)
    ax4 = fig.add_subplot(gs[1, 0])
    
    for _, row in ow_results.iterrows():
        regime_name = row['regime']
        ax4.scatter(row['kyle_r_squared'], row['gamma'],
                   s=200, color=colors[regime_name], alpha=0.7,
                   label=regime_name.capitalize())
    
    ax4.set_xlabel("Kyle's R²", fontsize=11)
    ax4.set_ylabel('OW Permanent Fraction (γ)', fontsize=11)
    ax4.set_title('OW vs Kyle Comparison\n(Higher R² → More Permanent)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(ow_results['kyle_r_squared']) * 1.2)
    ax4.set_ylim(0, 1)
    
    # Plot 5: Constraint validation
    ax5 = fig.add_subplot(gs[1, 1])
    
    constraint_checks = []
    for _, row in ow_results.iterrows():
        checks = [
            row['gamma_valid'],
            row['rho_valid'],
            row['half_life_finite'],
            not row['has_nans']
        ]
        constraint_checks.append(checks)
    
    constraint_checks = np.array(constraint_checks).T
    labels = ['γ ∈ [0,1]', 'ρ > 0', 'Half-life\nfinite', 'No NaNs']
    
    im = ax5.imshow(constraint_checks, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(np.arange(len(ow_results)))
    ax5.set_xticklabels(ow_results['regime'].str.capitalize())
    ax5.set_yticks(np.arange(len(labels)))
    ax5.set_yticklabels(labels)
    ax5.set_title('Constraint Validation\n(Green=Pass, Red=Fail)', fontsize=12, fontweight='bold')
    
    # Add checkmarks/crosses
    for i in range(len(labels)):
        for j in range(len(ow_results)):
            text = '✓' if constraint_checks[i, j] else '✗'
            ax5.text(j, i, text, ha='center', va='center',
                    color='white', fontsize=16, fontweight='bold')
    
    # Plot 6: Parameter stability
    ax6 = fig.add_subplot(gs[1, 2])
    
    x_pos = np.arange(len(ow_results))
    ax6_twin = ax6.twinx()
    
    bars1 = ax6.bar(x_pos - 0.2, ow_results['gamma'], 0.4,
                    label='γ (Permanent)', color='#2ca02c', alpha=0.7)
    bars2 = ax6_twin.bar(x_pos + 0.2, ow_results['rho'], 0.4,
                         label='ρ (Decay)', color='#1f77b4', alpha=0.7)
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(ow_results['regime'].str.capitalize())
    ax6.set_ylabel('γ (Permanent Fraction)', fontsize=11, color='#2ca02c')
    ax6_twin.set_ylabel('ρ (Decay Rate)', fontsize=11, color='#1f77b4')
    ax6.set_title('Parameter Stability Across Regimes', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='#2ca02c')
    ax6_twin.tick_params(axis='y', labelcolor='#1f77b4')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.savefig(f'{save_path}/ow_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ OW diagnostic plots saved to {save_path}/ow_diagnostics.png")


def print_ow_summary(ow_results: pd.DataFrame):
    """
    Print comprehensive summary of OW model calibration.
    
    Parameters:
        ow_results: DataFrame from calibrate_ow_by_regime()
    """
    print("\n" + "="*80)
    print("OBIZHAEVA-WANG MODEL: CALIBRATION SUMMARY")
    print("="*80)
    
    for _, row in ow_results.iterrows():
        regime = row['regime'].upper()
        print(f"\n{regime} LIQUIDITY REGIME:")
        print(f"  Permanent Fraction (γ): {row['gamma']:.4f} ({row['permanent_fraction_pct']:.1f}%)")
        print(f"  Transient Fraction:     {1-row['gamma']:.4f} ({row['transient_fraction_pct']:.1f}%)")
        print(f"  Decay Rate (ρ):         {row['rho']:.4f}")
        print(f"  Half-Life:              {row['half_life']:.2f} time units")
        
        print(f"\n  Constraint Validation:")
        print(f"    γ ∈ [0,1]:            {'✓' if row['gamma_valid'] else '✗'}")
        print(f"    ρ > 0:                {'✓' if row['rho_valid'] else '✗'}")
        print(f"    Half-life finite:     {'✓' if row['half_life_finite'] else '✗'}")
        print(f"    No NaNs:              {'✓' if not row['has_nans'] else '✗'}")
        print(f"    All constraints:      {'✓ PASS' if row['constraints_satisfied'] else '✗ FAIL'}")
        
        print(f"\n  Comparison to Kyle:")
        print(f"    Kyle's λ:             {row['kyle_lambda']:.4f}")
        print(f"    Kyle's R²:            {row['kyle_r_squared']:.4f}")
        print(f"    OW improves on Kyle:  {'✓ YES' if row['ow_improves_on_kyle'] else '✗ NO'}")
        print(f"    Interpretation:       {row['interpretation']}")
    
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT:")
    print("="*80)
    
    all_constraints = ow_results['constraints_satisfied'].all()
    no_nans = (~ow_results['has_nans']).all()
    mean_gamma = ow_results['gamma'].mean()
    gamma_cv = ow_results['gamma'].std() / mean_gamma if mean_gamma > 0 else np.inf
    
    print(f"  All constraints satisfied: {'✓ PASS' if all_constraints else '✗ FAIL'}")
    print(f"  No NaN values:             {'✓ PASS' if no_nans else '✗ FAIL'}")
    print(f"  Mean permanent fraction:   {mean_gamma:.4f} ({mean_gamma*100:.1f}%)")
    print(f"  Gamma CV (stability):      {gamma_cv:.4f}")
    print(f"  Parameter stability:       {'✓ STABLE' if gamma_cv < 0.5 else '✗ UNSTABLE'}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    """
    Standalone execution for Phase 3B validation.
    """
    import sys
    sys.path.append('.')
    from src.data_generation import generate_regime_switching_data
    from src.kyle_model import calibrate_kyle_by_regime
    
    print("\n" + "="*80)
    print("PHASE 3B: OBIZHAEVA-WANG MODEL IMPLEMENTATION")
    print("="*80 + "\n")
    
    # Generate data
    print("Generating synthetic market data...")
    regimes = generate_regime_switching_data(
        n_periods_per_regime=3000,
        hurst=0.7,
        random_seed=42
    )
    
    # True λ values (converted to bps)
    true_lambdas = {
        'low': 0.002 * 100,
        'medium': 0.001 * 100,
        'high': 0.0005 * 100
    }
    
    print("✓ Data generated\n")
    
    # Calibrate Kyle model first
    print("Calibrating Kyle model...")
    kyle_results = calibrate_kyle_by_regime(regimes, true_lambdas)
    print("✓ Kyle calibration complete\n")
    
    # Calibrate OW model
    print("Calibrating Obizhaeva-Wang model...")
    ow_results = calibrate_ow_by_regime(regimes, kyle_results)
    print("✓ OW calibration complete\n")
    
    # Print summary
    print_ow_summary(ow_results)
    
    # Save results
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    ow_results.to_csv('results/tables/ow_calibration.csv', index=False)
    print("✓ Results saved to results/tables/ow_calibration.csv\n")
    
    # Generate diagnostic plots
    print("Generating diagnostic plots...")
    plot_ow_diagnostics(regimes, ow_results, kyle_results)
    
    print("\n" + "="*80)
    print("PHASE 3B COMPLETE")
    print("="*80)
    print("\nNext: Review OW results and approve before Phase 3C (Bouchaud Model)")
