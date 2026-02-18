"""
Market Impact Analysis: Complete Pipeline

Executes end-to-end analysis of Kyle, Obizhaeva-Wang, and Bouchaud models
across three liquidity regimes with comprehensive validation.

Run: python main.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from data_generation import generate_regime_switching_data, compute_diagnostics
from kyle_model import KyleModel
from obizhaeva_wang import ObizhaevaWangModel
from bouchaud_model import BouchaudModel


def main():
    """Execute complete market impact analysis pipeline."""
    
    print("=" * 80)
    print("MARKET IMPACT ANALYSIS: KYLE, OBIZHAEVA-WANG, BOUCHAUD")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # =========================================================================
    # 1. DATA GENERATION
    # =========================================================================
    print("\n[1/4] Generating synthetic market data...")
    print("  - 3 liquidity regimes (low, medium, high)")
    print("  - 3,000 periods per regime")
    print("  - Fractional Brownian motion (H=0.7)")
    
    data_by_regime = generate_regime_switching_data(
        n_periods_per_regime=3000,
        hurst=0.7,
        random_seed=42
    )
    
    # Validate data properties
    print("\n  Data validation:")
    for regime, data in data_by_regime.items():
        props = compute_diagnostics(data)
        print(f"    {regime:6s}: return_acf={props['return_acf_lag1']:6.3f}, "
              f"volume_acf={props['volume_acf_lag1']:6.3f}, "
              f"impact_corr={props['impact_correlation']:6.3f}")
    
    # =========================================================================
    # 2. MODEL CALIBRATION
    # =========================================================================
    print("\n[2/4] Calibrating models on each regime...")
    
    calibration_results = {}
    
    for regime in ['low', 'medium', 'high']:
        print(f"\n  Calibrating {regime.upper()} regime...")
        
        data = data_by_regime[regime]
        order_flow = data['order_flow'].values
        returns = data['returns'].values
        
        # Kyle's Lambda
        kyle = KyleModel()
        kyle_results = kyle.calibrate(order_flow, returns)
        print(f"    Kyle λ = {kyle_results['lambda']:.6f} "
              f"(CI: [{kyle_results['ci_lower']:.6f}, {kyle_results['ci_upper']:.6f}]), "
              f"R² = {kyle_results['r_squared']:.4f}")
        
        # Obizhaeva-Wang
        ow = ObizhaevaWangModel()
        ow_results = ow.calibrate_from_kyle(order_flow, returns, kyle_results['lambda'])
        print(f"    OW γ = {ow_results['gamma']:.4f}, "
              f"ρ = {ow_results['rho']:.4f}, "
              f"half-life = {ow_results['half_life']:.4f}")
        
        # Bouchaud
        bouchaud = BouchaudModel(memory_horizon=60, tau_0=1.0)
        bouchaud_results = bouchaud.calibrate(order_flow, returns)
        long_memory = bouchaud.validate_long_memory(order_flow)
        bouchaud_results.update(long_memory)  # Add hurst_estimate to results
        print(f"    Bouchaud β = {bouchaud_results['beta']:.4f}, "
              f"A = {bouchaud_results['amplitude']:.6f}, "
              f"Hurst = {bouchaud_results['hurst_estimate']:.4f}")
        
        calibration_results[regime] = {
            'kyle': kyle_results,
            'ow': ow_results,
            'bouchaud': bouchaud_results
        }
    
    # =========================================================================
    # 3. PARAMETER STABILITY ANALYSIS
    # =========================================================================
    print("\n[3/4] Analyzing parameter stability across regimes...")
    
    # Extract parameters
    lambdas = [calibration_results[r]['kyle']['lambda'] for r in ['low', 'medium', 'high']]
    gammas = [calibration_results[r]['ow']['gamma'] for r in ['low', 'medium', 'high']]
    betas = [calibration_results[r]['bouchaud']['beta'] for r in ['low', 'medium', 'high']]
    
    # Compute coefficient of variation
    lambda_cv = np.std(lambdas) / np.mean(lambdas)
    gamma_cv = np.std(gammas) / np.mean(gammas)
    beta_cv = np.std(betas) / np.mean(betas)
    
    print(f"\n  Parameter Stability (CV < 0.5 = stable):")
    print(f"    Kyle λ:      CV = {lambda_cv:.4f}  {'✓ STABLE' if lambda_cv < 0.5 else '✗ UNSTABLE'}")
    print(f"    OW γ:        CV = {gamma_cv:.4f}  {'✓ STABLE' if gamma_cv < 0.5 else '✗ UNSTABLE'}")
    print(f"    Bouchaud β:  CV = {beta_cv:.4f}  {'✓ STABLE' if beta_cv < 0.5 else '✗ UNSTABLE'}")
    
    # =========================================================================
    # 4. SAVE RESULTS
    # =========================================================================
    print("\n[4/4] Saving results...")
    
    # Kyle calibration table
    kyle_table = []
    for regime in ['low', 'medium', 'high']:
        kyle_table.append({
            'regime': regime,
            **calibration_results[regime]['kyle']
        })
    pd.DataFrame(kyle_table).to_csv('results/tables/kyle_calibration.csv', index=False)
    print("  ✓ results/tables/kyle_calibration.csv")
    
    # OW calibration table
    ow_table = []
    for regime in ['low', 'medium', 'high']:
        ow_table.append({
            'regime': regime,
            **calibration_results[regime]['ow']
        })
    pd.DataFrame(ow_table).to_csv('results/tables/ow_calibration.csv', index=False)
    print("  ✓ results/tables/ow_calibration.csv")
    
    # Bouchaud calibration table
    bouchaud_table = []
    for regime in ['low', 'medium', 'high']:
        bouchaud_table.append({
            'regime': regime,
            **calibration_results[regime]['bouchaud']
        })
    pd.DataFrame(bouchaud_table).to_csv('results/tables/bouchaud_calibration.csv', index=False)
    print("  ✓ results/tables/bouchaud_calibration.csv")
    
    # Cross-regime validation
    print("\n  Computing cross-regime validation...")
    validation_results = []
    for train_regime in ['low', 'medium', 'high']:
        for test_regime in ['low', 'medium', 'high']:
            if train_regime == test_regime:
                continue
            
            # Get trained lambda
            lambda_train = calibration_results[train_regime]['kyle']['lambda']
            
            # Test on different regime
            test_data = data_by_regime[test_regime]
            order_flow_test = test_data['order_flow'].values
            returns_test = test_data['returns'].values
            
            # Predict and compute error
            predicted = lambda_train * order_flow_test
            mse = np.mean((returns_test - predicted)**2)
            mae = np.mean(np.abs(returns_test - predicted))
            relative_error = mae / np.std(returns_test)
            
            validation_results.append({
                'train_regime': train_regime,
                'test_regime': test_regime,
                'mse': mse,
                'mae': mae,
                'relative_error': relative_error
            })
    
    pd.DataFrame(validation_results).to_csv('results/tables/cross_regime_validation.csv', index=False)
    print("  ✓ results/tables/cross_regime_validation.csv")
    
    # Parameter stability table
    stability_table = pd.DataFrame({
        'parameter': ['kyle_lambda', 'ow_gamma', 'bouchaud_beta'],
        'mean': [np.mean(lambdas), np.mean(gammas), np.mean(betas)],
        'std': [np.std(lambdas), np.std(gammas), np.std(betas)],
        'cv': [lambda_cv, gamma_cv, beta_cv],
        'stable': [lambda_cv < 0.5, gamma_cv < 0.5, beta_cv < 0.5]
    })
    stability_table.to_csv('results/tables/parameter_stability.csv', index=False)
    print("  ✓ results/tables/parameter_stability.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print("\nKey Findings:")
    print(f"  • Kyle's λ ranges from {min(lambdas):.6f} to {max(lambdas):.6f} (varies {max(lambdas)/min(lambdas):.1f}×)")
    print(f"  • OW permanent fraction γ ranges from {min(gammas):.1%} to {max(gammas):.1%}")
    print(f"  • Bouchaud exponent β ranges from {min(betas):.2f} to {max(betas):.2f}")
    
    print("\nConstraint Validation:")
    all_gamma_valid = all(0 <= calibration_results[r]['ow']['gamma'] <= 1 for r in ['low', 'medium', 'high'])
    all_beta_valid = all(0.3 <= calibration_results[r]['bouchaud']['beta'] <= 0.8 for r in ['low', 'medium', 'high'])
    print(f"  • OW γ ∈ [0,1]: {'✓ PASS' if all_gamma_valid else '✗ FAIL'}")
    print(f"  • Bouchaud β ∈ [0.3,0.8]: {'✓ PASS' if all_beta_valid else '✗ FAIL'}")
    
    print("\nResults saved to:")
    print("  • results/tables/kyle_calibration.csv")
    print("  • results/tables/ow_calibration.csv")
    print("  • results/tables/bouchaud_calibration.csv")
    print("  • results/tables/cross_regime_validation.csv")
    print("  • results/tables/parameter_stability.csv")
    
    print("\nDocumentation:")
    print("  • report/market_impact_analysis.md")
    print("  • report/failure_analysis.md")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
