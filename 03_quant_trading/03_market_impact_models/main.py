"""
Market Impact Analysis: Complete Pipeline

Executes end-to-end analysis of Kyle, Obizhaeva-Wang, and Bouchaud models
across three liquidity regimes with validation tables.

Run: python main.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from bouchaud_model import BouchaudModel
from data_generation import compute_diagnostics, generate_regime_switching_data
from kyle_model import KyleModel
from obizhaeva_wang import ObizhaevaWangModel


REGIMES = ["low", "medium", "high"]


def status_label(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def main():
    """Execute complete market impact analysis pipeline."""

    print("=" * 80)
    print("MARKET IMPACT ANALYSIS: KYLE, OBIZHAEVA-WANG, BOUCHAUD")
    print("=" * 80)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("\n[1/4] Generating synthetic market data...")
    print("  - 3 liquidity regimes: low, medium, high")
    print("  - 3,000 periods per regime")
    print("  - Long-memory order-flow component with H=0.7")

    data_by_regime = generate_regime_switching_data(
        n_periods_per_regime=3000,
        hurst=0.7,
        random_seed=42,
    )

    print("\n  Data validation:")
    for regime, data in data_by_regime.items():
        props = compute_diagnostics(data)
        print(
            f"    {regime:6s}: return_acf={props['return_acf_lag1']:6.3f}, "
            f"volume_acf={props['volume_acf_lag1']:6.3f}, "
            f"impact_corr={props['impact_correlation']:6.3f}"
        )

    print("\n[2/4] Calibrating models on each regime...")
    calibration_results = {}

    for regime in REGIMES:
        print(f"\n  Calibrating {regime.upper()} regime...")

        data = data_by_regime[regime]
        order_flow = data["order_flow"].values
        returns = data["returns"].values

        kyle = KyleModel()
        kyle_results = kyle.calibrate(order_flow, returns)
        print(
            f"    Kyle lambda = {kyle_results['lambda']:.6f} "
            f"(CI: [{kyle_results['ci_lower']:.6f}, {kyle_results['ci_upper']:.6f}]), "
            f"R^2 = {kyle_results['r_squared']:.4f}"
        )

        ow = ObizhaevaWangModel()
        ow_results = ow.calibrate_from_kyle(order_flow, returns, kyle_results["lambda"])
        print(
            f"    OW gamma = {ow_results['gamma']:.4f}, "
            f"rho = {ow_results['rho']:.4f}, "
            f"half-life = {ow_results['half_life']:.4f}"
        )

        bouchaud = BouchaudModel(memory_horizon=60, tau_0=1.0)
        bouchaud_results = bouchaud.calibrate(order_flow, returns)
        long_memory = bouchaud.validate_long_memory(order_flow)
        bouchaud_results.update(long_memory)
        print(
            f"    Bouchaud beta = {bouchaud_results['beta']:.4f}, "
            f"A = {bouchaud_results['amplitude']:.6f}, "
            f"Hurst = {bouchaud_results['hurst_estimate']:.4f}"
        )

        calibration_results[regime] = {
            "kyle": kyle_results,
            "ow": ow_results,
            "bouchaud": bouchaud_results,
        }

    print("\n[3/4] Analyzing parameter stability across regimes...")

    lambdas = [calibration_results[r]["kyle"]["lambda"] for r in REGIMES]
    gammas = [calibration_results[r]["ow"]["gamma"] for r in REGIMES]
    betas = [calibration_results[r]["bouchaud"]["beta"] for r in REGIMES]

    lambda_cv = np.std(lambdas) / np.mean(lambdas)
    gamma_cv = np.std(gammas) / np.mean(gammas)
    beta_cv = np.std(betas) / np.mean(betas)

    print("\n  Parameter Stability (CV < 0.5 = stable):")
    print(f"    Kyle lambda:   CV = {lambda_cv:.4f}  {status_label(lambda_cv < 0.5)}")
    print(f"    OW gamma:      CV = {gamma_cv:.4f}  {status_label(gamma_cv < 0.5)}")
    print(f"    Bouchaud beta: CV = {beta_cv:.4f}  {status_label(beta_cv < 0.5)}")

    print("\n[4/4] Saving results...")

    kyle_table = [{"regime": r, **calibration_results[r]["kyle"]} for r in REGIMES]
    pd.DataFrame(kyle_table).to_csv("results/tables/kyle_calibration.csv", index=False)
    print("  [OK] results/tables/kyle_calibration.csv")

    ow_table = [{"regime": r, **calibration_results[r]["ow"]} for r in REGIMES]
    pd.DataFrame(ow_table).to_csv("results/tables/ow_calibration.csv", index=False)
    print("  [OK] results/tables/ow_calibration.csv")

    bouchaud_table = [{"regime": r, **calibration_results[r]["bouchaud"]} for r in REGIMES]
    pd.DataFrame(bouchaud_table).to_csv("results/tables/bouchaud_calibration.csv", index=False)
    print("  [OK] results/tables/bouchaud_calibration.csv")

    print("\n  Computing cross-regime validation...")
    validation_results = []
    for train_regime in REGIMES:
        for test_regime in REGIMES:
            if train_regime == test_regime:
                continue

            lambda_train = calibration_results[train_regime]["kyle"]["lambda"]
            test_data = data_by_regime[test_regime]
            order_flow_test = test_data["order_flow"].values
            returns_test = test_data["returns"].values

            predicted = lambda_train * order_flow_test
            mse = np.mean((returns_test - predicted) ** 2)
            mae = np.mean(np.abs(returns_test - predicted))
            relative_error = mae / np.std(returns_test)

            validation_results.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    "mse": mse,
                    "mae": mae,
                    "relative_error": relative_error,
                }
            )

    pd.DataFrame(validation_results).to_csv(
        "results/tables/cross_regime_validation.csv", index=False
    )
    print("  [OK] results/tables/cross_regime_validation.csv")

    stability_table = pd.DataFrame(
        {
            "parameter": ["kyle_lambda", "ow_gamma", "bouchaud_beta"],
            "mean": [np.mean(lambdas), np.mean(gammas), np.mean(betas)],
            "std": [np.std(lambdas), np.std(gammas), np.std(betas)],
            "cv": [lambda_cv, gamma_cv, beta_cv],
            "stable": [lambda_cv < 0.5, gamma_cv < 0.5, beta_cv < 0.5],
        }
    )
    stability_table.to_csv("results/tables/parameter_stability.csv", index=False)
    print("  [OK] results/tables/parameter_stability.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nKey Findings:")
    print(
        f"  - Kyle lambda ranges from {min(lambdas):.6f} to {max(lambdas):.6f} "
        f"(varies {max(lambdas) / min(lambdas):.1f}x)"
    )
    print(f"  - OW permanent fraction ranges from {min(gammas):.1%} to {max(gammas):.1%}")
    print(f"  - Bouchaud exponent ranges from {min(betas):.2f} to {max(betas):.2f}")

    all_gamma_valid = all(0 <= calibration_results[r]["ow"]["gamma"] <= 1 for r in REGIMES)
    all_beta_valid = all(0.3 <= calibration_results[r]["bouchaud"]["beta"] <= 0.8 for r in REGIMES)

    print("\nConstraint Validation:")
    print(f"  - OW gamma in [0, 1]: {status_label(all_gamma_valid)}")
    print(f"  - Bouchaud beta in [0.3, 0.8]: {status_label(all_beta_valid)}")

    print("\nResults saved to:")
    print("  - results/tables/kyle_calibration.csv")
    print("  - results/tables/ow_calibration.csv")
    print("  - results/tables/bouchaud_calibration.csv")
    print("  - results/tables/cross_regime_validation.csv")
    print("  - results/tables/parameter_stability.csv")

    print("\nDocumentation:")
    print("  - report/market_impact_analysis.md")
    print("  - report/failure_analysis.md")
    print("  - docs/ATS_SCREENING_PACK.md")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
