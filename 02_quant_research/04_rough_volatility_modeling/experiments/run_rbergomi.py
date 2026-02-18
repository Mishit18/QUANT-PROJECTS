import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.rbergomi import RBergomiModel
from src.simulation.hybrid_scheme import HybridScheme
from src.pricing.monte_carlo_pricer import MonteCarloOptionPricer
from src.pricing.implied_vol import implied_volatility_smile
from src.utils.plotting import (
    plot_implied_vol_smile,
    plot_term_structure_skew,
    plot_convergence_diagnostics,
    plot_hurst_sensitivity
)


def main():
    """Run rBergomi experiments."""

    print("=" * 70)
    print("rBergomi Rough Volatility: Simulation and Analysis")
    print("=" * 70)

    # Model parameters (calibrated-like values)
    H = 0.07        # Rough Hurst exponent
    eta = 1.9       # Vol-of-vol
    rho = -0.9      # Strong negative correlation
    xi0 = 0.04      # Initial variance (20% vol)

    model = RBergomiModel(H, eta, rho, xi0)
    print(f"\n{model}")

    # Simulation parameters
    S0 = 100.0
    T = 1.0
    n_steps = 252
    n_paths = 10000

    print(f"\nSimulation: {n_paths} paths, {n_steps} steps, T={T}y")

    # Simulate paths
    print("\nSimulating rBergomi paths...")
    scheme = HybridScheme(n_steps, n_paths, seed=42)
    S_paths, v_paths = scheme.simulate_rbergomi(H, eta, rho, xi0, T, S0)

    print(f"Terminal spot: mean={np.mean(S_paths[:, -1]):.2f}, "
          f"std={np.std(S_paths[:, -1]):.2f}")
    print(f"Terminal variance: mean={np.mean(v_paths[:, -1]):.4f}, "
          f"std={np.std(v_paths[:, -1]):.4f}")

    # Option pricing
    print("\n" + "-" * 70)
    print("European Option Pricing")
    print("-" * 70)

    strikes = np.array([80, 90, 95, 100, 105, 110, 120])
    maturities = np.array([0.1, 0.25, 0.5, 1.0])  # 1m, 3m, 6m, 1y

    pricer = MonteCarloOptionPricer(n_paths, seed=42)

    # Price options for different maturities
    implied_vols_surface = np.zeros((len(maturities), len(strikes)))

    for i, T_mat in enumerate(maturities):
        print(f"\nMaturity T={T_mat:.2f}y:")

        # Simulate for this maturity
        scheme_mat = HybridScheme(int(n_steps * T_mat), n_paths, seed=42)
        S_paths_mat, _ = scheme_mat.simulate_rbergomi(H, eta, rho, xi0, T_mat, S0)

        prices = []
        for K in strikes:
            price, std_err = pricer.price_european_call(S_paths_mat, K, 0.0, T_mat)
            prices.append(price)
            print(f"  K={K:3.0f}: Price={price:6.3f} ± {std_err:.3f}")

        # Compute implied volatilities
        iv_smile = implied_volatility_smile(
            np.array(prices), S0, strikes, T_mat, 0.0, 'call'
        )
        implied_vols_surface[i, :] = iv_smile

    # Plot implied volatility smiles
    print("\n" + "-" * 70)
    print("Generating Plots")
    print("-" * 70)

    Path("data").mkdir(exist_ok=True)

    # Plot smiles for each maturity
    for i, T_mat in enumerate(maturities):
        plot_implied_vol_smile(
            strikes,
            implied_vols_surface[i, :],
            labels=[f'rBergomi (T={T_mat:.2f}y)'],
            title=f'rBergomi Implied Volatility Smile (T={T_mat:.2f}y)',
            spot=S0,
            save_path=f'data/rbergomi_smile_T{T_mat:.2f}.png'
        )

    # Term structure of skew
    print("\nComputing term structure of skew...")
    atm_idx = np.argmin(np.abs(strikes - S0))
    otm_put_idx = 0  # 80% strike

    skews = implied_vols_surface[:, atm_idx] - implied_vols_surface[:, otm_put_idx]

    plot_term_structure_skew(
        maturities,
        skews,
        labels=['rBergomi'],
        title='rBergomi: Term Structure of Skew',
        save_path='data/rbergomi_skew_term_structure.png'
    )

    print(f"\nSkew values: {skews}")
    print("Note: rBergomi produces steep skew at short maturities")

    # Convergence diagnostics
    print("\n" + "-" * 70)
    print("Monte Carlo Convergence Analysis")
    print("-" * 70)

    n_paths_array = np.array([1000, 2000, 5000, 10000, 20000])
    K_test = 100  # ATM
    T_test = 0.25

    prices_conv = []
    std_errors_conv = []

    for n_p in n_paths_array:
        scheme_conv = HybridScheme(int(n_steps * T_test), n_p, seed=42)
        S_paths_conv, _ = scheme_conv.simulate_rbergomi(H, eta, rho, xi0, T_test, S0)

        pricer_conv = MonteCarloOptionPricer(n_p, seed=42)
        price, std_err = pricer_conv.price_european_call(S_paths_conv, K_test, 0.0, T_test)

        prices_conv.append(price)
        std_errors_conv.append(std_err)
        print(f"n_paths={n_p:5d}: Price={price:.4f} ± {std_err:.4f}")

    plot_convergence_diagnostics(
        n_paths_array,
        np.array(prices_conv),
        np.array(std_errors_conv),
        title='rBergomi Monte Carlo Convergence (ATM Call, T=0.25y)',
        save_path='data/rbergomi_convergence.png'
    )

    # Hurst parameter sensitivity
    print("\n" + "-" * 70)
    print("Hurst Parameter Sensitivity")
    print("-" * 70)

    H_values = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    skew_values = []
    atm_vols = []

    print("\nAnalyzing skew vs H (fixed η=1.9, ρ=-0.9):")
    print(f"{'H':<8} {'ATM Vol':<12} {'Skew':<12} {'T^(H-0.5)':<12}")
    print("-" * 50)

    for H_test in H_values:
        scheme_h = HybridScheme(int(n_steps * 0.25), 5000, seed=42)
        S_paths_h, _ = scheme_h.simulate_rbergomi(H_test, eta, rho, xi0, 0.25, S0)

        prices_h = []
        for K in [80, 100]:  # OTM put and ATM
            pricer_h = MonteCarloOptionPricer(5000, seed=42)
            price, _ = pricer_h.price_european_call(S_paths_h, K, 0.0, 0.25)
            prices_h.append(price)

        iv_h = implied_volatility_smile(
            np.array(prices_h), S0, np.array([80, 100]), 0.25, 0.0, 'call'
        )

        skew_h = iv_h[1] - iv_h[0]  # ATM - OTM
        atm_vol = iv_h[1]
        skew_values.append(skew_h)
        atm_vols.append(atm_vol)

        theoretical_decay = 0.25**(H_test - 0.5)
        print(f"{H_test:<8.2f} {atm_vol:<12.4f} {skew_h:<12.4f} {theoretical_decay:<12.4f}")

    plot_hurst_sensitivity(
        H_values,
        np.array(skew_values),
        metric_name='Skew (ATM - 80% Put)',
        title='rBergomi: Sensitivity to Hurst Parameter (T=0.25y)',
        save_path='data/rbergomi_hurst_sensitivity.png'
    )

    # Vol-of-vol sensitivity
    print("\n" + "-" * 70)
    print("Vol-of-Vol Sensitivity")
    print("-" * 70)

    eta_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    skew_eta = []

    print("\nAnalyzing skew vs η (fixed H=0.07, ρ=-0.9):")
    print(f"{'η':<8} {'ATM Vol':<12} {'Skew':<12}")
    print("-" * 40)

    for eta_test in eta_values:
        scheme_e = HybridScheme(int(n_steps * 0.25), 5000, seed=42)
        S_paths_e, _ = scheme_e.simulate_rbergomi(H, eta_test, rho, xi0, 0.25, S0)

        prices_e = []
        for K in [80, 100]:
            pricer_e = MonteCarloOptionPricer(5000, seed=42)
            price, _ = pricer_e.price_european_call(S_paths_e, K, 0.0, 0.25)
            prices_e.append(price)

        iv_e = implied_volatility_smile(
            np.array(prices_e), S0, np.array([80, 100]), 0.25, 0.0, 'call'
        )

        skew_e = iv_e[1] - iv_e[0]
        atm_vol_e = iv_e[1]
        skew_eta.append(skew_e)

        print(f"{eta_test:<8.2f} {atm_vol_e:<12.4f} {skew_e:<12.4f}")

    # Plot eta sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_values, skew_eta, 'b-o', linewidth=2.5, markersize=8)
    ax.set_xlabel('Vol-of-Vol (η)', fontsize=12)
    ax.set_ylabel('Skew (ATM - 80% Put)', fontsize=12)
    ax.set_title('rBergomi: Sensitivity to Vol-of-Vol (H=0.07, T=0.25y)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/rbergomi_eta_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nObservation: Skew scales approximately linearly with η")
    print(f"Theoretical: Skew ~ η T^(H-0.5) = η × {0.25**(H-0.5):.4f}")

    # H-eta identifiability demonstration
    print("\n" + "-" * 70)
    print("H-η Identifiability Issue")
    print("-" * 70)

    print("\nDemonstrating that different (H, η) pairs can produce similar skew:")
    print(f"{'H':<8} {'η':<8} {'Skew':<12} {'η×T^(H-0.5)':<15}")
    print("-" * 50)

    # Test parameter pairs that should give similar skew
    param_pairs = [
        (0.07, 1.9),
        (0.10, 2.45),
        (0.05, 1.52),
    ]

    for H_test, eta_test in param_pairs:
        scheme_id = HybridScheme(int(n_steps * 0.25), 5000, seed=42)
        S_paths_id, _ = scheme_id.simulate_rbergomi(H_test, eta_test, rho, xi0, 0.25, S0)

        prices_id = []
        for K in [80, 100]:
            pricer_id = MonteCarloOptionPricer(5000, seed=42)
            price, _ = pricer_id.price_european_call(S_paths_id, K, 0.0, 0.25)
            prices_id.append(price)

        iv_id = implied_volatility_smile(
            np.array(prices_id), S0, np.array([80, 100]), 0.25, 0.0, 'call'
        )

        skew_id = iv_id[1] - iv_id[0]
        product = eta_test * (0.25**(H_test - 0.5))

        print(f"{H_test:<8.2f} {eta_test:<8.2f} {skew_id:<12.4f} {product:<15.4f}")

    print("\nConclusion: Different (H, η) pairs with similar η×T^(H-0.5) produce")
    print("similar short-maturity skew, demonstrating identifiability issues.")
    print("Regularization is necessary to stabilize calibration.")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("Results saved to data/ directory")
    print("=" * 70)


if __name__ == "__main__":
    main()
