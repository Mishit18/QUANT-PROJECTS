"""
Plotting utilities for volatility surfaces and model comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, List

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_implied_vol_smile(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Implied Volatility Smile",
    spot: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot implied volatility smile(s).

    Args:
        strikes: Strike prices
        implied_vols: IV arrays (can be 2D for multiple models)
        labels: Model labels
        title: Plot title
        spot: Spot price for moneyness axis
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if implied_vols.ndim == 1:
        implied_vols = implied_vols.reshape(1, -1)
        labels = [labels] if labels else ['Model']

    if spot is not None:
        x_axis = strikes / spot
        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    else:
        x_axis = strikes
        ax.set_xlabel('Strike', fontsize=12)

    for i, iv in enumerate(implied_vols):
        label = labels[i] if labels and i < len(labels) else f'Model {i+1}'
        ax.plot(x_axis, iv, marker='o', linewidth=2, label=label, markersize=6)

    ax.set_ylabel('Implied Volatility', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_vol_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    title: str = "Implied Volatility Surface",
    save_path: Optional[str] = None
):
    """
    Plot 3D implied volatility surface.

    Args:
        strikes: Strike prices
        maturities: Time to maturity
        implied_vols: IV surface (n_maturities, n_strikes)
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    K, T = np.meshgrid(strikes, maturities)

    surf = ax.plot_surface(K, T, implied_vols, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.9)

    ax.set_xlabel('Strike', fontsize=11)
    ax.set_ylabel('Maturity', fontsize=11)
    ax.set_zlabel('Implied Volatility', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_term_structure_skew(
    maturities: np.ndarray,
    skews: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Term Structure of Skew",
    save_path: Optional[str] = None
):
    """
    Plot term structure of implied volatility skew.

    Args:
        maturities: Time to maturity
        skews: Skew values (ATM - OTM vol difference)
        labels: Model labels
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if skews.ndim == 1:
        skews = skews.reshape(1, -1)
        labels = [labels] if labels else ['Model']

    for i, skew in enumerate(skews):
        label = labels[i] if labels and i < len(labels) else f'Model {i+1}'
        ax.plot(maturities, skew, marker='s', linewidth=2, label=label, markersize=7)

    ax.set_xlabel('Time to Maturity (years)', fontsize=12)
    ax.set_ylabel('Skew (ATM - 90% Put IV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_model_comparison(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    model_vols_dict: dict,
    maturity: float,
    spot: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Compare multiple models against market data.

    Args:
        strikes: Strike prices
        market_vols: Market implied volatilities
        model_vols_dict: Dictionary {model_name: implied_vols}
        maturity: Time to maturity
        spot: Spot price
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    moneyness = strikes / spot

    # Left panel: Implied volatilities
    ax1.plot(moneyness, market_vols, 'ko-', linewidth=2,
             markersize=8, label='Market', zorder=10)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (model_name, model_vols) in enumerate(model_vols_dict.items()):
        ax1.plot(moneyness, model_vols, marker='s', linewidth=2,
                color=colors[i % len(colors)], label=model_name, alpha=0.8)

    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax1.set_ylabel('Implied Volatility', fontsize=12)
    ax1.set_title(f'Implied Volatility Smile (T={maturity:.2f}y)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Calibration errors
    for i, (model_name, model_vols) in enumerate(model_vols_dict.items()):
        errors = (model_vols - market_vols) * 100  # In percentage points
        ax2.plot(moneyness, errors, marker='o', linewidth=2,
                color=colors[i % len(colors)], label=model_name)

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax2.set_ylabel('Error (% points)', fontsize=12)
    ax2.set_title('Calibration Errors', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_convergence_diagnostics(
    n_paths_array: np.ndarray,
    prices: np.ndarray,
    std_errors: np.ndarray,
    true_price: Optional[float] = None,
    title: str = "Monte Carlo Convergence",
    save_path: Optional[str] = None
):
    """
    Plot Monte Carlo convergence diagnostics.

    Args:
        n_paths_array: Array of path counts
        prices: Estimated prices
        std_errors: Standard errors
        true_price: Reference price (if known)
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Price convergence
    ax1.plot(n_paths_array, prices, 'b-o', linewidth=2, markersize=6)
    ax1.fill_between(n_paths_array,
                     prices - 1.96 * std_errors,
                     prices + 1.96 * std_errors,
                     alpha=0.3, label='95% CI')

    if true_price is not None:
        ax1.axhline(true_price, color='red', linestyle='--',
                   linewidth=2, label='True Price')

    ax1.set_xlabel('Number of Paths', fontsize=12)
    ax1.set_ylabel('Option Price', fontsize=12)
    ax1.set_title('Price Convergence', fontsize=13)
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Standard error decay
    theoretical_decay = std_errors[0] * np.sqrt(n_paths_array[0] / n_paths_array)

    ax2.loglog(n_paths_array, std_errors, 'b-o', linewidth=2,
              markersize=6, label='Empirical')
    ax2.loglog(n_paths_array, theoretical_decay, 'r--',
              linewidth=2, label='O(1/√N)')

    ax2.set_xlabel('Number of Paths', fontsize=12)
    ax2.set_ylabel('Standard Error', fontsize=12)
    ax2.set_title('Error Decay Rate', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_hurst_sensitivity(
    H_values: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str = "Skew",
    title: str = "Sensitivity to Hurst Parameter",
    save_path: Optional[str] = None
):
    """
    Plot sensitivity to Hurst parameter.

    Args:
        H_values: Hurst parameter values
        metric_values: Metric values (e.g., skew, ATM vol)
        metric_name: Name of metric
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(H_values, metric_values, 'b-o', linewidth=2.5, markersize=8)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5,
              label='H=0.5 (Brownian)')

    ax.set_xlabel('Hurst Parameter H', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
