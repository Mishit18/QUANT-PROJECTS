"""
Visualization utilities for Kalman filters, HMM regimes, and trading strategies.

Generates publication-quality plots for research and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_kalman_filter_results(observations: np.ndarray,
                               filtered_states: np.ndarray,
                               smoothed_states: Optional[np.ndarray] = None,
                               true_states: Optional[np.ndarray] = None,
                               title: str = "Kalman Filter Results",
                               save_path: Optional[str] = None):
    """
    Plot Kalman filter estimation results.
    
    Parameters
    ----------
    observations : np.ndarray
        Observed data
    filtered_states : np.ndarray
        Filtered state estimates
    smoothed_states : np.ndarray, optional
        Smoothed state estimates
    true_states : np.ndarray, optional
        True states (if available)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    t = np.arange(len(observations))
    
    # Observations
    ax.plot(t, observations, 'o', alpha=0.3, markersize=3, label='Observations', color='gray')
    
    # Filtered states
    if filtered_states.ndim == 1:
        ax.plot(t, filtered_states, '-', linewidth=2, label='Filtered', color='blue')
    else:
        ax.plot(t, filtered_states[:, 0], '-', linewidth=2, label='Filtered', color='blue')
    
    # Smoothed states
    if smoothed_states is not None:
        if smoothed_states.ndim == 1:
            ax.plot(t, smoothed_states, '-', linewidth=2, label='Smoothed', color='green')
        else:
            ax.plot(t, smoothed_states[:, 0], '-', linewidth=2, label='Smoothed', color='green')
    
    # True states
    if true_states is not None:
        if true_states.ndim == 1:
            ax.plot(t, true_states, '--', linewidth=2, label='True State', color='red', alpha=0.7)
        else:
            ax.plot(t, true_states[:, 0], '--', linewidth=2, label='True State', color='red', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regime_probabilities(regime_probs: np.ndarray,
                              returns: Optional[np.ndarray] = None,
                              title: str = "Regime Probabilities",
                              save_path: Optional[str] = None):
    """
    Plot regime probabilities over time.
    
    Parameters
    ----------
    regime_probs : np.ndarray
        Regime probabilities (n_samples, n_regimes)
    returns : np.ndarray, optional
        Returns to plot alongside
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    n_regimes = regime_probs.shape[1]
    
    if returns is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))
    
    t = np.arange(len(regime_probs))
    colors = plt.cm.Set3(np.linspace(0, 1, n_regimes))
    
    # Stacked area plot of regime probabilities
    ax1.stackplot(t, *[regime_probs[:, k] for k in range(n_regimes)],
                 labels=[f'Regime {k}' for k in range(n_regimes)],
                 colors=colors, alpha=0.7)
    
    ax1.set_ylabel('Regime Probability')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Plot returns if provided
    if returns is not None:
        ax2.plot(t, returns, color='black', alpha=0.6, linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Returns')
        ax2.set_title('Returns')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regime_labeled_series(data: np.ndarray,
                               regimes: np.ndarray,
                               title: str = "Regime-Labeled Time Series",
                               ylabel: str = "Value",
                               save_path: Optional[str] = None):
    """
    Plot time series with regime-based coloring.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    regimes : np.ndarray
        Regime labels
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    t = np.arange(len(data))
    unique_regimes = np.unique(regimes)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
    
    for i, regime in enumerate(unique_regimes):
        mask = regimes == regime
        ax.scatter(t[mask], data[mask], c=[colors[i]], label=f'Regime {regime}',
                  alpha=0.6, s=10)
    
    ax.plot(t, data, color='black', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_equity_curve(equity_curve: np.ndarray,
                     benchmark: Optional[np.ndarray] = None,
                     title: str = "Equity Curve",
                     save_path: Optional[str] = None):
    """
    Plot strategy equity curve.
    
    Parameters
    ----------
    equity_curve : np.ndarray
        Equity curve
    benchmark : np.ndarray, optional
        Benchmark equity curve
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    t = np.arange(len(equity_curve))
    
    ax.plot(t, equity_curve, linewidth=2, label='Strategy', color='blue')
    
    if benchmark is not None:
        ax.plot(t, benchmark, linewidth=2, label='Benchmark', color='gray', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add drawdown shading
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    ax2 = ax.twinx()
    ax2.fill_between(t, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_ylim([min(drawdown) * 1.1, 0.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_returns_distribution(returns: np.ndarray,
                              title: str = "Returns Distribution",
                              save_path: Optional[str] = None):
    """
    Plot returns distribution with statistics.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            'r-', linewidth=2, label='Normal fit')
    
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title}\nMean: {mu:.4f}, Std: {sigma:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_dynamic_beta(beta: np.ndarray,
                     market_returns: Optional[np.ndarray] = None,
                     title: str = "Dynamic Beta",
                     save_path: Optional[str] = None):
    """
    Plot time-varying beta estimates.
    
    Parameters
    ----------
    beta : np.ndarray
        Beta estimates over time
    market_returns : np.ndarray, optional
        Market returns to plot alongside
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    if market_returns is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))
    
    t = np.arange(len(beta))
    
    # Beta
    ax1.plot(t, beta, linewidth=2, color='blue')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Beta = 1')
    ax1.fill_between(t, beta, 1.0, where=(beta > 1.0), alpha=0.3, color='green', label='Beta > 1')
    ax1.fill_between(t, beta, 1.0, where=(beta < 1.0), alpha=0.3, color='red', label='Beta < 1')
    ax1.set_ylabel('Beta')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Market returns
    if market_returns is not None:
        ax2.plot(t, market_returns, color='black', alpha=0.6, linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Market Returns')
        ax2.set_title('Market Returns')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_volatility_comparison(latent_vol: np.ndarray,
                               realized_vol: np.ndarray,
                               title: str = "Volatility Comparison",
                               save_path: Optional[str] = None):
    """
    Compare latent and realized volatility.
    
    Parameters
    ----------
    latent_vol : np.ndarray
        Kalman-filtered latent volatility
    realized_vol : np.ndarray
        Realized volatility
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    t = np.arange(len(latent_vol))
    
    ax.plot(t, latent_vol, linewidth=2, label='Latent Volatility (Kalman)', color='blue')
    ax.plot(t, realized_vol, linewidth=1, label='Realized Volatility', color='red', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Volatility')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    valid_mask = ~(np.isnan(latent_vol) | np.isnan(realized_vol))
    if valid_mask.sum() > 0:
        corr = np.corrcoef(latent_vol[valid_mask], realized_vol[valid_mask])[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regime_transition_matrix(transition_matrix: np.ndarray,
                                  title: str = "Regime Transition Matrix",
                                  save_path: Optional[str] = None):
    """
    Plot HMM transition matrix as heatmap.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probability matrix
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_regimes = len(transition_matrix)
    
    sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=[f'Regime {i}' for i in range(n_regimes)],
               yticklabels=[f'Regime {i}' for i in range(n_regimes)],
               ax=ax, cbar_kws={'label': 'Probability'})
    
    ax.set_xlabel('To Regime')
    ax.set_ylabel('From Regime')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_comparison(strategies: dict,
                               title: str = "Strategy Performance Comparison",
                               save_path: Optional[str] = None):
    """
    Compare multiple strategies.
    
    Parameters
    ----------
    strategies : dict
        Dictionary of strategy_name -> equity_curve
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, equity in strategies.items():
        t = np.arange(len(equity))
        ax.plot(t, equity, linewidth=2, label=name, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_signals_and_positions(returns: np.ndarray,
                              signals: np.ndarray,
                              positions: np.ndarray,
                              title: str = "Trading Signals and Positions",
                              save_path: Optional[str] = None):
    """
    Plot trading signals and positions over time.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    signals : np.ndarray
        Trading signals
    positions : np.ndarray
        Actual positions taken
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    t = np.arange(len(returns))
    
    # Cumulative returns
    cum_returns = np.cumsum(returns)
    ax1.plot(t, cum_returns, linewidth=2, color='blue', label='Cumulative Returns')
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signals
    ax2.plot(t, signals, linewidth=1.5, color='green', alpha=0.7, label='Signals')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.fill_between(t, 0, signals, where=(signals > 0), color='green', alpha=0.3, label='Long')
    ax2.fill_between(t, 0, signals, where=(signals < 0), color='red', alpha=0.3, label='Short')
    ax2.set_ylabel('Signal Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Positions
    ax3.plot(t, positions, linewidth=1.5, color='purple', alpha=0.7, label='Positions')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax3.fill_between(t, 0, positions, where=(positions > 0), color='green', alpha=0.3)
    ax3.fill_between(t, 0, positions, where=(positions < 0), color='red', alpha=0.3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_dashboard(returns: np.ndarray,
                            equity_curve: np.ndarray,
                            regime_probs: np.ndarray,
                            filtered_states: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Create comprehensive dashboard with multiple subplots.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    equity_curve : np.ndarray
        Equity curve
    regime_probs : np.ndarray
        Regime probabilities
    filtered_states : np.ndarray
        Kalman-filtered states
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    t = np.arange(len(returns))
    n_regimes = regime_probs.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, n_regimes))
    
    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, equity_curve, linewidth=2, color='blue')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity')
    ax1.grid(True, alpha=0.3)
    
    # Regime probabilities
    ax2 = fig.add_subplot(gs[1, :])
    ax2.stackplot(t, *[regime_probs[:, k] for k in range(n_regimes)],
                 labels=[f'Regime {k}' for k in range(n_regimes)],
                 colors=colors, alpha=0.7)
    ax2.set_title('Regime Probabilities', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Returns distribution
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    
    # Kalman filtered states
    ax4 = fig.add_subplot(gs[2, 1])
    if filtered_states.ndim == 1:
        ax4.plot(t, filtered_states, linewidth=2, color='green')
    else:
        ax4.plot(t, filtered_states[:, 0], linewidth=2, color='green')
    ax4.set_title('Kalman Filtered State', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('State')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Strategy Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization functions...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    
    observations = np.cumsum(np.random.randn(n) * 0.1) + np.random.randn(n) * 0.5
    filtered = np.cumsum(np.random.randn(n) * 0.1)
    
    # Test plot
    plot_kalman_filter_results(observations, filtered, title="Test Kalman Filter")
    
    print("Visualization tests complete.")
