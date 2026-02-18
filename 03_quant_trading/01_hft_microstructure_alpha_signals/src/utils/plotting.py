"""
Plotting utilities for HFT research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_pnl_curve(results_df: pd.DataFrame, save_path: str = None):
    """Plot cumulative PnL over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cumulative PnL
    ax1.plot(results_df['event_idx'], results_df['total_pnl'], linewidth=1.5)
    ax1.set_xlabel('Event Index')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.set_title('Cumulative PnL Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Position over time
    ax2.plot(results_df['event_idx'], results_df['position'], linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Event Index')
    ax2.set_ylabel('Position Size')
    ax2.set_title('Position Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"PnL curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                           save_path: str = None):
    """Plot feature importance."""
    # Take top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_returns_distribution(returns: pd.Series, save_path: str = None):
    """Plot distribution of returns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distribution', fontweight='bold')
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Returns distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdown(pnl_series: pd.Series, save_path: str = None):
    """Plot drawdown over time."""
    running_max = pnl_series.expanding().max()
    drawdown = pnl_series - running_max
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # PnL with running max
    ax1.plot(pnl_series.index, pnl_series.values, label='PnL', linewidth=1.5)
    ax1.plot(running_max.index, running_max.values, label='Running Max', 
             linestyle='--', alpha=0.7)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title('PnL and Drawdown', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax2.set_xlabel('Event Index')
    ax2.set_ylabel('Drawdown ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Drawdown plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                                save_path: str = None):
    """Plot prediction calibration curve."""
    from sklearn.calibration import calibration_curve
    
    # For binary classification (up vs down, ignoring flat)
    binary_mask = y_true != 0
    y_binary = (y_true[binary_mask] == 1).astype(int)
    
    # Get probability of up class
    if y_proba.shape[1] == 3:
        # Assuming columns are [0=down, 1=flat, 2=up]
        prob_up = y_proba[binary_mask, 2]
    else:
        prob_up = y_proba[binary_mask, 1]
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary, prob_up, n_bins=10
    )
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label='Model', linewidth=2, markersize=8)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Calibration plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test plotting functions
    np.random.seed(42)
    
    # Synthetic data
    pnl = np.cumsum(np.random.normal(0.1, 1, 1000))
    results_df = pd.DataFrame({
        'event_idx': range(1000),
        'total_pnl': pnl,
        'position': np.random.randint(-100, 100, 1000)
    })
    
    plot_pnl_curve(results_df)
    plot_drawdown(results_df['total_pnl'])
