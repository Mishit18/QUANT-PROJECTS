import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_ic_series(ic_series: pd.Series, output_path: Path):
    """IC time series with rolling average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(ic_series.index, ic_series.values, alpha=0.3, color='steelblue', label='Daily IC')
    ax.plot(ic_series.index, ic_series.rolling(20).mean(), color='darkblue', linewidth=2, label='20D MA')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(ic_series.mean(), color='red', linestyle='--', linewidth=0.8, label=f'Mean: {ic_series.mean():.3f}')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Information Coefficient')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decay_curve(decay_series: pd.Series, output_path: Path):
    """Alpha decay over horizons."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [int(h.split('_')[1]) for h in decay_series.index]
    ax.plot(horizons, decay_series.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Information Coefficient')
    ax.set_title('Alpha Decay Curve')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_equity_curve(returns: pd.Series, output_path: Path):
    """Cumulative returns equity curve."""
    cum_returns = (1 + returns).cumprod()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(cum_returns.index, cum_returns.values, linewidth=2, color='darkblue')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Equity Curve')
    ax1.grid(alpha=0.3)
    
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_quantile_returns(quantile_df: pd.DataFrame, output_path: Path):
    """Average returns by prediction quantile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_returns = quantile_df.mean() * 252
    std_returns = quantile_df.std() / np.sqrt(len(quantile_df)) * 252
    
    x = range(len(mean_returns))
    ax.bar(x, mean_returns.values, yerr=std_returns.values, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(mean_returns.index)
    ax.set_xlabel('Prediction Quantile')
    ax.set_ylabel('Annualized Return')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_path: Path):
    """Alpha correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance: pd.Series, top_n: int = 20, output_path: Path = None):
    """Feature importance bar chart."""
    top_features = importance.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features.values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
