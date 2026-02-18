"""
Plotting utilities for execution diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def plot_execution_comparison(metrics_old: Dict, metrics_new: Dict, 
                              exec_diagnostics: Dict, save_path: str = None):
    """
    Create comprehensive comparison plot of execution improvements.
    
    Args:
        metrics_old: Original execution metrics
        metrics_new: Improved execution metrics
        exec_diagnostics: Execution diagnostics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Execution Improvement Analysis', fontsize=16, fontweight='bold')
    
    # 1. PnL Comparison
    ax = axes[0, 0]
    pnl_data = [metrics_old['total_pnl'], metrics_new['total_pnl']]
    colors = ['#e74c3c', '#27ae60']
    bars = ax.bar(['Original\n(Market Orders)', 'Improved\n(Limit Orders)'], 
                   pnl_data, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Total PnL ($)', fontsize=11, fontweight='bold')
    ax.set_title('PnL Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 2. Sharpe Ratio Comparison
    ax = axes[0, 1]
    sharpe_data = [metrics_old['sharpe_ratio'], metrics_new['sharpe_ratio']]
    bars = ax.bar(['Original', 'Improved'], sharpe_data, 
                   color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 3. Number of Trades
    ax = axes[0, 2]
    trades_data = [metrics_old['num_trades'], metrics_new['num_trades']]
    bars = ax.bar(['Original', 'Improved'], trades_data, 
                   color=['#3498db', '#9b59b6'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Trades', fontsize=11, fontweight='bold')
    ax.set_title('Trading Frequency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 4. Signal Filtering Breakdown
    ax = axes[1, 0]
    filter_data = [
        exec_diagnostics['signals_filtered_confidence'],
        exec_diagnostics['signals_filtered_volatility'],
        exec_diagnostics['signals_executed']
    ]
    labels = ['Filtered\n(Confidence)', 'Filtered\n(Volatility)', 'Executed']
    colors_filter = ['#e67e22', '#e74c3c', '#27ae60']
    bars = ax.bar(labels, filter_data, color=colors_filter, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Signals', fontsize=11, fontweight='bold')
    ax.set_title('Signal Filtering Breakdown', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # 5. Limit Order Fill Rate
    ax = axes[1, 1]
    fill_rate = exec_diagnostics['fill_rate'] * 100
    cancel_rate = (1 - exec_diagnostics['fill_rate']) * 100
    
    wedges, texts, autotexts = ax.pie(
        [fill_rate, cancel_rate],
        labels=['Filled', 'Cancelled'],
        autopct='%1.1f%%',
        colors=['#27ae60', '#95a5a6'],
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax.set_title('Limit Order Fill Rate', fontsize=12, fontweight='bold')
    
    # 6. Key Metrics Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    EXECUTION IMPROVEMENTS
    
    PnL Improvement:
    ${metrics_new['total_pnl'] - metrics_old['total_pnl']:.2f}
    ({(metrics_new['total_pnl'] - metrics_old['total_pnl'])/abs(metrics_old['total_pnl'])*100:.1f}%)
    
    Sharpe Improvement:
    {metrics_new['sharpe_ratio'] - metrics_old['sharpe_ratio']:.2f}
    
    Trade Reduction:
    {int(metrics_old['num_trades'] - metrics_new['num_trades'])} trades
    ({(metrics_old['num_trades'] - metrics_new['num_trades'])/metrics_old['num_trades']*100:.1f}%)
    
    Filter Rate: {exec_diagnostics['filter_rate']*100:.1f}%
    Fill Rate: {exec_diagnostics['fill_rate']*100:.1f}%
    Win Rate: {exec_diagnostics['win_rate']*100:.1f}%
    """
    
    ax.text(0.1, 0.5, summary_text, 
            fontsize=11, 
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Execution comparison plot saved to {save_path}")
    
    plt.close()


def plot_confidence_distribution(confidence_scores: List[float], save_path: str = None):
    """
    Plot distribution of confidence scores with threshold.
    
    Args:
        confidence_scores: List of confidence scores
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidence_scores = np.array(confidence_scores)
    
    # Histogram
    ax.hist(confidence_scores, bins=50, alpha=0.7, color='steelblue', 
            edgecolor='black', label='All Signals')
    
    # Mark threshold
    threshold = 0.20
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold: ±{threshold}')
    ax.axvline(x=-threshold, color='red', linestyle='--', linewidth=2)
    
    # Shade filtered regions
    ax.axvspan(-threshold, threshold, alpha=0.2, color='red', 
               label='Filtered Region')
    
    ax.set_xlabel('Confidence Score (P(up) - P(down))', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Score Distribution with Filtering Threshold', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add statistics
    stats_text = f"""
    Mean: {confidence_scores.mean():.4f}
    Std: {confidence_scores.std():.4f}
    Filtered: {(np.abs(confidence_scores) < threshold).sum()} ({(np.abs(confidence_scores) < threshold).mean()*100:.1f}%)
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test plotting functions
    metrics_old = {
        'total_pnl': -58.47,
        'sharpe_ratio': -105.018,
        'num_trades': 116
    }
    
    metrics_new = {
        'total_pnl': -3.30,
        'sharpe_ratio': -2.166,
        'num_trades': 53
    }
    
    exec_diagnostics = {
        'signals_generated': 17280,
        'signals_filtered_confidence': 14610,
        'signals_filtered_volatility': 2410,
        'signals_executed': 53,
        'filter_rate': 0.985,
        'fill_rate': 0.29,
        'win_rate': 0.0
    }
    
    plot_execution_comparison(metrics_old, metrics_new, exec_diagnostics,
                             'reports/figures/execution_comparison.png')
