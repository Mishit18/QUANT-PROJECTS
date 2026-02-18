"""
PnL attribution and execution diagnostic plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def plot_pnl_attribution(attribution: Dict, save_path: str = None):
    """
    Plot detailed PnL attribution breakdown.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PnL Attribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. PnL Components Waterfall
    ax = axes[0, 0]
    components = [
        ('Directional\nPnL', attribution['directional_pnl']),
        ('Spread\nCapture', attribution['spread_capture']),
        ('Transaction\nCosts', -attribution['transaction_costs']),
        ('Maker\nRebates', attribution['maker_rebates']),
        ('Adverse\nSelection', -attribution['adverse_selection'])
    ]
    
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('PnL ($)', fontsize=11, fontweight='bold')
    ax.set_title('PnL Components', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # 2. PnL by Confidence Bucket
    ax = axes[0, 1]
    if attribution['pnl_by_confidence']:
        confidences, pnls = zip(*attribution['pnl_by_confidence'])
        
        # Create confidence buckets
        conf_bins = np.linspace(-1, 1, 11)
        pnl_by_bucket = []
        bucket_labels = []
        
        for i in range(len(conf_bins)-1):
            mask = [(c >= conf_bins[i] and c < conf_bins[i+1]) 
                   for c in confidences]
            if any(mask):
                bucket_pnl = np.mean([p for p, m in zip(pnls, mask) if m])
                pnl_by_bucket.append(bucket_pnl)
                bucket_labels.append(f'{conf_bins[i]:.1f}')
        
        colors_bucket = ['#27ae60' if p > 0 else '#e74c3c' for p in pnl_by_bucket]
        ax.bar(range(len(pnl_by_bucket)), pnl_by_bucket, 
               color=colors_bucket, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Confidence Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg PnL ($)', fontsize=11, fontweight='bold')
        ax.set_title('PnL by Confidence Level', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(bucket_labels)))
        ax.set_xticklabels(bucket_labels, rotation=45)
        ax.grid(axis='y', alpha=0.3)

    
    # 3. PnL by Spread Regime
    ax = axes[1, 0]
    if attribution['pnl_by_spread']:
        spreads, pnls = zip(*attribution['pnl_by_spread'])
        
        # Create spread buckets
        spread_bins = np.percentile(spreads, [0, 25, 50, 75, 100])
        pnl_by_spread_bucket = []
        spread_labels = ['Tight', 'Normal', 'Wide', 'Very Wide']
        
        for i in range(len(spread_bins)-1):
            mask = [(s >= spread_bins[i] and s < spread_bins[i+1]) 
                   for s in spreads]
            if any(mask):
                bucket_pnl = np.mean([p for p, m in zip(pnls, mask) if m])
                pnl_by_spread_bucket.append(bucket_pnl)
        
        colors_spread = ['#27ae60' if p > 0 else '#e74c3c' for p in pnl_by_spread_bucket]
        ax.bar(spread_labels[:len(pnl_by_spread_bucket)], pnl_by_spread_bucket,
               color=colors_spread, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Spread Regime', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg PnL ($)', fontsize=11, fontweight='bold')
        ax.set_title('PnL by Spread Regime', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Trade Outcomes Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    net_pnl = (attribution['directional_pnl'] + 
              attribution['spread_capture'] -
              attribution['transaction_costs'] +
              attribution['maker_rebates'] -
              attribution['adverse_selection'])
    
    summary_text = f"""
    ATTRIBUTION SUMMARY
    
    Directional PnL:    ${attribution['directional_pnl']:.2f}
    Spread Capture:     ${attribution['spread_capture']:.2f}
    Transaction Costs: -${attribution['transaction_costs']:.2f}
    Maker Rebates:     +${attribution['maker_rebates']:.2f}
    Adverse Selection: -${attribution['adverse_selection']:.2f}
    ─────────────────────────────
    Net PnL:            ${net_pnl:.2f}
    
    Trade Outcomes:
    Filled:             {attribution['filled_trades']}
    Cancelled:          {attribution['cancelled_orders']}
    Missed (neg EV):    {attribution['missed_opportunities']}
    """
    
    ax.text(0.1, 0.5, summary_text,
            fontsize=11,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PnL attribution plot saved to {save_path}")
    
    plt.close()



def plot_ev_analysis(diagnostics: Dict, save_path: str = None):
    """
    Plot Expected Value analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Expected Value Analysis', fontsize=16, fontweight='bold')
    
    # 1. EV Distribution
    ax = axes[0]
    ev_scores = diagnostics.get('ev_scores', [])
    
    if ev_scores:
        ax.hist(ev_scores, bins=50, alpha=0.7, color='steelblue',
                edgecolor='black', label='All Signals')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                   label='EV = 0 (Filter Threshold)')
        ax.axvspan(-np.inf, 0, alpha=0.2, color='red', label='Filtered (EV ≤ 0)')
        
        ax.set_xlabel('Expected Value ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('EV Distribution with Filtering', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Add statistics
        ev_array = np.array(ev_scores)
        stats_text = f"""
        Mean EV: ${ev_array.mean():.6f}
        Positive EV: {(ev_array > 0).sum()} ({(ev_array > 0).mean()*100:.1f}%)
        Negative EV: {(ev_array <= 0).sum()} ({(ev_array <= 0).mean()*100:.1f}%)
        """
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Signal Filtering Breakdown
    ax = axes[1]
    filter_data = [
        diagnostics.get('signals_filtered_ev', 0),
        diagnostics.get('signals_filtered_spread', 0),
        diagnostics.get('signals_executed', 0)
    ]
    labels = ['Filtered\n(Negative EV)', 'Filtered\n(Wide Spread)', 'Executed']
    colors_filter = ['#e74c3c', '#e67e22', '#27ae60']
    
    wedges, texts, autotexts = ax.pie(
        filter_data,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors_filter,
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax.set_title('Signal Filtering Breakdown', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"EV analysis plot saved to {save_path}")
    
    plt.close()



def plot_order_management(diagnostics: Dict, save_path: str = None):
    """
    Plot queue-aware order management analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Queue-Aware Order Management', fontsize=16, fontweight='bold')
    
    # 1. Order Cancellation Reasons
    ax = axes[0]
    cancel_data = [
        diagnostics.get('limit_orders_cancelled_ofi', 0),
        diagnostics.get('limit_orders_cancelled_queue', 0),
        diagnostics.get('limit_orders_cancelled_spread', 0),
        diagnostics.get('limit_orders_cancelled_timeout', 0)
    ]
    labels = ['OFI Flip', 'Queue\nDeterioration', 'Spread\nWidening', 'Timeout']
    colors_cancel = ['#e74c3c', '#e67e22', '#f39c12', '#95a5a6']
    
    bars = ax.bar(labels, cancel_data, color=colors_cancel, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Cancellations', fontsize=11, fontweight='bold')
    ax.set_title('Order Cancellation Reasons', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    # 2. Fill vs Cancel Outcomes
    ax = axes[1]
    filled = diagnostics.get('limit_orders_filled', 0)
    cancelled = sum(cancel_data)
    
    outcome_data = [filled, cancelled]
    outcome_labels = ['Filled', 'Cancelled']
    outcome_colors = ['#27ae60', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(
        outcome_data,
        labels=outcome_labels,
        autopct='%1.1f%%',
        colors=outcome_colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax.set_title('Order Outcomes', fontsize=12, fontweight='bold')
    
    # Add summary text
    fill_rate = filled / max(filled + cancelled, 1) * 100
    summary = f"Fill Rate: {fill_rate:.1f}%\nTotal Orders: {filled + cancelled}"
    ax.text(0, -1.3, summary, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Order management plot saved to {save_path}")
    
    plt.close()
