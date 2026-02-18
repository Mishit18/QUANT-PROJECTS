"""
PnL Decomposition Experiment

Explicit self-financing accounting-based decomposition of total PnL into:
1. Spread capture
2. Inventory PnL
3. Adverse selection cost

Enforces: Total PnL = Cash + Inventory × MidPrice - Initial_Wealth
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.models.avellaneda_stoikov import AvellanedaStoikov
from src.models.intensity_models import ExponentialIntensity
from src.market.market_environment import MarketEnvironment
from src.agents.market_maker import MarketMaker
from src.simulation.single_agent_simulator import SingleAgentSimulator
from src.analysis.pnl_attribution import Transaction, PnLAttribution


def run_pnl_decomposition():
    """Run PnL decomposition analysis with self-financing verification."""
    
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Running PnL decomposition with self-financing verification...")
    
    # Run simulation
    intensity_model = ExponentialIntensity(
        arrival_rate=config['order_flow']['arrival_rate'],
        decay_rate=config['order_flow']['intensity_decay']
    )
    
    as_model = AvellanedaStoikov(
        risk_aversion=config['market_maker']['risk_aversion'],
        volatility=config['market']['volatility'],
        terminal_time=config['simulation']['T'],
        intensity_decay=config['order_flow']['intensity_decay']
    )
    
    market_env = MarketEnvironment(
        initial_price=config['market']['initial_price'],
        volatility=config['market']['volatility'],
        dt=config['simulation']['dt'],
        intensity_model=intensity_model,
        adverse_selection_coef=config['adverse_selection']['impact_coefficient'],
        random_seed=config['simulation']['random_seed']
    )
    
    agent = MarketMaker(model=as_model, initial_inventory=0, initial_cash=0.0)
    simulator = SingleAgentSimulator(market_env, agent, config['simulation']['T'])
    results = simulator.run(verbose=False)
    
    # Convert transactions to Transaction objects
    transactions = []
    for txn in agent.transactions:
        transactions.append(Transaction(
            time=txn['time'],
            side=txn['side'],
            price=txn['price'],
            size=txn['size'],
            mid_price=txn['mid_price']
        ))
    
    # Decompose PnL with self-financing verification
    history = results['history']
    
    # Ensure histories are aligned
    min_len = min(len(history['cash']), len(history['inventory']), len(history['mid_price']))
    
    decomposition = PnLAttribution.decompose(
        transactions=transactions,
        cash_history=history['cash'][:min_len],
        inventory_history=history['inventory'][:min_len],
        mid_price_history=history['mid_price'][:min_len],
        time_history=history['time'][:min_len],
        initial_cash=0.0,
        initial_inventory=0
    )
    
    # Print results
    print("\n" + "="*70)
    print("PNL DECOMPOSITION (Self-Financing Accounting)")
    print("="*70)
    print(f"\nSelf-Financing Valid: {decomposition['self_financing_valid']}")
    print(f"Max Accounting Error: {decomposition['max_accounting_error']:.2e}")
    print(f"\nTotal PnL:              ${decomposition['total_pnl']:>10.2f}")
    print(f"  Spread Capture:       ${decomposition['spread_capture']:>10.2f}  ({decomposition['spread_pct']:>5.1f}%)")
    print(f"  Inventory PnL:        ${decomposition['inventory_pnl']:>10.2f}  ({decomposition['inventory_pct']:>5.1f}%)")
    print(f"  Adverse Selection:   -${decomposition['adverse_selection_cost']:>10.2f}  ({-decomposition['adverse_sel_pct']:>5.1f}%)")
    print(f"  Residual:             ${decomposition['residual']:>10.2f}")
    
    # Verify sum
    explained = (decomposition['spread_capture'] + 
                decomposition['inventory_pnl'] - 
                decomposition['adverse_selection_cost'])
    print(f"\nVerification:")
    print(f"  Sum of components:    ${explained:>10.2f}")
    print(f"  Total PnL:            ${decomposition['total_pnl']:>10.2f}")
    print(f"  Difference:           ${abs(decomposition['total_pnl'] - explained):>10.2f}")
    
    # Plot decomposition
    fig_dir = Path(config['output']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ['Spread\nCapture', 'Inventory\nPnL', 'Adverse\nSelection', 'Total\nPnL']
    values = [
        decomposition['spread_capture'],
        decomposition['inventory_pnl'],
        -decomposition['adverse_selection_cost'],
        decomposition['total_pnl']
    ]
    colors = ['green', 'blue', 'red', 'purple']
    
    bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('PnL ($)', fontsize=12)
    ax.set_title('PnL Decomposition (Self-Financing Accounting)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'pnl_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save table
    table_dir = Path(config['output']['tables_dir'])
    table_dir.mkdir(parents=True, exist_ok=True)
    
    with open(table_dir / 'pnl_decomposition.txt', 'w', encoding='utf-8') as f:
        f.write("PNL DECOMPOSITION (SELF-FINANCING ACCOUNTING)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Self-Financing Valid: {decomposition['self_financing_valid']}\n")
        f.write(f"Max Accounting Error: {decomposition['max_accounting_error']:.2e}\n\n")
        f.write(f"{'Component':<30} {'Value ($)':>15} {'Percentage':>12}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Spread Capture':<30} {decomposition['spread_capture']:>15.2f} {decomposition['spread_pct']:>11.1f}%\n")
        f.write(f"{'Inventory PnL':<30} {decomposition['inventory_pnl']:>15.2f} {decomposition['inventory_pct']:>11.1f}%\n")
        f.write(f"{'Adverse Selection Cost':<30} {-decomposition['adverse_selection_cost']:>15.2f} {-decomposition['adverse_sel_pct']:>11.1f}%\n")
        f.write(f"{'Residual':<30} {decomposition['residual']:>15.2f}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Total PnL':<30} {decomposition['total_pnl']:>15.2f} {'100.0%':>12}\n")
        f.write("\n")
        f.write("SELF-FINANCING CONSTRAINT:\n")
        f.write("  Total PnL = Cash + Inventory × MidPrice - Initial_Wealth\n")
        f.write("\nINTERPRETATION:\n")
        f.write("- Spread capture: Profit from bid-ask spread\n")
        f.write("- Inventory PnL: Profit from holding inventory during price moves\n")
        f.write("- Adverse selection: Cost from informed trading\n")
        f.write("- Components sum to total PnL (residual should be small)\n")
    
    print(f"\nFigure saved to {fig_dir}/pnl_decomposition.png")
    print(f"Table saved to {table_dir}/pnl_decomposition.txt")
    
    return decomposition


if __name__ == '__main__':
    run_pnl_decomposition()
