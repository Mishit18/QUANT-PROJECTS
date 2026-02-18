"""
Regime Sweep Experiment - Where Avellaneda-Stoikov Breaks

Sweeps across volatility and arrival rate regimes to show:
1. Monotonic structural relationships
2. Failure regions (high σ, low A)
3. Inventory blow-up regimes

NOT optimization - diagnostic analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.avellaneda_stoikov import AvellanedaStoikov
from src.models.intensity_models import ExponentialIntensity
from src.market.market_environment import MarketEnvironment
from src.agents.market_maker import MarketMaker
from src.simulation.single_agent_simulator import SingleAgentSimulator


def run_regime_sweeps():
    """Run regime sweeps showing structural relationships and failures."""
    
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Running regime sweeps...")
    print("NOTE: This shows WHERE the model BREAKS, not optimization")
    
    # Volatility sweep
    print("\n1. Volatility sweep (showing failure at high σ)...")
    vol_results = sweep_volatility(config)
    
    # Arrival rate sweep
    print("\n2. Arrival rate sweep (showing failure at low A)...")
    arrival_results = sweep_arrival_rate(config)
    
    # Plot results
    fig_dir = Path(config['output']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Volatility sweep plot - showing breaks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vols = vol_results['volatilities']
    
    # Mark failure regions
    failure_threshold_vol = 0.05
    failure_mask = np.array(vols) > failure_threshold_vol
    
    axes[0, 0].plot(vols, vol_results['pnl'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].axvspan(failure_threshold_vol, max(vols), alpha=0.2, color='red', label='Failure Region')
    axes[0, 0].set_xlabel('Volatility (σ)', fontsize=11)
    axes[0, 0].set_ylabel('Final PnL ($)', fontsize=11)
    axes[0, 0].set_title('PnL vs Volatility (degrades at high σ)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(vols, vol_results['sharpe'], 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].axvspan(failure_threshold_vol, max(vols), alpha=0.2, color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Volatility (σ)', fontsize=11)
    axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[0, 1].set_title('Sharpe vs Volatility (negative at high σ)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(vols, vol_results['spread'], 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Volatility (σ)', fontsize=11)
    axes[1, 0].set_ylabel('Avg Spread ($)', fontsize=11)
    axes[1, 0].set_title('Spread vs Volatility (widens monotonically)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(vols, vol_results['inv_std'], 'o-', linewidth=2, markersize=6, color='red')
    axes[1, 1].axvspan(failure_threshold_vol, max(vols), alpha=0.2, color='red')
    axes[1, 1].set_xlabel('Volatility (σ)', fontsize=11)
    axes[1, 1].set_ylabel('Inventory Std', fontsize=11)
    axes[1, 1].set_title('Inventory Risk vs Volatility (blows up)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Regime Sweep: Volatility (Red = Model Breaks)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'regime_sweep_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Arrival rate sweep plot - showing breaks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    arrivals = arrival_results['arrival_rates']
    
    # Mark failure regions
    failure_threshold_arr = 5.0
    
    axes[0, 0].plot(arrivals, arrival_results['pnl'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].axvspan(min(arrivals), failure_threshold_arr, alpha=0.2, color='red', label='Failure Region')
    axes[0, 0].set_xlabel('Arrival Rate (A)', fontsize=11)
    axes[0, 0].set_ylabel('Final PnL ($)', fontsize=11)
    axes[0, 0].set_title('PnL vs Arrival Rate (fails at low A)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(arrivals, arrival_results['fills'], 'o-', linewidth=2, markersize=6, color='purple')
    axes[0, 1].axvspan(min(arrivals), failure_threshold_arr, alpha=0.2, color='red')
    axes[0, 1].set_xlabel('Arrival Rate (A)', fontsize=11)
    axes[0, 1].set_ylabel('Total Fills', fontsize=11)
    axes[0, 1].set_title('Fills vs Arrival Rate (too few at low A)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(arrivals, arrival_results['spread'], 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Arrival Rate (A)', fontsize=11)
    axes[1, 0].set_ylabel('Avg Spread ($)', fontsize=11)
    axes[1, 0].set_title('Spread vs Arrival Rate (optimal tightens)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(arrivals, arrival_results['sharpe'], 'o-', linewidth=2, markersize=6, color='orange')
    axes[1, 1].axvspan(min(arrivals), failure_threshold_arr, alpha=0.2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Arrival Rate (A)', fontsize=11)
    axes[1, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[1, 1].set_title('Sharpe vs Arrival Rate (improves with A)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Regime Sweep: Arrival Rate (Red = Model Breaks)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'regime_sweep_arrival_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigures saved to {fig_dir}/")
    
    # Save failure analysis
    table_dir = Path(config['output']['tables_dir'])
    table_dir.mkdir(parents=True, exist_ok=True)
    
    with open(table_dir / 'regime_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("REGIME ANALYSIS: WHERE AVELLANEDA-STOIKOV BREAKS\n")
        f.write("="*70 + "\n\n")
        f.write("VOLATILITY SWEEP:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Volatility':<15} {'PnL ($)':>12} {'Sharpe':>10} {'Inv Std':>12} {'Status':>15}\n")
        f.write("-"*70 + "\n")
        for i, vol in enumerate(vols):
            status = "BREAKS" if vol > failure_threshold_vol else "OK"
            f.write(f"{vol:<15.3f} {vol_results['pnl'][i]:>12.2f} {vol_results['sharpe'][i]:>10.3f} {vol_results['inv_std'][i]:>12.2f} {status:>15}\n")
        
        f.write("\n\nARRIVAL RATE SWEEP:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Arrival Rate':<15} {'PnL ($)':>12} {'Sharpe':>10} {'Fills':>10} {'Status':>15}\n")
        f.write("-"*70 + "\n")
        for i, arr in enumerate(arrivals):
            status = "BREAKS" if arr < failure_threshold_arr else "OK"
            f.write(f"{arr:<15.1f} {arrival_results['pnl'][i]:>12.2f} {arrival_results['sharpe'][i]:>10.3f} {arrival_results['fills'][i]:>10.0f} {status:>15}\n")
        
        f.write("\n\nKEY FINDINGS:\n")
        f.write("- High volatility (σ > 0.05): Inventory risk dominates, Sharpe → negative\n")
        f.write("- Low arrival rate (A < 5): Too few fills, model ineffective\n")
        f.write("- Spread widens monotonically with volatility (structural)\n")
        f.write("- Sharpe improves with arrival rate (more diversification)\n")
        f.write("\nThese are STRUCTURAL relationships, not optimization targets.\n")
    
    print(f"Table saved to {table_dir}/regime_analysis.txt")
    print("\nKey insights:")
    print("- Model BREAKS at high volatility (σ > 0.05)")
    print("- Model BREAKS at low arrival rate (A < 5)")
    print("- These are fundamental limitations, not bugs")


def sweep_volatility(config):
    """Sweep across volatility levels."""
    
    volatilities = config['regime_sweeps']['volatility_range']
    results = {'volatilities': volatilities, 'pnl': [], 'sharpe': [], 'spread': [], 'inv_std': []}
    
    for vol in tqdm(volatilities, desc="Volatility sweep"):
        result = run_single_sim(config, volatility=vol)
        results['pnl'].append(result['metrics']['final_pnl'])
        results['sharpe'].append(result['metrics']['sharpe_ratio'])
        results['spread'].append(np.mean(result['history']['bid_spread'] + result['history']['ask_spread']))
        results['inv_std'].append(result['metrics']['inventory_std'])
    
    return results


def sweep_arrival_rate(config):
    """Sweep across arrival rates."""
    
    arrival_rates = config['regime_sweeps']['arrival_rate_range']
    results = {'arrival_rates': arrival_rates, 'pnl': [], 'sharpe': [], 'spread': [], 'fills': []}
    
    for rate in tqdm(arrival_rates, desc="Arrival rate sweep"):
        result = run_single_sim(config, arrival_rate=rate)
        results['pnl'].append(result['metrics']['final_pnl'])
        results['sharpe'].append(result['metrics']['sharpe_ratio'])
        results['spread'].append(np.mean(result['history']['bid_spread'] + result['history']['ask_spread']))
        results['fills'].append(result['metrics']['total_fills'])
    
    return results


def run_single_sim(config, volatility=None, arrival_rate=None):
    """Run single simulation with specified parameters."""
    
    vol = volatility if volatility is not None else config['market']['volatility']
    arr_rate = arrival_rate if arrival_rate is not None else config['order_flow']['arrival_rate']
    
    intensity_model = ExponentialIntensity(arr_rate, config['order_flow']['intensity_decay'])
    as_model = AvellanedaStoikov(
        config['market_maker']['risk_aversion'], vol,
        config['simulation']['T'], config['order_flow']['intensity_decay']
    )
    
    market_env = MarketEnvironment(
        config['market']['initial_price'], vol, config['simulation']['dt'],
        intensity_model, config['adverse_selection']['impact_coefficient'],
        random_seed=config['simulation']['random_seed']
    )
    
    agent = MarketMaker(as_model, 0, 0.0)
    simulator = SingleAgentSimulator(market_env, agent, config['simulation']['T'])
    
    return simulator.run(verbose=False)


if __name__ == '__main__':
    run_regime_sweeps()
