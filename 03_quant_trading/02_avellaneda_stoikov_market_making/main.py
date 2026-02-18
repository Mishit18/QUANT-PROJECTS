"""
Avellaneda-Stoikov Market Making Research System

Clean entry point for all experiments.
Run with: python main.py [experiment_name]

Available experiments:
- baseline: Single-agent with mid-price
- microprice: Single-agent with microprice comparison
- queue: Queue position analysis
- pnl_decomposition: Detailed PnL attribution
- regime_sweep: Volatility and arrival rate sweeps
- competition: Multi-agent zero-profit equilibrium
- all: Run all experiments
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.baseline_experiment import run_baseline
from experiments.microprice_experiment import run_microprice_comparison
from experiments.queue_experiment import run_queue_analysis
from experiments.pnl_decomposition_experiment import run_pnl_decomposition
from experiments.regime_sweep_experiment import run_regime_sweeps
from experiments.competition_experiment import run_competition_analysis


EXPERIMENTS = {
    'baseline': ('Baseline single-agent market making', run_baseline),
    'microprice': ('Mid-price vs microprice comparison', run_microprice_comparison),
    'queue': ('Queue position and fill probability', run_queue_analysis),
    'pnl_decomposition': ('Detailed PnL attribution', run_pnl_decomposition),
    'regime_sweep': ('Volatility and arrival rate sweeps', run_regime_sweeps),
    'competition': ('Multi-agent competition analysis', run_competition_analysis),
}


def main():
    parser = argparse.ArgumentParser(
        description='Avellaneda-Stoikov Market Making Research System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py baseline              # Run baseline experiment
  python main.py microprice            # Compare mid vs microprice
  python main.py all                   # Run all experiments
  
Available experiments:
""" + '\n'.join(f"  {name:20s} - {desc}" for name, (desc, _) in EXPERIMENTS.items())
    )
    
    parser.add_argument(
        'experiment',
        nargs='?',
        default='baseline',
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help='Experiment to run (default: baseline)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("AVELLANEDA-STOIKOV MARKET MAKING RESEARCH SYSTEM")
    print("="*70)
    print()
    
    if args.experiment == 'all':
        print("Running all experiments...\n")
        for name, (desc, func) in EXPERIMENTS.items():
            print(f"\n{'='*70}")
            print(f"EXPERIMENT: {name.upper()}")
            print(f"Description: {desc}")
            print(f"{'='*70}\n")
            try:
                func()
                print(f"\n✓ {name} completed successfully")
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        desc, func = EXPERIMENTS[args.experiment]
        print(f"Running: {desc}\n")
        func()
        print(f"\n✓ Experiment completed successfully")
    
    print("\n" + "="*70)
    print("Results saved to:")
    print("  - Figures: results/figures/")
    print("  - Tables: results/tables/")
    print("="*70)


if __name__ == '__main__':
    main()
