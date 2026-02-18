"""
Queue Position Experiment

Demonstrates structural relationship between queue position and fill probability.
No calibration - pure structural model.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.market.limit_order_book import SimpleLOB


def run_queue_analysis():
    """Analyze queue position effects on fill probability."""
    
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Running queue position analysis...")
    print("NOTE: This is a STRUCTURAL model, not calibrated")
    
    lob = SimpleLOB(tick_size=config['market']['tick_size'])
    
    # Simulate queue positions and fill probabilities
    queue_positions = np.arange(0, 50)
    base_intensity = config['order_flow']['arrival_rate']
    dt = config['simulation']['dt'] * 1000  # Scale up for visibility
    
    # Calculate fill probabilities
    fill_probs_at_best = []
    fill_probs_off_best = []
    
    for pos in queue_positions:
        # At best price
        prob_best = lob.queue_position_fill_probability(
            quote_price=100.0,
            best_price=100.0,
            queue_position=pos,
            base_intensity=base_intensity,
            dt=dt,
            is_bid=True
        )
        fill_probs_at_best.append(prob_best)
        
        # Off best price (1 tick away)
        prob_off = lob.queue_position_fill_probability(
            quote_price=99.99,
            best_price=100.0,
            queue_position=pos,
            base_intensity=base_intensity,
            dt=dt,
            is_bid=True
        )
        fill_probs_off_best.append(prob_off)
    
    # Plot results
    fig_dir = Path(config['output']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(queue_positions, fill_probs_at_best, 'o-', label='At Best Price', linewidth=2, markersize=4)
    ax.plot(queue_positions, fill_probs_off_best, 's-', label='Off Best (1 tick)', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Queue Position', fontsize=12)
    ax.set_ylabel('Fill Probability', fontsize=12)
    ax.set_title('Fill Probability vs Queue Position (Structural Model)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(fill_probs_at_best) * 1.1])
    plt.tight_layout()
    plt.savefig(fig_dir / 'queue_position_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nKey findings:")
    print(f"  - Front of queue (pos=0) has highest fill probability: {fill_probs_at_best[0]:.3f}")
    print(f"  - Fill probability decays exponentially with queue position")
    print(f"  - Off-best quotes have much lower fill probability: {fill_probs_off_best[0]:.3f}")
    print(f"  - This is STRUCTURAL, not calibrated to data")
    print(f"\nFigure saved to {fig_dir}/queue_position_analysis.png")


if __name__ == '__main__':
    run_queue_analysis()
