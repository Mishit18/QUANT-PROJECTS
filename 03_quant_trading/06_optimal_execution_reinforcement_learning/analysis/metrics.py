"""Performance metrics for execution strategies."""

import numpy as np
from typing import List, Dict


def execution_cost_metrics(costs: List[float]) -> Dict[str, float]:
    """
    Compute execution cost statistics.
    
    Args:
        costs: List of execution costs
    
    Returns:
        Dictionary of metrics
    """
    costs = np.array(costs)
    
    return {
        'mean': np.mean(costs),
        'median': np.median(costs),
        'std': np.std(costs),
        'min': np.min(costs),
        'max': np.max(costs),
        'q5': np.percentile(costs, 5),
        'q25': np.percentile(costs, 25),
        'q75': np.percentile(costs, 75),
        'q95': np.percentile(costs, 95),
        'iqr': np.percentile(costs, 75) - np.percentile(costs, 25),
        'cvar_95': np.mean(costs[costs >= np.percentile(costs, 95)])
    }


def sharpe_ratio(costs: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe-like ratio for execution.
    
    Higher is better (lower cost, lower variance).
    """
    costs = np.array(costs)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    
    if std_cost == 0:
        return 0.0
    
    return -(mean_cost - risk_free_rate) / std_cost


def tail_risk(costs: List[float], percentile: float = 95) -> float:
    """
    Compute tail risk: difference between percentile and mean.
    
    Measures exposure to extreme costs.
    """
    costs = np.array(costs)
    return np.percentile(costs, percentile) - np.mean(costs)


def implementation_shortfall(execution_prices: List[float], 
                             arrival_price: float,
                             quantities: List[float]) -> float:
    """
    Compute implementation shortfall.
    
    IS = Σ q_i * (P_i - P_0) / Σ q_i
    
    Args:
        execution_prices: Prices at which trades executed
        arrival_price: Initial mid-price
        quantities: Trade sizes
    
    Returns:
        Implementation shortfall (negative = worse execution)
    """
    execution_prices = np.array(execution_prices)
    quantities = np.array(quantities)
    
    total_quantity = np.sum(quantities)
    if total_quantity == 0:
        return 0.0
    
    weighted_price = np.sum(execution_prices * quantities) / total_quantity
    shortfall = (arrival_price - weighted_price) / arrival_price
    
    return shortfall


def compare_strategies(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print comparison table of strategies.
    
    Args:
        results: Dictionary mapping strategy names to their metrics
    """
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<15} {'Mean Cost':<12} {'Std Cost':<12} {'Sharpe':<10} {'Tail Risk':<12}")
    print("-"*80)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<15} "
              f"{metrics.get('mean_cost', 0):<12.2f} "
              f"{metrics.get('std_cost', 0):<12.2f} "
              f"{metrics.get('sharpe', 0):<10.4f} "
              f"{metrics.get('tail_risk', 0):<12.2f}")
    
    print("="*80)


def relative_performance(strategy_costs: List[float], 
                        baseline_costs: List[float]) -> Dict[str, float]:
    """
    Compute performance relative to baseline.
    
    Args:
        strategy_costs: Costs from strategy
        baseline_costs: Costs from baseline (e.g., TWAP)
    
    Returns:
        Relative performance metrics
    """
    strategy_costs = np.array(strategy_costs)
    baseline_costs = np.array(baseline_costs)
    
    cost_improvement = (np.mean(baseline_costs) - np.mean(strategy_costs)) / np.mean(baseline_costs)
    risk_improvement = (np.std(baseline_costs) - np.std(strategy_costs)) / np.std(baseline_costs)
    
    return {
        'cost_improvement': cost_improvement,
        'risk_improvement': risk_improvement,
        'win_rate': np.mean(strategy_costs < baseline_costs)
    }
