import pandas as pd
import numpy as np
from typing import List
from scipy import stats


def superior_predictive_ability(strategy_returns: pd.Series, benchmark_returns: List[pd.Series],
                                n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Hansen's Superior Predictive Ability test."""
    
    n_benchmarks = len(benchmark_returns)
    n_obs = len(strategy_returns)
    
    performance_diffs = np.zeros((n_obs, n_benchmarks))
    for i, bench in enumerate(benchmark_returns):
        performance_diffs[:, i] = strategy_returns.values - bench.values
    
    mean_diffs = performance_diffs.mean(axis=0)
    std_diffs = performance_diffs.std(axis=0)
    
    t_stats = mean_diffs / (std_diffs / np.sqrt(n_obs))
    max_t_stat = t_stats.max()
    
    bootstrap_max_stats = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_obs, n_obs, replace=True)
        boot_diffs = performance_diffs[indices]
        
        boot_mean = boot_diffs.mean(axis=0) - mean_diffs
        boot_std = boot_diffs.std(axis=0)
        
        boot_t = boot_mean / (boot_std / np.sqrt(n_obs))
        bootstrap_max_stats.append(boot_t.max())
    
    bootstrap_max_stats = np.array(bootstrap_max_stats)
    p_value = (bootstrap_max_stats >= max_t_stat).mean()
    
    critical_value = np.percentile(bootstrap_max_stats, confidence * 100)
    
    return {
        'max_t_statistic': max_t_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'significant': p_value < (1 - confidence),
        'n_benchmarks': n_benchmarks
    }


def stepwise_spa(strategy_returns: pd.Series, benchmark_returns: List[pd.Series],
                n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Stepwise SPA with elimination of dominated benchmarks."""
    
    active_benchmarks = list(range(len(benchmark_returns)))
    eliminated = []
    
    while len(active_benchmarks) > 0:
        active_bench_returns = [benchmark_returns[i] for i in active_benchmarks]
        
        result = superior_predictive_ability(strategy_returns, active_bench_returns, 
                                            n_bootstrap, confidence)
        
        if result['significant']:
            break
        
        performance_diffs = np.array([
            (strategy_returns - bench).mean() for bench in active_bench_returns
        ])
        
        worst_idx = performance_diffs.argmin()
        eliminated.append(active_benchmarks[worst_idx])
        active_benchmarks.pop(worst_idx)
    
    return {
        'significant': len(active_benchmarks) == 0,
        'remaining_benchmarks': len(active_benchmarks),
        'eliminated_benchmarks': len(eliminated),
        'final_p_value': result['p_value'] if active_benchmarks else 1.0
    }
