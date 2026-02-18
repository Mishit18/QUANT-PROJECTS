import pandas as pd
import numpy as np
from typing import List


def whites_reality_check(strategy_returns: pd.Series, benchmark_returns: List[pd.Series],
                        n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """White's Reality Check for data snooping."""
    
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    benchmark_sharpes = []
    for bench in benchmark_returns:
        bench_sharpe = bench.mean() / bench.std() * np.sqrt(252)
        benchmark_sharpes.append(bench_sharpe)
    
    max_benchmark_sharpe = max(benchmark_sharpes)
    
    outperformance = strategy_returns.values[:, np.newaxis] - np.column_stack([b.values for b in benchmark_returns])
    
    bootstrap_stats = []
    n_obs = len(strategy_returns)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_obs, n_obs, replace=True)
        boot_outperf = outperformance[indices]
        
        boot_mean = boot_outperf.mean(axis=0)
        boot_std = boot_outperf.std(axis=0)
        boot_sharpe = boot_mean / boot_std * np.sqrt(252)
        
        bootstrap_stats.append(boot_sharpe.max())
    
    bootstrap_stats = np.array(bootstrap_stats)
    critical_value = np.percentile(bootstrap_stats, confidence * 100)
    
    p_value = (bootstrap_stats >= (strategy_sharpe - max_benchmark_sharpe)).mean()
    
    return {
        'strategy_sharpe': strategy_sharpe,
        'max_benchmark_sharpe': max_benchmark_sharpe,
        'outperformance': strategy_sharpe - max_benchmark_sharpe,
        'critical_value': critical_value,
        'p_value': p_value,
        'significant': p_value < (1 - confidence)
    }


def bootstrap_sharpe_distribution(returns: pd.Series, n_bootstrap: int = 1000) -> np.ndarray:
    """Bootstrap distribution of Sharpe ratio."""
    sharpes = []
    n_obs = len(returns)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_obs, n_obs, replace=True)
        boot_returns = returns.iloc[indices]
        sharpe = boot_returns.mean() / boot_returns.std() * np.sqrt(252)
        sharpes.append(sharpe)
    
    return np.array(sharpes)


def compute_confidence_interval(returns: pd.Series, confidence: float = 0.95,
                               n_bootstrap: int = 1000) -> dict:
    """Bootstrap confidence interval for Sharpe."""
    sharpe_dist = bootstrap_sharpe_distribution(returns, n_bootstrap)
    
    lower = np.percentile(sharpe_dist, (1 - confidence) / 2 * 100)
    upper = np.percentile(sharpe_dist, (1 + confidence) / 2 * 100)
    
    return {
        'sharpe': returns.mean() / returns.std() * np.sqrt(252),
        'lower_bound': lower,
        'upper_bound': upper,
        'confidence': confidence
    }
