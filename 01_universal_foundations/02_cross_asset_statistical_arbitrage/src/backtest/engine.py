import pandas as pd
import numpy as np
from typing import Dict, Optional
from .portfolio import Portfolio
from .costs import compute_total_costs


class BacktestEngine:
    """Walk-forward backtest with transaction costs."""
    
    def __init__(self, config: dict):
        self.config = config
        self.portfolio = Portfolio(
            leverage=config['backtest']['leverage'],
            long_short=config['backtest']['long_short']
        )
        self.tcost_bps = config['backtest']['tcost_bps']
        self.slippage_bps = config['backtest']['slippage_bps']
        self.rebalance_freq = config['backtest']['rebalance_freq']
        
        self.weights_history = []
        self.returns_history = []
        self.costs_history = []
        
    def run(self, predictions: pd.DataFrame, returns: pd.DataFrame, 
            volumes: pd.DataFrame) -> Dict[str, pd.Series]:
        """Execute backtest."""
        if len(predictions) == 0 or predictions.empty:
            return {
                'returns': pd.Series(dtype=float),
                'weights': pd.DataFrame(),
                'costs': pd.Series(dtype=float)
            }
        
        dates = predictions.index.intersection(returns.index)
        
        if len(dates) == 0:
            return {
                'returns': pd.Series(dtype=float),
                'weights': pd.DataFrame(),
                'costs': pd.Series(dtype=float)
            }
        
        prev_weights = pd.Series(0.0, index=predictions.columns)
        equity_curve = []
        
        for i, date in enumerate(dates):
            if i % self.rebalance_freq != 0 and i > 0:
                continue
            
            alpha = predictions.loc[date]
            target_weights = self.portfolio.construct_weights(alpha, method='rank')
            
            if i > 0:
                costs = compute_total_costs(
                    prev_weights, target_weights, 
                    volumes.loc[date] if date in volumes.index else pd.Series(),
                    self.tcost_bps, self.slippage_bps
                )
                self.costs_history.append((date, costs))
            else:
                costs = 0.0
            
            forward_ret = returns.loc[date] if date in returns.index else pd.Series(0.0, index=target_weights.index)
            
            common_idx = target_weights.index.intersection(forward_ret.index)
            portfolio_ret = (target_weights[common_idx] * forward_ret[common_idx]).sum() - costs
            
            self.weights_history.append((date, target_weights))
            self.returns_history.append((date, portfolio_ret))
            
            prev_weights = target_weights
            equity_curve.append((date, portfolio_ret))
        
        return {
            'returns': pd.Series(dict(self.returns_history)),
            'weights': pd.DataFrame([w for _, w in self.weights_history], 
                                   index=[d for d, _ in self.weights_history]),
            'costs': pd.Series(dict(self.costs_history))
        }
    
    def compute_metrics(self, returns: pd.Series) -> dict:
        """Backtest performance metrics."""
        from ..utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio
        
        cum_returns = (1 + returns).cumprod()
        
        return {
            'total_return': cum_returns.iloc[-1] - 1,
            'sharpe': sharpe_ratio(returns),
            'sortino': sortino_ratio(returns),
            'max_drawdown': max_drawdown(cum_returns),
            'calmar': calmar_ratio(returns),
            'avg_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
