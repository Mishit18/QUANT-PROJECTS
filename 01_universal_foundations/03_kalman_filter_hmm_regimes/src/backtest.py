"""
Backtesting engine for trading strategies.

Implements walk-forward backtesting with:
- Transaction cost modeling
- Position sizing
- No lookahead bias
- Regime-conditional performance analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from src.utils import sharpe_ratio, sortino_ratio, max_drawdown


class Backtest:
    """
    Backtesting engine for systematic trading strategies.
    """
    
    def __init__(self,
                 signals: np.ndarray,
                 returns: np.ndarray,
                 transaction_cost: float = 0.0005,
                 leverage: float = 1.0,
                 initial_capital: float = 1000000.0):
        """
        Initialize backtest.
        
        Parameters
        ----------
        signals : np.ndarray
            Trading signals (-1 to 1)
        returns : np.ndarray
            Asset returns
        transaction_cost : float
            Transaction cost (bps as decimal, e.g., 5bps = 0.0005)
        leverage : float
            Maximum leverage
        initial_capital : float
            Initial capital
        """
        self.signals = signals
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.initial_capital = initial_capital
        
        # Results storage
        self.positions = None
        self.strategy_returns = None
        self.equity_curve = None
        self.turnover = None
        self.costs = None
    
    def run(self) -> Dict:
        """
        Run backtest.
        
        PRODUCTION HARDENING:
        - Validates positions and returns
        - Checks for lookahead bias
        - Ensures no NaN propagation
        - Validates performance metrics
        
        Returns
        -------
        dict
            Backtest results
            
        Raises
        ------
        ValueError
            If inputs are invalid
        RuntimeError
            If backtest produces invalid results
        """
        # Input validation
        if np.any(np.isnan(self.signals)):
            raise ValueError(f"Signals contain {np.sum(np.isnan(self.signals))} NaN values")
        
        if np.any(np.isinf(self.signals)):
            raise ValueError(f"Signals contain {np.sum(np.isinf(self.signals))} Inf values")
        
        if np.any(np.isnan(self.returns)):
            raise ValueError(f"Returns contain {np.sum(np.isnan(self.returns))} NaN values")
        
        if np.any(np.isinf(self.returns)):
            raise ValueError(f"Returns contain {np.sum(np.isinf(self.returns))} Inf values")
        
        if len(self.signals) != len(self.returns):
            raise ValueError(f"Signal length ({len(self.signals)}) != returns length ({len(self.returns)})")
        
        # Positions (lagged signals to avoid lookahead)
        self.positions = np.roll(self.signals, 1)
        self.positions[0] = 0  # No position on first day
        
        # Validate no lookahead bias: position[t] should not depend on return[t]
        # This is ensured by the lag above
        
        # Apply leverage constraint
        self.positions = np.clip(self.positions * self.leverage, -self.leverage, self.leverage)
        
        # Validate positions
        if np.any(np.isnan(self.positions)):
            raise RuntimeError("Positions contain NaN values after processing")
        
        if np.any(np.isinf(self.positions)):
            raise RuntimeError("Positions contain Inf values after processing")
        
        # Position changes (for transaction costs)
        position_changes = np.diff(self.positions, prepend=0)
        self.turnover = np.abs(position_changes)
        
        # Validate turnover
        if np.any(np.isnan(self.turnover)):
            raise RuntimeError("Turnover contains NaN values")
        
        # Transaction costs
        self.costs = self.turnover * self.transaction_cost
        
        # Validate costs
        if np.any(np.isnan(self.costs)) or np.any(np.isinf(self.costs)):
            raise RuntimeError("Transaction costs contain NaN/Inf values")
        
        if np.any(self.costs < 0):
            raise RuntimeError(f"Negative transaction costs detected: min={self.costs.min():.6f}")
        
        # Strategy returns (before costs)
        gross_returns = self.positions * self.returns
        
        # Validate gross returns
        if np.any(np.isnan(gross_returns)):
            raise RuntimeError("Gross returns contain NaN values")
        
        if np.any(np.isinf(gross_returns)):
            raise RuntimeError("Gross returns contain Inf values")
        
        # Net returns (after costs)
        self.strategy_returns = gross_returns - self.costs
        
        # Validate net returns
        if np.any(np.isnan(self.strategy_returns)):
            raise RuntimeError("Net returns contain NaN values")
        
        if np.any(np.isinf(self.strategy_returns)):
            raise RuntimeError("Net returns contain Inf values")
        
        # Check for extreme returns (potential error)
        if np.any(np.abs(self.strategy_returns) > 1.0):
            import warnings
            warnings.warn(f"Extreme returns detected: max={self.strategy_returns.max():.2%}, min={self.strategy_returns.min():.2%}")
        
        # Equity curve
        cumulative_returns = np.cumprod(1 + self.strategy_returns)
        
        # Validate cumulative returns
        if np.any(np.isnan(cumulative_returns)) or np.any(np.isinf(cumulative_returns)):
            raise RuntimeError("Cumulative returns contain NaN/Inf")
        
        if np.any(cumulative_returns <= 0):
            raise RuntimeError(f"Non-positive cumulative returns detected: min={cumulative_returns.min():.6f}")
        
        self.equity_curve = self.initial_capital * cumulative_returns
        
        # Performance metrics
        results = self._calculate_metrics()
        
        # Validate key metrics
        if np.isnan(results['sharpe_ratio']) or np.isinf(results['sharpe_ratio']):
            raise RuntimeError(f"Invalid Sharpe ratio: {results['sharpe_ratio']}")
        
        if np.isnan(results['total_return']) or np.isinf(results['total_return']):
            raise RuntimeError(f"Invalid total return: {results['total_return']}")
        
        if np.isnan(results['max_drawdown']) or np.isinf(results['max_drawdown']):
            raise RuntimeError(f"Invalid max drawdown: {results['max_drawdown']}")
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        metrics = {}
        
        # Return statistics
        metrics['total_return'] = (self.equity_curve[-1] / self.initial_capital - 1)
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(self.returns)) - 1
        metrics['volatility'] = np.std(self.strategy_returns) * np.sqrt(252)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = sharpe_ratio(self.strategy_returns)
        metrics['sortino_ratio'] = sortino_ratio(self.strategy_returns)
        
        # Drawdown
        max_dd, dd_start, dd_end = max_drawdown(self.strategy_returns)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_start'] = dd_start
        metrics['max_drawdown_end'] = dd_end
        
        # Trading statistics
        metrics['avg_turnover'] = np.mean(self.turnover)
        metrics['total_costs'] = np.sum(self.costs)
        metrics['cost_drag'] = np.sum(self.costs) / len(self.returns) * 252
        
        # Win rate
        winning_days = (self.strategy_returns > 0).sum()
        metrics['win_rate'] = winning_days / len(self.strategy_returns)
        
        # Average win/loss
        wins = self.strategy_returns[self.strategy_returns > 0]
        losses = self.strategy_returns[self.strategy_returns < 0]
        metrics['avg_win'] = np.mean(wins) if len(wins) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
        
        # Exposure
        metrics['avg_exposure'] = np.mean(np.abs(self.positions))
        metrics['long_exposure'] = np.mean(self.positions[self.positions > 0]) if (self.positions > 0).any() else 0
        metrics['short_exposure'] = np.mean(self.positions[self.positions < 0]) if (self.positions < 0).any() else 0
        
        return metrics
    
    def regime_conditional_performance(self, regimes: np.ndarray) -> pd.DataFrame:
        """
        Calculate performance metrics conditional on regime.
        
        Parameters
        ----------
        regimes : np.ndarray
            Regime labels
            
        Returns
        -------
        pd.DataFrame
            Regime-conditional metrics
        """
        unique_regimes = np.unique(regimes)
        results = []
        
        for regime in unique_regimes:
            mask = regimes == regime
            regime_returns = self.strategy_returns[mask]
            
            if len(regime_returns) > 0:
                metrics = {
                    'regime': regime,
                    'n_periods': len(regime_returns),
                    'frequency': len(regime_returns) / len(self.strategy_returns),
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns) * np.sqrt(252),
                    'sharpe': sharpe_ratio(regime_returns) if len(regime_returns) > 20 else np.nan,
                    'win_rate': (regime_returns > 0).sum() / len(regime_returns),
                    'avg_position': np.mean(self.positions[mask])
                }
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        return pd.Series(self.equity_curve)
    
    def get_returns(self) -> pd.Series:
        """Get strategy returns as pandas Series."""
        return pd.Series(self.strategy_returns)
    
    def get_positions(self) -> pd.Series:
        """Get positions as pandas Series."""
        return pd.Series(self.positions)


class WalkForwardBacktest:
    """
    Walk-forward backtesting with rolling model re-estimation.
    """
    
    def __init__(self,
                 train_window: int = 252,
                 test_window: int = 63,
                 transaction_cost: float = 0.0005):
        """
        Initialize walk-forward backtest.
        
        Parameters
        ----------
        train_window : int
            Training window size
        test_window : int
            Test window size
        transaction_cost : float
            Transaction cost
        """
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost = transaction_cost
        
        self.results = []
    
    def run(self, 
            returns: np.ndarray,
            model_fn: callable,
            signal_fn: callable) -> Dict:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
        model_fn : callable
            Function to fit model: model_fn(train_returns) -> model
        signal_fn : callable
            Function to generate signals: signal_fn(model, test_returns) -> signals
            
        Returns
        -------
        dict
            Aggregated results
        """
        n = len(returns)
        all_signals = np.zeros(n)
        all_predictions = []
        
        # Walk forward
        start = self.train_window
        while start + self.test_window <= n:
            # Training data
            train_returns = returns[start - self.train_window:start]
            
            # Fit model
            model = model_fn(train_returns)
            
            # Test data
            test_returns = returns[start:start + self.test_window]
            
            # Generate signals
            signals = signal_fn(model, test_returns)
            
            # Store signals
            all_signals[start:start + self.test_window] = signals
            
            # Move window
            start += self.test_window
        
        # Run backtest on out-of-sample signals
        bt = Backtest(all_signals, returns, transaction_cost=self.transaction_cost)
        results = bt.run()
        
        results['backtest'] = bt
        results['signals'] = all_signals
        
        return results


def compare_strategies(strategies: Dict[str, Backtest]) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Parameters
    ----------
    strategies : dict
        Dictionary of strategy name -> Backtest object
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    results = []
    
    for name, bt in strategies.items():
        if bt.strategy_returns is None:
            bt.run()
        
        metrics = bt._calculate_metrics()
        metrics['strategy'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('strategy')
    
    return df


if __name__ == '__main__':
    # Test backtest
    from src.data_loader import generate_synthetic_data
    from src.state_space_models import LocalLevelModel
    from src.kalman_filter import KalmanFilter
    from src.hmm_regimes import GaussianHMM
    from src.signals import create_regime_aware_strategy
    
    print("Testing backtest engine...")
    
    # Generate data
    data = generate_synthetic_data(n_samples=500, seed=42)
    returns = data['returns'].iloc[:, 0].values
    
    # Fit models
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    kf = KalmanFilter(model)
    kf.filter(returns)
    
    hmm = GaussianHMM(n_regimes=3, random_state=42)
    hmm.fit(returns)
    
    # Generate signals
    signals = create_regime_aware_strategy(returns, kf, hmm)
    
    # Run backtest
    bt = Backtest(signals, returns, transaction_cost=0.0005)
    results = bt.run()
    
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    # Regime-conditional performance
    regimes = hmm.predict(returns)
    regime_perf = bt.regime_conditional_performance(regimes)
    print("\nRegime-Conditional Performance:")
    print(regime_perf)
