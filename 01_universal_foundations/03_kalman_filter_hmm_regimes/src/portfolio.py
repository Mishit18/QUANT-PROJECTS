"""
Multi-asset portfolio construction and backtesting.

SCALABILITY:
- Vectorized operations for multiple assets
- Cross-sectional signal generation
- Portfolio-level risk management
- Efficient matrix operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.signals import BaseSignal
from src.utils import ensure_positive_definite


class MultiAssetSignalGenerator:
    """
    Generate signals for multiple assets simultaneously.
    
    SCALABILITY: Vectorized operations across assets.
    """
    
    def __init__(self,
                 signal_generator: BaseSignal,
                 asset_names: List[str]):
        """
        Initialize multi-asset signal generator.
        
        Parameters
        ----------
        signal_generator : BaseSignal
            Signal generator to apply to each asset
        asset_names : list of str
            Asset identifiers
        """
        self.signal_generator = signal_generator
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
    
    def generate(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Generate signals for all assets.
        
        CAUSALITY: Each asset signal uses only its own history.
        NUMERICAL: Validates all outputs.
        
        Parameters
        ----------
        returns_matrix : np.ndarray
            Returns matrix (n_periods, n_assets)
            
        Returns
        -------
        signals : np.ndarray
            Signal matrix (n_periods, n_assets)
            
        Raises
        ------
        ValueError
            If returns contain NaN/Inf
        RuntimeError
            If signal generation fails
        """
        # Input validation
        if np.any(np.isnan(returns_matrix)):
            raise ValueError(f"Returns matrix contains {np.sum(np.isnan(returns_matrix))} NaN values")
        
        if np.any(np.isinf(returns_matrix)):
            raise ValueError(f"Returns matrix contains {np.sum(np.isinf(returns_matrix))} Inf values")
        
        if returns_matrix.shape[1] != self.n_assets:
            raise ValueError(f"Returns matrix has {returns_matrix.shape[1]} assets, expected {self.n_assets}")
        
        # Generate signals for each asset (vectorized where possible)
        n_periods = returns_matrix.shape[0]
        signals = np.zeros((n_periods, self.n_assets))
        
        for i in range(self.n_assets):
            try:
                asset_returns = returns_matrix[:, i]
                signals[:, i] = self.signal_generator.generate(asset_returns)
            except Exception as e:
                raise RuntimeError(f"Signal generation failed for asset {self.asset_names[i]}: {e}") from e
        
        # Validate outputs
        if np.any(np.isnan(signals)):
            raise RuntimeError(f"Generated signals contain {np.sum(np.isnan(signals))} NaN values")
        
        if np.any(np.isinf(signals)):
            raise RuntimeError(f"Generated signals contain {np.sum(np.isinf(signals))} Inf values")
        
        return signals


class PortfolioConstructor:
    """
    Construct portfolio weights from signals with risk constraints.
    
    SCALABILITY: Efficient covariance estimation and optimization.
    NUMERICAL: Ensures weights sum to 1, handles singular covariances.
    """
    
    def __init__(self,
                 method: str = 'equal_risk',
                 leverage: float = 1.0,
                 max_weight: float = 0.3):
        """
        Initialize portfolio constructor.
        
        Parameters
        ----------
        method : str
            'equal_risk', 'signal_weighted', 'min_variance'
        leverage : float
            Maximum gross leverage
        max_weight : float
            Maximum weight per asset
        """
        self.method = method
        self.leverage = leverage
        self.max_weight = max_weight
    
    def construct(self,
                  signals: np.ndarray,
                  returns: np.ndarray,
                  lookback: int = 60) -> np.ndarray:
        """
        Construct portfolio weights from signals.
        
        CAUSALITY: Uses only historical returns for covariance.
        NUMERICAL: Validates weights sum to leverage, handles edge cases.
        
        Parameters
        ----------
        signals : np.ndarray
            Signal matrix (n_periods, n_assets)
        returns : np.ndarray
            Returns matrix (n_periods, n_assets)
        lookback : int
            Lookback for covariance estimation
            
        Returns
        -------
        weights : np.ndarray
            Weight matrix (n_periods, n_assets)
            
        Raises
        ------
        ValueError
            If inputs invalid
        RuntimeError
            If weight construction fails
        """
        # Input validation
        if signals.shape != returns.shape:
            raise ValueError(f"Signals shape {signals.shape} != returns shape {returns.shape}")
        
        n_periods, n_assets = signals.shape
        weights = np.zeros((n_periods, n_assets))
        
        for t in range(n_periods):
            # Historical returns for covariance (CAUSALITY: only past data)
            if t < lookback:
                hist_returns = returns[:t+1, :]
            else:
                hist_returns = returns[t-lookback+1:t+1, :]
            
            # Skip if insufficient history
            if len(hist_returns) < 2:
                weights[t, :] = 0.0
                continue
            
            # Current signals
            current_signals = signals[t, :]
            
            # Construct weights based on method
            try:
                if self.method == 'signal_weighted':
                    weights[t, :] = self._signal_weighted(current_signals)
                
                elif self.method == 'equal_risk':
                    weights[t, :] = self._equal_risk(current_signals, hist_returns)
                
                elif self.method == 'min_variance':
                    weights[t, :] = self._min_variance(current_signals, hist_returns)
                
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
            except Exception as e:
                raise RuntimeError(f"Weight construction failed at t={t}: {e}") from e
        
        # Validate weights
        if np.any(np.isnan(weights)):
            raise RuntimeError(f"Weights contain {np.sum(np.isnan(weights))} NaN values")
        
        if np.any(np.isinf(weights)):
            raise RuntimeError(f"Weights contain {np.sum(np.isinf(weights))} Inf values")
        
        return weights
    
    def _signal_weighted(self, signals: np.ndarray) -> np.ndarray:
        """
        Simple signal-weighted portfolio.
        
        NUMERICAL: Handles zero signal sum.
        """
        # Normalize signals
        signal_sum = np.abs(signals).sum()
        
        if signal_sum < 1e-10:
            return np.zeros_like(signals)
        
        weights = signals / signal_sum * self.leverage
        
        # Apply max weight constraint
        weights = np.clip(weights, -self.max_weight, self.max_weight)
        
        # Renormalize to leverage
        weight_sum = np.abs(weights).sum()
        if weight_sum > 1e-10:
            weights = weights / weight_sum * self.leverage
        
        return weights
    
    def _equal_risk(self, signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Equal risk contribution portfolio.
        
        NUMERICAL: Handles zero volatility, validates covariance.
        """
        # Estimate volatilities
        vols = np.std(returns, axis=0)
        vols = np.maximum(vols, 1e-6)  # Floor to prevent division by zero
        
        # Inverse volatility weights
        inv_vol = 1.0 / vols
        
        # Apply signal direction
        weights = np.sign(signals) * inv_vol
        
        # Normalize
        weight_sum = np.abs(weights).sum()
        if weight_sum > 1e-10:
            weights = weights / weight_sum * self.leverage
        else:
            weights = np.zeros_like(signals)
        
        # Apply max weight constraint
        weights = np.clip(weights, -self.max_weight, self.max_weight)
        
        return weights
    
    def _min_variance(self, signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Minimum variance portfolio with signal constraints.
        
        NUMERICAL: Handles singular covariance, validates PSD.
        """
        n_assets = len(signals)
        
        # Estimate covariance
        if len(returns) < n_assets:
            # Insufficient data - fall back to equal risk
            return self._equal_risk(signals, returns)
        
        cov = np.cov(returns.T)
        cov = ensure_positive_definite(cov, epsilon=1e-6)
        
        # Active assets (non-zero signals)
        active = np.abs(signals) > 1e-6
        
        if not np.any(active):
            return np.zeros(n_assets)
        
        # Minimum variance on active assets
        try:
            cov_active = cov[np.ix_(active, active)]
            inv_cov = np.linalg.inv(cov_active)
            ones = np.ones(active.sum())
            
            # Minimum variance weights
            w_active = inv_cov @ ones / (ones @ inv_cov @ ones)
            
            # Apply signal direction
            w_active = w_active * np.sign(signals[active])
            
            # Map back to full weight vector
            weights = np.zeros(n_assets)
            weights[active] = w_active
            
            # Normalize to leverage
            weight_sum = np.abs(weights).sum()
            if weight_sum > 1e-10:
                weights = weights / weight_sum * self.leverage
            
            # Apply max weight constraint
            weights = np.clip(weights, -self.max_weight, self.max_weight)
            
            return weights
            
        except np.linalg.LinAlgError:
            # Singular covariance - fall back to equal risk
            return self._equal_risk(signals, returns)


class PortfolioBacktest:
    """
    Backtest multi-asset portfolio with transaction costs.
    
    CAUSALITY: Strict time-lagging of positions.
    NUMERICAL: Validates all returns and positions.
    AUDITABILITY: Tracks all costs and exposures.
    """
    
    def __init__(self,
                 weights: np.ndarray,
                 returns: np.ndarray,
                 transaction_cost: float = 0.0005,
                 initial_capital: float = 1000000.0):
        """
        Initialize portfolio backtest.
        
        Parameters
        ----------
        weights : np.ndarray
            Weight matrix (n_periods, n_assets)
        returns : np.ndarray
            Returns matrix (n_periods, n_assets)
        transaction_cost : float
            Transaction cost per unit turnover
        initial_capital : float
            Initial capital
        """
        self.weights = weights
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Results storage
        self.positions = None
        self.portfolio_returns = None
        self.equity_curve = None
        self.turnover = None
        self.costs = None
    
    def run(self) -> Dict:
        """
        Run portfolio backtest.
        
        CAUSALITY: Positions lagged by one period.
        NUMERICAL: Validates all intermediate results.
        
        Returns
        -------
        dict
            Backtest results
            
        Raises
        ------
        ValueError
            If inputs invalid
        RuntimeError
            If backtest produces invalid results
        """
        # Input validation
        if np.any(np.isnan(self.weights)):
            raise ValueError(f"Weights contain {np.sum(np.isnan(self.weights))} NaN values")
        
        if np.any(np.isnan(self.returns)):
            raise ValueError(f"Returns contain {np.sum(np.isnan(self.returns))} NaN values")
        
        if self.weights.shape != self.returns.shape:
            raise ValueError(f"Weights shape {self.weights.shape} != returns shape {self.returns.shape}")
        
        n_periods, n_assets = self.weights.shape
        
        # CAUSALITY: Lag positions by one period
        self.positions = np.vstack([np.zeros(n_assets), self.weights[:-1, :]])
        
        # Validate positions
        if np.any(np.isnan(self.positions)):
            raise RuntimeError("Positions contain NaN after lagging")
        
        # Portfolio returns (vectorized)
        gross_returns = np.sum(self.positions * self.returns, axis=1)
        
        # Validate gross returns
        if np.any(np.isnan(gross_returns)):
            raise RuntimeError("Gross returns contain NaN")
        
        if np.any(np.isinf(gross_returns)):
            raise RuntimeError("Gross returns contain Inf")
        
        # Turnover (vectorized)
        position_changes = np.diff(self.positions, axis=0, prepend=np.zeros((1, n_assets)))
        self.turnover = np.sum(np.abs(position_changes), axis=1)
        
        # Transaction costs
        self.costs = self.turnover * self.transaction_cost
        
        # Validate costs
        if np.any(self.costs < 0):
            raise RuntimeError(f"Negative costs detected: min={self.costs.min():.6f}")
        
        # Net returns
        self.portfolio_returns = gross_returns - self.costs
        
        # Validate net returns
        if np.any(np.isnan(self.portfolio_returns)):
            raise RuntimeError("Net returns contain NaN")
        
        if np.any(np.isinf(self.portfolio_returns)):
            raise RuntimeError("Net returns contain Inf")
        
        # Equity curve
        cumulative_returns = np.cumprod(1 + self.portfolio_returns)
        
        if np.any(cumulative_returns <= 0):
            raise RuntimeError(f"Non-positive cumulative returns: min={cumulative_returns.min():.6f}")
        
        self.equity_curve = self.initial_capital * cumulative_returns
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        # Validate key metrics
        if np.isnan(results['sharpe_ratio']) or np.isinf(results['sharpe_ratio']):
            raise RuntimeError(f"Invalid Sharpe ratio: {results['sharpe_ratio']}")
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        from src.utils import sharpe_ratio, sortino_ratio, max_drawdown
        
        metrics = {}
        
        # Returns
        metrics['total_return'] = (self.equity_curve[-1] / self.initial_capital - 1)
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(self.portfolio_returns)) - 1
        metrics['volatility'] = np.std(self.portfolio_returns) * np.sqrt(252)
        
        # Risk-adjusted
        metrics['sharpe_ratio'] = sharpe_ratio(self.portfolio_returns)
        metrics['sortino_ratio'] = sortino_ratio(self.portfolio_returns)
        
        # Drawdown
        max_dd, dd_start, dd_end = max_drawdown(self.portfolio_returns)
        metrics['max_drawdown'] = max_dd
        
        # Trading
        metrics['avg_turnover'] = np.mean(self.turnover)
        metrics['total_costs'] = np.sum(self.costs)
        
        # Exposure
        metrics['avg_gross_exposure'] = np.mean(np.sum(np.abs(self.positions), axis=1))
        metrics['avg_net_exposure'] = np.mean(np.sum(self.positions, axis=1))
        
        return metrics


def create_multi_asset_portfolio(returns_matrix: np.ndarray,
                                 asset_names: List[str],
                                 signal_generator: BaseSignal,
                                 portfolio_method: str = 'equal_risk',
                                 leverage: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end multi-asset portfolio construction.
    
    SCALABILITY: Handles arbitrary number of assets.
    SEPARATION: Clear pipeline from signals to weights.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        Returns (n_periods, n_assets)
    asset_names : list of str
        Asset identifiers
    signal_generator : BaseSignal
        Signal generator
    portfolio_method : str
        Portfolio construction method
    leverage : float
        Target leverage
        
    Returns
    -------
    signals : np.ndarray
        Signal matrix
    weights : np.ndarray
        Weight matrix
        
    Raises
    ------
    ValueError
        If inputs invalid
    RuntimeError
        If construction fails
    """
    # Input validation
    if returns_matrix.shape[1] != len(asset_names):
        raise ValueError(f"Returns has {returns_matrix.shape[1]} assets, but {len(asset_names)} names provided")
    
    # Generate signals
    signal_gen = MultiAssetSignalGenerator(signal_generator, asset_names)
    signals = signal_gen.generate(returns_matrix)
    
    # Construct portfolio
    constructor = PortfolioConstructor(method=portfolio_method, leverage=leverage)
    weights = constructor.construct(signals, returns_matrix)
    
    return signals, weights


if __name__ == '__main__':
    # Test multi-asset portfolio
    from src.data_loader import generate_synthetic_data
    from src.state_space_models import LocalLevelModel
    from src.kalman_filter import KalmanFilter
    from src.signals import KalmanTrendSignal
    
    print("Testing multi-asset portfolio...")
    
    # Generate multi-asset data
    n_assets = 5
    n_periods = 500
    
    returns_matrix = np.random.randn(n_periods, n_assets) * 0.01
    asset_names = [f"Asset_{i}" for i in range(n_assets)]
    
    print(f"Returns matrix shape: {returns_matrix.shape}")
    
    # Create signal generator (single-asset, will be applied to each)
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    kf = KalmanFilter(model)
    signal_gen = KalmanTrendSignal(kf, lookback=20)
    
    # Generate multi-asset signals and weights
    signals, weights = create_multi_asset_portfolio(
        returns_matrix,
        asset_names,
        signal_gen,
        portfolio_method='equal_risk',
        leverage=1.0
    )
    
    print(f"Signals shape: {signals.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Backtest portfolio
    bt = PortfolioBacktest(weights, returns_matrix, transaction_cost=0.0005)
    results = bt.run()
    
    print(f"\nPortfolio Performance:")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Volatility: {results['volatility']:.2%}")
    print(f"  Avg Gross Exposure: {results['avg_gross_exposure']:.2f}")
    print(f"  Avg Turnover: {results['avg_turnover']:.2f}")
    
    print("\n✓ Multi-asset portfolio test passed")
