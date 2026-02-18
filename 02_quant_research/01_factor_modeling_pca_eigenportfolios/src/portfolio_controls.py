"""
Portfolio Controls: Volatility Targeting and Turnover Management
Implements risk management and transaction cost analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging

from utils import compute_turnover, apply_transaction_costs, save_results

logger = logging.getLogger(__name__)


class PortfolioControls:
    """
    Portfolio risk management and transaction cost controls
    """
    
    def __init__(self, config: dict):
        """
        Initialize portfolio controls
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.portfolio_config = config['portfolio']
        
    def apply_volatility_targeting(self, returns: pd.Series,
                                   target_vol: Optional[float] = None,
                                   lookback: int = 63) -> Tuple[pd.Series, pd.Series]:
        """
        Apply volatility targeting to factor returns
        
        Parameters:
        -----------
        returns : pd.Series
            Factor returns
        target_vol : float, optional
            Target annualized volatility
        lookback : int
            Lookback window for volatility estimation
        
        Returns:
        --------
        tuple : (scaled_returns, leverage)
        """
        if target_vol is None:
            target_vol = self.portfolio_config['target_volatility']
        
        logger.info(f"Applying volatility targeting: {target_vol:.1%} target")
        
        # Estimate realized volatility
        realized_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
        
        # Compute leverage (target / realized)
        leverage = target_vol / realized_vol
        
        # Cap leverage at reasonable levels
        leverage = leverage.clip(0.5, 2.0)
        
        # Apply leverage
        scaled_returns = returns * leverage.shift(1)  # Use lagged leverage
        
        # Remove NaN from initial period
        scaled_returns = scaled_returns.dropna()
        leverage = leverage.dropna()
        
        logger.info(f"Mean leverage: {leverage.mean():.2f}")
        
        return scaled_returns, leverage
    
    def compute_portfolio_turnover(self, weights: pd.DataFrame,
                                   rebalance_freq: Optional[int] = None) -> pd.Series:
        """
        Compute portfolio turnover
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights over time
        rebalance_freq : int, optional
            Rebalancing frequency in days
        
        Returns:
        --------
        pd.Series : Turnover
        """
        if rebalance_freq is None:
            rebalance_freq = self.portfolio_config['rebalance_frequency']
        
        logger.info(f"Computing turnover with rebalance frequency: {rebalance_freq} days")
        
        # Sample weights at rebalance frequency
        rebalance_dates = weights.index[::rebalance_freq]
        weights_rebalanced = weights.loc[rebalance_dates]
        
        # Compute turnover
        turnover = weights_rebalanced.diff().abs().sum(axis=1)
        
        # Annualize
        turnover_annual = turnover * (252 / rebalance_freq)
        
        logger.info(f"Mean annual turnover: {turnover_annual.mean():.2f}")
        
        return turnover_annual
    
    def apply_turnover_constraint(self, weights_target: pd.DataFrame,
                                  weights_current: pd.DataFrame,
                                  max_turnover: Optional[float] = None) -> pd.DataFrame:
        """
        Apply turnover constraint to portfolio weights
        
        Parameters:
        -----------
        weights_target : pd.DataFrame
            Target portfolio weights
        weights_current : pd.DataFrame
            Current portfolio weights
        max_turnover : float, optional
            Maximum allowed turnover
        
        Returns:
        --------
        pd.DataFrame : Constrained weights
        """
        if max_turnover is None:
            max_turnover = self.portfolio_config['max_turnover']
        
        logger.info(f"Applying turnover constraint: {max_turnover:.1%} max")
        
        # Compute desired trade
        trade = weights_target - weights_current
        
        # Compute turnover
        turnover = trade.abs().sum(axis=1)
        
        # Scale trade if turnover exceeds limit
        scale_factor = np.minimum(1.0, max_turnover / turnover)
        
        # Apply scaling
        constrained_trade = trade.multiply(scale_factor, axis=0)
        constrained_weights = weights_current + constrained_trade
        
        # Renormalize
        constrained_weights = constrained_weights.div(
            constrained_weights.abs().sum(axis=1), axis=0
        )
        
        return constrained_weights
    
    def compute_transaction_costs(self, returns: pd.Series,
                                  weights: pd.DataFrame,
                                  cost_bps: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Compute returns after transaction costs
        
        Parameters:
        -----------
        returns : pd.Series
            Gross returns
        weights : pd.DataFrame
            Portfolio weights
        cost_bps : float, optional
            Transaction cost in basis points
        
        Returns:
        --------
        tuple : (net_returns, costs)
        """
        if cost_bps is None:
            cost_bps = self.portfolio_config['cost_bps']
        
        logger.info(f"Computing transaction costs: {cost_bps} bps")
        
        # Compute turnover
        turnover = weights.diff().abs().sum(axis=1)
        
        # Compute costs
        costs = turnover * (cost_bps / 10000)
        
        # Apply to returns
        net_returns = returns - costs
        
        logger.info(f"Mean annual cost: {costs.mean() * 252:.2%}")
        
        return net_returns, costs
    
    def analyze_turnover_sharpe_tradeoff(self, factor_returns: pd.DataFrame,
                                        factor_weights: Dict[str, pd.DataFrame],
                                        rebalance_frequencies: list = None) -> pd.DataFrame:
        """
        Analyze tradeoff between turnover and Sharpe ratio
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        factor_weights : dict
            Portfolio weights for each factor
        rebalance_frequencies : list, optional
            List of rebalancing frequencies to test
        
        Returns:
        --------
        pd.DataFrame : Turnover-Sharpe analysis
        """
        if rebalance_frequencies is None:
            rebalance_frequencies = [1, 5, 21, 63, 126]  # Daily, weekly, monthly, quarterly, semi-annual
        
        logger.info("Analyzing turnover-Sharpe tradeoff...")
        
        results = []
        
        for factor in factor_returns.columns:
            if factor not in factor_weights:
                continue
            
            weights = factor_weights[factor]
            
            for freq in rebalance_frequencies:
                # Compute turnover
                turnover = self.compute_portfolio_turnover(weights, freq)
                
                # Compute net returns after costs
                gross_returns = factor_returns[factor]
                net_returns, costs = self.compute_transaction_costs(
                    gross_returns, weights
                )
                
                # Compute Sharpe ratios
                gross_sharpe = (gross_returns.mean() / gross_returns.std()) * np.sqrt(252)
                net_sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
                
                results.append({
                    'Factor': factor,
                    'Rebalance_Freq': freq,
                    'Turnover': turnover.mean(),
                    'Gross_Sharpe': gross_sharpe,
                    'Net_Sharpe': net_sharpe,
                    'Sharpe_Decay': gross_sharpe - net_sharpe,
                    'Cost_Impact': costs.mean() * 252
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info("Turnover-Sharpe analysis complete")
        
        return results_df
    
    def optimize_rebalance_frequency(self, returns: pd.Series,
                                     weights: pd.DataFrame) -> int:
        """
        Find optimal rebalancing frequency
        
        Parameters:
        -----------
        returns : pd.Series
            Factor returns
        weights : pd.DataFrame
            Portfolio weights
        
        Returns:
        --------
        int : Optimal rebalancing frequency
        """
        logger.info("Optimizing rebalancing frequency...")
        
        frequencies = [1, 5, 10, 21, 42, 63]
        sharpe_ratios = []
        
        for freq in frequencies:
            # Compute net returns
            net_returns, _ = self.compute_transaction_costs(returns, weights)
            sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
            sharpe_ratios.append(sharpe)
        
        # Find frequency with highest Sharpe
        optimal_idx = np.argmax(sharpe_ratios)
        optimal_freq = frequencies[optimal_idx]
        
        logger.info(f"Optimal rebalancing frequency: {optimal_freq} days")
        
        return optimal_freq
    
    def apply_risk_limits(self, weights: pd.DataFrame,
                         max_weight: float = 0.10,
                         max_sector_weight: Optional[float] = None) -> pd.DataFrame:
        """
        Apply risk limits to portfolio weights
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights
        max_weight : float
            Maximum weight per asset
        max_sector_weight : float, optional
            Maximum weight per sector
        
        Returns:
        --------
        pd.DataFrame : Constrained weights
        """
        logger.info(f"Applying risk limits: {max_weight:.1%} max per asset")
        
        # Clip weights
        constrained_weights = weights.clip(-max_weight, max_weight)
        
        # Renormalize
        constrained_weights = constrained_weights.div(
            constrained_weights.abs().sum(axis=1), axis=0
        )
        
        return constrained_weights
    
    def backtest_with_controls(self, factor_returns: pd.Series,
                              apply_vol_targeting: bool = True,
                              apply_costs: bool = True) -> Dict:
        """
        Backtest factor with portfolio controls
        
        Parameters:
        -----------
        factor_returns : pd.Series
            Factor returns
        apply_vol_targeting : bool
            Whether to apply volatility targeting
        apply_costs : bool
            Whether to apply transaction costs
        
        Returns:
        --------
        dict : Backtest results
        """
        logger.info("Running backtest with portfolio controls...")
        
        results = {}
        
        # Baseline (no controls)
        results['gross_returns'] = factor_returns
        results['gross_sharpe'] = (
            factor_returns.mean() / factor_returns.std()
        ) * np.sqrt(252)
        
        # Apply volatility targeting
        if apply_vol_targeting:
            scaled_returns, leverage = self.apply_volatility_targeting(factor_returns)
            results['scaled_returns'] = scaled_returns
            results['leverage'] = leverage
            results['scaled_sharpe'] = (
                scaled_returns.mean() / scaled_returns.std()
            ) * np.sqrt(252)
        
        # Note: Transaction costs require portfolio weights
        # This is a simplified version
        
        logger.info("Backtest complete")
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = "results") -> None:
        """
        Save portfolio control results
        
        Parameters:
        -----------
        results : dict
            Results dictionary
        output_dir : str
            Output directory
        """
        logger.info("Saving portfolio control results...")
        
        for key, value in results.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                filename = f'portfolio_{key}.csv'
                save_results(value, filename, output_dir)
        
        logger.info("Portfolio control results saved")


class RiskMetrics:
    """
    Compute portfolio risk metrics
    """
    
    def __init__(self, config: dict):
        self.config = config
        
    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Value at Risk"""
        return returns.quantile(1 - confidence)
    
    def compute_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Conditional Value at Risk (Expected Shortfall)"""
        var = self.compute_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def compute_downside_deviation(self, returns: pd.Series, 
                                   target: float = 0.0) -> float:
        """Compute downside deviation"""
        downside_returns = returns[returns < target]
        return downside_returns.std() * np.sqrt(252)
    
    def compute_sortino_ratio(self, returns: pd.Series,
                             target: float = 0.0) -> float:
        """Compute Sortino ratio"""
        excess_return = returns.mean() * 252
        downside_dev = self.compute_downside_deviation(returns, target)
        return excess_return / downside_dev
    
    def compute_calmar_ratio(self, returns: pd.Series) -> float:
        """Compute Calmar ratio"""
        annual_return = returns.mean() * 252
        
        # Compute max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        return annual_return / max_dd if max_dd > 0 else np.nan
    
    def compute_all_metrics(self, returns: pd.Series) -> pd.Series:
        """Compute all risk metrics"""
        metrics = pd.Series({
            'VaR_95': self.compute_var(returns, 0.95),
            'CVaR_95': self.compute_cvar(returns, 0.95),
            'Downside_Dev': self.compute_downside_deviation(returns),
            'Sortino_Ratio': self.compute_sortino_ratio(returns),
            'Calmar_Ratio': self.compute_calmar_ratio(returns)
        })
        
        return metrics


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    # Load config and data
    config = load_config()
    ensure_directories(config)
    
    pca_factors = pd.read_parquet(f"{config['paths']['results']}/pca_factor_returns.parquet")
    classical_factors = pd.read_parquet(f"{config['paths']['results']}/classical_factor_returns.parquet")
    
    # Initialize portfolio controls
    portfolio_controls = PortfolioControls(config)
    
    # Apply volatility targeting to first PCA factor
    pc1_returns = pca_factors.iloc[:, 0]
    scaled_returns, leverage = portfolio_controls.apply_volatility_targeting(pc1_returns)
    
    print("\nPortfolio Controls Analysis:")
    print(f"Original Sharpe: {(pc1_returns.mean() / pc1_returns.std() * np.sqrt(252)):.2f}")
    print(f"Scaled Sharpe: {(scaled_returns.mean() / scaled_returns.std() * np.sqrt(252)):.2f}")
    print(f"Mean Leverage: {leverage.mean():.2f}")
