"""
Classical Factor Construction
Implements value, momentum, size, quality, and low-volatility factors
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

from utils import rank_transform, standardize, compute_returns, save_results

logger = logging.getLogger(__name__)


class ClassicalFactors:
    """
    Construction of classical equity factors
    Implements long-short factor portfolios
    """
    
    def __init__(self, config: dict):
        """
        Initialize factor constructor
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.factor_config = config['factors']
        self.factor_returns = {}
        self.factor_portfolios = {}
        
    def construct_momentum_factor(self, returns: pd.DataFrame) -> pd.Series:
        """
        Construct momentum factor (past returns)
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.Series : Momentum factor returns
        """
        logger.info("Constructing momentum factor...")
        
        lookback = self.factor_config['momentum']['lookback']
        skip_days = self.factor_config['momentum']['skip_days']
        
        # Compute cumulative returns over lookback period, skipping recent days
        momentum_signal = returns.rolling(window=lookback).apply(
            lambda x: (1 + x[:-skip_days]).prod() - 1 if len(x) >= skip_days else np.nan
        )
        
        # Construct long-short portfolio
        factor_returns = self._construct_long_short_portfolio(
            returns, momentum_signal, 'Momentum'
        )
        
        return factor_returns
    
    def construct_value_factor(self, prices: pd.DataFrame,
                              returns: pd.DataFrame) -> pd.Series:
        """
        Construct value factor (book-to-market proxy using price reversal)
        
        Note: In production, this would use fundamental data
        Here we use long-term price reversal as a proxy
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset prices
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.Series : Value factor returns
        """
        logger.info("Constructing value factor...")
        
        lookback = self.factor_config['value']['lookback']
        
        # Use long-term reversal as value proxy
        # Low past returns -> high book-to-market (value)
        value_signal = -returns.rolling(window=lookback).mean()
        
        # Construct long-short portfolio
        factor_returns = self._construct_long_short_portfolio(
            returns, value_signal, 'Value'
        )
        
        return factor_returns
    
    def construct_size_factor(self, prices: pd.DataFrame,
                             returns: pd.DataFrame) -> pd.Series:
        """
        Construct size factor (market cap proxy using price level)
        
        Note: In production, this would use actual market cap
        Here we use price level as a proxy
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset prices
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.Series : Size factor returns
        """
        logger.info("Constructing size factor...")
        
        # Use rolling average price as size proxy
        # Small cap = low price (inverse relationship)
        size_signal = -prices.rolling(window=63).mean()
        
        # Construct long-short portfolio
        factor_returns = self._construct_long_short_portfolio(
            returns, size_signal, 'Size'
        )
        
        return factor_returns
    
    def construct_quality_factor(self, returns: pd.DataFrame) -> pd.Series:
        """
        Construct quality factor (profitability/stability proxy)
        
        Use return stability and consistency as quality proxy
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.Series : Quality factor returns
        """
        logger.info("Constructing quality factor...")
        
        # Quality = high mean return, low volatility
        # Compute rolling Sharpe ratio as quality signal
        rolling_mean = returns.rolling(window=126).mean()
        rolling_std = returns.rolling(window=126).std()
        quality_signal = rolling_mean / rolling_std
        
        # Construct long-short portfolio
        factor_returns = self._construct_long_short_portfolio(
            returns, quality_signal, 'Quality'
        )
        
        return factor_returns
    
    def construct_low_vol_factor(self, returns: pd.DataFrame) -> pd.Series:
        """
        Construct low-volatility factor
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.Series : Low-vol factor returns
        """
        logger.info("Constructing low-volatility factor...")
        
        lookback = self.factor_config['low_vol']['lookback']
        
        # Compute rolling volatility
        volatility = returns.rolling(window=lookback).std()
        
        # Low vol signal (negative so low vol = high signal)
        low_vol_signal = -volatility
        
        # Construct long-short portfolio
        factor_returns = self._construct_long_short_portfolio(
            returns, low_vol_signal, 'LowVol'
        )
        
        return factor_returns
    
    def _construct_long_short_portfolio(self, returns: pd.DataFrame,
                                       signal: pd.DataFrame,
                                       factor_name: str) -> pd.Series:
        """
        Construct long-short portfolio from signal
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        signal : pd.DataFrame
            Factor signal (higher = better)
        factor_name : str
            Name of factor
        
        Returns:
        --------
        pd.Series : Factor returns
        """
        quantile_split = self.factor_config['quantile_split']
        
        # Rank assets by signal (cross-sectional)
        signal_rank = signal.rank(axis=1, pct=True)
        
        # Long top quantile, short bottom quantile
        long_mask = signal_rank >= (1 - quantile_split)
        short_mask = signal_rank <= quantile_split
        
        # Compute portfolio weights
        if self.factor_config['weighting'] == 'equal':
            long_weights = long_mask.div(long_mask.sum(axis=1), axis=0)
            short_weights = short_mask.div(short_mask.sum(axis=1), axis=0)
        else:
            # Value-weighted (use signal strength)
            long_weights = (signal * long_mask).div((signal * long_mask).sum(axis=1), axis=0)
            short_weights = (signal * short_mask).div((signal * short_mask).sum(axis=1), axis=0)
        
        # Compute factor returns (long - short)
        long_returns = (returns * long_weights).sum(axis=1)
        short_returns = (returns * short_weights).sum(axis=1)
        factor_returns = long_returns - short_returns
        
        # Store portfolio weights
        self.factor_portfolios[factor_name] = {
            'long_weights': long_weights,
            'short_weights': short_weights
        }
        
        return factor_returns
    
    def construct_all_factors(self, prices: pd.DataFrame,
                             returns: pd.DataFrame) -> pd.DataFrame:
        """
        Construct all classical factors
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset prices
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.DataFrame : All factor returns
        """
        logger.info("=" * 60)
        logger.info("CONSTRUCTING CLASSICAL FACTORS")
        logger.info("=" * 60)
        
        # Construct each factor
        self.factor_returns['Momentum'] = self.construct_momentum_factor(returns)
        self.factor_returns['Value'] = self.construct_value_factor(prices, returns)
        self.factor_returns['Size'] = self.construct_size_factor(prices, returns)
        self.factor_returns['Quality'] = self.construct_quality_factor(returns)
        self.factor_returns['LowVol'] = self.construct_low_vol_factor(returns)
        
        # Combine into DataFrame
        factor_returns_df = pd.DataFrame(self.factor_returns)
        
        # Remove NaN rows (from rolling windows)
        factor_returns_df = factor_returns_df.dropna()
        
        logger.info(f"Classical factors constructed: {factor_returns_df.shape}")
        logger.info("=" * 60)
        
        return factor_returns_df
    
    def compute_factor_statistics(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive factor statistics
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        
        Returns:
        --------
        pd.DataFrame : Factor statistics
        """
        logger.info("Computing factor statistics...")
        
        stats = pd.DataFrame(index=factor_returns.columns)
        
        # Annualized returns
        stats['Mean_Return'] = factor_returns.mean() * 252
        
        # Volatility
        stats['Volatility'] = factor_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        stats['Sharpe_Ratio'] = stats['Mean_Return'] / stats['Volatility']
        
        # t-statistic
        stats['t_Statistic'] = (factor_returns.mean() / factor_returns.std()) * np.sqrt(len(factor_returns))
        
        # Skewness and kurtosis
        stats['Skewness'] = factor_returns.skew()
        stats['Kurtosis'] = factor_returns.kurtosis()
        
        # Maximum drawdown
        for factor in factor_returns.columns:
            cum_returns = (1 + factor_returns[factor]).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            stats.loc[factor, 'Max_Drawdown'] = drawdown.min()
        
        # Hit rate (% positive days)
        stats['Hit_Rate'] = (factor_returns > 0).mean()
        
        # Correlation with market (if available)
        # This would be computed separately with market returns
        
        return stats
    
    def compute_factor_correlations(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute factor correlation matrix
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        return factor_returns.corr()
    
    def save_results(self, factor_returns: pd.DataFrame,
                    output_dir: str = "results") -> None:
        """
        Save factor results
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        output_dir : str
            Output directory
        """
        logger.info("Saving factor results...")
        
        # Save factor returns
        save_results(factor_returns, 'classical_factor_returns.csv', output_dir)
        
        # Save factor statistics
        stats = self.compute_factor_statistics(factor_returns)
        save_results(stats, 'factor_statistics.csv', output_dir)
        
        # Save factor correlations
        corr = self.compute_factor_correlations(factor_returns)
        save_results(corr, 'factor_correlations.csv', output_dir)
        
        logger.info("Factor results saved")


class FactorPortfolioAnalyzer:
    """
    Analyze factor portfolio characteristics
    """
    
    def __init__(self, config: dict):
        self.config = config
        
    def compute_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio turnover over time
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights over time
        
        Returns:
        --------
        pd.Series : Turnover
        """
        turnover = weights.diff().abs().sum(axis=1)
        return turnover
    
    def compute_concentration(self, weights: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio concentration (Herfindahl index)
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights
        
        Returns:
        --------
        pd.Series : Concentration index
        """
        concentration = (weights ** 2).sum(axis=1)
        return concentration
    
    def analyze_factor_exposures(self, factor_returns: pd.DataFrame,
                                 pca_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze classical factor exposures to PCA factors
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Classical factor returns
        pca_factors : pd.DataFrame
            PCA factor returns
        
        Returns:
        --------
        pd.DataFrame : Factor exposures (betas)
        """
        logger.info("Analyzing factor exposures to PCA factors...")
        
        # Align dates
        common_dates = factor_returns.index.intersection(pca_factors.index)
        factor_returns_aligned = factor_returns.loc[common_dates]
        pca_factors_aligned = pca_factors.loc[common_dates]
        
        # Compute betas via regression
        exposures = pd.DataFrame(
            index=factor_returns.columns,
            columns=pca_factors.columns
        )
        
        for factor in factor_returns.columns:
            y = factor_returns_aligned[factor].values
            X = pca_factors_aligned.values
            
            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # OLS regression
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            exposures.loc[factor] = betas[1:]  # Exclude intercept
        
        return exposures.astype(float)


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    # Load config and data
    config = load_config()
    ensure_directories(config)
    
    prices = pd.read_parquet(f"{config['paths']['data_processed']}/prices_clean.parquet")
    returns = pd.read_parquet(f"{config['paths']['data_processed']}/returns.parquet")
    
    # Construct factors
    factor_constructor = ClassicalFactors(config)
    factor_returns = factor_constructor.construct_all_factors(prices, returns)
    
    # Save results
    factor_constructor.save_results(factor_returns, config['paths']['results'])
    
    # Print statistics
    stats = factor_constructor.compute_factor_statistics(factor_returns)
    print("\nFactor Statistics:")
    print(stats.round(3))
