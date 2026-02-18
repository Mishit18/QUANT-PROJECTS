"""
Regime-Dependent Factor Analysis
Analyzes factor performance across different market regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

from utils import save_results, compute_summary_statistics

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """
    Identify market regimes and analyze factor performance
    """
    
    def __init__(self, config: dict):
        """
        Initialize regime analyzer
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.regime_config = config['regime']
        self.regimes = None
        self.regime_stats = {}
        
    def identify_volatility_regimes(self, returns: pd.DataFrame,
                                   window: int = 63) -> pd.Series:
        """
        Identify high/low volatility regimes
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        window : int
            Rolling window for volatility estimation
        
        Returns:
        --------
        pd.Series : Regime labels
        """
        logger.info("Identifying volatility regimes...")
        
        # Compute realized volatility
        realized_vol = returns.std(axis=1).rolling(window=window).mean()
        
        # Define threshold
        threshold_method = self.regime_config['volatility_threshold']
        if threshold_method == 'median':
            threshold = realized_vol.median()
        else:
            threshold = float(threshold_method)
        
        # Classify regimes
        regimes = pd.Series('low_vol', index=returns.index)
        regimes[realized_vol > threshold] = 'high_vol'
        
        logger.info(f"Volatility regimes identified: "
                   f"{(regimes == 'high_vol').sum()} high-vol days, "
                   f"{(regimes == 'low_vol').sum()} low-vol days")
        
        return regimes
    
    def identify_market_regimes(self, market_returns: pd.Series,
                               window: int = 126) -> pd.Series:
        """
        Identify bull/bear market regimes
        
        Parameters:
        -----------
        market_returns : pd.Series
            Market index returns
        window : int
            Rolling window for trend estimation
        
        Returns:
        --------
        pd.Series : Regime labels
        """
        logger.info("Identifying market regimes...")
        
        # Compute rolling cumulative returns
        rolling_cum_returns = market_returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Define threshold
        threshold = self.regime_config['market_threshold']
        
        # Classify regimes
        regimes = pd.Series('bear', index=market_returns.index)
        regimes[rolling_cum_returns > threshold] = 'bull'
        
        logger.info(f"Market regimes identified: "
                   f"{(regimes == 'bull').sum()} bull days, "
                   f"{(regimes == 'bear').sum()} bear days")
        
        return regimes
    
    def identify_crisis_regimes(self, returns: pd.DataFrame,
                               vix_threshold: float = 30,
                               drawdown_threshold: float = -0.20) -> pd.Series:
        """
        Identify crisis periods
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        vix_threshold : float
            VIX level for crisis (proxy using volatility)
        drawdown_threshold : float
            Drawdown threshold for crisis
        
        Returns:
        --------
        pd.Series : Regime labels
        """
        logger.info("Identifying crisis regimes...")
        
        # Compute market volatility (VIX proxy)
        market_vol = returns.std(axis=1).rolling(window=21).mean() * np.sqrt(252) * 100
        
        # Compute drawdown
        cum_returns = (1 + returns.mean(axis=1)).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Identify crisis
        regimes = pd.Series('normal', index=returns.index)
        crisis_mask = (market_vol > vix_threshold) | (drawdown < drawdown_threshold)
        regimes[crisis_mask] = 'crisis'
        
        logger.info(f"Crisis regimes identified: {(regimes == 'crisis').sum()} crisis days")
        
        return regimes
    
    def identify_all_regimes(self, returns: pd.DataFrame,
                            market_returns: pd.Series) -> pd.DataFrame:
        """
        Identify all regime types
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        market_returns : pd.Series
            Market returns
        
        Returns:
        --------
        pd.DataFrame : All regime classifications
        """
        logger.info("=" * 60)
        logger.info("IDENTIFYING MARKET REGIMES")
        logger.info("=" * 60)
        
        regimes_df = pd.DataFrame(index=returns.index)
        
        # Volatility regimes
        regimes_df['volatility'] = self.identify_volatility_regimes(returns)
        
        # Market regimes
        regimes_df['market'] = self.identify_market_regimes(market_returns)
        
        # Crisis regimes
        regimes_df['crisis'] = self.identify_crisis_regimes(returns)
        
        self.regimes = regimes_df
        
        logger.info("=" * 60)
        
        return regimes_df
    
    def analyze_factor_by_regime(self, factor_returns: pd.DataFrame,
                                 regime: pd.Series,
                                 regime_name: str) -> pd.DataFrame:
        """
        Analyze factor performance within each regime
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        regime : pd.Series
            Regime classifications
        regime_name : str
            Name of regime type
        
        Returns:
        --------
        pd.DataFrame : Regime-specific statistics
        """
        logger.info(f"Analyzing factors by {regime_name} regime...")
        
        # Align dates
        common_dates = factor_returns.index.intersection(regime.index)
        factor_aligned = factor_returns.loc[common_dates]
        regime_aligned = regime.loc[common_dates]
        
        # Get unique regimes
        unique_regimes = regime_aligned.unique()
        
        # Compute statistics for each regime
        regime_stats = {}
        
        for reg in unique_regimes:
            mask = regime_aligned == reg
            regime_returns = factor_aligned[mask]
            
            if len(regime_returns) < 20:  # Skip if too few observations
                continue
            
            stats = pd.DataFrame(index=factor_returns.columns)
            stats['Regime'] = reg
            stats['N_Obs'] = len(regime_returns)
            stats['Mean_Return'] = regime_returns.mean() * 252
            stats['Volatility'] = regime_returns.std() * np.sqrt(252)
            stats['Sharpe_Ratio'] = stats['Mean_Return'] / stats['Volatility']
            stats['Skewness'] = regime_returns.skew()
            stats['Max_Drawdown'] = self._compute_max_drawdown(regime_returns)
            stats['Hit_Rate'] = (regime_returns > 0).mean()
            
            regime_stats[reg] = stats
        
        # Combine all regimes
        combined_stats = pd.concat(regime_stats.values(), keys=regime_stats.keys())
        
        return combined_stats
    
    def _compute_max_drawdown(self, returns: pd.DataFrame) -> pd.Series:
        """Compute maximum drawdown for each factor"""
        max_dd = pd.Series(index=returns.columns)
        
        for col in returns.columns:
            cum_returns = (1 + returns[col]).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd[col] = drawdown.min()
        
        return max_dd
    
    def compare_regimes(self, factor_returns: pd.DataFrame,
                       regime: pd.Series,
                       metric: str = 'Sharpe_Ratio') -> pd.DataFrame:
        """
        Compare factor performance across regimes
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        regime : pd.Series
            Regime classifications
        metric : str
            Metric to compare
        
        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        # Get regime statistics
        regime_stats = self.analyze_factor_by_regime(
            factor_returns, regime, 'comparison'
        )
        
        # Pivot to get comparison table
        comparison = regime_stats.reset_index()
        comparison = comparison.pivot(
            index='level_1',
            columns='Regime',
            values=metric
        )
        
        return comparison
    
    def test_regime_differences(self, factor_returns: pd.DataFrame,
                               regime: pd.Series) -> pd.DataFrame:
        """
        Test statistical significance of regime differences
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        regime : pd.Series
            Regime classifications
        
        Returns:
        --------
        pd.DataFrame : Test statistics
        """
        from scipy.stats import ttest_ind
        
        logger.info("Testing regime differences...")
        
        # Align dates
        common_dates = factor_returns.index.intersection(regime.index)
        factor_aligned = factor_returns.loc[common_dates]
        regime_aligned = regime.loc[common_dates]
        
        unique_regimes = regime_aligned.unique()
        
        if len(unique_regimes) != 2:
            logger.warning("t-test requires exactly 2 regimes")
            return None
        
        # Split by regime
        regime1, regime2 = unique_regimes
        returns1 = factor_aligned[regime_aligned == regime1]
        returns2 = factor_aligned[regime_aligned == regime2]
        
        # Perform t-tests
        test_results = pd.DataFrame(index=factor_returns.columns)
        
        for factor in factor_returns.columns:
            stat, pval = ttest_ind(
                returns1[factor].dropna(),
                returns2[factor].dropna(),
                equal_var=False
            )
            test_results.loc[factor, 't_statistic'] = stat
            test_results.loc[factor, 'p_value'] = pval
            test_results.loc[factor, 'significant'] = pval < 0.05
        
        return test_results
    
    def analyze_all_regimes(self, factor_returns: pd.DataFrame) -> Dict:
        """
        Analyze factors across all regime types
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        
        Returns:
        --------
        dict : All regime analyses
        """
        if self.regimes is None:
            raise ValueError("Must identify regimes first")
        
        logger.info("=" * 60)
        logger.info("ANALYZING FACTORS BY REGIME")
        logger.info("=" * 60)
        
        results = {}
        
        for regime_type in self.regimes.columns:
            logger.info(f"\nAnalyzing {regime_type} regimes...")
            
            regime_stats = self.analyze_factor_by_regime(
                factor_returns,
                self.regimes[regime_type],
                regime_type
            )
            
            results[regime_type] = regime_stats
            
            # Test differences if binary regime
            if len(self.regimes[regime_type].unique()) == 2:
                test_results = self.test_regime_differences(
                    factor_returns,
                    self.regimes[regime_type]
                )
                results[f'{regime_type}_tests'] = test_results
        
        self.regime_stats = results
        
        logger.info("=" * 60)
        
        return results
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save regime analysis results
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        logger.info("Saving regime analysis results...")
        
        if self.regimes is not None:
            save_results(self.regimes, 'regime_classifications.csv', output_dir)
        
        if self.regime_stats:
            for regime_type, stats in self.regime_stats.items():
                filename = f'regime_stats_{regime_type}.csv'
                save_results(stats, filename, output_dir)
        
        logger.info("Regime analysis results saved")


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    # Load config and data
    config = load_config()
    ensure_directories(config)
    
    returns = pd.read_parquet(f"{config['paths']['data_processed']}/returns.parquet")
    market_returns = pd.read_parquet(f"{config['paths']['data_processed']}/market_returns.parquet")
    pca_factors = pd.read_parquet(f"{config['paths']['results']}/pca_factor_returns.parquet")
    classical_factors = pd.read_parquet(f"{config['paths']['results']}/classical_factor_returns.parquet")
    
    # Identify regimes
    regime_analyzer = RegimeAnalyzer(config)
    regimes = regime_analyzer.identify_all_regimes(returns, market_returns.squeeze())
    
    # Analyze PCA factors by regime
    pca_regime_stats = regime_analyzer.analyze_all_regimes(pca_factors)
    
    # Analyze classical factors by regime
    classical_regime_stats = regime_analyzer.analyze_all_regimes(classical_factors)
    
    # Save results
    regime_analyzer.save_results(config['paths']['results'])
    
    print("\nRegime Analysis Complete")
    print(f"Regimes identified: {regimes.columns.tolist()}")
