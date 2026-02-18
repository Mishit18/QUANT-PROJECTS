"""
Factor Model Regression and Risk Premia Estimation
Implements cross-sectional and time-series factor models
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

from utils import save_results

logger = logging.getLogger(__name__)


class FactorRegressionModel:
    """
    Factor model regression: r_it = alpha_i + beta_i' f_t + epsilon_it
    Estimates factor loadings and risk premia
    """
    
    def __init__(self, config: dict):
        """
        Initialize regression model
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.regression_config = config['regression']
        
        # Results storage
        self.betas = None
        self.alphas = None
        self.residuals = None
        self.r_squared = None
        self.factor_premia = None
        
    def estimate_time_series_regression(self, returns: pd.DataFrame,
                                       factors: pd.DataFrame) -> Dict:
        """
        Estimate time-series regression for each asset
        r_it = alpha_i + beta_i' f_t + epsilon_it
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (T x N)
        factors : pd.DataFrame
            Factor returns (T x K)
        
        Returns:
        --------
        dict : Regression results
        """
        logger.info("Estimating time-series regressions...")
        
        # Align dates
        common_dates = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        # Remove any NaN or inf
        returns_aligned = returns_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        factors_aligned = factors_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        n_assets = len(returns_aligned.columns)
        n_factors = len(factors_aligned.columns)
        
        # Initialize result arrays
        betas = np.zeros((n_assets, n_factors))
        alphas = np.zeros(n_assets)
        residuals = np.zeros((len(common_dates), n_assets))
        r_squared = np.zeros(n_assets)
        t_stats_beta = np.zeros((n_assets, n_factors))
        t_stats_alpha = np.zeros(n_assets)
        
        # Run regression for each asset
        for i, asset in enumerate(returns_aligned.columns):
            y = returns_aligned[asset].values
            X = factors_aligned.values
            X_with_const = add_constant(X)
            
            # OLS regression
            model = OLS(y, X_with_const).fit()
            
            # Store results
            alphas[i] = model.params[0]
            betas[i, :] = model.params[1:]
            residuals[:, i] = model.resid
            r_squared[i] = model.rsquared
            
            # t-statistics
            t_stats_alpha[i] = model.tvalues[0]
            t_stats_beta[i, :] = model.tvalues[1:]
        
        # Convert to DataFrames
        self.betas = pd.DataFrame(
            betas,
            index=returns_aligned.columns,
            columns=factors_aligned.columns
        )
        
        self.alphas = pd.Series(alphas, index=returns_aligned.columns)
        
        self.residuals = pd.DataFrame(
            residuals,
            index=common_dates,
            columns=returns_aligned.columns
        )
        
        self.r_squared = pd.Series(r_squared, index=returns_aligned.columns)
        
        results = {
            'betas': self.betas,
            'alphas': self.alphas,
            'residuals': self.residuals,
            'r_squared': self.r_squared,
            't_stats_beta': pd.DataFrame(
                t_stats_beta,
                index=returns_aligned.columns,
                columns=factors_aligned.columns
            ),
            't_stats_alpha': pd.Series(t_stats_alpha, index=returns_aligned.columns)
        }
        
        logger.info(f"Time-series regressions complete: {n_assets} assets, {n_factors} factors")
        logger.info(f"Mean R-squared: {r_squared.mean():.3f}")
        
        return results
    
    def estimate_cross_sectional_regression(self, returns: pd.DataFrame,
                                           betas: pd.DataFrame) -> Dict:
        """
        Estimate cross-sectional regression (Fama-MacBeth)
        r_i = lambda_0 + beta_i' lambda + eta_i
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (T x N)
        betas : pd.DataFrame
            Factor loadings (N x K)
        
        Returns:
        --------
        dict : Risk premia estimates
        """
        logger.info("Estimating cross-sectional regressions (Fama-MacBeth)...")
        
        n_periods = len(returns)
        n_factors = len(betas.columns)
        
        # Initialize arrays for period-by-period estimates
        lambdas = np.zeros((n_periods, n_factors))
        lambda_0 = np.zeros(n_periods)
        
        # Run cross-sectional regression for each period
        for t in range(n_periods):
            # Get returns for this period
            y = returns.iloc[t].values
            X = betas.values
            X_with_const = add_constant(X)
            
            # Handle missing values
            valid_idx = ~np.isnan(y)
            if valid_idx.sum() < n_factors + 1:
                continue
            
            y_valid = y[valid_idx]
            X_valid = X_with_const[valid_idx]
            
            # OLS regression
            try:
                params = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                lambda_0[t] = params[0]
                lambdas[t, :] = params[1:]
            except:
                continue
        
        # Average across periods (Fama-MacBeth)
        lambda_mean = lambdas.mean(axis=0)
        lambda_std = lambdas.std(axis=0)
        lambda_tstat = lambda_mean / (lambda_std / np.sqrt(n_periods))
        
        # Annualize
        lambda_mean_annual = lambda_mean * 252
        lambda_std_annual = lambda_std * np.sqrt(252)
        
        self.factor_premia = pd.DataFrame({
            'Risk_Premium': lambda_mean_annual,
            'Std_Error': lambda_std_annual,
            't_Statistic': lambda_tstat,
            'p_Value': 2 * (1 - stats.t.cdf(np.abs(lambda_tstat), n_periods - 1))
        }, index=betas.columns)
        
        logger.info("Cross-sectional regressions complete")
        
        return {
            'factor_premia': self.factor_premia,
            'lambda_series': pd.DataFrame(lambdas, columns=betas.columns, index=returns.index)
        }
    
    def estimate_factor_risk_premia(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate factor risk premia directly from factor returns
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        
        Returns:
        --------
        pd.DataFrame : Risk premia estimates
        """
        logger.info("Estimating factor risk premia from factor returns...")
        
        # Mean returns (annualized)
        mean_returns = factor_returns.mean() * 252
        
        # Standard errors (Newey-West HAC)
        std_errors = []
        t_stats = []
        
        for factor in factor_returns.columns:
            y = factor_returns[factor].values
            X = np.ones((len(y), 1))
            
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': self.regression_config['newey_west_lags']})
            
            std_error = model.bse[0] * np.sqrt(252)
            t_stat = model.tvalues[0]
            
            std_errors.append(std_error)
            t_stats.append(t_stat)
        
        risk_premia = pd.DataFrame({
            'Mean_Return': mean_returns,
            'Std_Error': std_errors,
            't_Statistic': t_stats,
            'Sharpe_Ratio': mean_returns / (factor_returns.std() * np.sqrt(252))
        }, index=factor_returns.columns)
        
        return risk_premia
    
    def rolling_regression(self, returns: pd.DataFrame,
                          factors: pd.DataFrame,
                          window: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate rolling factor loadings
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        factors : pd.DataFrame
            Factor returns
        window : int, optional
            Rolling window size
        
        Returns:
        --------
        pd.DataFrame : Rolling betas
        """
        if window is None:
            window = self.regression_config['estimation_window']
        
        logger.info(f"Estimating rolling regressions with window={window}...")
        
        # Align dates
        common_dates = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        # Initialize results
        rolling_betas = {}
        
        # For each asset, compute rolling betas
        for asset in returns_aligned.columns:
            asset_betas = []
            dates = []
            
            for i in range(window, len(returns_aligned)):
                window_returns = returns_aligned[asset].iloc[i-window:i]
                window_factors = factors_aligned.iloc[i-window:i]
                
                # OLS regression
                X = add_constant(window_factors.values)
                y = window_returns.values
                
                try:
                    params = np.linalg.lstsq(X, y, rcond=None)[0]
                    asset_betas.append(params[1:])  # Exclude intercept
                    dates.append(returns_aligned.index[i])
                except:
                    continue
            
            rolling_betas[asset] = pd.DataFrame(
                asset_betas,
                index=dates,
                columns=factors_aligned.columns
            )
        
        logger.info("Rolling regressions complete")
        
        return rolling_betas
    
    def compute_residual_diagnostics(self) -> Dict:
        """
        Compute diagnostics on regression residuals
        
        Returns:
        --------
        dict : Diagnostic statistics
        """
        if self.residuals is None:
            raise ValueError("Must run regression first")
        
        logger.info("Computing residual diagnostics...")
        
        diagnostics = {}
        
        # Mean (should be ~0)
        diagnostics['mean'] = self.residuals.mean().mean()
        
        # Volatility
        diagnostics['volatility'] = self.residuals.std().mean() * np.sqrt(252)
        
        # Autocorrelation
        diagnostics['autocorr_lag1'] = self.residuals.apply(
            lambda x: x.autocorr(lag=1)
        ).mean()
        
        # Normality test (Jarque-Bera)
        jb_stats = []
        jb_pvals = []
        for col in self.residuals.columns:
            stat, pval = stats.jarque_bera(self.residuals[col].dropna())
            jb_stats.append(stat)
            jb_pvals.append(pval)
        
        diagnostics['jarque_bera_stat'] = np.mean(jb_stats)
        diagnostics['jarque_bera_pval'] = np.mean(jb_pvals)
        
        # Cross-sectional correlation
        residual_corr = self.residuals.corr()
        diagnostics['mean_correlation'] = residual_corr.values[
            np.triu_indices_from(residual_corr.values, k=1)
        ].mean()
        
        return diagnostics
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save regression results
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        logger.info("Saving regression results...")
        
        if self.betas is not None:
            save_results(self.betas, 'factor_betas.csv', output_dir)
        
        if self.alphas is not None:
            save_results(self.alphas, 'factor_alphas.csv', output_dir)
        
        if self.residuals is not None:
            save_results(self.residuals, 'regression_residuals.csv', output_dir)
        
        if self.r_squared is not None:
            save_results(self.r_squared, 'r_squared.csv', output_dir)
        
        if self.factor_premia is not None:
            save_results(self.factor_premia, 'factor_risk_premia.csv', output_dir)
        
        # Save diagnostics
        if self.residuals is not None:
            diagnostics = self.compute_residual_diagnostics()
            save_results(diagnostics, 'residual_diagnostics.json', output_dir)
        
        logger.info("Regression results saved")


class FactorComparison:
    """
    Compare PCA factors vs classical factors
    """
    
    def __init__(self, config: dict):
        self.config = config
        
    def compare_factor_performance(self, pca_factors: pd.DataFrame,
                                   classical_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Compare performance of PCA vs classical factors
        
        Parameters:
        -----------
        pca_factors : pd.DataFrame
            PCA factor returns
        classical_factors : pd.DataFrame
            Classical factor returns
        
        Returns:
        --------
        pd.DataFrame : Comparison statistics
        """
        logger.info("Comparing PCA vs classical factors...")
        
        # Align dates
        common_dates = pca_factors.index.intersection(classical_factors.index)
        pca_aligned = pca_factors.loc[common_dates]
        classical_aligned = classical_factors.loc[common_dates]
        
        # Compute statistics for each set
        stats_pca = self._compute_factor_stats(pca_aligned, 'PCA')
        stats_classical = self._compute_factor_stats(classical_aligned, 'Classical')
        
        # Combine
        comparison = pd.concat([stats_pca, stats_classical])
        
        return comparison
    
    def _compute_factor_stats(self, factors: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Compute factor statistics"""
        stats = pd.DataFrame(index=factors.columns)
        stats['Type'] = prefix
        stats['Mean_Return'] = factors.mean() * 252
        stats['Volatility'] = factors.std() * np.sqrt(252)
        stats['Sharpe_Ratio'] = stats['Mean_Return'] / stats['Volatility']
        stats['Skewness'] = factors.skew()
        stats['Kurtosis'] = factors.kurtosis()
        
        return stats


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    # Load config and data
    config = load_config()
    ensure_directories(config)
    
    returns = pd.read_parquet(f"{config['paths']['data_processed']}/returns.parquet")
    pca_factors = pd.read_parquet(f"{config['paths']['results']}/pca_factor_returns.parquet")
    classical_factors = pd.read_parquet(f"{config['paths']['results']}/classical_factor_returns.parquet")
    
    # Estimate regressions
    regression_model = FactorRegressionModel(config)
    
    # Time-series regression with PCA factors
    ts_results = regression_model.estimate_time_series_regression(returns, pca_factors)
    
    # Estimate risk premia
    pca_premia = regression_model.estimate_factor_risk_premia(pca_factors)
    classical_premia = regression_model.estimate_factor_risk_premia(classical_factors)
    
    # Save results
    regression_model.save_results(config['paths']['results'])
    
    print("\nPCA Factor Risk Premia:")
    print(pca_premia.round(3))
    print("\nClassical Factor Risk Premia:")
    print(classical_premia.round(3))
