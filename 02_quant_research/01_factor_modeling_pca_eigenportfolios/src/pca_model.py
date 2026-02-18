"""
PCA-Based Factor Model
Implements eigen-decomposition, factor extraction, and eigen-portfolio construction

FINAL VERSION - DO NOT MODIFY
advanced methodology: Standardize for covariance, apply to raw returns
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import logging
from scipy import linalg
from sklearn.decomposition import PCA

from utils import save_results, standardize

logger = logging.getLogger(__name__)


class PCAFactorModel:
    """
    Principal Component Analysis for factor extraction
    Implements eigen-decomposition and eigen-portfolio construction
    """
    
    def __init__(self, config: dict):
        """
        Initialize PCA factor model
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.pca_config = config['pca']
        self.n_components = self.pca_config['n_components']
        
        # Results storage
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance = None
        self.cumulative_variance = None
        self.factor_returns = None
        self.eigen_portfolios = None
        self.loadings = None
        
    def fit(self, returns: pd.DataFrame) -> 'PCAFactorModel':
        """
        Fit PCA model to return data
        
        advanced METHODOLOGY:
        - Perform PCA on standardized returns to extract covariance structure
        - Use eigenvectors to construct eigen-portfolios
        - Compute factor returns using RAW returns (not standardized)
        - This ensures economically interpretable factor returns with meaningful premia
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (T x N) - raw excess returns
        
        Returns:
        --------
        self : PCAFactorModel
        """
        logger.info("Fitting PCA model...")
        
        # Remove any NaN or inf values
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Store raw returns for factor return computation
        self.raw_returns = returns_clean
        
        # Standardize returns ONLY for covariance estimation
        # This ensures PCA captures correlation structure, not scale effects
        returns_std = standardize(returns_clean, method='zscore')
        
        # Replace any remaining NaN with 0
        returns_std = returns_std.fillna(0)
        
        # Compute covariance matrix on standardized returns
        cov_matrix = returns_std.cov()
        
        # Eigen-decomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Explained variance
        total_variance = self.eigenvalues.sum()
        self.explained_variance = self.eigenvalues / total_variance
        self.cumulative_variance = np.cumsum(self.explained_variance)
        
        # Keep top n_components
        self.eigenvalues = self.eigenvalues[:self.n_components]
        self.eigenvectors = self.eigenvectors[:, :self.n_components]
        self.explained_variance = self.explained_variance[:self.n_components]
        
        logger.info(f"PCA fitted: {self.n_components} components explain "
                   f"{self.cumulative_variance[self.n_components-1]:.2%} of variance")
        
        # Construct eigen-portfolios
        self._construct_eigen_portfolios(returns_clean.columns)
        
        # CRITICAL: Compute factor returns using RAW returns
        # This preserves economic interpretation and allows meaningful Sharpe ratios
        self._compute_factor_returns(returns_clean)
        
        # Compute factor loadings
        self._compute_loadings(returns_clean)
        
        return self
    
    def _construct_eigen_portfolios(self, asset_names: pd.Index) -> None:
        """
        Construct eigen-portfolios from eigenvectors
        
        Parameters:
        -----------
        asset_names : pd.Index
            Asset names
        """
        logger.info("Constructing eigen-portfolios...")
        
        # Eigenvectors are the portfolio weights
        eigen_portfolios = pd.DataFrame(
            self.eigenvectors,
            index=asset_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # Normalize weights if configured
        if self.pca_config['normalize_weights']:
            # L2 normalization (already done by eigen-decomposition)
            # Optionally apply weight cap
            weight_cap = self.pca_config.get('weight_cap', None)
            if weight_cap:
                eigen_portfolios = eigen_portfolios.clip(-weight_cap, weight_cap)
                # Re-normalize after capping
                eigen_portfolios = eigen_portfolios.div(
                    eigen_portfolios.abs().sum(axis=0), axis=1
                )
        
        self.eigen_portfolios = eigen_portfolios
        
        logger.info(f"Eigen-portfolios constructed: {eigen_portfolios.shape}")
    
    def _compute_factor_returns(self, returns_raw: pd.DataFrame) -> None:
        """
        Compute time series of factor returns
        
        CRITICAL: Uses RAW returns, not standardized returns
        This ensures factor returns have economically meaningful scale,
        allowing proper interpretation of risk premia and Sharpe ratios.
        
        Parameters:
        -----------
        returns_raw : pd.DataFrame
            Raw excess returns (not standardized)
        """
        logger.info("Computing factor returns from raw excess returns...")
        
        # Ensure returns and eigenvectors are aligned
        # Factor returns = Raw Returns @ Eigenvectors
        # Eigenvectors capture covariance structure from standardized returns
        # But we apply them to raw returns to preserve economic scale
        
        # Convert to numpy for matrix multiplication
        returns_array = returns_raw.values
        eigenvectors_array = self.eigenvectors
        
        # Matrix multiplication: (T x N) @ (N x K) = (T x K)
        factor_returns_array = returns_array @ eigenvectors_array
        
        self.factor_returns = pd.DataFrame(
            factor_returns_array,
            index=returns_raw.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        logger.info(f"Factor returns computed: {self.factor_returns.shape}")
        
        # Log annualized means for verification
        annual_means = self.factor_returns.mean() * 252
        logger.info(f"Factor return means (annualized): {annual_means.round(4).to_dict()}")
    
    def _compute_loadings(self, returns: pd.DataFrame) -> None:
        """
        Compute factor loadings (betas) for each asset
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        """
        logger.info("Computing factor loadings...")
        
        # Loadings = Eigenvectors * sqrt(Eigenvalues)
        # This gives the correlation between assets and factors
        loadings = self.eigenvectors * np.sqrt(self.eigenvalues)
        
        self.loadings = pd.DataFrame(
            loadings,
            index=returns.columns,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        logger.info(f"Factor loadings computed: {self.loadings.shape}")
    
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Project returns onto principal components
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        pd.DataFrame : Factor returns
        """
        returns_std = standardize(returns, method='zscore')
        factor_returns = returns_std @ self.eigenvectors
        
        return pd.DataFrame(
            factor_returns,
            index=returns.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
    
    def reconstruct_returns(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """
        Reconstruct returns using n principal components
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to use (default: all)
        
        Returns:
        --------
        pd.DataFrame : Reconstructed returns
        """
        if n_components is None:
            n_components = self.n_components
        
        # Reconstructed returns = Factor returns @ Eigenvectors.T
        reconstructed = self.factor_returns.iloc[:, :n_components] @ \
                       self.eigenvectors[:, :n_components].T
        
        return pd.DataFrame(
            reconstructed,
            index=self.factor_returns.index,
            columns=self.eigen_portfolios.index
        )
    
    def compute_residuals(self, returns: pd.DataFrame,
                         n_components: Optional[int] = None) -> pd.DataFrame:
        """
        Compute idiosyncratic returns (residuals)
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Original returns
        n_components : int, optional
            Number of components to use
        
        Returns:
        --------
        pd.DataFrame : Residual returns
        """
        returns_std = standardize(returns, method='zscore')
        reconstructed = self.reconstruct_returns(n_components)
        
        residuals = returns_std - reconstructed
        
        return residuals
    
    def rolling_pca(self, returns: pd.DataFrame,
                   window: Optional[int] = None) -> pd.DataFrame:
        """
        Compute rolling PCA to assess factor stability
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        window : int, optional
            Rolling window size (default: from config)
        
        Returns:
        --------
        pd.DataFrame : Rolling explained variance
        """
        if window is None:
            window = self.pca_config['rolling_window']
        
        logger.info(f"Computing rolling PCA with window={window}...")
        
        rolling_variance = []
        dates = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            # Fit PCA on window
            returns_std = standardize(window_returns, method='zscore')
            cov_matrix = returns_std.cov()
            eigenvalues = linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Explained variance
            explained_var = eigenvalues[:self.n_components] / eigenvalues.sum()
            
            rolling_variance.append(explained_var)
            dates.append(returns.index[i])
        
        rolling_variance_df = pd.DataFrame(
            rolling_variance,
            index=dates,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        logger.info("Rolling PCA complete")
        
        return rolling_variance_df
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of PCA model
        
        Returns:
        --------
        dict : Summary statistics
        """
        summary = {
            'n_components': self.n_components,
            'total_variance_explained': self.cumulative_variance[self.n_components-1],
            'eigenvalues': self.eigenvalues.tolist(),
            'explained_variance': self.explained_variance.tolist(),
            'factor_sharpe_ratios': (
                self.factor_returns.mean() / self.factor_returns.std() * np.sqrt(252)
            ).to_dict(),
            'factor_correlations': self.factor_returns.corr().values.tolist()
        }
        
        return summary
    
    def evaluate_factors_Advanced(self) -> dict:
        """
        advanced EVALUATION FRAMEWORK
        
        Evaluate PCA factors using three distinct lenses:
        A) Statistical importance (variance explained, stability)
        B) Explanatory power (R², incremental fit)
        C) Economic relevance (returns, Sharpe, regime behavior)
        
        This framework recognizes that:
        - Not all PCA factors earn premia
        - Some are pure risk factors (zero expected return)
        - Statistical importance ≠ economic profitability
        
        Returns:
        --------
        dict : Multi-lens evaluation metrics
        """
        evaluation = {}
        
        # A) STATISTICAL IMPORTANCE
        evaluation['statistical'] = {
            'eigenvalues': self.eigenvalues.tolist(),
            'variance_explained': self.explained_variance.tolist(),
            'cumulative_variance': self.cumulative_variance[:self.n_components].tolist(),
            'interpretation': 'Measures covariance structure, not expected returns'
        }
        
        # B) EXPLANATORY POWER (computed later in regression)
        evaluation['explanatory'] = {
            'note': 'R² and incremental fit computed in time-series regression',
            'interpretation': 'Measures ability to explain cross-sectional variation'
        }
        
        # C) ECONOMIC RELEVANCE
        factor_stats = {}
        for col in self.factor_returns.columns:
            returns = self.factor_returns[col]
            factor_stats[col] = {
                'mean_return_daily': float(returns.mean()),
                'mean_return_annual': float(returns.mean() * 252),
                'volatility_daily': float(returns.std()),
                'volatility_annual': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0,
                'interpretation': 'PC1 typically = market factor; others may have zero premia'
            }
        
        evaluation['economic'] = factor_stats
        
        # advanced INTERPRETATION GUIDE
        evaluation['interpretation_guide'] = {
            'statistical_vs_economic': 'High variance explained does NOT imply high Sharpe ratio',
            'zero_premia_factors': 'Many PCA factors are risk factors with zero expected return',
            'pc1_interpretation': 'PC1 typically captures market factor (positive premia expected)',
            'higher_pcs': 'PC2+ capture sector/style rotations (may have zero or negative premia)',
            'professional_use': 'PCA factors used for risk decomposition, not alpha generation'
        }
        
        return evaluation
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save PCA results
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        logger.info("Saving PCA results...")
        
        save_results(self.eigen_portfolios, 'eigen_portfolios.csv', output_dir)
        save_results(self.factor_returns, 'pca_factor_returns.csv', output_dir)
        save_results(self.loadings, 'factor_loadings.csv', output_dir)
        
        # Save summary
        summary = self.get_summary()
        save_results(summary, 'pca_summary.json', output_dir)
        
        # Save advanced evaluation framework
        Advanced_eval = self.evaluate_factors_Advanced()
        save_results(Advanced_eval, 'pca_Advanced_evaluation.json', output_dir)
        
        # Save eigenvalues and variance explained
        variance_df = pd.DataFrame({
            'Eigenvalue': self.eigenvalues,
            'Explained_Variance': self.explained_variance,
            'Cumulative_Variance': self.cumulative_variance[:self.n_components]
        }, index=[f'PC{i+1}' for i in range(self.n_components)])
        
        save_results(variance_df, 'pca_variance_explained.csv', output_dir)
        
        logger.info("PCA results saved")


class AdaptivePCA:
    """
    Adaptive PCA with automatic component selection
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.pca_config = config['pca']
        self.min_variance = self.pca_config['min_variance_explained']
        
    def select_components(self, returns: pd.DataFrame) -> int:
        """
        Automatically select number of components
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        
        Returns:
        --------
        int : Optimal number of components
        """
        returns_std = standardize(returns, method='zscore')
        cov_matrix = returns_std.cov()
        
        eigenvalues = linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        explained_variance = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components to reach target variance
        n_components = np.argmax(cumulative_variance >= self.min_variance) + 1
        
        logger.info(f"Selected {n_components} components to explain "
                   f"{cumulative_variance[n_components-1]:.2%} of variance")
        
        return n_components


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    # Load config and data
    config = load_config()
    ensure_directories(config)
    
    returns = pd.read_parquet(f"{config['paths']['data_processed']}/returns.parquet")
    
    # Fit PCA model
    pca_model = PCAFactorModel(config)
    pca_model.fit(returns)
    
    # Save results
    pca_model.save_results(config['paths']['results'])
    
    # Print summary
    summary = pca_model.get_summary()
    print("\nPCA Model Summary:")
    print(f"Components: {summary['n_components']}")
    print(f"Variance explained: {summary['total_variance_explained']:.2%}")
    print("\nFactor Sharpe Ratios:")
    for factor, sharpe in summary['factor_sharpe_ratios'].items():
        print(f"  {factor}: {sharpe:.2f}")
