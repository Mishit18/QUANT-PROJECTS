"""
Visualization Module
Professional plots for factor analysis and research reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class FactorVisualizer:
    """
    Create professional visualizations for factor analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.plot_dir = Path(config['paths']['plots'])
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config['analysis'].get('figure_dpi', 300)
        
    def plot_scree(self, eigenvalues: np.ndarray,
                  explained_variance: np.ndarray,
                  save_name: str = 'scree_plot.png') -> None:
        """
        Plot scree plot of eigenvalues
        
        Parameters:
        -----------
        eigenvalues : np.ndarray
            Eigenvalues from PCA
        explained_variance : np.ndarray
            Explained variance ratios
        save_name : str
            Filename to save
        """
        logger.info("Creating scree plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Eigenvalue plot
        n_components = len(eigenvalues)
        ax1.bar(range(1, n_components + 1), eigenvalues, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Scree Plot: Eigenvalues')
        ax1.grid(True, alpha=0.3)
        
        # Explained variance plot
        cumulative_var = np.cumsum(explained_variance)
        ax2.plot(range(1, n_components + 1), explained_variance * 100, 
                'o-', label='Individual', color='steelblue', linewidth=2)
        ax2.plot(range(1, n_components + 1), cumulative_var * 100,
                's-', label='Cumulative', color='coral', linewidth=2)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.set_title('Variance Explained by Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scree plot saved: {save_name}")
    
    def plot_eigen_portfolio_weights(self, eigen_portfolios: pd.DataFrame,
                                    n_components: int = 5,
                                    save_name: str = 'eigen_portfolio_weights.png') -> None:
        """
        Plot heatmap of eigen-portfolio weights
        
        Parameters:
        -----------
        eigen_portfolios : pd.DataFrame
            Eigen-portfolio weights
        n_components : int
            Number of components to plot
        save_name : str
            Filename to save
        """
        logger.info("Creating eigen-portfolio weight heatmap...")
        
        # Select top components
        weights = eigen_portfolios.iloc[:, :n_components]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(weights, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Weight'},
                   ax=ax, vmin=-0.2, vmax=0.2)
        ax.set_title('Eigen-Portfolio Weights (First 5 Components)')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Asset')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Eigen-portfolio weights saved: {save_name}")
    
    def plot_factor_returns(self, factor_returns: pd.DataFrame,
                           title: str = 'Factor Returns',
                           save_name: str = 'factor_returns.png') -> None:
        """
        Plot cumulative factor returns
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        logger.info(f"Creating factor returns plot: {title}...")
        
        # Compute cumulative returns
        cum_returns = (1 + factor_returns).cumprod()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for col in cum_returns.columns:
            ax.plot(cum_returns.index, cum_returns[col], label=col, linewidth=1.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(title)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Factor returns plot saved: {save_name}")
    
    def plot_factor_statistics(self, stats: pd.DataFrame,
                              metric: str = 'Sharpe_Ratio',
                              save_name: str = 'factor_sharpe_ratios.png') -> None:
        """
        Plot bar chart of factor statistics
        
        Parameters:
        -----------
        stats : pd.DataFrame
            Factor statistics
        metric : str
            Metric to plot
        save_name : str
            Filename to save
        """
        logger.info(f"Creating factor statistics plot: {metric}...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stats[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_xlabel('Factor')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'Factor {metric.replace("_", " ")}')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Factor statistics plot saved: {save_name}")
    
    def plot_factor_correlation(self, factor_returns: pd.DataFrame,
                               title: str = 'Factor Correlation Matrix',
                               save_name: str = 'factor_correlation.png') -> None:
        """
        Plot factor correlation heatmap
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        logger.info("Creating factor correlation heatmap...")
        
        corr = factor_returns.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Factor correlation plot saved: {save_name}")
    
    def plot_regime_comparison(self, regime_stats: pd.DataFrame,
                              metric: str = 'Sharpe_Ratio',
                              save_name: str = 'regime_comparison.png') -> None:
        """
        Plot factor performance across regimes
        
        Parameters:
        -----------
        regime_stats : pd.DataFrame
            Regime statistics
        metric : str
            Metric to compare
        save_name : str
            Filename to save
        """
        logger.info(f"Creating regime comparison plot: {metric}...")
        
        # Pivot data for plotting
        plot_data = regime_stats.reset_index()
        plot_data = plot_data.pivot(index='level_1', columns='Regime', values=metric)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Factor')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} by Regime')
        ax.legend(title='Regime')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Regime comparison plot saved: {save_name}")
    
    def plot_rolling_sharpe(self, factor_returns: pd.DataFrame,
                           window: int = 252,
                           save_name: str = 'rolling_sharpe.png') -> None:
        """
        Plot rolling Sharpe ratios
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        window : int
            Rolling window size
        save_name : str
            Filename to save
        """
        logger.info("Creating rolling Sharpe ratio plot...")
        
        # Compute rolling Sharpe
        rolling_sharpe = factor_returns.rolling(window=window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252)
        )
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for col in rolling_sharpe.columns:
            ax.plot(rolling_sharpe.index, rolling_sharpe[col], 
                   label=col, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.set_title(f'Rolling Sharpe Ratio ({window}-day window)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Rolling Sharpe plot saved: {save_name}")
    
    def plot_drawdown(self, factor_returns: pd.DataFrame,
                     save_name: str = 'drawdown.png') -> None:
        """
        Plot drawdown analysis
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        save_name : str
            Filename to save
        """
        logger.info("Creating drawdown plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Cumulative returns
        cum_returns = (1 + factor_returns).cumprod()
        for col in cum_returns.columns:
            axes[0].plot(cum_returns.index, cum_returns[col], 
                        label=col, linewidth=1.5)
        
        axes[0].set_ylabel('Cumulative Return')
        axes[0].set_title('Cumulative Returns')
        axes[0].legend(loc='best', ncol=2)
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        for col in factor_returns.columns:
            cum_ret = (1 + factor_returns[col]).cumprod()
            running_max = cum_ret.expanding().max()
            drawdown = (cum_ret - running_max) / running_max
            axes[1].fill_between(drawdown.index, drawdown, 0, 
                                alpha=0.3, label=col)
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown')
        axes[1].set_title('Drawdown')
        axes[1].legend(loc='best', ncol=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drawdown plot saved: {save_name}")
    
    def plot_risk_return_scatter(self, stats: pd.DataFrame,
                                save_name: str = 'risk_return_scatter.png') -> None:
        """
        Plot risk-return scatter
        
        Parameters:
        -----------
        stats : pd.DataFrame
            Factor statistics with Mean_Return and Volatility
        save_name : str
            Filename to save
        """
        logger.info("Creating risk-return scatter plot...")
        
        # Filter out rows with NaN values
        stats_clean = stats.dropna(subset=['Volatility', 'Mean_Return', 'Sharpe_Ratio'])
        
        if len(stats_clean) == 0:
            logger.warning(f"No valid data for scatter plot {save_name}, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(stats_clean['Volatility'], stats_clean['Mean_Return'], 
                  s=100, alpha=0.6, c=stats_clean['Sharpe_Ratio'],
                  cmap='viridis')
        
        # Add labels
        for idx, row in stats_clean.iterrows():
            ax.annotate(idx, (row['Volatility'], row['Mean_Return']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
        
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Mean Return (Annualized)')
        ax.set_title('Risk-Return Profile')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(stats_clean['Sharpe_Ratio'])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Risk-return scatter saved: {save_name}")
    
    def create_all_plots(self, pca_model, classical_factors_obj,
                        regime_analyzer) -> None:
        """
        Create all standard plots
        
        Parameters:
        -----------
        pca_model : PCAFactorModel
            Fitted PCA model
        classical_factors_obj : ClassicalFactors
            Classical factors object
        regime_analyzer : RegimeAnalyzer
            Regime analyzer object
        """
        logger.info("=" * 60)
        logger.info("CREATING ALL VISUALIZATIONS")
        logger.info("=" * 60)
        
        # PCA plots
        self.plot_scree(pca_model.eigenvalues, pca_model.explained_variance)
        self.plot_eigen_portfolio_weights(pca_model.eigen_portfolios)
        self.plot_factor_returns(pca_model.factor_returns, 
                                'PCA Factor Returns',
                                'pca_factor_returns.png')
        
        # Classical factor plots
        classical_returns = pd.DataFrame(classical_factors_obj.factor_returns)
        self.plot_factor_returns(classical_returns,
                                'Classical Factor Returns',
                                'classical_factor_returns.png')
        
        # Statistics plots
        pca_stats = classical_factors_obj.compute_factor_statistics(pca_model.factor_returns)
        classical_stats = classical_factors_obj.compute_factor_statistics(classical_returns)
        
        self.plot_factor_statistics(pca_stats, 'Sharpe_Ratio', 'pca_sharpe_ratios.png')
        self.plot_factor_statistics(classical_stats, 'Sharpe_Ratio', 'classical_sharpe_ratios.png')
        
        # Correlation plots
        self.plot_factor_correlation(pca_model.factor_returns, 
                                    'PCA Factor Correlations',
                                    'pca_correlation.png')
        self.plot_factor_correlation(classical_returns,
                                    'Classical Factor Correlations',
                                    'classical_correlation.png')
        
        # Drawdown plots
        self.plot_drawdown(pca_model.factor_returns.iloc[:, :5], 'pca_drawdown.png')
        self.plot_drawdown(classical_returns, 'classical_drawdown.png')
        
        # Risk-return scatter
        self.plot_risk_return_scatter(pca_stats, 'pca_risk_return.png')
        self.plot_risk_return_scatter(classical_stats, 'classical_risk_return.png')
        
        logger.info("=" * 60)
        logger.info("ALL VISUALIZATIONS COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    import pandas as pd
    
    config = load_config()
    ensure_directories(config)
    
    visualizer = FactorVisualizer(config)
    
    # Load data
    pca_factors = pd.read_parquet(f"{config['paths']['results']}/pca_factor_returns.parquet")
    
    # Create sample plots
    visualizer.plot_factor_returns(pca_factors, 'PCA Factors', 'test_plot.png')
    
    print("Visualization test complete")
