# src/visualizations.py
"""
Generate diagnostic plots and visualizations for the pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sample_for_plot(df: pd.DataFrame, max_points: int = 100_000) -> pd.DataFrame:
    """Downsample large intraday dataframes for diagnostic plotting."""
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def plot_cleaning_diagnostics(df: pd.DataFrame, output_dir: Path):
    """
    Plot before/after cleaning diagnostics.
    """
    df = _sample_for_plot(df)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Price with flagged points
    axes[0].plot(df.index, df['close'], linewidth=0.5, alpha=0.7, label='Original')
    if 'flag_bad_tick' in df.columns:
        flagged = df[df['flag_bad_tick']]
        axes[0].scatter(flagged.index, flagged['close'], 
                       c='red', s=10, alpha=0.5, label='Flagged')
    axes[0].set_title('Price Series with Flagged Outliers')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Returns distribution
    if 'log_return' in df.columns:
        returns = df['log_return'].dropna()
        axes[1].hist(returns, bins=100, alpha=0.7, edgecolor='black')
        axes[1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.6f}')
        axes[1].axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.6f}')
        axes[1].set_title('Log Returns Distribution')
        axes[1].set_xlabel('Log Return')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling volatility
    if 'log_return' in df.columns:
        rolling_vol = df['log_return'].rolling(60).std()
        axes[2].plot(df.index, rolling_vol, linewidth=0.7)
        axes[2].set_title('Rolling Volatility (60-period)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Volatility')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'cleaning_diagnostics.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved cleaning diagnostics to {output_path}")


def plot_stationarity_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Compare price levels vs returns (stationarity).
    """
    try:
        plot_df = _sample_for_plot(df)
        acf_df = df.tail(50_000)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price level
        axes[0, 0].plot(plot_df.index, plot_df['close'], linewidth=0.7)
        axes[0, 0].set_title('Price Level (Non-Stationary)')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price ACF
        if 'close' in df.columns:
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(acf_df['close'].dropna(), lags=40, ax=axes[0, 1])
                axes[0, 1].set_title('Price ACF')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'ACF plot failed: {str(e)}', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Returns
        if 'log_return' in df.columns:
            axes[1, 0].plot(plot_df.index, plot_df['log_return'], linewidth=0.5, alpha=0.7)
            axes[1, 0].set_title('Log Returns (Stationary)')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Returns ACF
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(acf_df['log_return'].dropna(), lags=40, ax=axes[1, 1])
                axes[1, 1].set_title('Returns ACF')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'ACF plot failed: {str(e)}', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        output_path = output_dir / 'stationarity_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved stationarity comparison to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create stationarity comparison plot: {e}")


def plot_hp_filter_decomposition(df: pd.DataFrame, output_dir: Path):
    """
    Plot HP filter decomposition results.
    """
    if 'close_trend' not in df.columns:
        logger.warning("HP filter decomposition not found in DataFrame")
        return
    df = _sample_for_plot(df)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original
    axes[0].plot(df.index, df['close'], linewidth=0.7)
    axes[0].set_title('Original Price Series')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(df.index, df['close_trend'], linewidth=0.7, color='orange')
    axes[1].set_title('HP Filter: Trend Component')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Cycle
    if 'close_cycle' in df.columns:
        axes[2].plot(df.index, df['close_cycle'], linewidth=0.7, color='green')
        axes[2].set_title('HP Filter: Cyclical Component')
        axes[2].set_ylabel('Cycle')
        axes[2].grid(True, alpha=0.3)
    
    # Detrended (residual)
    axes[3].plot(df.index, df['close_detrended'], linewidth=0.5, alpha=0.7, color='red')
    axes[3].set_title('HP Filter: Detrended (Stationary Residual)')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'hp_filter_decomposition.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved HP filter decomposition to {output_path}")


def plot_feature_correlations(df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """
    Plot correlation heatmap of top features.
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude index and flag columns
    feature_cols = [col for col in numeric_cols 
                   if not col.startswith('flag_')
                   and not col.startswith('target_')
                   and not col.endswith('_raw')
                   and col not in ['hour', 'day_of_week', 'minute']]
    
    if len(feature_cols) == 0:
        logger.warning("No feature columns found")
        return
    
    df = _sample_for_plot(df[feature_cols], max_points=150_000)
    feature_cols = df.columns.tolist()
    
    # Select top N features by variance
    variances = df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    # Compute correlation matrix
    corr_matrix = df[top_features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(top_features, fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    ax.set_title(f'Feature Correlation Heatmap (Top {top_n} by Variance)')
    
    plt.tight_layout()
    output_path = output_dir / 'feature_correlations.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved feature correlations to {output_path}")


def plot_backtest_results(df: pd.DataFrame, output_dir: Path):
    """
    Plot backtest equity curve and drawdown.
    """
    if 'equity' not in df.columns:
        logger.warning("Backtest results not found in DataFrame")
        return
    df = _sample_for_plot(df)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Equity curve
    axes[0].plot(df.index, df['equity'], linewidth=0.8)
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('Equity ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Returns
    if 'returns' in df.columns:
        axes[1].plot(df.index, df['returns'], linewidth=0.5, alpha=0.7)
        axes[1].set_title('Strategy Returns')
        axes[1].set_ylabel('Return')
        axes[1].grid(True, alpha=0.3)
    
    # Drawdown
    if 'returns' in df.columns:
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[2].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        axes[2].plot(df.index, drawdown, linewidth=0.8, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'backtest_equity_curve.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved backtest results to {output_path}")


def generate_all_plots():
    """
    Generate all diagnostic plots.
    """
    from config import CLEANED_PARQUET, STATIONARY_PARQUET, FEATURES_PARQUET, PLOTS_DIR, REPORTS_DIR
    
    logger.info("\n" + "="*60)
    logger.info("GENERATING DIAGNOSTIC PLOTS")
    logger.info("="*60 + "\n")
    
    # Cleaning diagnostics
    if CLEANED_PARQUET.exists():
        logger.info("Generating cleaning diagnostics...")
        df = pd.read_parquet(CLEANED_PARQUET)
        plot_cleaning_diagnostics(df, PLOTS_DIR)
        plot_stationarity_comparison(df, PLOTS_DIR)
    
    # HP filter decomposition
    if STATIONARY_PARQUET.exists():
        logger.info("Generating HP filter decomposition plots...")
        df = pd.read_parquet(STATIONARY_PARQUET)
        plot_hp_filter_decomposition(df, PLOTS_DIR)
    
    # Feature correlations
    if FEATURES_PARQUET.exists():
        logger.info("Generating feature correlation plots...")
        df = pd.read_parquet(FEATURES_PARQUET)
        plot_feature_correlations(df, PLOTS_DIR, top_n=20)
    
    # Backtest results
    backtest_file = REPORTS_DIR / 'backtest_results.parquet'
    if backtest_file.exists():
        logger.info("Generating backtest plots...")
        df = pd.read_parquet(backtest_file)
        plot_backtest_results(df, PLOTS_DIR)
    
    logger.info("\n" + "="*60)
    logger.info(f"All plots saved to {PLOTS_DIR}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    generate_all_plots()
