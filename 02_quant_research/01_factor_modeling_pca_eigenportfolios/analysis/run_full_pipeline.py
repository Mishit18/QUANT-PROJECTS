"""
Full Factor Modeling Pipeline
Executes complete analysis from data to results

FINAL VERSION - DO NOT MODIFY
Single command execution: python analysis/run_full_pipeline.py
"""

import sys
sys.path.append('src')

import logging
from pathlib import Path

from utils import load_config, ensure_directories
from data_pipeline import EquityDataPipeline
from pca_model import PCAFactorModel
from factor_construction import ClassicalFactors
from regression import FactorRegressionModel, FactorComparison
from regime_analysis import RegimeAnalyzer
from portfolio_controls import PortfolioControls, RiskMetrics
from visualization import FactorVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Execute full factor modeling pipeline"""
    
    logger.info("=" * 80)
    logger.info("SYSTEMATIC FACTOR MODELING PIPELINE")
    logger.info("PCA, Eigen-Portfolios, and Cross-Sectional Return Decomposition")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: DATA PIPELINE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA ACQUISITION AND PREPROCESSING")
    logger.info("=" * 80)
    
    data_pipeline = EquityDataPipeline(config)
    prices, returns, excess_returns, diagnostics = data_pipeline.run()
    market_returns = data_pipeline.market_data
    
    logger.info(f"\nData Summary:")
    logger.info(f"  Assets: {len(returns.columns)}")
    logger.info(f"  Time period: {returns.index[0]} to {returns.index[-1]}")
    logger.info(f"  Trading days: {len(returns)}")
    logger.info(f"  Mean annual return: {diagnostics['mean_return']:.2%}")
    logger.info(f"  Mean volatility: {diagnostics['mean_volatility']:.2%}")
    
    # ========================================================================
    # STEP 2: PCA FACTOR MODEL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PCA FACTOR EXTRACTION")
    logger.info("=" * 80)
    
    pca_model = PCAFactorModel(config)
    pca_model.fit(returns)
    pca_model.save_results(config['paths']['results'])
    
    summary = pca_model.get_summary()
    logger.info(f"\nPCA Summary:")
    logger.info(f"  Components: {summary['n_components']}")
    logger.info(f"  Variance explained: {summary['total_variance_explained']:.2%}")
    
    # advanced evaluation
    Advanced_eval = pca_model.evaluate_factors_Advanced()
    logger.info(f"\nPCA Factor Evaluation (advanced Framework):")
    logger.info(f"  Statistical: {summary['n_components']} components, {summary['total_variance_explained']:.2%} variance")
    logger.info(f"  Economic: PC1 Sharpe = {Advanced_eval['economic']['PC1']['sharpe_ratio']:.2f}")
    logger.info(f"  Note: Not all PCA factors expected to earn premia (see advanced evaluation)")
    
    logger.info(f"\nTop 3 Factor Sharpe Ratios:")
    for i, (factor, sharpe) in enumerate(list(summary['factor_sharpe_ratios'].items())[:3]):
        logger.info(f"  {factor}: {sharpe:.2f}")
    
    # Compute residuals
    residuals = pca_model.compute_residuals(returns)
    residuals.to_parquet(f"{config['paths']['results']}/idiosyncratic_returns.parquet")
    
    # Rolling PCA for stability analysis
    rolling_variance = pca_model.rolling_pca(returns)
    rolling_variance.to_csv(f"{config['paths']['results']}/rolling_pca_variance.csv")
    
    # ========================================================================
    # STEP 3: CLASSICAL FACTORS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CLASSICAL FACTOR CONSTRUCTION")
    logger.info("=" * 80)
    
    classical_factors = ClassicalFactors(config)
    classical_factor_returns = classical_factors.construct_all_factors(prices, returns)
    classical_factors.save_results(classical_factor_returns, config['paths']['results'])
    
    classical_stats = classical_factors.compute_factor_statistics(classical_factor_returns)
    logger.info(f"\nClassical Factor Statistics:")
    for idx in classical_stats.index:
        logger.info(f"  {idx}: Return={classical_stats.loc[idx, 'Mean_Return']:.3f}, "
                   f"Sharpe={classical_stats.loc[idx, 'Sharpe_Ratio']:.3f}")
    
    # ========================================================================
    # STEP 4: FACTOR REGRESSION AND RISK PREMIA
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: FACTOR MODEL REGRESSION")
    logger.info("=" * 80)
    
    regression_model = FactorRegressionModel(config)
    
    # Time-series regression with PCA factors
    logger.info("\nEstimating time-series regressions with PCA factors...")
    ts_results_pca = regression_model.estimate_time_series_regression(
        returns, pca_model.factor_returns
    )
    
    # Time-series regression with classical factors
    logger.info("\nEstimating time-series regressions with classical factors...")
    ts_results_classical = regression_model.estimate_time_series_regression(
        returns, classical_factor_returns
    )
    
    # Estimate risk premia
    logger.info("\nEstimating factor risk premia...")
    pca_premia = regression_model.estimate_factor_risk_premia(pca_model.factor_returns)
    classical_premia = regression_model.estimate_factor_risk_premia(classical_factor_returns)
    
    regression_model.save_results(config['paths']['results'])
    
    logger.info(f"\nPCA Factor Risk Premia:")
    for idx in pca_premia.index:
        logger.info(f"  {idx}: {pca_premia.loc[idx, 'Mean_Return']:.3f}")
    logger.info(f"\nClassical Factor Risk Premia:")
    for idx in classical_premia.index:
        logger.info(f"  {idx}: {classical_premia.loc[idx, 'Mean_Return']:.3f}")
    
    # Compare PCA vs Classical
    factor_comparison = FactorComparison(config)
    comparison = factor_comparison.compare_factor_performance(
        pca_model.factor_returns, classical_factor_returns
    )
    comparison.to_csv(f"{config['paths']['results']}/factor_comparison.csv")
    
    # ========================================================================
    # STEP 5: REGIME ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: REGIME-DEPENDENT ANALYSIS")
    logger.info("=" * 80)
    
    regime_analyzer = RegimeAnalyzer(config)
    regimes = regime_analyzer.identify_all_regimes(returns, market_returns)
    
    # Analyze PCA factors by regime
    logger.info("\nAnalyzing PCA factors by regime...")
    pca_regime_stats = regime_analyzer.analyze_all_regimes(pca_model.factor_returns)
    
    # Analyze classical factors by regime
    logger.info("\nAnalyzing classical factors by regime...")
    classical_regime_stats = regime_analyzer.analyze_all_regimes(classical_factor_returns)
    
    regime_analyzer.save_results(config['paths']['results'])
    
    # ========================================================================
    # STEP 6: PORTFOLIO CONTROLS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: PORTFOLIO CONTROLS AND RISK MANAGEMENT")
    logger.info("=" * 80)
    
    portfolio_controls = PortfolioControls(config)
    risk_metrics = RiskMetrics(config)
    
    # Apply volatility targeting to top factors
    logger.info("\nApplying volatility targeting...")
    vol_target_results = {}
    
    for factor in pca_model.factor_returns.columns[:3]:
        scaled_returns, leverage = portfolio_controls.apply_volatility_targeting(
            pca_model.factor_returns[factor]
        )
        vol_target_results[factor] = {
            'scaled_returns': scaled_returns,
            'leverage': leverage
        }
        
        original_sharpe = (pca_model.factor_returns[factor].mean() / 
                          pca_model.factor_returns[factor].std()) * np.sqrt(252)
        scaled_sharpe = (scaled_returns.mean() / scaled_returns.std()) * np.sqrt(252)
        
        logger.info(f"\n{factor}:")
        logger.info(f"  Original Sharpe: {original_sharpe:.2f}")
        logger.info(f"  Scaled Sharpe: {scaled_sharpe:.2f}")
        logger.info(f"  Mean Leverage: {leverage.mean():.2f}")
    
    # Compute risk metrics
    logger.info("\nComputing risk metrics...")
    risk_summary = {}
    for factor in classical_factor_returns.columns:
        metrics = risk_metrics.compute_all_metrics(classical_factor_returns[factor])
        risk_summary[factor] = metrics
    
    risk_summary_df = pd.DataFrame(risk_summary).T
    risk_summary_df.to_csv(f"{config['paths']['results']}/risk_metrics.csv")
    
    # ========================================================================
    # STEP 7: VISUALIZATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: CREATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    visualizer = FactorVisualizer(config)
    visualizer.create_all_plots(pca_model, classical_factors, regime_analyzer)
    
    # Additional regime plots
    if 'volatility' in regime_analyzer.regime_stats:
        vol_stats = regime_analyzer.regime_stats['volatility']
        visualizer.plot_regime_comparison(
            vol_stats, 'Sharpe_Ratio', 'regime_volatility_sharpe.png'
        )
    
    if 'market' in regime_analyzer.regime_stats:
        market_stats = regime_analyzer.regime_stats['market']
        visualizer.plot_regime_comparison(
            market_stats, 'Sharpe_Ratio', 'regime_market_sharpe.png'
        )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nData:")
    logger.info(f"  Assets: {len(returns.columns)}")
    logger.info(f"  Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    logger.info(f"  Days: {len(returns)}")
    
    logger.info(f"\nPCA Factors:")
    logger.info(f"  Components: {pca_model.n_components}")
    logger.info(f"  Variance explained: {summary['total_variance_explained']:.2%}")
    pc1_sharpe = Advanced_eval['economic']['PC1']['sharpe_ratio']
    logger.info(f"  PC1 Sharpe: {pc1_sharpe:.2f} (market factor)")
    logger.info(f"  Interpretation: Statistical risk factors, not all earn premia")
    
    logger.info(f"\nClassical Factors:")
    logger.info(f"  Factors: {len(classical_factor_returns.columns)}")
    logger.info(f"  Mean Sharpe: {classical_premia['Sharpe_Ratio'].mean():.2f}")
    
    logger.info(f"\nRegimes:")
    logger.info(f"  Types analyzed: {len(regimes.columns)}")
    
    logger.info(f"\nOutputs:")
    logger.info(f"  Results: {config['paths']['results']}")
    logger.info(f"  Plots: {config['paths']['plots']}")
    logger.info(f"  Logs: logs/pipeline.log")
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
