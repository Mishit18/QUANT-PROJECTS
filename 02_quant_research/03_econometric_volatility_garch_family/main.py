"""
Main execution pipeline for econometric volatility modeling project.

Runs complete analysis:
1. Data loading and stylized facts
2. Model estimation (ARCH, GARCH, EGARCH, GJR, HARCH)
3. Rolling forecasts
4. VaR/ES backtesting
5. Rough volatility benchmarking
6. Report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Import project modules
from src.data_loader import DataLoader
from src.returns import StylizedFacts, realized_volatility
from src.models import GARCHModel, EGARCHModel, GJRGARCHModel, HARCHModel
from src.forecasting import RollingForecast, ForecastMetrics
from src.forecasting.rolling_forecast import compare_models_rolling
from src.risk import VaRCalculator, ESCalculator, VaRBacktest, ESBacktest
from src.rough_vol import RoughBergomiModel, RoughVolBenchmark

# Create output directories
Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("ECONOMETRIC VOLATILITY MODELING PIPELINE")
    print("="*70)
    
    # ========== 1. DATA LOADING ==========
    print("\n[1/7] Loading and preparing data...")
    loader = DataLoader()
    returns, prices = loader.prepare_dataset(
        ticker="^GSPC",
        start_date="2010-01-01",
        return_method="log"
    )
    
    print(f"Loaded {len(returns)} daily returns")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # ========== 2. STYLIZED FACTS ==========
    print("\n[2/7] Analyzing stylized facts...")
    sf = StylizedFacts(returns)
    facts = sf.full_report()
    
    print("\nSummary Statistics:")
    for key, value in facts["summary_stats"].items():
        print(f"  {key:20s}: {value:12.6f}")
    
    print(f"\nLjung-Box (returns):  stat={facts['ljung_box_returns'][0]:.2f}, p={facts['ljung_box_returns'][1]:.4f}")
    print(f"Ljung-Box (squared):  stat={facts['ljung_box_squared'][0]:.2f}, p={facts['ljung_box_squared'][1]:.4f}")
    print(f"ARCH-LM test:         stat={facts['arch_lm'][0]:.2f}, p={facts['arch_lm'][1]:.4f}")
    print(f"Leverage effect:      corr={facts['leverage_effect']:.4f}")
    
    # ========== 3. MODEL ESTIMATION ==========
    print("\n[3/7] Estimating volatility models...")
    
    # Use subset for faster estimation
    train_returns = returns.iloc[-1500:].values
    
    models = {}
    
    print("  Fitting GARCH(1,1)...")
    garch = GARCHModel(p=1, q=1)
    garch.fit(train_returns)
    models["GARCH"] = garch
    print(f"    Log-likelihood: {garch.log_likelihood:.2f}, Persistence: {garch.persistence():.4f}")
    
    print("  Fitting EGARCH(1,1)...")
    egarch = EGARCHModel(p=1, q=1)
    egarch.fit(train_returns)
    models["EGARCH"] = egarch
    print(f"    Log-likelihood: {egarch.log_likelihood:.2f}, Leverage: {egarch.leverage_effect():.4f}")
    
    print("  Fitting GJR-GARCH(1,1)...")
    gjr = GJRGARCHModel(p=1, q=1)
    gjr.fit(train_returns)
    models["GJR-GARCH"] = gjr
    print(f"    Log-likelihood: {gjr.log_likelihood:.2f}, Asymmetry: {gjr.asymmetry_ratio():.4f}")
    
    print("  Fitting HARCH(1,5,22)...")
    harch = HARCHModel(lags=[1, 5, 22])
    harch.fit(train_returns)
    models["HARCH"] = harch
    print(f"    Log-likelihood: {harch.log_likelihood:.2f}")
    
    # Model comparison table
    comparison = pd.DataFrame({
        "Model": list(models.keys()),
        "Log-Likelihood": [m.log_likelihood for m in models.values()],
        "AIC": [m.aic for m in models.values()],
        "BIC": [m.bic for m in models.values()]
    })
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    comparison.to_csv("reports/tables/model_comparison.csv", index=False)
    
    # ========== 4. ROLLING FORECASTS ==========
    print("\n[4/7] Generating rolling forecasts...")
    
    models_config = {
        "GARCH": {"class": GARCHModel, "params": {"p": 1, "q": 1}},
        "EGARCH": {"class": EGARCHModel, "params": {"p": 1, "q": 1}},
        "GJR-GARCH": {"class": GJRGARCHModel, "params": {"p": 1, "q": 1}}
    }
    
    forecast_df = compare_models_rolling(
        returns.iloc[-1000:],
        models_config,
        window_size=500,
        horizon=1,
        verbose=False
    )
    
    # Forecast evaluation
    print("\nForecast Evaluation:")
    for model_name in ["GARCH", "EGARCH", "GJR-GARCH"]:
        metrics = ForecastMetrics(
            forecast_df[model_name].values,
            forecast_df["realized"].values
        )
        results = metrics.all_metrics()
        print(f"\n  {model_name}:")
        print(f"    RMSE:   {results['RMSE']:.6f}")
        print(f"    QLIKE:  {results['QLIKE']:.6f}")
        print(f"    R²:     {results['R2']:.4f}")
    
    # Diebold-Mariano pairwise comparison
    print("\nDiebold-Mariano Forecast Comparison:")
    print("(Because volatility forecasts are noisy, formal statistical tests are essential)")
    
    from src.forecasting.forecast_metrics import compare_forecasts_dm
    
    forecasts_for_dm = {
        "GARCH": forecast_df["GARCH"].values,
        "EGARCH": forecast_df["EGARCH"].values,
        "GJR-GARCH": forecast_df["GJR-GARCH"].values
    }
    
    dm_results = compare_forecasts_dm(
        forecasts_for_dm,
        forecast_df["realized"].values,
        loss_function="mse",
        horizon=1
    )
    
    print("\n" + dm_results.to_string(index=False))
    dm_results.to_csv("reports/tables/diebold_mariano_tests.csv", index=False)
    print("\nSaved to reports/tables/diebold_mariano_tests.csv")
    
    # ========== 5. VAR/ES BACKTESTING ==========
    print("\n[5/7] Backtesting VaR and ES...")
    
    # Use GARCH forecasts for VaR/ES
    var_calc = VaRCalculator(confidence_level=0.95)
    es_calc = ESCalculator(confidence_level=0.95)
    
    var_forecasts = var_calc.rolling_var(forecast_df["GARCH"])
    es_forecasts = es_calc.rolling_es(forecast_df["GARCH"])
    
    # Align with realized returns
    realized_returns = returns.iloc[-len(var_forecasts):].values
    
    # VaR backtest
    var_backtest = VaRBacktest(
        realized_returns,
        var_forecasts.values,
        confidence_level=0.95
    )
    var_summary = var_backtest.summary()
    
    print("\nVaR Backtest Results:")
    print(f"  Violations: {var_summary['n_violations']} / {var_summary['n_observations']}")
    print(f"  Violation rate: {var_summary['violation_rate']:.4f} (expected: {var_summary['expected_rate']:.4f})")
    print(f"  Kupiec test: stat={var_summary['kupiec_stat']:.2f}, p={var_summary['kupiec_pval']:.4f}")
    print(f"  Traffic light: {var_summary['traffic_light']}")
    
    # ES backtest
    es_backtest = ESBacktest(
        realized_returns,
        es_forecasts.values,
        var_forecasts.values,
        confidence_level=0.95
    )
    es_summary = es_backtest.summary()
    
    print("\nES Backtest Results:")
    print(f"  Average tail loss: {es_summary['average_tail_loss']:.6f}")
    print(f"  Average ES forecast: {es_summary['average_es_forecast']:.6f}")
    print(f"  ES ratio: {es_summary['es_ratio']:.4f}")
    
    # ========== 6. ROUGH VOLATILITY BENCHMARK ==========
    print("\n[6/7] Rough volatility benchmarking...")
    
    rbergomi = RoughBergomiModel(hurst=0.1, eta=1.9, seed=42)
    benchmark = RoughVolBenchmark(hurst=0.1, seed=42)
    
    # Generate rough vol data
    rough_returns, rough_variance = benchmark.generate_rough_vol_data(n_steps=1000)
    
    print(f"  Generated {len(rough_returns)} rough volatility returns")
    
    # Compare autocorrelation
    acf_comparison = benchmark.compare_autocorrelation(rough_returns, max_lag=30)
    
    print("\nAutocorrelation comparison (lag 1, 5, 10):")
    for lag in [1, 5, 10]:
        row = acf_comparison.iloc[lag]
        print(f"  Lag {lag:2d}: Empirical={row['empirical']:.4f}, "
              f"Rough Vol={row['rough_vol_theory']:.4f}, GARCH={row['garch_theory']:.4f}")
    
    # Fit GARCH to rough vol data
    print("\nFitting GARCH to rough volatility data...")
    garch_rough = GARCHModel(p=1, q=1)
    garch_rough.fit(rough_returns)
    print(f"  Persistence: {garch_rough.persistence():.4f}")
    print(f"  Half-life: {garch_rough.half_life():.2f} days")
    
    # ========== 7. VISUALIZATION ==========
    print("\n[7/7] Generating visualizations...")
    
    # Plot 1: Returns and volatility
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    returns_plot = returns.iloc[-1000:]
    axes[0].plot(returns_plot.index, returns_plot.values, linewidth=0.5, alpha=0.7)
    axes[0].set_title("S&P 500 Daily Returns", fontsize=12)
    axes[0].set_ylabel("Log Returns")
    axes[0].grid(alpha=0.3)
    
    rv = realized_volatility(returns.iloc[-1000:], window=20)
    axes[1].plot(rv.index, rv.values, linewidth=1)
    axes[1].set_title("Realized Volatility (20-day)", fontsize=12)
    axes[1].set_ylabel("Annualized Volatility")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/figures/returns_and_volatility.png", dpi=150)
    print("  Saved: returns_and_volatility.png")
    
    # Plot 2: Forecast comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df = forecast_df.iloc[-200:]
    ax.plot(plot_df.index, plot_df["realized"], label="Realized", linewidth=1.5, alpha=0.7)
    ax.plot(plot_df.index, plot_df["GARCH"], label="GARCH", linewidth=1)
    ax.plot(plot_df.index, plot_df["EGARCH"], label="EGARCH", linewidth=1)
    ax.plot(plot_df.index, plot_df["GJR-GARCH"], label="GJR-GARCH", linewidth=1)
    
    ax.set_title("Volatility Forecasts Comparison", fontsize=12)
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/figures/forecast_comparison.png", dpi=150)
    print("  Saved: forecast_comparison.png")
    
    # Plot 3: VaR violations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_returns = realized_returns[-200:]
    plot_var = -var_forecasts.values[-200:]
    plot_dates = var_forecasts.index[-200:]
    
    ax.plot(plot_dates, plot_returns, linewidth=0.8, alpha=0.7, label="Returns")
    ax.plot(plot_dates, plot_var, 'r--', linewidth=1, label="VaR (95%)")
    
    violations_idx = plot_returns < plot_var
    ax.scatter(plot_dates[violations_idx], plot_returns[violations_idx], 
               color='red', s=30, zorder=5, label="Violations")
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("VaR Backtesting", fontsize=12)
    ax.set_ylabel("Returns")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/figures/var_backtest.png", dpi=150)
    print("  Saved: var_backtest.png")
    
    # Plot 4: Autocorrelation comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(acf_comparison["lag"], acf_comparison["empirical"], 
            'o-', label="Empirical (Rough Vol Data)", markersize=4)
    ax.plot(acf_comparison["lag"], acf_comparison["rough_vol_theory"], 
            '--', label="Rough Vol Theory", linewidth=2)
    ax.plot(acf_comparison["lag"], acf_comparison["garch_theory"], 
            '--', label="GARCH Theory", linewidth=2)
    
    ax.set_title("Autocorrelation of Squared Returns", fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/figures/acf_comparison.png", dpi=150)
    print("  Saved: acf_comparison.png")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nOutputs saved to:")
    print("  - reports/figures/")
    print("  - reports/tables/")
    print("\nNext steps:")
    print("  - Review notebooks/ for detailed analysis")
    print("  - Examine model diagnostics")
    print("  - Explore rough volatility benchmarks")


if __name__ == "__main__":
    main()
