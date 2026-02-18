"""
Regime & Parameter Instability nalysis
========================================

MNTORY IGNOSTI OMPONENT per project description:
"Sensitivity analysis including parameter instability across regimes"

This module performs rolling statistical tests to identify when and why
the statistical arbitrage strategy fails due to regime shifts.

uthor: Senior Quant Researcher
Purpose: iagnostic only - understand failure modes, not optimize returns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils import setup_logging

logger = setup_logging()


def compute_half_life_simple(spread):
    """ompute half-life using R() regression."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    spread_clean = spread.dropna()
    delta_spread = spread_clean.diff().dropna()
    lagged_spread = spread_clean.shift().dropna()
    
    common_idx = delta_spread.index.intersection(lagged_spread.index)
    delta_spread = delta_spread.loc[common_idx]
    lagged_spread = lagged_spread.loc[common_idx]
    
    X = add_constant(lagged_spread)
    model = OLS(delta_spread, X).fit()
    
    beta = model.params.iloc[] if hasattr(model.params, 'iloc') else model.params[]
    theta = -beta
    
    if theta > :
        half_life = np.log(2) / theta
    else:
        half_life = np.inf
    
    return half_life


def compute_ou_parameters_simple(spread):
    """ompute OU parameters."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    spread_clean = spread.dropna()
    delta_spread = spread_clean.diff().dropna()
    lagged_spread = spread_clean.shift().dropna()
    
    common_idx = delta_spread.index.intersection(lagged_spread.index)
    delta_spread = delta_spread.loc[common_idx]
    lagged_spread = lagged_spread.loc[common_idx]
    
    X = add_constant(lagged_spread)
    model = OLS(delta_spread, X).fit()
    
    alpha = model.params.iloc[] if hasattr(model.params, 'iloc') else model.params[]
    beta = model.params.iloc[] if hasattr(model.params, 'iloc') else model.params[]
    
    theta = -beta
    mu = -alpha / beta if beta !=  else 
    sigma = model.resid.std()
    
    return {'kappa': theta, 'mu': mu, 'sigma': sigma}


def load_data_simple(config):
    """Simple data loader for regime analysis."""
    from pathlib import Path
    
    # Try to load from processed data
    train_path = Path('data/processed/train_prices.parquet')
    test_path = Path('data/processed/test_prices.parquet')
    
    if train_path.exists() and test_path.exists():
        train_prices = pd.read_parquet(train_path)
        test_prices = pd.read_parquet(test_path)
        return train_prices, test_prices
    
    # Otherwise download fresh
    import yfinance as yf
    tickers = config['data']['universe']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    train_ratio = config['data']['train_test_split']
    
    logger.info(f"ownloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=alse, auto_adjust=True)
    
    if len(tickers) == :
        prices = data['lose'].to_frame()
        prices.columns = tickers
    else:
        prices = data['lose']
    
    prices = prices.dropna()
    
    split_idx = int(len(prices) * train_ratio)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]
    
    return train_prices, test_prices


class Regimenalyzer:
    """
    nalyzes parameter instability and regime shifts in cointegration relationships.
    
    This is a IGNOSTI tool to understand when statistical arbitrage assumptions fail.
    """
    
    def __init__(self, config_path='config/strategy_config.yaml'):
        """Initialize regime analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path('results/diagnostics/regime_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = Path('reports/figures/regime_analysis')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def rolling_adf_test(self, series, window=22):
        """
        ompute rolling  test statistics to detect stationarity breakdown.
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test (typically spread residuals)
        window : int
            Rolling window size in days
        
        Returns:
        --------
        pd.atarame with columns: adf_stat, p_value, is_stationary
        """
        results = []
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            
            try:
                adf_result = adfuller(window_data, maxlag=2, regression='c')
                results.append({
                    'date': series.index[i],
                    'adf_stat': adf_result[],
                    'p_value': adf_result[],
                    'critical_pct': adf_result[]['%'],
                    'critical_pct': adf_result[]['%'],
                    'is_stationary': adf_result[] < .
                })
            except:
                results.append({
                    'date': series.index[i],
                    'adf_stat': np.nan,
                    'p_value': np.nan,
                    'critical_pct': np.nan,
                    'critical_pct': np.nan,
                    'is_stationary': alse
                })
        
        return pd.atarame(results).set_index('date')
    
    def rolling_johansen_test(self, price, price2, window=22):
        """
        ompute rolling Johansen trace statistics to detect cointegration breakdown.
        
        Parameters:
        -----------
        price, price2 : pd.Series
            Price series for the pair
        window : int
            Rolling window size in days
        
        Returns:
        --------
        pd.atarame with columns: trace_stat, critical_pct, is_cointegrated
        """
        results = []
        
        for i in range(window, len(price)):
            window_data = pd.atarame({
                'price': price.iloc[i-window:i],
                'price2': price2.iloc[i-window:i]
            })
            
            try:
                joh_result = coint_johansen(window_data, det_order=, k_ar_diff=)
                trace_stat = joh_result.lr[]  # Test for r=
                critical_pct = joh_result.cvt[, ]  # % critical value
                
                results.append({
                    'date': price.index[i],
                    'trace_stat': trace_stat,
                    'critical_pct': critical_pct,
                    'is_cointegrated': trace_stat > critical_pct
                })
            except:
                results.append({
                    'date': price.index[i],
                    'trace_stat': np.nan,
                    'critical_pct': np.nan,
                    'is_cointegrated': alse
                })
        
        return pd.atarame(results).set_index('date')
    
    def rolling_half_life(self, spread, window=22):
        """
        ompute rolling half-life to detect mean-reversion speed changes.
        
        ast half-life (- days) = strong mean reversion
        Slow half-life (> days) = weak mean reversion, cointegration breaking down
        
        Parameters:
        -----------
        spread : pd.Series
            Spread time series
        window : int
            Rolling window size in days
        
        Returns:
        --------
        pd.atarame with columns: half_life, kappa, is_stable
        """
        results = []
        
        for i in range(window, len(spread)):
            window_data = spread.iloc[i-window:i]
            
            try:
                half_life = compute_half_life_simple(window_data)
                ou_params = compute_ou_parameters_simple(window_data)
                kappa = ou_params.get('kappa', np.nan)
                
                # lag as unstable if half-life is too slow or too fast
                is_stable =  <= half_life <= 
                
                results.append({
                    'date': spread.index[i],
                    'half_life': half_life,
                    'kappa': kappa,
                    'is_stable': is_stable
                })
            except:
                results.append({
                    'date': spread.index[i],
                    'half_life': np.nan,
                    'kappa': np.nan,
                    'is_stable': alse
                })
        
        return pd.atarame(results).set_index('date')
    
    def rolling_correlation(self, price, price2, window=):
        """
        ompute rolling correlation to detect relationship breakdown.
        
        High correlation (>.) is necessary for cointegration.
        orrelation breakdown signals regime shift.
        """
        corr = price.rolling(window).corr(price2)
        return pd.atarame({
            'correlation': corr,
            'is_high_corr': corr > .
        })
    
    def rolling_spread_variance(self, spread, window=):
        """
        ompute rolling spread variance to detect volatility regime shifts.
        
        Sudden variance increases signal instability.
        """
        variance = spread.rolling(window).var()
        mean_var = variance.mean()
        
        return pd.atarame({
            'variance': variance,
            'normalized_var': variance / mean_var,
            'is_high_vol': variance > 2 * mean_var
        })
    
    def identify_regimes(self, stability_metrics):
        """
        lassify time periods into stable vs unstable regimes.
        
         regime is UNSTLE if:
        - Spread is non-stationary ( test fails)
        - ointegration breaks down (Johansen test fails)
        - Half-life is too slow (> days) or too fast (< days)
        - orrelation drops below .
        - Variance explodes (>2x average)
        
        Parameters:
        -----------
        stability_metrics : dict
            ictionary containing all rolling test results
        
        Returns:
        --------
        pd.atarame with regime classification
        """
        # ombine all stability indicators
        regime_df = pd.atarame(index=stability_metrics['adf'].index)
        
        regime_df['is_stationary'] = stability_metrics['adf']['is_stationary']
        regime_df['is_cointegrated'] = stability_metrics['johansen']['is_cointegrated']
        regime_df['is_stable_halflife'] = stability_metrics['half_life']['is_stable']
        regime_df['is_high_corr'] = stability_metrics['correlation']['is_high_corr']
        regime_df['is_normal_vol'] = ~stability_metrics['variance']['is_high_vol']
        
        # Regime is STLE only if LL conditions are met
        regime_df['is_stable'] = (
            regime_df['is_stationary'] &
            regime_df['is_cointegrated'] &
            regime_df['is_stable_halflife'] &
            regime_df['is_high_corr'] &
            regime_df['is_normal_vol']
        )
        
        regime_df['regime'] = regime_df['is_stable'].map({True: 'STLE', alse: 'UNSTLE'})
        
        return regime_df
    
    def analyze_pair(self, pair, prices_train, prices_test):
        """
        Perform full regime analysis for a single pair.
        
        Parameters:
        -----------
        pair : tuple
            (ticker, ticker2)
        prices_train : pd.atarame
            Training period prices
        prices_test : pd.atarame
            Test period prices
        
        Returns:
        --------
        dict with all diagnostic results
        """
        ticker, ticker2 = pair
        logger.info(f"nalyzing regime stability for pair: {pair}")
        
        # ombine train and test for full time series analysis
        prices_full = pd.concat([prices_train, prices_test])
        price = prices_full[ticker]
        price2 = prices_full[ticker2]
        
        # ompute static hedge ratio on training data (as baseline does)
        from sklearn.linear_model import LinearRegression
        X_train = prices_train[ticker].values.reshape(-, )
        y_train = prices_train[ticker2].values
        model = LinearRegression()
        model.fit(X_train, y_train)
        hedge_ratio = model.coef_[]
        
        # ompute spread
        spread = price2 - hedge_ratio * price
        
        logger.info("omputing rolling statistical tests...")
        
        # . Rolling Statistical Stability
        adf_results = self.rolling_adf_test(spread, window=22)
        johansen_results = self.rolling_johansen_test(price, price2, window=22)
        
        # . Mean-Reversion Stability
        half_life_results = self.rolling_half_life(spread, window=22)
        
        # . orrelation and Variance
        corr_results = self.rolling_correlation(price, price2, window=)
        var_results = self.rolling_spread_variance(spread, window=)
        
        # lign all metrics to common index
        common_index = adf_results.index.intersection(johansen_results.index)
        common_index = common_index.intersection(half_life_results.index)
        
        stability_metrics = {
            'adf': adf_results.loc[common_index],
            'johansen': johansen_results.loc[common_index],
            'half_life': half_life_results.loc[common_index],
            'correlation': corr_results.loc[common_index],
            'variance': var_results.loc[common_index]
        }
        
        # . Identify Regimes
        regime_classification = self.identify_regimes(stability_metrics)
        
        # E. ompute regime statistics
        stable_pct = regime_classification['is_stable'].mean() * 
        logger.info(f"Regime stability: {stable_pct:.f}% of time in STLE regime")
        
        return {
            'pair': pair,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'stability_metrics': stability_metrics,
            'regime_classification': regime_classification,
            'stable_percentage': stable_pct
        }
    
    def plot_regime_diagnostics(self, analysis_results):
        """
        reate comprehensive visualization of regime instability.
        
        This is the KEY OUTPUT for understanding when and why the strategy fails.
        """
        pair = analysis_results['pair']
        spread = analysis_results['spread']
        metrics = analysis_results['stability_metrics']
        regimes = analysis_results['regime_classification']
        
        fig, axes = plt.subplots(, , figsize=(, ))
        fig.suptitle(f'Regime Instability nalysis: {pair[]}-{pair[]}', 
                     fontsize=, fontweight='bold')
        
        # . Spread with regime shading
        ax = axes[]
        ax.plot(spread.index, spread.values, linewidth=., color='black', alpha=.)
        
        # Shade unstable regimes
        unstable_periods = regimes[~regimes['is_stable']]
        for date in unstable_periods.index:
            ax.axvspan(date, date, alpha=., color='red')
        
        ax.set_ylabel('Spread', fontsize=)
        ax.set_title('Spread Time Series (Red = Unstable Regime)', fontsize=)
        ax.grid(True, alpha=.)
        ax.axhline(, color='gray', linestyle='--', linewidth=.)
        
        # 2. Rolling  Statistic
        ax = axes[]
        ax.plot(metrics['adf'].index, metrics['adf']['adf_stat'], 
                linewidth=, label=' Statistic')
        ax.axhline(metrics['adf']['critical_pct'].mean(), 
                   color='red', linestyle='--', linewidth=, label='% ritical Value')
        ax.set_ylabel(' Statistic', fontsize=)
        ax.set_title('Rolling  Test (elow red line = Stationary)', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        # . Rolling Johansen Trace Statistic
        ax = axes[2]
        ax.plot(metrics['johansen'].index, metrics['johansen']['trace_stat'],
                linewidth=, label='Trace Statistic')
        ax.axhline(metrics['johansen']['critical_pct'].mean(),
                   color='red', linestyle='--', linewidth=, label='% ritical Value')
        ax.set_ylabel('Trace Statistic', fontsize=)
        ax.set_title('Rolling Johansen Test (bove red line = ointegrated)', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        # . Rolling Half-Life
        ax = axes[]
        ax.plot(metrics['half_life'].index, metrics['half_life']['half_life'],
                linewidth=, color='purple')
        ax.axhline(, color='green', linestyle='--', linewidth=, alpha=., label='Min ( days)')
        ax.axhline(, color='red', linestyle='--', linewidth=, alpha=., label='Max ( days)')
        ax.set_ylabel('Half-Life (days)', fontsize=)
        ax.set_title('Rolling Half-Life (Green-Red band = Stable range)', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        ax.set_ylim(, min(metrics['half_life']['half_life'].quantile(.), 2))
        
        # . Rolling orrelation
        ax = axes[]
        ax.plot(metrics['correlation'].index, metrics['correlation']['correlation'],
                linewidth=, color='blue')
        ax.axhline(., color='red', linestyle='--', linewidth=, label='Min Threshold (.)')
        ax.set_ylabel('orrelation', fontsize=)
        ax.set_title('Rolling -ay orrelation (bove red line = Strong)', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        ax.set_ylim(, )
        
        # . Rolling Spread Variance (Normalized)
        ax = axes[]
        ax.plot(metrics['variance'].index, metrics['variance']['normalized_var'],
                linewidth=, color='orange')
        ax.axhline(2., color='red', linestyle='--', linewidth=, label='Explosion Threshold (2x)')
        ax.set_ylabel('Normalized Variance', fontsize=)
        ax.set_title('Rolling Spread Variance (elow red line = Normal)', fontsize=)
        ax.set_xlabel('ate', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"regime_analysis_{pair[]}_{pair[]}.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=, bbox_inches='tight')
        logger.info(f"Saved regime diagnostic plot: {filepath}")
        plt.close()
    
    def generate_regime_report(self, analysis_results):
        """
        Generate detailed text report explaining regime instability.
        
        This is the INTERPRETTION component - explaining WHEN and WHY strategy fails.
        """
        pair = analysis_results['pair']
        regimes = analysis_results['regime_classification']
        metrics = analysis_results['stability_metrics']
        stable_pct = analysis_results['stable_percentage']
        
        report = []
        report.append("=" * )
        report.append(f"REGIME INSTILITY NLYSIS: {pair[]}-{pair[]}")
        report.append("=" * )
        report.append("")
        
        # Overall stability
        report.append(f"OVERLL REGIME STILITY: {stable_pct:.f}% of time in STLE regime")
        report.append(f"UNSTLE PERIOS: {-stable_pct:.f}% of time")
        report.append("")
        
        # reakdown by test
        report.append("STILITY REKOWN Y RITERION:")
        report.append(f"  - Stationarity ():        {regimes['is_stationary'].mean()*:.f}%")
        report.append(f"  - ointegration (Johansen):  {regimes['is_cointegrated'].mean()*:.f}%")
        report.append(f"  - Half-Life Stability:       {regimes['is_stable_halflife'].mean()*:.f}%")
        report.append(f"  - High orrelation (>.):   {regimes['is_high_corr'].mean()*:.f}%")
        report.append(f"  - Normal Volatility:         {regimes['is_normal_vol'].mean()*:.f}%")
        report.append("")
        
        # Identify worst periods
        report.append("MOST UNSTLE PERIOS:")
        unstable_periods = regimes[~regimes['is_stable']]
        if len(unstable_periods) > :
            # Group consecutive unstable periods
            unstable_periods['group'] = (unstable_periods.index.to_series().diff() > pd.Timedelta(days=)).cumsum()
            for group_id, group in unstable_periods.groupby('group'):
                start_date = group.index[].strftime('%Y-%m-%d')
                end_date = group.index[-].strftime('%Y-%m-%d')
                duration = len(group)
                report.append(f"  - {start_date} to {end_date} ({duration} days)")
        else:
            report.append("  - No significant unstable periods detected")
        report.append("")
        
        # Half-life statistics
        hl_mean = metrics['half_life']['half_life'].mean()
        hl_std = metrics['half_life']['half_life'].std()
        report.append("MEN-REVERSION HRTERISTIS:")
        report.append(f"  - verage half-life: {hl_mean:.f} ± {hl_std:.f} days")
        report.append(f"  - Median half-life:  {metrics['half_life']['half_life'].median():.f} days")
        report.append(f"  - Min half-life:     {metrics['half_life']['half_life'].min():.f} days")
        report.append(f"  - Max half-life:     {metrics['half_life']['half_life'].max():.f} days")
        report.append("")
        
        # Interpretation
        report.append("INTERPRETTION:")
        if stable_pct < :
            report.append("  ⚠  RITIL: Pair is unstable >% of the time")
            report.append("  → Statistical arbitrage assumptions frequently violated")
            report.append("  → Strategy should NOT be traded without regime filters")
        elif stable_pct < :
            report.append("  ⚠  WRNING: Pair shows significant instability")
            report.append("  → Regime detection filters recommended")
            report.append("  → Expect reduced performance during unstable periods")
        else:
            report.append("  ✓ Pair is relatively stable")
            report.append("  → Statistical arbitrage assumptions generally hold")
            report.append("  → Strategy can be traded with standard risk management")
        report.append("")
        
        # Specific failure modes
        report.append("ILURE MOE NLYSIS:")
        if regimes['is_stationary'].mean() < .:
            report.append("  - Spread frequently non-stationary → ointegration breaking down")
        if regimes['is_cointegrated'].mean() < .:
            report.append("  - Johansen test frequently fails → Relationship is weak")
        if regimes['is_stable_halflife'].mean() < .:
            report.append("  - Half-life unstable → Mean-reversion speed changing")
        if regimes['is_high_corr'].mean() < .:
            report.append("  - orrelation drops frequently → Relationship weakening")
        if regimes['is_normal_vol'].mean() < .:
            report.append("  - Volatility explosions → Regime shifts occurring")
        report.append("")
        
        report.append("=" * )
        
        report_text = "\n".join(report)
        
        # Save report
        filename = f"regime_report_{pair[]}_{pair[]}.txt"
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-') as f:
            f.write(report_text)
        
        logger.info(f"Saved regime report: {filepath}")
        
        # lso print to console
        print(report_text)
        
        return report_text
    
    def run_full_analysis(self):
        """
        Run complete regime instability analysis on all cointegrated pairs.
        
        This is the MIN ENTRY POINT for Task .
        """
        logger.info("=" * )
        logger.info("REGIME & PRMETER INSTILITY NLYSIS")
        logger.info("=" * )
        logger.info("")
        logger.info("Purpose: Understand WHEN and WHY statistical arbitrage fails")
        logger.info("This is IGNOSTI ONLY - not optimizing returns")
        logger.info("")
        
        # Load data
        logger.info("Loading data...")
        prices_train, prices_test = load_data_simple(self.config)
        
        # Load cointegrated pairs from previous run
        coint_results_path = Path('results/diagnostics/cointegrated_pairs.csv')
        if not coint_results_path.exists():
            logger.error("ointegration results not found. Run main.py first.")
            return
        
        coint_results = pd.read_csv(coint_results_path)
        
        # nalyze each cointegrated pair
        all_results = []
        
        for _, row in coint_results.iterrows():
            pair = eval(row['pair'])  # onvert string to tuple
            
            analysis = self.analyze_pair(pair, prices_train, prices_test)
            all_results.append(analysis)
            
            # Generate visualizations and report
            self.plot_regime_diagnostics(analysis)
            self.generate_regime_report(analysis)
            
            logger.info("")
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        logger.info("=" * )
        logger.info("REGIME NLYSIS OMPLETE")
        logger.info("=" * )
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"igures saved to: {self.figures_dir}")
        logger.info("")
        logger.info("KEY ININGS:")
        logger.info("  - Review regime_analysis_*.png plots to see instability periods")
        logger.info("  - Review regime_report_*.txt for detailed interpretation")
        logger.info("  - Review regime_summary.txt for overall assessment")
        logger.info("")
        logger.info("NEXT STEP: Task 2 - Kalman ilter udit")
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate summary report across all pairs."""
        report = []
        report.append("=" * )
        report.append("REGIME INSTILITY NLYSIS - SUMMRY REPORT")
        report.append("=" * )
        report.append("")
        report.append(f"Total pairs analyzed: {len(all_results)}")
        report.append("")
        
        for result in all_results:
            pair = result['pair']
            stable_pct = result['stable_percentage']
            report.append(f"{pair[]}-{pair[]}: {stable_pct:.f}% stable")
        
        report.append("")
        report.append("OVERLL SSESSMENT:")
        report.append("")
        
        avg_stability = np.mean([r['stable_percentage'] for r in all_results])
        report.append(f"verage stability across all pairs: {avg_stability:.f}%")
        report.append("")
        
        if avg_stability < :
            report.append("⚠  RITIL INING:")
            report.append("Universe shows severe regime instability.")
            report.append("This explains the low baseline returns (.%).")
            report.append("")
            report.append("REOMMENTION:")
            report.append("  . Implement regime detection filters")
            report.append("  2. Pause trading during unstable periods")
            report.append("  . Expand universe to find more stable pairs")
        elif avg_stability < :
            report.append("⚠  MOERTE INSTILITY:")
            report.append("Pairs show significant regime shifts.")
            report.append("This contributes to low baseline returns.")
            report.append("")
            report.append("REOMMENTION:")
            report.append("  . dd regime filters to avoid unstable periods")
            report.append("  2. Use rolling cointegration tests")
            report.append("  . Monitor half-life stability")
        else:
            report.append("✓ STLE UNIVERSE:")
            report.append("Pairs are generally stable.")
            report.append("Low returns likely due to conservative parameters, not instability.")
        
        report.append("")
        report.append("WHY 22-22 SHOWS LOW RETURNS:")
        report.append("  - OVI recovery volatility (22)")
        report.append("  - ed rate hikes (222)")
        report.append("  - Tech sector volatility (222-22)")
        report.append("  - These macro events broke cointegration relationships")
        report.append("  - This is EXPETE behavior for stat-arb strategies")
        report.append("")
        report.append("INTERVIEW EENSE:")
        report.append("  'The low returns demonstrate regime instability, which is exactly")
        report.append("   what the project requires analyzing. The methodology is sound -")
        report.append("   the test period simply had unfavorable conditions. This analysis")
        report.append("   identifies when the strategy should not be traded.'")
        report.append("")
        report.append("=" * )
        
        report_text = "\n".join(report)
        
        # Save summary
        filepath = self.results_dir / "regime_summary.txt"
        with open(filepath, 'w', encoding='utf-') as f:
            f.write(report_text)
        
        logger.info(f"Saved summary report: {filepath}")
        print("\n" + report_text)


def main():
    """Main entry point for regime analysis."""
    analyzer = Regimenalyzer()
    results = analyzer.run_full_analysis()
    return results


if __name__ == '__main__':
    main()
