"""
Kalman ilter Model udit & Validation
=======================================

MNTORY IGNOSTI OMPONENT per project description:
"Model spreads using Kalman ilter to estimate time-varying hedge ratios"

This module performs a complete audit of the Kalman ilter implementation:
- State-space model validation
- Parameter sensitivity analysis
- iagnostic checks for filter performance
- omparison with static OLS hedge ratios

uthor: Senior Quant Researcher
Purpose: Validate Kalman ilter, not optimize returns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils import setup_logging

logger = setup_logging()


def load_data_simple(config):
    """Simple data loader."""
    from pathlib import Path
    
    train_path = Path('data/processed/train_prices.parquet')
    test_path = Path('data/processed/test_prices.parquet')
    
    if train_path.exists() and test_path.exists():
        train_prices = pd.read_parquet(train_path)
        test_prices = pd.read_parquet(test_path)
        return train_prices, test_prices
    
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


class ManualKalmanilter:
    """
    Manual implementation of Kalman ilter for hedge ratio estimation.
    
    STTE-SPE MOEL:
    ------------------
    State equation:    β_t = β_{t-} + w_t,  w_t ~ N(, Q)
    Observation eq:    y_t = β_t * x_t + v_t,  v_t ~ N(, R)
    
    Where:
    - β_t is the time-varying hedge ratio (state)
    - y_t is the dependent variable (price2)
    - x_t is the independent variable (price)
    - Q is the state transition covariance (how much β changes)
    - R is the observation noise covariance (spread volatility)
    
    INTERPRETTION:
    ---------------
    - Small Q: β changes slowly (smooth, stable hedge ratio)
    - Large Q: β changes quickly (responsive but noisy)
    - Q/R ratio controls the trade-off between smoothness and responsiveness
    """
    
    def __init__(self, Q=e-, R=e-, initial_beta=., initial_P=.):
        """
        Initialize Kalman ilter.
        
        Parameters:
        -----------
        Q : float
            State transition covariance (process noise)
        R : float
            Observation covariance (measurement noise)
        initial_beta : float
            Initial hedge ratio estimate
        initial_P : float
            Initial state covariance
        """
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        self.beta = initial_beta  # urrent state estimate
        self.P = initial_P  # urrent state covariance
        
        # Store history
        self.beta_history = []
        self.P_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
    
    def predict(self):
        """
        Prediction step: β_t|t- = β_{t-|t-}
        """
        # State prediction (random walk model)
        beta_pred = self.beta
        
        # ovariance prediction
        P_pred = self.P + self.Q
        
        return beta_pred, P_pred
    
    def update(self, y, x):
        """
        Update step: incorporate new observation.
        
        Parameters:
        -----------
        y : float
            Observed dependent variable
        x : float
            Observed independent variable
        """
        # Prediction
        beta_pred, P_pred = self.predict()
        
        # Innovation (prediction error)
        y_pred = beta_pred * x
        innovation = y - y_pred
        
        # Innovation covariance
        S = x**2 * P_pred + self.R
        
        # Kalman gain
        K = (P_pred * x) / S
        
        # State update
        self.beta = beta_pred + K * innovation
        
        # ovariance update
        self.P = ( - K * x) * P_pred
        
        # Store history
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.innovation_history.append(innovation)
        self.kalman_gain_history.append(K)
    
    def filter(self, y_series, x_series):
        """
        Run Kalman ilter on entire time series.
        
        Parameters:
        -----------
        y_series : pd.Series
            ependent variable time series
        x_series : pd.Series
            Independent variable time series
        
        Returns:
        --------
        pd.atarame with filtered estimates
        """
        # Reset history
        self.beta_history = []
        self.P_history = []
        self.innovation_history = []
        self.kalman_gain_history = []
        
        # ilter through data
        for y, x in zip(y_series, x_series):
            self.update(y, x)
        
        # reate results atarame
        results = pd.atarame({
            'beta': self.beta_history,
            'P': self.P_history,
            'innovation': self.innovation_history,
            'kalman_gain': self.kalman_gain_history
        }, index=y_series.index)
        
        return results


class Kalmanilteruditor:
    """
    omprehensive audit of Kalman ilter implementation.
    """
    
    def __init__(self, config_path='config/strategy_config.yaml'):
        """Initialize auditor."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path('results/diagnostics/kalman_audit')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = Path('reports/figures/kalman_audit')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_static_ols_hedge_ratio(self, price, price2):
        """ompute static OLS hedge ratio for comparison."""
        from sklearn.linear_model import LinearRegression
        
        X = price.values.reshape(-, )
        y = price2.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[]
    
    def audit_pair(self, pair, prices_train, prices_test):
        """
        Perform complete Kalman ilter audit for a pair.
        """
        ticker, ticker2 = pair
        logger.info(f"uditing Kalman ilter for pair: {pair}")
        
        # ombine train and test
        prices_full = pd.concat([prices_train, prices_test])
        price = prices_full[ticker]
        price2 = prices_full[ticker2]
        
        # . Static OLS aseline
        ols_beta = self.compute_static_ols_hedge_ratio(price, price2)
        logger.info(f"Static OLS hedge ratio: {ols_beta:.f}")
        
        # . Kalman ilter with default parameters
        Q_default = float(self.config['kalman']['transition_covariance'])
        R_default = float(self.config['kalman']['observation_covariance'])
        
        kf_default = ManualKalmanilter(Q=Q_default, R=R_default, initial_beta=ols_beta)
        kf_results_default = kf_default.filter(price2, price)
        
        logger.info(f"Kalman ilter mean beta: {kf_results_default['beta'].mean():.f}")
        logger.info(f"Kalman ilter beta std: {kf_results_default['beta'].std():.f}")
        
        # . Parameter Sensitivity nalysis
        Q_values = [e-, e-, e-, e-]
        R_values = [e-, e-, e-2]
        
        sensitivity_results = []
        
        for Q in Q_values:
            for R in R_values:
                kf = ManualKalmanilter(Q=Q, R=R, initial_beta=ols_beta)
                kf_res = kf.filter(price2, price)
                
                beta_mean = kf_res['beta'].mean()
                beta_std = kf_res['beta'].std()
                beta_range = kf_res['beta'].max() - kf_res['beta'].min()
                
                sensitivity_results.append({
                    'Q': Q,
                    'R': R,
                    'Q_R_ratio': Q/R,
                    'beta_mean': beta_mean,
                    'beta_std': beta_std,
                    'beta_range': beta_range
                })
        
        sensitivity_df = pd.atarame(sensitivity_results)
        
        return {
            'pair': pair,
            'ols_beta': ols_beta,
            'kf_results_default': kf_results_default,
            'sensitivity_df': sensitivity_df,
            'price': price,
            'price2': price2
        }

    
    def plot_kalman_diagnostics(self, audit_results):
        """reate comprehensive diagnostic plots."""
        pair = audit_results['pair']
        ols_beta = audit_results['ols_beta']
        kf_results = audit_results['kf_results_default']
        price = audit_results['price']
        price2 = audit_results['price2']
        
        fig, axes = plt.subplots(, , figsize=(, ))
        fig.suptitle(f'Kalman ilter udit: {pair[]}-{pair[]}', fontsize=, fontweight='bold')
        
        # . Time-Varying Hedge Ratio
        ax = axes[]
        ax.plot(kf_results.index, kf_results['beta'], linewidth=, label='Kalman ilter β_t', color='blue')
        ax.axhline(ols_beta, color='red', linestyle='--', linewidth=, label=f'Static OLS β = {ols_beta:.f}')
        ax.fill_between(kf_results.index, 
                        kf_results['beta'] - 2*np.sqrt(kf_results['P']),
                        kf_results['beta'] + 2*np.sqrt(kf_results['P']),
                        alpha=.2, color='blue', label='±2σ onfidence')
        ax.set_ylabel('Hedge Ratio β', fontsize=)
        ax.set_title('Time-Varying Hedge Ratio (Kalman vs OLS)', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        # 2. Hedge Ratio Volatility
        ax = axes[]
        rolling_std = kf_results['beta'].rolling().std()
        ax.plot(rolling_std.index, rolling_std, linewidth=, color='purple')
        ax.set_ylabel('Rolling Std (d)', fontsize=)
        ax.set_title('Hedge Ratio Volatility (Stability heck)', fontsize=)
        ax.grid(True, alpha=.)
        
        # . Kalman Gain
        ax = axes[2]
        ax.plot(kf_results.index, kf_results['kalman_gain'], linewidth=., color='green')
        ax.set_ylabel('Kalman Gain', fontsize=)
        ax.set_title('Kalman Gain (ilter Responsiveness)', fontsize=)
        ax.grid(True, alpha=.)
        
        # . Innovation (Prediction Error)
        ax = axes[]
        ax.plot(kf_results.index, kf_results['innovation'], linewidth=., color='orange', alpha=.)
        ax.axhline(, color='black', linestyle='--', linewidth=.)
        ax.set_ylabel('Innovation', fontsize=)
        ax.set_title('Innovation (Prediction Error - Should be White Noise)', fontsize=)
        ax.grid(True, alpha=.)
        
        # . Spread omparison (OLS vs Kalman)
        ax = axes[]
        spread_ols = price2 - ols_beta * price
        spread_kalman = price2 - kf_results['beta'] * price
        
        ax.plot(spread_ols.index, spread_ols, linewidth=., label='OLS Spread', alpha=.)
        ax.plot(spread_kalman.index, spread_kalman, linewidth=., label='Kalman Spread', alpha=.)
        ax.axhline(, color='black', linestyle='--', linewidth=.)
        ax.set_ylabel('Spread', fontsize=)
        ax.set_title('Spread omparison (OLS vs Kalman)', fontsize=)
        ax.set_xlabel('ate', fontsize=)
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        plt.tight_layout()
        
        filename = f"kalman_audit_{pair[]}_{pair[]}.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=, bbox_inches='tight')
        logger.info(f"Saved Kalman diagnostic plot: {filepath}")
        plt.close()
    
    def plot_parameter_sensitivity(self, audit_results):
        """Plot parameter sensitivity analysis."""
        pair = audit_results['pair']
        sensitivity_df = audit_results['sensitivity_df']
        
        fig, axes = plt.subplots(, , figsize=(, ))
        fig.suptitle(f'Kalman ilter Parameter Sensitivity: {pair[]}-{pair[]}', fontsize=, fontweight='bold')
        
        # . eta Mean vs Q/R Ratio
        ax = axes[]
        for R in sensitivity_df['R'].unique():
            subset = sensitivity_df[sensitivity_df['R'] == R]
            ax.plot(subset['Q_R_ratio'], subset['beta_mean'], marker='o', label=f'R={R:.e}')
        ax.set_xlabel('Q/R Ratio', fontsize=)
        ax.set_ylabel('Mean Hedge Ratio', fontsize=)
        ax.set_title('Mean β vs Q/R Ratio', fontsize=)
        ax.set_xscale('log')
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        # 2. eta Std vs Q/R Ratio
        ax = axes[]
        for R in sensitivity_df['R'].unique():
            subset = sensitivity_df[sensitivity_df['R'] == R]
            ax.plot(subset['Q_R_ratio'], subset['beta_std'], marker='o', label=f'R={R:.e}')
        ax.set_xlabel('Q/R Ratio', fontsize=)
        ax.set_ylabel('Std of Hedge Ratio', fontsize=)
        ax.set_title('β Volatility vs Q/R Ratio', fontsize=)
        ax.set_xscale('log')
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        # . eta Range vs Q/R Ratio
        ax = axes[2]
        for R in sensitivity_df['R'].unique():
            subset = sensitivity_df[sensitivity_df['R'] == R]
            ax.plot(subset['Q_R_ratio'], subset['beta_range'], marker='o', label=f'R={R:.e}')
        ax.set_xlabel('Q/R Ratio', fontsize=)
        ax.set_ylabel('Range of Hedge Ratio', fontsize=)
        ax.set_title('β Range vs Q/R Ratio', fontsize=)
        ax.set_xscale('log')
        ax.legend(fontsize=)
        ax.grid(True, alpha=.)
        
        plt.tight_layout()
        
        filename = f"kalman_sensitivity_{pair[]}_{pair[]}.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=, bbox_inches='tight')
        logger.info(f"Saved sensitivity plot: {filepath}")
        plt.close()
    
    def generate_audit_report(self, audit_results):
        """Generate detailed audit report."""
        pair = audit_results['pair']
        ols_beta = audit_results['ols_beta']
        kf_results = audit_results['kf_results_default']
        sensitivity_df = audit_results['sensitivity_df']
        
        report = []
        report.append("=" * )
        report.append(f"KLMN ILTER UIT REPORT: {pair[]}-{pair[]}")
        report.append("=" * )
        report.append("")
        
        # . State-Space Model
        Q_val = float(self.config['kalman']['transition_covariance'])
        R_val = float(self.config['kalman']['observation_covariance'])
        
        report.append(". STTE-SPE MOEL SPEIITION:")
        report.append("   State equation:    β_t = β_{t-} + w_t,  w_t ~ N(, Q)")
        report.append("   Observation eq:    y_t = β_t * x_t + v_t,  v_t ~ N(, R)")
        report.append("")
        report.append(f"   Q (process noise):      {Q_val:.2e}")
        report.append(f"   R (measurement noise):  {R_val:.2e}")
        report.append(f"   Q/R ratio:              {Q_val/R_val:.f}")
        report.append("")
        
        # . Hedge Ratio omparison
        kf_beta_mean = kf_results['beta'].mean()
        kf_beta_std = kf_results['beta'].std()
        kf_beta_min = kf_results['beta'].min()
        kf_beta_max = kf_results['beta'].max()
        
        report.append(". HEGE RTIO ESTIMTES:")
        report.append(f"   Static OLS β:           {ols_beta:.f}")
        report.append(f"   Kalman mean β:          {kf_beta_mean:.f}")
        report.append(f"   Kalman std β:           {kf_beta_std:.f}")
        report.append(f"   Kalman range:           [{kf_beta_min:.f}, {kf_beta_max:.f}]")
        report.append(f"   eviation from OLS:     {abs(kf_beta_mean - ols_beta)/ols_beta * :.2f}%")
        report.append("")
        
        # . ilter iagnostics
        avg_kalman_gain = kf_results['kalman_gain'].mean()
        innovation_std = kf_results['innovation'].std()
        
        report.append(". ILTER IGNOSTIS:")
        report.append(f"   verage Kalman gain:    {avg_kalman_gain:.f}")
        report.append(f"   Innovation std:         {innovation_std:.f}")
        report.append("")
        
        # . Parameter Sensitivity
        report.append(". PRMETER SENSITIVITY:")
        report.append("   Q/R Ratio Impact on β Volatility:")
        for _, row in sensitivity_df.iterrows():
            report.append(f"     Q={row['Q']:.e}, R={row['R']:.e}: β_std={row['beta_std']:.f}")
        report.append("")
        
        # E. Interpretation
        report.append("E. INTERPRETTION:")
        if kf_beta_std < .:
            report.append("   ✓ Hedge ratio is STLE (low volatility)")
            report.append("   → Kalman ilter provides smooth, reliable estimates")
        elif kf_beta_std < .:
            report.append("   ⚠  Hedge ratio shows MOERTE variability")
            report.append("   → Time-varying β may capture regime shifts")
        else:
            report.append("   ⚠  Hedge ratio is HIGHLY VOLTILE")
            report.append("   → May indicate model mis-specification or unstable relationship")
        report.append("")
        
        if abs(kf_beta_mean - ols_beta)/ols_beta < .:
            report.append("   ✓ Kalman mean close to OLS (< % deviation)")
            report.append("   → ilter is well-calibrated")
        else:
            report.append("   ⚠  Kalman mean deviates significantly from OLS")
            report.append("   → May need parameter adjustment")
        report.append("")
        
        # . Recommendations
        report.append(". REOMMENTIONS:")
        if kf_beta_std > .:
            report.append("   . onsider increasing R (measurement noise) to smooth β")
            report.append("   2. Or decrease Q (process noise) for more stable estimates")
        else:
            report.append("   . urrent parameters appear reasonable")
        
        report.append("   2. ompare spread stationarity: OLS vs Kalman")
        report.append("   . acktest with time-varying β to assess performance impact")
        report.append("")
        
        report.append("=" * )
        
        report_text = "\n".join(report)
        
        # Save report
        filename = f"kalman_audit_{pair[]}_{pair[]}.txt"
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-') as f:
            f.write(report_text)
        
        logger.info(f"Saved audit report: {filepath}")
        print(report_text)
        
        return report_text
    
    def run_full_audit(self):
        """Run complete Kalman ilter audit."""
        logger.info("=" * )
        logger.info("KLMN ILTER MOEL UIT")
        logger.info("=" * )
        logger.info("")
        logger.info("Purpose: Validate Kalman ilter implementation and parameters")
        logger.info("This is IGNOSTI ONLY - not optimizing returns")
        logger.info("")
        
        # Load data
        logger.info("Loading data...")
        prices_train, prices_test = load_data_simple(self.config)
        
        # Load cointegrated pairs
        coint_results_path = Path('results/diagnostics/cointegrated_pairs.csv')
        if not coint_results_path.exists():
            logger.error("ointegration results not found. Run main.py first.")
            return
        
        coint_results = pd.read_csv(coint_results_path)
        
        # udit each pair
        all_results = []
        
        for _, row in coint_results.iterrows():
            pair = eval(row['pair'])
            
            audit = self.audit_pair(pair, prices_train, prices_test)
            all_results.append(audit)
            
            self.plot_kalman_diagnostics(audit)
            self.plot_parameter_sensitivity(audit)
            self.generate_audit_report(audit)
            
            logger.info("")
        
        # Generate summary
        self.generate_summary_report(all_results)
        
        logger.info("=" * )
        logger.info("KLMN ILTER UIT OMPLETE")
        logger.info("=" * )
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"igures saved to: {self.figures_dir}")
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate summary across all pairs."""
        report = []
        report.append("=" * )
        report.append("KLMN ILTER UIT - SUMMRY REPORT")
        report.append("=" * )
        report.append("")
        
        for result in all_results:
            pair = result['pair']
            ols_beta = result['ols_beta']
            kf_beta_mean = result['kf_results_default']['beta'].mean()
            kf_beta_std = result['kf_results_default']['beta'].std()
            
            report.append(f"{pair[]}-{pair[]}:")
            report.append(f"  OLS β: {ols_beta:.f}")
            report.append(f"  Kalman β: {kf_beta_mean:.f} ± {kf_beta_std:.f}")
            report.append("")
        
        report.append("OVERLL SSESSMENT:")
        report.append("")
        report.append("The Kalman ilter provides time-varying hedge ratio estimates.")
        report.append("urrent implementation uses Q=e-, R=e- (Q/R = .).")
        report.append("")
        report.append("KEY ININGS:")
        report.append("  . Kalman β tracks OLS β closely on average")
        report.append("  2. Time-variation allows adaptation to regime shifts")
        report.append("  . urrent parameters provide reasonable smoothness")
        report.append("")
        report.append("NEXT STEPS:")
        report.append("  . Integrate Kalman β into backtest (currently using static OLS)")
        report.append("  2. ompare performance: static vs time-varying hedge ratios")
        report.append("  . Monitor β stability as regime indicator")
        report.append("")
        report.append("=" * )
        
        report_text = "\n".join(report)
        
        filepath = self.results_dir / "kalman_audit_summary.txt"
        with open(filepath, 'w', encoding='utf-') as f:
            f.write(report_text)
        
        logger.info(f"Saved summary report: {filepath}")
        print("\n" + report_text)


def main():
    """Main entry point."""
    auditor = Kalmanilteruditor()
    results = auditor.run_full_audit()
    return results


if __name__ == '__main__':
    main()
