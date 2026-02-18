"""
Stylized facts and descriptive statistics for financial returns.

Key empirical regularities:
- Fat tails (excess kurtosis)
- Volatility clustering (ARCH effects)
- Leverage effect (asymmetric volatility response)
- Near-zero autocorrelation in returns
- Strong autocorrelation in absolute/squared returns
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class StylizedFacts:
    """Analyze empirical properties of return series."""
    
    def __init__(self, returns: pd.Series):
        # Convert to pandas Series if numpy array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        self.returns = returns.dropna()
        self.n = len(self.returns)
    
    def summary_statistics(self) -> Dict[str, float]:
        """
        Core distributional statistics.
        
        Financial returns typically show:
        - Mean ≈ 0 (after scaling)
        - Std > 0 (volatility)
        - Skewness < 0 (left tail heavier)
        - Kurtosis > 3 (fat tails)
        """
        r = self.returns
        
        stats_dict = {
            "mean": r.mean(),
            "std": r.std(),
            "skewness": stats.skew(r),
            "kurtosis": stats.kurtosis(r, fisher=False),  # Pearson kurtosis
            "excess_kurtosis": stats.kurtosis(r, fisher=True),
            "min": r.min(),
            "max": r.max(),
            "median": r.median(),
            "n_obs": self.n
        }
        
        return stats_dict
    
    def normality_tests(self) -> Dict[str, Tuple[float, float]]:
        """
        Test for normality.
        
        Returns are typically non-normal due to fat tails.
        Jarque-Bera test combines skewness and kurtosis.
        """
        r = self.returns
        
        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(r)
        
        # Shapiro-Wilk test (more powerful for small samples)
        if self.n < 5000:
            sw_stat, sw_pval = stats.shapiro(r)
        else:
            sw_stat, sw_pval = np.nan, np.nan
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(r, 'norm', args=(r.mean(), r.std()))
        
        return {
            "jarque_bera": (jb_stat, jb_pval),
            "shapiro_wilk": (sw_stat, sw_pval),
            "kolmogorov_smirnov": (ks_stat, ks_pval)
        }
    
    def autocorrelation_structure(self, max_lag: int = 20) -> pd.DataFrame:
        """
        Autocorrelation in returns, absolute returns, squared returns.
        
        Stylized fact: returns show little autocorrelation,
        but |returns| and returns^2 show strong persistence.
        This is the signature of volatility clustering.
        """
        r = self.returns
        
        acf_returns = [r.autocorr(lag=i) for i in range(1, max_lag + 1)]
        acf_abs = [r.abs().autocorr(lag=i) for i in range(1, max_lag + 1)]
        acf_squared = [(r**2).autocorr(lag=i) for i in range(1, max_lag + 1)]
        
        df = pd.DataFrame({
            "lag": range(1, max_lag + 1),
            "acf_returns": acf_returns,
            "acf_abs_returns": acf_abs,
            "acf_squared_returns": acf_squared
        })
        
        return df
    
    def ljung_box_test(self, lags: int = 10, series_type: str = "returns") -> Tuple[float, float]:
        """
        Ljung-Box test for serial correlation.
        
        H0: No autocorrelation up to lag k
        Used to test for ARCH effects in squared returns.
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            if series_type == "returns":
                series = self.returns
            elif series_type == "squared":
                series = self.returns ** 2
            elif series_type == "absolute":
                series = self.returns.abs()
            else:
                raise ValueError(f"Unknown series_type: {series_type}")
            
            result = acorr_ljungbox(series, lags=lags, return_df=False)
            
            # Return test statistic and p-value for the highest lag
            return result[0][-1], result[1][-1]
        except Exception as e:
            # Fallback: manual calculation
            print(f"Warning: Using manual Ljung-Box calculation due to: {e}")
            if series_type == "returns":
                series = self.returns
            elif series_type == "squared":
                series = self.returns ** 2
            elif series_type == "absolute":
                series = self.returns.abs()
            else:
                raise ValueError(f"Unknown series_type: {series_type}")
            
            n = len(series)
            acf_vals = [series.autocorr(lag=i) for i in range(1, lags + 1)]
            lb_stat = n * (n + 2) * sum(acf**2 / (n - i) for i, acf in enumerate(acf_vals, 1))
            
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lb_stat, df=lags)
            
            return lb_stat, p_value
    
    def arch_lm_test(self, lags: int = 5) -> Tuple[float, float]:
        """
        ARCH-LM test for conditional heteroskedasticity.
        
        H0: No ARCH effects
        Rejection indicates time-varying volatility.
        """
        try:
            from statsmodels.stats.diagnostic import het_arch
            lm_stat, lm_pval, f_stat, f_pval = het_arch(self.returns, nlags=lags)
            return lm_stat, lm_pval
        except Exception as e:
            # Fallback: manual calculation
            print(f"Warning: Using manual ARCH-LM calculation due to: {e}")
            from scipy.stats import chi2
            
            # Regress squared returns on lagged squared returns
            r_sq = self.returns ** 2
            n = len(r_sq)
            
            # Create lagged matrix
            X = np.column_stack([r_sq.shift(i).values for i in range(1, lags + 1)])
            y = r_sq.values
            
            # Remove NaN rows
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid]
            y = y[valid]
            
            # Add constant
            X = np.column_stack([np.ones(len(X)), X])
            
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # LM statistic
            lm_stat = len(y) * r_squared
            p_value = 1 - chi2.cdf(lm_stat, df=lags)
            
            return lm_stat, p_value
    
    def leverage_effect_correlation(self, lag: int = 1) -> float:
        """
        Correlation between returns and future squared returns.
        
        Negative correlation indicates leverage effect:
        negative shocks increase volatility more than positive shocks.
        """
        r = self.returns
        r_squared_lead = (r ** 2).shift(-lag)
        
        corr = r.corr(r_squared_lead)
        
        return corr
    
    def tail_index_estimation(self, tail_fraction: float = 0.05) -> Dict[str, float]:
        """
        Estimate tail index using Hill estimator.
        
        Lower tail index = fatter tails.
        Normal distribution has infinite tail index.
        """
        r = self.returns
        
        # Left tail (negative returns)
        left_tail = -r[r < 0].sort_values()
        n_left = int(len(left_tail) * tail_fraction)
        if n_left > 1:
            left_tail_sorted = left_tail.iloc[:n_left]
            hill_left = np.mean(np.log(left_tail_sorted / left_tail_sorted.iloc[n_left-1]))
            alpha_left = 1 / hill_left if hill_left > 0 else np.nan
        else:
            alpha_left = np.nan
        
        # Right tail (positive returns)
        right_tail = r[r > 0].sort_values(ascending=False)
        n_right = int(len(right_tail) * tail_fraction)
        if n_right > 1:
            right_tail_sorted = right_tail.iloc[:n_right]
            hill_right = np.mean(np.log(right_tail_sorted / right_tail_sorted.iloc[n_right-1]))
            alpha_right = 1 / hill_right if hill_right > 0 else np.nan
        else:
            alpha_right = np.nan
        
        return {
            "left_tail_index": alpha_left,
            "right_tail_index": alpha_right
        }
    
    def full_report(self) -> Dict:
        """Generate complete stylized facts report."""
        report = {
            "summary_stats": self.summary_statistics(),
            "normality_tests": self.normality_tests(),
            "ljung_box_returns": self.ljung_box_test(lags=10, series_type="returns"),
            "ljung_box_squared": self.ljung_box_test(lags=10, series_type="squared"),
            "arch_lm": self.arch_lm_test(lags=5),
            "leverage_effect": self.leverage_effect_correlation(lag=1),
            "tail_indices": self.tail_index_estimation()
        }
        
        return report


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute realized volatility as rolling standard deviation.
    
    Simple proxy for latent volatility. More sophisticated measures
    (realized kernel, bipower variation) can be added for high-freq data.
    """
    rv = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return rv.dropna()
