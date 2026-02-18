"""
Model diagnostics for GARCH-type models.

Post-estimation checks:
- Standardized residuals should be i.i.d.
- No remaining ARCH effects
- Parameter stability
- Persistence analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelDiagnostics:
    """Diagnostic tests for fitted volatility models."""
    
    def __init__(self, standardized_residuals: np.ndarray, model_name: str = "Model"):
        self.std_resid = standardized_residuals
        self.model_name = model_name
        self.n = len(standardized_residuals)
    
    def residual_statistics(self) -> Dict[str, float]:
        """
        Standardized residuals should have:
        - Mean ≈ 0
        - Variance ≈ 1
        - Reduced kurtosis (compared to raw returns)
        """
        resid = self.std_resid
        
        return {
            "mean": np.mean(resid),
            "std": np.std(resid),
            "skewness": stats.skew(resid),
            "kurtosis": stats.kurtosis(resid, fisher=False),
            "excess_kurtosis": stats.kurtosis(resid, fisher=True),
            "min": np.min(resid),
            "max": np.max(resid)
        }
    
    def ljung_box_test(self, lags: int = 10, squared: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test for remaining autocorrelation.
        
        If model is correctly specified:
        - Residuals should show no autocorrelation
        - Squared residuals should show no autocorrelation (no ARCH effects)
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        series = self.std_resid ** 2 if squared else self.std_resid
        result = acorr_ljungbox(series, lags=lags, return_df=True)
        
        return result['lb_stat'].values, result['lb_pvalue'].values
    
    def arch_lm_test(self, lags: int = 5) -> Tuple[float, float]:
        """
        Test for remaining ARCH effects in standardized residuals.
        
        Should not reject H0 if model captured all heteroskedasticity.
        """
        from statsmodels.stats.diagnostic import het_arch
        
        lm_stat, lm_pval, _, _ = het_arch(self.std_resid, nlags=lags)
        
        return lm_stat, lm_pval
    
    def jarque_bera_test(self) -> Tuple[float, float]:
        """Test for normality of standardized residuals."""
        jb_stat, jb_pval = stats.jarque_bera(self.std_resid)
        return jb_stat, jb_pval
    
    def sign_bias_test(self, returns: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Engle-Ng sign bias test for asymmetric effects.
        
        Tests whether positive and negative shocks have different
        impacts on volatility beyond what the model captures.
        """
        resid = self.std_resid
        resid_sq = resid ** 2
        
        # Indicator for negative returns
        S_neg = (returns < 0).astype(float)
        S_pos = (returns >= 0).astype(float)
        
        # Sign bias: effect of sign on squared residuals
        from scipy.stats import pearsonr
        
        sign_bias_stat, sign_bias_pval = pearsonr(S_neg[1:], resid_sq[1:])
        
        # Negative size bias
        neg_size_bias_stat, neg_size_bias_pval = pearsonr(
            S_neg[1:] * returns[1:], resid_sq[1:]
        )
        
        # Positive size bias
        pos_size_bias_stat, pos_size_bias_pval = pearsonr(
            S_pos[1:] * returns[1:], resid_sq[1:]
        )
        
        return {
            "sign_bias": (sign_bias_stat, sign_bias_pval),
            "negative_size_bias": (neg_size_bias_stat, neg_size_bias_pval),
            "positive_size_bias": (pos_size_bias_stat, pos_size_bias_pval)
        }
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Generate diagnostic plots:
        1. Standardized residuals over time
        2. Histogram with normal overlay
        3. Q-Q plot
        4. ACF of residuals
        5. ACF of squared residuals
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f"{self.model_name} - Diagnostic Plots", fontsize=14, y=0.995)
        
        resid = self.std_resid
        
        # 1. Time series of standardized residuals
        axes[0, 0].plot(resid, linewidth=0.5, alpha=0.7)
        axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[0, 0].axhline(2, color='red', linestyle='--', linewidth=0.6, alpha=0.5)
        axes[0, 0].axhline(-2, color='red', linestyle='--', linewidth=0.6, alpha=0.5)
        axes[0, 0].set_title("Standardized Residuals")
        axes[0, 0].set_ylabel("Std. Residuals")
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Histogram with normal overlay
        axes[0, 1].hist(resid, bins=50, density=True, alpha=0.7, edgecolor='black')
        x_range = np.linspace(resid.min(), resid.max(), 100)
        axes[0, 1].plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', linewidth=2, label='N(0,1)')
        axes[0, 1].set_title("Histogram vs Normal")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        axes[1, 0].grid(alpha=0.3)
        
        # 4. ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid, lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title("ACF of Standardized Residuals")
        axes[1, 1].grid(alpha=0.3)
        
        # 5. ACF of squared residuals
        plot_acf(resid**2, lags=20, ax=axes[2, 0], alpha=0.05)
        axes[2, 0].set_title("ACF of Squared Standardized Residuals")
        axes[2, 0].grid(alpha=0.3)
        
        # 6. Squared residuals over time
        axes[2, 1].plot(resid**2, linewidth=0.5, alpha=0.7)
        axes[2, 1].axhline(1, color='black', linestyle='--', linewidth=0.8)
        axes[2, 1].set_title("Squared Standardized Residuals")
        axes[2, 1].set_ylabel("Squared Std. Residuals")
        axes[2, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def full_diagnostic_report(self, returns: Optional[np.ndarray] = None) -> Dict:
        """Generate complete diagnostic report."""
        report = {
            "residual_stats": self.residual_statistics(),
            "ljung_box_resid": self.ljung_box_test(lags=10, squared=False),
            "ljung_box_squared": self.ljung_box_test(lags=10, squared=True),
            "arch_lm": self.arch_lm_test(lags=5),
            "jarque_bera": self.jarque_bera_test()
        }
        
        if returns is not None:
            report["sign_bias"] = self.sign_bias_test(returns)
        
        return report


def persistence_analysis(params: Dict[str, float], model_type: str) -> Dict[str, float]:
    """
    Analyze volatility persistence and half-life.
    
    For GARCH(1,1): persistence = alpha + beta
    Half-life: time for shock to decay to 50%
    """
    if model_type == "GARCH":
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        persistence = alpha + beta
        
        if persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf
        
        return {
            "persistence": persistence,
            "half_life_days": half_life,
            "mean_reverting": persistence < 1
        }
    
    elif model_type == "EGARCH":
        # For EGARCH, persistence is beta
        beta = params.get("beta[1]", 0)
        persistence = beta
        
        if persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf
        
        return {
            "persistence": persistence,
            "half_life_days": half_life,
            "mean_reverting": persistence < 1
        }
    
    else:
        return {"persistence": np.nan, "half_life_days": np.nan, "mean_reverting": False}
