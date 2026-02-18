"""
Backtesting procedures for VaR and ES.

Statistical tests for model validation:
- Kupiec unconditional coverage test
- Christoffersen independence test
- ES consistency checks
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


class VaRBacktest:
    """Backtest Value-at-Risk forecasts."""
    
    def __init__(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.95
    ):
        """
        Initialize VaR backtest.
        
        Args:
            returns: Realized returns
            var_forecasts: VaR forecasts (positive values)
            confidence_level: Confidence level used for VaR
        """
        # Remove NaN values
        mask = ~(np.isnan(returns) | np.isnan(var_forecasts))
        self.returns = returns[mask]
        self.var_forecasts = var_forecasts[mask]
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n = len(self.returns)
        
        # Compute violations (hits)
        self.violations = self.returns < -self.var_forecasts
        self.n_violations = np.sum(self.violations)
        self.violation_rate = self.n_violations / self.n
    
    def hit_ratio(self) -> float:
        """
        Proportion of VaR violations.
        
        Should be close to alpha if model is correctly specified.
        """
        return self.violation_rate
    
    def kupiec_test(self) -> Tuple[float, float]:
        """
        Kupiec unconditional coverage test (1995).
        
        H0: Violation rate = alpha (model is correctly specified)
        
        Test statistic: LR = -2 * log(L(alpha) / L(p_hat))
        where p_hat = observed violation rate
        
        Returns:
            (test_statistic, p_value)
        """
        n = self.n
        x = self.n_violations
        p = self.alpha
        p_hat = self.violation_rate
        
        if x == 0 or x == n:
            # Degenerate case
            return np.inf, 0.0
        
        # Log-likelihood ratio
        lr_stat = -2 * (
            x * np.log(p / p_hat) + (n - x) * np.log((1 - p) / (1 - p_hat))
        )
        
        # Chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return lr_stat, p_value
    
    def christoffersen_test(self) -> Tuple[float, float]:
        """
        Christoffersen independence test (1998).
        
        H0: Violations are independent (no clustering)
        
        Tests whether violations are serially independent.
        
        Returns:
            (test_statistic, p_value)
        """
        violations = self.violations.astype(int)
        
        # Transition counts
        n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Transition probabilities
        if n_00 + n_01 == 0 or n_10 + n_11 == 0:
            return np.nan, np.nan
        
        p_01 = n_01 / (n_00 + n_01)
        p_11 = n_11 / (n_10 + n_11)
        p_hat = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)
        
        if p_01 == 0 or p_01 == 1 or p_11 == 0 or p_11 == 1:
            return np.nan, np.nan
        
        # Log-likelihood ratio
        lr_stat = -2 * (
            n_00 * np.log(1 - p_hat) + n_01 * np.log(p_hat) +
            n_10 * np.log(1 - p_hat) + n_11 * np.log(p_hat) -
            n_00 * np.log(1 - p_01) - n_01 * np.log(p_01) -
            n_10 * np.log(1 - p_11) - n_11 * np.log(p_11)
        )
        
        # Chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return lr_stat, p_value
    
    def conditional_coverage_test(self) -> Tuple[float, float]:
        """
        Christoffersen conditional coverage test (1998).
        
        Combines unconditional coverage and independence tests.
        
        H0: Correct unconditional coverage AND independence
        
        Returns:
            (test_statistic, p_value)
        """
        uc_stat, _ = self.kupiec_test()
        ind_stat, _ = self.christoffersen_test()
        
        if np.isnan(uc_stat) or np.isnan(ind_stat):
            return np.nan, np.nan
        
        # Combined test statistic
        cc_stat = uc_stat + ind_stat
        
        # Chi-squared distribution with 2 degrees of freedom
        p_value = 1 - stats.chi2.cdf(cc_stat, df=2)
        
        return cc_stat, p_value
    
    def traffic_light_test(self) -> str:
        """
        Basel traffic light test for VaR models.
        
        Green zone: 0-4 violations (99% VaR, 250 days)
        Yellow zone: 5-9 violations
        Red zone: 10+ violations
        
        Returns:
            "green", "yellow", or "red"
        """
        # Adjust for sample size and confidence level
        expected_violations = self.n * self.alpha
        
        # Basel thresholds (scaled)
        green_threshold = expected_violations + 1.65 * np.sqrt(expected_violations * (1 - self.alpha))
        yellow_threshold = expected_violations + 2.33 * np.sqrt(expected_violations * (1 - self.alpha))
        
        if self.n_violations <= green_threshold:
            return "green"
        elif self.n_violations <= yellow_threshold:
            return "yellow"
        else:
            return "red"
    
    def summary(self) -> Dict:
        """Generate backtest summary."""
        kupiec_stat, kupiec_pval = self.kupiec_test()
        christ_stat, christ_pval = self.christoffersen_test()
        cc_stat, cc_pval = self.conditional_coverage_test()
        
        return {
            "n_observations": self.n,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "expected_rate": self.alpha,
            "kupiec_stat": kupiec_stat,
            "kupiec_pval": kupiec_pval,
            "christoffersen_stat": christ_stat,
            "christoffersen_pval": christ_pval,
            "conditional_coverage_stat": cc_stat,
            "conditional_coverage_pval": cc_pval,
            "traffic_light": self.traffic_light_test()
        }


class ESBacktest:
    """Backtest Expected Shortfall forecasts."""
    
    def __init__(
        self,
        returns: np.ndarray,
        es_forecasts: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.95
    ):
        """
        Initialize ES backtest.
        
        Args:
            returns: Realized returns
            es_forecasts: ES forecasts (positive values)
            var_forecasts: VaR forecasts (for identifying violations)
            confidence_level: Confidence level
        """
        # Remove NaN values
        mask = ~(np.isnan(returns) | np.isnan(es_forecasts) | np.isnan(var_forecasts))
        self.returns = returns[mask]
        self.es_forecasts = es_forecasts[mask]
        self.var_forecasts = var_forecasts[mask]
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n = len(self.returns)
        
        # Identify VaR violations
        self.violations = self.returns < -self.var_forecasts
        self.n_violations = np.sum(self.violations)
    
    def average_tail_loss(self) -> float:
        """
        Average realized loss on days when VaR was violated.
        
        Should be close to average ES forecast if model is correct.
        """
        if self.n_violations == 0:
            return np.nan
        
        tail_losses = -self.returns[self.violations]
        return np.mean(tail_losses)
    
    def average_es_forecast(self) -> float:
        """Average ES forecast on violation days."""
        if self.n_violations == 0:
            return np.nan
        
        return np.mean(self.es_forecasts[self.violations])
    
    def es_ratio(self) -> float:
        """
        Ratio of realized tail loss to forecasted ES.
        
        Should be close to 1 if ES is correctly specified.
        """
        atl = self.average_tail_loss()
        aes = self.average_es_forecast()
        
        if np.isnan(atl) or np.isnan(aes) or aes == 0:
            return np.nan
        
        return atl / aes
    
    def mcneil_frey_test(self) -> Tuple[float, float]:
        """
        McNeil-Frey test for ES (2000).
        
        H0: ES is correctly specified
        
        Tests whether average tail loss equals average ES forecast.
        
        Returns:
            (test_statistic, p_value)
        """
        if self.n_violations < 2:
            return np.nan, np.nan
        
        tail_losses = -self.returns[self.violations]
        es_forecasts_violations = self.es_forecasts[self.violations]
        
        # Differences
        diff = tail_losses - es_forecasts_violations
        
        # t-test
        t_stat, p_value = stats.ttest_1samp(diff, 0)
        
        return t_stat, p_value
    
    def summary(self) -> Dict:
        """Generate ES backtest summary."""
        mf_stat, mf_pval = self.mcneil_frey_test()
        
        return {
            "n_observations": self.n,
            "n_violations": self.n_violations,
            "average_tail_loss": self.average_tail_loss(),
            "average_es_forecast": self.average_es_forecast(),
            "es_ratio": self.es_ratio(),
            "mcneil_frey_stat": mf_stat,
            "mcneil_frey_pval": mf_pval
        }
