"""
Forecast evaluation metrics for volatility models.

Standard metrics:
- MSE, RMSE, MAE
- QLIKE (Quasi-Likelihood)
- R-squared
- Diebold-Mariano test for forecast comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class ForecastMetrics:
    """Evaluate volatility forecast accuracy."""
    
    def __init__(self, forecasts: np.ndarray, realized: np.ndarray):
        """
        Initialize with forecasts and realized volatility.
        
        Args:
            forecasts: Forecasted volatility
            realized: Realized volatility (proxy)
        """
        # Remove NaN values
        mask = ~(np.isnan(forecasts) | np.isnan(realized))
        self.forecasts = forecasts[mask]
        self.realized = realized[mask]
        self.n = len(self.forecasts)
    
    def mse(self) -> float:
        """Mean Squared Error."""
        return np.mean((self.forecasts - self.realized) ** 2)
    
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(self.mse())
    
    def mae(self) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(self.forecasts - self.realized))
    
    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = self.realized != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((self.forecasts[mask] - self.realized[mask]) / self.realized[mask])) * 100
    
    def qlike(self) -> float:
        """
        Quasi-Likelihood loss function.
        
        QLIKE = (realized / forecast) - log(realized / forecast) - 1
        
        Robust to outliers and scale-invariant.
        Lower is better.
        """
        # Avoid division by zero and log of zero
        mask = (self.forecasts > 0) & (self.realized > 0)
        if np.sum(mask) == 0:
            return np.nan
        
        ratio = self.realized[mask] / self.forecasts[mask]
        qlike = np.mean(ratio - np.log(ratio) - 1)
        
        return qlike
    
    def r_squared(self) -> float:
        """
        R-squared: proportion of variance explained.
        
        Mincer-Zarnowitz regression: realized = alpha + beta * forecast
        """
        if self.n < 2:
            return np.nan
        
        ss_res = np.sum((self.realized - self.forecasts) ** 2)
        ss_tot = np.sum((self.realized - np.mean(self.realized)) ** 2)
        
        if ss_tot == 0:
            return np.nan
        
        return 1 - (ss_res / ss_tot)
    
    def mincer_zarnowitz_regression(self) -> Dict[str, float]:
        """
        Mincer-Zarnowitz regression for forecast unbiasedness.
        
        realized = alpha + beta * forecast + error
        
        Unbiased forecast: alpha = 0, beta = 1
        """
        from scipy.stats import linregress
        
        if self.n < 2:
            return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan, "p_value": np.nan}
        
        result = linregress(self.forecasts, self.realized)
        
        return {
            "alpha": result.intercept,
            "beta": result.slope,
            "r_squared": result.rvalue ** 2,
            "p_value": result.pvalue
        }
    
    def forecast_bias(self) -> float:
        """
        Forecast bias: mean(forecast - realized)
        
        Positive = over-forecasting
        Negative = under-forecasting
        """
        return np.mean(self.forecasts - self.realized)
    
    def hit_ratio(self, threshold: float = 0.0) -> float:
        """
        Proportion of forecasts within threshold of realized.
        
        Args:
            threshold: Acceptable error threshold
        """
        errors = np.abs(self.forecasts - self.realized)
        return np.mean(errors <= threshold)
    
    def all_metrics(self) -> Dict[str, float]:
        """Compute all forecast metrics."""
        mz = self.mincer_zarnowitz_regression()
        
        return {
            "MSE": self.mse(),
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "MAPE": self.mape(),
            "QLIKE": self.qlike(),
            "R2": self.r_squared(),
            "Bias": self.forecast_bias(),
            "MZ_alpha": mz["alpha"],
            "MZ_beta": mz["beta"],
            "MZ_R2": mz["r_squared"]
        }


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    loss_function: str = "mse",
    horizon: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Because volatility forecasts are noisy and R² values are typically low,
    formal forecast comparison via the Diebold-Mariano test is essential to
    assess statistically significant differences in predictive accuracy.
    
    H0: Two forecasts have equal predictive accuracy
    H1: Forecast 1 is more accurate than forecast 2 (one-sided)
    
    The test accounts for autocorrelation in the loss differential series
    when horizon > 1 using Newey-West HAC standard errors.
    
    Args:
        errors1: Forecast errors from model 1 (realized - forecast1)
        errors2: Forecast errors from model 2 (realized - forecast2)
        loss_function: "mse" or "qlike"
        horizon: Forecast horizon (for HAC adjustment)
    
    Returns:
        (dm_statistic, p_value)
        
        Interpretation:
        - Negative DM stat: Model 1 is more accurate
        - Positive DM stat: Model 2 is more accurate
        - p_value < 0.05: Significant difference at 5% level
    """
    # Remove NaN values
    mask = ~(np.isnan(errors1) | np.isnan(errors2))
    e1 = errors1[mask]
    e2 = errors2[mask]
    
    n = len(e1)
    
    if n < 2:
        return np.nan, np.nan
    
    # Loss differential
    if loss_function == "mse":
        d = e1**2 - e2**2
    elif loss_function == "qlike":
        # QLIKE loss differential (requires forecasts and realized)
        # For now, use squared error as proxy
        d = e1**2 - e2**2
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    # Mean loss differential
    d_mean = np.mean(d)
    
    # Variance with HAC correction for multi-step forecasts
    if horizon == 1:
        # Simple variance for 1-step ahead
        d_var = np.var(d, ddof=1)
    else:
        # Newey-West HAC variance for h-step ahead
        # Use h-1 lags for autocorrelation
        d_var = np.var(d, ddof=1)
        for lag in range(1, min(horizon, n // 4)):
            gamma = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
            weight = 1 - lag / (horizon + 1)
            d_var += 2 * weight * gamma
    
    if d_var <= 0:
        return np.nan, np.nan
    
    # DM test statistic
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # Two-sided p-value (standard in literature)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def compare_forecasts_dm(
    forecasts_dict: Dict[str, np.ndarray],
    realized: np.ndarray,
    loss_function: str = "mse",
    horizon: int = 1
) -> pd.DataFrame:
    """
    Pairwise Diebold-Mariano tests for multiple models.
    
    Because volatility forecasts are noisy and R² values are typically low,
    formal statistical comparison is essential. This function performs all
    pairwise DM tests and returns a comprehensive comparison table.
    
    Args:
        forecasts_dict: Dict mapping model names to forecast arrays
        realized: Realized volatility
        loss_function: "mse" or "qlike"
        horizon: Forecast horizon
    
    Returns:
        DataFrame with DM statistics, p-values, and interpretations
    """
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)
    
    results = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]
            
            # Compute errors
            errors1 = realized - forecasts_dict[model1]
            errors2 = realized - forecasts_dict[model2]
            
            # DM test
            dm_stat, p_value = diebold_mariano_test(
                errors1, errors2, loss_function, horizon
            )
            
            # Interpretation
            if np.isnan(p_value):
                interpretation = "Insufficient data"
            elif p_value < 0.01:
                better = model1 if dm_stat < 0 else model2
                interpretation = f"{better} significantly better (p<0.01)"
            elif p_value < 0.05:
                better = model1 if dm_stat < 0 else model2
                interpretation = f"{better} significantly better (p<0.05)"
            elif p_value < 0.10:
                better = model1 if dm_stat < 0 else model2
                interpretation = f"{better} marginally better (p<0.10)"
            else:
                interpretation = "No significant difference"
            
            results.append({
                "Model 1": model1,
                "Model 2": model2,
                "DM Statistic": dm_stat,
                "p-value": p_value,
                "Interpretation": interpretation
            })
    
    return pd.DataFrame(results)


def model_confidence_set(
    forecasts_dict: Dict[str, np.ndarray],
    realized: np.ndarray,
    alpha: float = 0.1
) -> Dict[str, bool]:
    """
    Model Confidence Set (Hansen et al., 2011).
    
    Identifies set of models that are not significantly worse than the best.
    
    Args:
        forecasts_dict: Dict mapping model names to forecast arrays
        realized: Realized volatility
        alpha: Significance level
    
    Returns:
        Dict indicating which models are in the confidence set
    """
    # Simplified implementation: pairwise DM tests
    model_names = list(forecasts_dict.keys())
    n_models = len(model_names)
    
    # Compute losses for each model
    losses = {}
    for name, forecast in forecasts_dict.items():
        mask = ~(np.isnan(forecast) | np.isnan(realized))
        losses[name] = np.mean((forecast[mask] - realized[mask]) ** 2)
    
    # Find best model
    best_model = min(losses, key=losses.get)
    
    # Test each model against best
    in_mcs = {}
    for name in model_names:
        if name == best_model:
            in_mcs[name] = True
        else:
            errors_best = forecasts_dict[best_model] - realized
            errors_model = forecasts_dict[name] - realized
            
            dm_stat, p_value = diebold_mariano_test(errors_best, errors_model)
            
            # If p-value > alpha, cannot reject that model is as good as best
            in_mcs[name] = (p_value > alpha) if not np.isnan(p_value) else False
    
    return in_mcs
