"""
Value-at-Risk (VaR) calculation.

Parametric VaR using GARCH-type models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union


class VaRCalculator:
    """Calculate Value-at-Risk from volatility forecasts."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def parametric_var(
        self,
        volatility_forecast: Union[float, np.ndarray],
        distribution: str = "normal",
        df: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute parametric VaR.
        
        VaR_alpha = -mu - sigma * z_alpha
        
        where z_alpha is the alpha-quantile of the distribution.
        
        Args:
            volatility_forecast: Forecasted volatility (sigma)
            distribution: "normal" or "t"
            df: Degrees of freedom for Student-t distribution
        
        Returns:
            VaR (positive number representing potential loss)
        """
        if distribution == "normal":
            z_alpha = stats.norm.ppf(self.alpha)
        elif distribution == "t":
            if df is None:
                raise ValueError("Degrees of freedom required for t-distribution")
            z_alpha = stats.t.ppf(self.alpha, df)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # VaR is positive (loss)
        var = -z_alpha * volatility_forecast
        
        return var
    
    def historical_var(
        self,
        returns: np.ndarray,
        window: Optional[int] = None
    ) -> float:
        """
        Compute historical VaR (non-parametric).
        
        Args:
            returns: Historical returns
            window: Use last N observations (None = all)
        
        Returns:
            VaR
        """
        if window is not None:
            returns = returns[-window:]
        
        # VaR is the alpha-quantile (as positive loss)
        var = -np.quantile(returns, self.alpha)
        
        return var
    
    def rolling_var(
        self,
        volatility_forecasts: pd.Series,
        distribution: str = "normal",
        df: Optional[float] = None
    ) -> pd.Series:
        """
        Compute rolling VaR from volatility forecasts.
        
        Args:
            volatility_forecasts: Series of volatility forecasts
            distribution: Distribution assumption
            df: Degrees of freedom for t-distribution
        
        Returns:
            Series of VaR values
        """
        var_values = self.parametric_var(
            volatility_forecasts.values,
            distribution=distribution,
            df=df
        )
        
        return pd.Series(var_values, index=volatility_forecasts.index, name=f"VaR_{self.confidence_level}")
    
    def var_from_model(
        self,
        model,
        returns: np.ndarray,
        horizon: int = 1,
        distribution: str = "normal"
    ) -> float:
        """
        Compute VaR directly from fitted model.
        
        Args:
            model: Fitted volatility model
            returns: Return series used for fitting
            horizon: Forecast horizon
            distribution: Distribution assumption
        
        Returns:
            VaR
        """
        # Get volatility forecast
        vol_forecast = model.forecast(returns - np.mean(returns), horizon=horizon)
        vol = np.sqrt(vol_forecast[-1])
        
        # Compute VaR
        var = self.parametric_var(vol, distribution=distribution)
        
        return var
