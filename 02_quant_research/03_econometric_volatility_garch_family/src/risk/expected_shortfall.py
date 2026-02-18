"""
Expected Shortfall (ES) / Conditional Value-at-Risk (CVaR) calculation.

ES is the expected loss given that VaR is exceeded.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union


class ESCalculator:
    """Calculate Expected Shortfall from volatility forecasts."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ES calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% ES)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def parametric_es(
        self,
        volatility_forecast: Union[float, np.ndarray],
        distribution: str = "normal",
        df: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute parametric Expected Shortfall.
        
        For normal distribution:
        ES_alpha = sigma * phi(z_alpha) / alpha
        
        where phi is the standard normal PDF.
        
        Args:
            volatility_forecast: Forecasted volatility (sigma)
            distribution: "normal" or "t"
            df: Degrees of freedom for Student-t distribution
        
        Returns:
            ES (positive number representing expected loss beyond VaR)
        """
        if distribution == "normal":
            z_alpha = stats.norm.ppf(self.alpha)
            phi_z = stats.norm.pdf(z_alpha)
            es = volatility_forecast * phi_z / self.alpha
        
        elif distribution == "t":
            if df is None:
                raise ValueError("Degrees of freedom required for t-distribution")
            
            z_alpha = stats.t.ppf(self.alpha, df)
            pdf_z = stats.t.pdf(z_alpha, df)
            
            # ES for Student-t
            es = volatility_forecast * pdf_z / self.alpha * (df + z_alpha**2) / (df - 1)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return es
    
    def historical_es(
        self,
        returns: np.ndarray,
        window: Optional[int] = None
    ) -> float:
        """
        Compute historical ES (non-parametric).
        
        ES is the average of returns below the VaR threshold.
        
        Args:
            returns: Historical returns
            window: Use last N observations (None = all)
        
        Returns:
            ES
        """
        if window is not None:
            returns = returns[-window:]
        
        # VaR threshold
        var_threshold = np.quantile(returns, self.alpha)
        
        # ES is average of returns below VaR (as positive loss)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return np.nan
        
        es = -np.mean(tail_returns)
        
        return es
    
    def rolling_es(
        self,
        volatility_forecasts: pd.Series,
        distribution: str = "normal",
        df: Optional[float] = None
    ) -> pd.Series:
        """
        Compute rolling ES from volatility forecasts.
        
        Args:
            volatility_forecasts: Series of volatility forecasts
            distribution: Distribution assumption
            df: Degrees of freedom for t-distribution
        
        Returns:
            Series of ES values
        """
        es_values = self.parametric_es(
            volatility_forecasts.values,
            distribution=distribution,
            df=df
        )
        
        return pd.Series(es_values, index=volatility_forecasts.index, name=f"ES_{self.confidence_level}")
    
    def es_from_model(
        self,
        model,
        returns: np.ndarray,
        horizon: int = 1,
        distribution: str = "normal"
    ) -> float:
        """
        Compute ES directly from fitted model.
        
        Args:
            model: Fitted volatility model
            returns: Return series used for fitting
            horizon: Forecast horizon
            distribution: Distribution assumption
        
        Returns:
            ES
        """
        # Get volatility forecast
        vol_forecast = model.forecast(returns - np.mean(returns), horizon=horizon)
        vol = np.sqrt(vol_forecast[-1])
        
        # Compute ES
        es = self.parametric_es(vol, distribution=distribution)
        
        return es
