"""
Rolling-window volatility forecasting.

Implements out-of-sample forecast generation with:
- Fixed or expanding window
- Multiple forecast horizons
- Model re-estimation at each step
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from tqdm import tqdm


class RollingForecast:
    """Rolling-window forecast generator for volatility models."""
    
    def __init__(
        self,
        model_class: Callable,
        model_params: Dict,
        window_size: int = 1000,
        window_type: str = "fixed"
    ):
        """
        Initialize rolling forecast.
        
        Args:
            model_class: Model class (e.g., GARCHModel)
            model_params: Parameters to pass to model constructor
            window_size: Size of estimation window
            window_type: "fixed" or "expanding"
        """
        self.model_class = model_class
        self.model_params = model_params
        self.window_size = window_size
        self.window_type = window_type
        
        self.forecasts = None
        self.realized_vol = None
        self.forecast_dates = None
    
    def generate_forecasts(
        self,
        returns: pd.Series,
        horizon: int = 1,
        refit_frequency: int = 1,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate rolling out-of-sample forecasts.
        
        Args:
            returns: Full return series
            horizon: Forecast horizon (days ahead)
            refit_frequency: Re-estimate model every N days
            verbose: Show progress bar
        
        Returns:
            DataFrame with forecasts and realized volatility
        """
        n = len(returns)
        
        if self.window_type == "fixed":
            start_idx = self.window_size
        else:  # expanding
            start_idx = self.window_size
        
        n_forecasts = n - start_idx - horizon + 1
        
        if n_forecasts <= 0:
            raise ValueError("Not enough data for forecasting")
        
        forecasts = []
        realized_vols = []
        dates = []
        
        iterator = range(start_idx, n - horizon + 1, refit_frequency)
        if verbose:
            iterator = tqdm(iterator, desc="Rolling forecasts")
        
        for t in iterator:
            # Estimation window
            if self.window_type == "fixed":
                train_returns = returns.iloc[t-self.window_size:t].values
            else:  # expanding
                train_returns = returns.iloc[:t].values
            
            # Fit model
            try:
                model = self.model_class(**self.model_params)
                model.fit(train_returns, verbose=False)
                
                # Generate forecast
                forecast_var = model.forecast(train_returns - np.mean(train_returns), horizon=horizon)
                forecast_vol = np.sqrt(forecast_var[-1])  # Last horizon
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Model fitting failed at t={t}: {e}")
                forecast_vol = np.nan
            
            # Realized volatility (proxy: absolute return or squared return)
            if t + horizon <= n:
                # Use squared return as realized variance proxy
                realized_var = returns.iloc[t:t+horizon].values ** 2
                realized_vol = np.sqrt(np.mean(realized_var))
            else:
                realized_vol = np.nan
            
            forecasts.append(forecast_vol)
            realized_vols.append(realized_vol)
            dates.append(returns.index[t])
        
        # Create DataFrame
        df = pd.DataFrame({
            "forecast": forecasts,
            "realized": realized_vols
        }, index=dates)
        
        self.forecasts = df["forecast"]
        self.realized_vol = df["realized"]
        self.forecast_dates = dates
        
        return df
    
    def get_forecast_errors(self) -> pd.Series:
        """Compute forecast errors: realized - forecast."""
        if self.forecasts is None:
            raise ValueError("No forecasts generated yet")
        
        return self.realized_vol - self.forecasts
    
    def get_squared_forecast_errors(self) -> pd.Series:
        """Compute squared forecast errors."""
        errors = self.get_forecast_errors()
        return errors ** 2


def compare_models_rolling(
    returns: pd.Series,
    models_config: Dict[str, Dict],
    window_size: int = 1000,
    horizon: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models using rolling forecasts.
    
    Args:
        returns: Return series
        models_config: Dict mapping model names to {class, params}
        window_size: Estimation window size
        horizon: Forecast horizon
        verbose: Show progress
    
    Returns:
        DataFrame with forecasts from all models
    """
    results = {}
    
    for model_name, config in models_config.items():
        if verbose:
            print(f"\nGenerating forecasts for {model_name}...")
        
        roller = RollingForecast(
            model_class=config["class"],
            model_params=config["params"],
            window_size=window_size,
            window_type="fixed"
        )
        
        df = roller.generate_forecasts(returns, horizon=horizon, verbose=verbose)
        results[model_name] = df["forecast"]
    
    # Add realized volatility
    results["realized"] = df["realized"]
    
    return pd.DataFrame(results)
