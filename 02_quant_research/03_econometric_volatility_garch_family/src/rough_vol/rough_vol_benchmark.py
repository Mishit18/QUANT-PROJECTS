"""
Rough volatility benchmarking against classical GARCH models.

Compare:
1. Autocorrelation structure
2. Impulse response to shocks
3. Model mis-specification: fit GARCH to rough-vol data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .rbergomi import RoughBergomiModel


class RoughVolBenchmark:
    """Benchmark classical models against rough volatility."""
    
    def __init__(self, hurst: float = 0.1, seed: Optional[int] = None):
        """
        Initialize benchmark.
        
        Args:
            hurst: Hurst parameter for rough volatility
            seed: Random seed
        """
        self.hurst = hurst
        self.seed = seed
        self.rbergomi = RoughBergomiModel(hurst=hurst, seed=seed)
    
    def generate_rough_vol_data(
        self,
        n_steps: int = 2500,
        dt: float = 1/252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data from rough volatility model.
        
        Args:
            n_steps: Number of time steps (default: ~10 years daily)
            dt: Time step size
        
        Returns:
            (returns, variance)
        """
        returns, variance = self.rbergomi.simulate_returns(
            n_steps, dt, return_variance=True
        )
        
        return returns, variance
    
    def compare_autocorrelation(
        self,
        returns: np.ndarray,
        max_lag: int = 50
    ) -> pd.DataFrame:
        """
        Compare autocorrelation structure.
        
        Args:
            returns: Return series
            max_lag: Maximum lag
        
        Returns:
            DataFrame with empirical and theoretical ACF
        """
        # Empirical ACF of squared returns
        squared_returns = returns ** 2
        empirical_acf = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                empirical_acf.append(1.0)
            else:
                acf = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
                empirical_acf.append(acf)
        
        # Theoretical rough vol ACF
        theoretical_acf = self.rbergomi.theoretical_autocorrelation(max_lag)
        
        # Theoretical GARCH(1,1) ACF (exponential decay)
        # Assume typical persistence of 0.95
        garch_persistence = 0.95
        garch_acf = garch_persistence ** np.arange(max_lag + 1)
        
        df = pd.DataFrame({
            "lag": range(max_lag + 1),
            "empirical": empirical_acf,
            "rough_vol_theory": theoretical_acf,
            "garch_theory": garch_acf
        })
        
        return df
    
    def compare_impulse_response(
        self,
        garch_model,
        returns: np.ndarray,
        max_lag: int = 50
    ) -> pd.DataFrame:
        """
        Compare impulse response functions.
        
        Args:
            garch_model: Fitted GARCH model
            returns: Return series
            max_lag: Maximum lag
        
        Returns:
            DataFrame with impulse responses
        """
        # Rough vol impulse response
        rough_ir = self.rbergomi.impulse_response(max_lag)
        
        # GARCH impulse response (exponential decay)
        if hasattr(garch_model, 'persistence'):
            pers = garch_model.persistence()
        else:
            pers = 0.95
        
        garch_ir = pers ** np.arange(1, max_lag + 1)
        
        df = pd.DataFrame({
            "lag": range(1, max_lag + 1),
            "rough_vol": rough_ir,
            "garch": garch_ir
        })
        
        return df
    
    def fit_garch_to_rough_vol(
        self,
        garch_model_class,
        n_simulations: int = 100,
        n_steps: int = 1000
    ) -> Dict[str, List[float]]:
        """
        Fit GARCH model to rough volatility data.
        
        Shows model mis-specification effects.
        
        Args:
            garch_model_class: GARCH model class
            n_simulations: Number of Monte Carlo simulations
            n_steps: Length of each simulation
        
        Returns:
            Dict of parameter distributions
        """
        results = {
            "omega": [],
            "alpha": [],
            "beta": [],
            "persistence": []
        }
        
        for _ in range(n_simulations):
            # Generate rough vol data
            returns, _ = self.generate_rough_vol_data(n_steps)
            
            # Fit GARCH(1,1)
            try:
                model = garch_model_class(p=1, q=1)
                model.fit(returns, verbose=False)
                
                params = model.get_params()
                results["omega"].append(params.get("omega", np.nan))
                results["alpha"].append(params.get("alpha[1]", np.nan))
                results["beta"].append(params.get("beta[1]", np.nan))
                results["persistence"].append(model.persistence())
            except:
                # Skip failed fits
                continue
        
        return results
    
    def regime_comparison(
        self,
        returns: np.ndarray,
        variance: np.ndarray,
        threshold_percentile: float = 75
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare behavior in calm vs volatile regimes.
        
        Args:
            returns: Return series
            variance: Variance series
            threshold_percentile: Percentile for regime split
        
        Returns:
            Dict with statistics for each regime
        """
        # Define regimes based on variance
        threshold = np.percentile(variance, threshold_percentile)
        
        calm_regime = variance[:-1] < threshold
        volatile_regime = variance[:-1] >= threshold
        
        # Statistics for each regime
        calm_stats = {
            "mean_return": np.mean(returns[calm_regime]),
            "std_return": np.std(returns[calm_regime]),
            "skewness": pd.Series(returns[calm_regime]).skew(),
            "kurtosis": pd.Series(returns[calm_regime]).kurtosis()
        }
        
        volatile_stats = {
            "mean_return": np.mean(returns[volatile_regime]),
            "std_return": np.std(returns[volatile_regime]),
            "skewness": pd.Series(returns[volatile_regime]).skew(),
            "kurtosis": pd.Series(returns[volatile_regime]).kurtosis()
        }
        
        return {
            "calm": calm_stats,
            "volatile": volatile_stats
        }
    
    def short_maturity_behavior(
        self,
        n_paths: int = 1000,
        n_steps: int = 20,
        dt: float = 1/252
    ) -> Dict[str, np.ndarray]:
        """
        Analyze short-maturity volatility behavior.
        
        Rough volatility shows faster reaction to shocks at short maturities.
        
        Args:
            n_paths: Number of paths
            n_steps: Number of steps (short horizon)
            dt: Time step
        
        Returns:
            Dict with volatility statistics
        """
        returns_array, variance_array = self.rbergomi.simulate_multiple_paths(
            n_paths, n_steps, dt
        )
        
        # Compute realized volatility at different horizons
        horizons = [1, 5, 10, 20]
        realized_vols = {}
        
        for h in horizons:
            if h <= n_steps:
                # Rolling realized vol
                rv = np.sqrt(np.mean(returns_array[:, :h]**2, axis=1))
                realized_vols[f"{h}_day"] = rv
        
        return realized_vols
