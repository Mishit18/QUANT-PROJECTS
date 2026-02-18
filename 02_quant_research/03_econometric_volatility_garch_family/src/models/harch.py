"""
HARCH model implementation.

Heterogeneous ARCH (Müller et al., 1997).

Model specification:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
    sigma_t^2 = omega + sum_{i=1}^m alpha_i * (sum_{j=0}^{i-1} epsilon_{t-j})^2 / i

Key feature:
- Aggregates returns over multiple horizons
- Captures heterogeneous market participants (day traders, swing traders, investors)
- Each component represents different trading horizons
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional
import warnings


class HARCHModel:
    """HARCH model with maximum likelihood estimation."""
    
    def __init__(self, lags: List[int] = [1, 5, 22]):
        """
        Initialize HARCH model.
        
        Args:
            lags: List of aggregation horizons (e.g., [1, 5, 22] for daily, weekly, monthly)
        """
        self.lags = sorted(lags)
        self.m = len(lags)
        self.params = None
        self.fitted_variance = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
    
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters.
        
        omega, alpha_1, alpha_2, ..., alpha_m
        """
        var_unconditional = np.var(returns)
        
        omega = var_unconditional * 0.1
        alphas = np.ones(self.m) * 0.8 / self.m
        
        return np.concatenate([[omega], alphas])
    
    def _compute_aggregated_squared_returns(self, returns: np.ndarray, horizon: int, t: int) -> float:
        """
        Compute aggregated squared returns over horizon.
        
        (sum_{j=0}^{horizon-1} r_{t-j})^2 / horizon
        """
        if t < horizon:
            return 0.0
        
        aggregated = np.sum(returns[t-horizon:t])
        return (aggregated ** 2) / horizon
    
    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        sigma_t^2 = omega + sum_{i=1}^m alpha_i * (sum_{j=0}^{lag_i-1} epsilon_{t-j})^2 / lag_i
        """
        n = len(returns)
        variance = np.zeros(n)
        
        omega = params[0]
        alphas = params[1:]
        
        # Initialize with unconditional variance
        var_unconditional = np.var(returns)
        max_lag = max(self.lags)
        variance[:max_lag] = var_unconditional
        
        # Recursive computation
        for t in range(max_lag, n):
            harch_terms = 0.0
            
            for i, lag in enumerate(self.lags):
                agg_sq_ret = self._compute_aggregated_squared_returns(returns, lag, t)
                harch_terms += alphas[i] * agg_sq_ret
            
            variance[t] = omega + harch_terms
        
        return variance
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for minimization.
        """
        # Parameter constraints
        if params[0] <= 0:  # omega > 0
            return 1e10
        if np.any(params[1:] < 0):  # alpha_i >= 0
            return 1e10
        if np.sum(params[1:]) >= 1:  # sum(alpha_i) < 1 for stationarity
            return 1e10
        
        variance = self._compute_variance(params, returns)
        
        # Avoid numerical issues
        variance = np.maximum(variance, 1e-8)
        
        # Log-likelihood
        log_lik = -0.5 * np.sum(np.log(variance) + returns**2 / variance)
        
        if np.isnan(log_lik) or np.isinf(log_lik):
            return 1e10
        
        return -log_lik
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'HARCHModel':
        """
        Estimate HARCH model via maximum likelihood.
        """
        # Demean returns
        returns = returns - np.mean(returns)
        
        # Initialize parameters
        x0 = self._initialize_params(returns)
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._negative_log_likelihood,
                x0,
                args=(returns,),
                method='L-BFGS-B',
                options={'maxiter': 2000}
            )
        
        if not result.success and verbose:
            print(f"Warning: Optimization did not converge - {result.message}")
        
        self.params = result.x
        self.fitted_variance = self._compute_variance(self.params, returns)
        self.log_likelihood = -result.fun
        
        # Information criteria
        k = len(self.params)
        n = len(returns)
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n) - 2 * self.log_likelihood
        
        return self
    
    def get_params(self) -> Dict[str, float]:
        """Return fitted parameters as dictionary."""
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        param_dict = {"omega": self.params[0]}
        
        for i, lag in enumerate(self.lags):
            param_dict[f"alpha[{lag}]"] = self.params[1+i]
        
        return param_dict
    
    def persistence(self) -> float:
        """
        Persistence: sum of alpha parameters.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:]
        return np.sum(alphas)
    
    def component_contributions(self) -> Dict[int, float]:
        """
        Relative contribution of each horizon component.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:]
        total = np.sum(alphas)
        
        contributions = {}
        for i, lag in enumerate(self.lags):
            contributions[lag] = alphas[i] / total if total > 0 else 0.0
        
        return contributions
    
    def standardized_residuals(self, returns: np.ndarray) -> np.ndarray:
        """Compute standardized residuals."""
        if self.fitted_variance is None:
            raise ValueError("Model not fitted yet")
        
        returns = returns - np.mean(returns)
        std_resid = returns / np.sqrt(self.fitted_variance)
        
        return std_resid
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Forecast conditional variance h steps ahead.
        
        For HARCH, forecasting is complex due to aggregation.
        We provide 1-step ahead forecast only.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        alphas = self.params[1:]
        
        forecasts = np.zeros(horizon)
        
        # 1-step ahead
        harch_terms = 0.0
        for i, lag in enumerate(self.lags):
            agg_sq_ret = self._compute_aggregated_squared_returns(returns, lag, len(returns))
            harch_terms += alphas[i] * agg_sq_ret
        
        forecasts[0] = omega + harch_terms
        
        # Multi-step: use persistence approximation
        if horizon > 1:
            pers = self.persistence()
            for h in range(1, horizon):
                forecasts[h] = omega + pers * forecasts[h-1]
        
        return forecasts
    
    def summary(self) -> str:
        """Print model summary."""
        if self.params is None:
            return "Model not fitted"
        
        params = self.get_params()
        contributions = self.component_contributions()
        
        summary_str = f"\n{'='*50}\n"
        summary_str += f"HARCH Model Summary (lags: {self.lags})\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Log-Likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"\nParameters:\n"
        for name, value in params.items():
            summary_str += f"  {name:12s}: {value:10.6f}\n"
        summary_str += f"\nComponent Contributions:\n"
        for lag, contrib in contributions.items():
            summary_str += f"  {lag:3d}-day: {contrib:10.2%}\n"
        summary_str += f"\nPersistence: {self.persistence():.6f}\n"
        summary_str += f"{'='*50}\n"
        
        return summary_str
