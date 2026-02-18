"""
ARCH(q) model implementation.

Autoregressive Conditional Heteroskedasticity (Engle, 1982).

Model specification:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
    sigma_t^2 = omega + sum_{i=1}^q alpha_i * epsilon_{t-i}^2

Captures volatility clustering through lagged squared residuals.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings


class ARCHModel:
    """ARCH(q) model with maximum likelihood estimation."""
    
    def __init__(self, q: int = 5):
        """
        Initialize ARCH model.
        
        Args:
            q: Number of ARCH lags
        """
        self.q = q
        self.params = None
        self.fitted_variance = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
    
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters for optimization.
        
        Start with unconditional variance and equal weights.
        """
        var_unconditional = np.var(returns)
        
        # omega, alpha_1, ..., alpha_q
        omega = var_unconditional * 0.1
        alphas = np.ones(self.q) * 0.8 / self.q
        
        return np.concatenate([[omega], alphas])
    
    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        sigma_t^2 = omega + sum alpha_i * epsilon_{t-i}^2
        """
        n = len(returns)
        variance = np.zeros(n)
        
        omega = params[0]
        alphas = params[1:]
        
        # Initialize with unconditional variance
        var_unconditional = np.var(returns)
        variance[:self.q] = var_unconditional
        
        # Recursive computation
        for t in range(self.q, n):
            arch_term = np.sum(alphas * returns[t-self.q:t][::-1]**2)
            variance[t] = omega + arch_term
        
        return variance
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for minimization.
        
        Assumes conditional normality: epsilon_t | F_{t-1} ~ N(0, sigma_t^2)
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
        
        # Log-likelihood (ignoring constant term)
        log_lik = -0.5 * np.sum(np.log(variance) + returns**2 / variance)
        
        if np.isnan(log_lik) or np.isinf(log_lik):
            return 1e10
        
        return -log_lik
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'ARCHModel':
        """
        Estimate ARCH model via maximum likelihood.
        
        Args:
            returns: Return series (mean should be close to zero)
            verbose: Print optimization details
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
                options={'maxiter': 1000}
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
        for i in range(self.q):
            param_dict[f"alpha[{i+1}]"] = self.params[i+1]
        
        return param_dict
    
    def standardized_residuals(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute standardized residuals: epsilon_t / sigma_t
        
        Should be approximately i.i.d. N(0,1) if model is correct.
        """
        if self.fitted_variance is None:
            raise ValueError("Model not fitted yet")
        
        returns = returns - np.mean(returns)
        std_resid = returns / np.sqrt(self.fitted_variance)
        
        return std_resid
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Forecast conditional variance h steps ahead.
        
        For ARCH(q), multi-step forecast requires iterating the model.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        alphas = self.params[1:]
        
        # Start from last q squared returns
        last_sq_returns = returns[-self.q:][::-1]**2
        
        forecasts = np.zeros(horizon)
        
        for h in range(horizon):
            if h == 0:
                # 1-step ahead
                forecasts[h] = omega + np.sum(alphas * last_sq_returns)
            else:
                # Multi-step: use forecasted variances
                # E[epsilon_{t+h}^2 | F_t] = sigma_{t+h}^2
                recent_vars = np.concatenate([last_sq_returns[:(self.q-h)], forecasts[:h]])
                forecasts[h] = omega + np.sum(alphas * recent_vars[::-1])
        
        return forecasts
    
    def summary(self) -> str:
        """Print model summary."""
        if self.params is None:
            return "Model not fitted"
        
        params = self.get_params()
        
        summary_str = f"\n{'='*50}\n"
        summary_str += f"ARCH({self.q}) Model Summary\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Log-Likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"\nParameters:\n"
        for name, value in params.items():
            summary_str += f"  {name:12s}: {value:10.6f}\n"
        summary_str += f"{'='*50}\n"
        
        return summary_str
