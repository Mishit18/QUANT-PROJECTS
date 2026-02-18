"""
GARCH(p,q) model implementation.

Generalized ARCH (Bollerslev, 1986).

Model specification:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
    sigma_t^2 = omega + sum_{i=1}^q alpha_i * epsilon_{t-i}^2 + sum_{j=1}^p beta_j * sigma_{t-j}^2

GARCH(1,1) is the workhorse model in finance:
- Parsimonious (3 parameters)
- Captures volatility clustering
- Mean-reverting if alpha + beta < 1
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings


class GARCHModel:
    """GARCH(p,q) model with maximum likelihood estimation."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.
        
        Args:
            p: Number of GARCH lags (lagged variance)
            q: Number of ARCH lags (lagged squared residuals)
        """
        self.p = p
        self.q = q
        self.params = None
        self.fitted_variance = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
    
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters for optimization.
        
        Common starting values for GARCH(1,1):
        omega = 0.01 * var, alpha = 0.1, beta = 0.85
        """
        var_unconditional = np.var(returns)
        
        omega = var_unconditional * 0.01
        alphas = np.ones(self.q) * 0.1 / self.q
        betas = np.ones(self.p) * 0.85 / self.p
        
        return np.concatenate([[omega], alphas, betas])
    
    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        sigma_t^2 = omega + sum alpha_i * epsilon_{t-i}^2 + sum beta_j * sigma_{t-j}^2
        """
        n = len(returns)
        variance = np.zeros(n)
        
        omega = params[0]
        alphas = params[1:1+self.q]
        betas = params[1+self.q:]
        
        # Initialize with unconditional variance
        var_unconditional = np.var(returns)
        max_lag = max(self.p, self.q)
        variance[:max_lag] = var_unconditional
        
        # Recursive computation
        for t in range(max_lag, n):
            arch_term = np.sum(alphas * returns[t-self.q:t][::-1]**2)
            garch_term = np.sum(betas * variance[t-self.p:t][::-1])
            variance[t] = omega + arch_term + garch_term
        
        return variance
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for minimization.
        
        Assumes conditional normality.
        """
        # Parameter constraints
        if params[0] <= 0:  # omega > 0
            return 1e10
        if np.any(params[1:] < 0):  # alpha_i, beta_j >= 0
            return 1e10
        
        # Stationarity: sum(alpha) + sum(beta) < 1
        alphas = params[1:1+self.q]
        betas = params[1+self.q:]
        if np.sum(alphas) + np.sum(betas) >= 0.9999:  # Slightly below 1 for numerical stability
            return 1e10
        
        variance = self._compute_variance(params, returns)
        
        # Avoid numerical issues
        variance = np.maximum(variance, 1e-8)
        
        # Log-likelihood
        log_lik = -0.5 * np.sum(np.log(variance) + returns**2 / variance)
        
        if np.isnan(log_lik) or np.isinf(log_lik):
            return 1e10
        
        return -log_lik
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'GARCHModel':
        """
        Estimate GARCH model via maximum likelihood.
        
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
        
        for i in range(self.q):
            param_dict[f"alpha[{i+1}]"] = self.params[1+i]
        
        for j in range(self.p):
            param_dict[f"beta[{j+1}]"] = self.params[1+self.q+j]
        
        return param_dict
    
    def persistence(self) -> float:
        """
        Compute persistence parameter: sum(alpha) + sum(beta)
        
        Measures how long volatility shocks persist.
        Close to 1 = highly persistent (slow mean reversion)
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:1+self.q]
        betas = self.params[1+self.q:]
        
        return np.sum(alphas) + np.sum(betas)
    
    def half_life(self) -> float:
        """
        Compute half-life of volatility shocks.
        
        Time for shock to decay to 50% of initial impact.
        """
        pers = self.persistence()
        
        if pers >= 1:
            return np.inf
        
        return np.log(0.5) / np.log(pers)
    
    def unconditional_variance(self) -> float:
        """
        Compute unconditional variance: omega / (1 - alpha - beta)
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        pers = self.persistence()
        
        if pers >= 1:
            return np.inf
        
        return omega / (1 - pers)
    
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
        
        For GARCH(1,1):
        h=1: sigma_{t+1}^2 = omega + alpha * epsilon_t^2 + beta * sigma_t^2
        h>1: sigma_{t+h}^2 = omega * sum_{i=0}^{h-2} (alpha+beta)^i + (alpha+beta)^{h-1} * sigma_{t+1}^2
        
        Converges to unconditional variance as h -> infinity.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        alphas = self.params[1:1+self.q]
        betas = self.params[1+self.q:]
        
        # Current variance
        current_variance = self.fitted_variance[-1]
        
        # Last squared returns
        last_sq_returns = returns[-self.q:][::-1]**2
        
        forecasts = np.zeros(horizon)
        
        # 1-step ahead
        arch_term = np.sum(alphas * last_sq_returns)
        garch_term = np.sum(betas * self.fitted_variance[-self.p:][::-1])
        forecasts[0] = omega + arch_term + garch_term
        
        # Multi-step ahead (simplified for GARCH(1,1))
        if self.p == 1 and self.q == 1:
            alpha = alphas[0]
            beta = betas[0]
            pers = alpha + beta
            uncond_var = self.unconditional_variance()
            
            for h in range(1, horizon):
                forecasts[h] = uncond_var + (pers ** h) * (forecasts[0] - uncond_var)
        else:
            # General case: iterate forward
            for h in range(1, horizon):
                forecasts[h] = omega + (np.sum(alphas) + np.sum(betas)) * forecasts[h-1]
        
        return forecasts
    
    def summary(self) -> str:
        """Print model summary."""
        if self.params is None:
            return "Model not fitted"
        
        params = self.get_params()
        
        summary_str = f"\n{'='*50}\n"
        summary_str += f"GARCH({self.p},{self.q}) Model Summary\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Log-Likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"\nParameters:\n"
        for name, value in params.items():
            summary_str += f"  {name:12s}: {value:10.6f}\n"
        summary_str += f"\nPersistence: {self.persistence():.6f}\n"
        summary_str += f"Half-life: {self.half_life():.2f} days\n"
        summary_str += f"Unconditional Variance: {self.unconditional_variance():.6f}\n"
        summary_str += f"{'='*50}\n"
        
        return summary_str
