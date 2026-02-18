"""
EGARCH model implementation.

Exponential GARCH (Nelson, 1991).

Model specification:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
    log(sigma_t^2) = omega + sum_{i=1}^q alpha_i * g(z_{t-i}) + sum_{j=1}^p beta_j * log(sigma_{t-j}^2)
    
    where g(z) = theta * z + gamma * (|z| - E[|z|])

Key features:
- Log formulation ensures sigma_t > 0 without constraints
- Asymmetric response: theta captures leverage effect
- gamma captures magnitude effect
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional
import warnings


class EGARCHModel:
    """EGARCH(p,q) model with maximum likelihood estimation."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize EGARCH model.
        
        Args:
            p: Number of GARCH lags
            q: Number of ARCH lags
        """
        self.p = p
        self.q = q
        self.params = None
        self.fitted_log_variance = None
        self.fitted_variance = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
    
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters.
        
        omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_q, beta_1, ..., beta_p
        """
        log_var = np.log(np.var(returns))
        
        omega = log_var * 0.1
        alphas = np.ones(self.q) * (-0.1)  # Leverage effect (negative)
        gammas = np.ones(self.q) * 0.2     # Magnitude effect
        betas = np.ones(self.p) * 0.9      # Persistence
        
        return np.concatenate([[omega], alphas, gammas, betas])
    
    def _compute_log_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Compute log conditional variance series.
        
        log(sigma_t^2) = omega + sum alpha_i * g(z_{t-i}) + sum beta_j * log(sigma_{t-j}^2)
        g(z) = theta * z + gamma * (|z| - sqrt(2/pi))
        """
        n = len(returns)
        log_variance = np.zeros(n)
        
        omega = params[0]
        alphas = params[1:1+self.q]  # theta parameters
        gammas = params[1+self.q:1+2*self.q]
        betas = params[1+2*self.q:]
        
        # Initialize with log of unconditional variance
        log_var_unconditional = np.log(np.var(returns))
        max_lag = max(self.p, self.q)
        log_variance[:max_lag] = log_var_unconditional
        
        # Expected value of |z| for standard normal
        ez = np.sqrt(2 / np.pi)
        
        # Recursive computation
        for t in range(max_lag, n):
            # Standardized residuals from previous periods
            z_prev = returns[t-self.q:t][::-1] / np.sqrt(np.exp(log_variance[t-self.q:t][::-1]))
            
            # g(z) function
            g_z = alphas * z_prev + gammas * (np.abs(z_prev) - ez)
            
            # GARCH term
            garch_term = np.sum(betas * log_variance[t-self.p:t][::-1])
            
            log_variance[t] = omega + np.sum(g_z) + garch_term
        
        return log_variance
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for minimization.
        """
        # No strict parameter constraints needed (log formulation)
        # But beta should be < 1 for stationarity
        betas = params[1+2*self.q:]
        if np.any(np.abs(betas) >= 0.9999):
            return 1e10
        
        log_variance = self._compute_log_variance(params, returns)
        variance = np.exp(log_variance)
        
        # Avoid numerical issues
        variance = np.maximum(variance, 1e-8)
        
        # Log-likelihood
        log_lik = -0.5 * np.sum(np.log(variance) + returns**2 / variance)
        
        if np.isnan(log_lik) or np.isinf(log_lik):
            return 1e10
        
        return -log_lik
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'EGARCHModel':
        """
        Estimate EGARCH model via maximum likelihood.
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
        self.fitted_log_variance = self._compute_log_variance(self.params, returns)
        self.fitted_variance = np.exp(self.fitted_log_variance)
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
        
        for i in range(self.q):
            param_dict[f"gamma[{i+1}]"] = self.params[1+self.q+i]
        
        for j in range(self.p):
            param_dict[f"beta[{j+1}]"] = self.params[1+2*self.q+j]
        
        return param_dict
    
    def leverage_effect(self) -> float:
        """
        Measure leverage effect from alpha parameter.
        
        Negative alpha indicates leverage effect:
        negative shocks increase volatility more than positive shocks.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:1+self.q]
        return np.mean(alphas)
    
    def persistence(self) -> float:
        """
        Persistence in EGARCH is measured by sum of beta parameters.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        betas = self.params[1+2*self.q:]
        return np.sum(betas)
    
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
        
        For EGARCH, multi-step forecasting is complex due to nonlinearity.
        We use a simplified approach assuming E[g(z)] = 0.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        betas = self.params[1+2*self.q:]
        
        # Current log variance
        current_log_var = self.fitted_log_variance[-1]
        
        forecasts = np.zeros(horizon)
        
        # Simplified forecast: assume g(z) terms average to zero
        log_var_forecast = current_log_var
        
        for h in range(horizon):
            log_var_forecast = omega + np.sum(betas) * log_var_forecast
            forecasts[h] = np.exp(log_var_forecast)
        
        return forecasts
    
    def summary(self) -> str:
        """Print model summary."""
        if self.params is None:
            return "Model not fitted"
        
        params = self.get_params()
        
        summary_str = f"\n{'='*50}\n"
        summary_str += f"EGARCH({self.p},{self.q}) Model Summary\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Log-Likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"\nParameters:\n"
        for name, value in params.items():
            summary_str += f"  {name:12s}: {value:10.6f}\n"
        summary_str += f"\nLeverage Effect (alpha): {self.leverage_effect():.6f}\n"
        summary_str += f"Persistence (beta): {self.persistence():.6f}\n"
        summary_str += f"{'='*50}\n"
        
        return summary_str
