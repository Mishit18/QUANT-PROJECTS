"""
GJR-GARCH model implementation.

Glosten-Jagannathan-Runkle GARCH (1993).

Model specification:
    r_t = mu + epsilon_t
    epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
    sigma_t^2 = omega + sum_{i=1}^q (alpha_i + gamma_i * I_{t-i}) * epsilon_{t-i}^2 + sum_{j=1}^p beta_j * sigma_{t-j}^2
    
    where I_t = 1 if epsilon_t < 0, else 0

Key feature:
- Captures leverage effect through gamma parameter
- Negative shocks have impact (alpha + gamma)
- Positive shocks have impact alpha
- If gamma > 0, leverage effect is present
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional
import warnings


class GJRGARCHModel:
    """GJR-GARCH(p,q) model with maximum likelihood estimation."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GJR-GARCH model.
        
        Args:
            p: Number of GARCH lags
            q: Number of ARCH lags
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
        Initialize parameters.
        
        omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_q, beta_1, ..., beta_p
        """
        var_unconditional = np.var(returns)
        
        omega = var_unconditional * 0.01
        alphas = np.ones(self.q) * 0.05
        gammas = np.ones(self.q) * 0.05  # Leverage effect
        betas = np.ones(self.p) * 0.85
        
        return np.concatenate([[omega], alphas, gammas, betas])
    
    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        sigma_t^2 = omega + sum (alpha_i + gamma_i * I_{t-i}) * epsilon_{t-i}^2 + sum beta_j * sigma_{t-j}^2
        """
        n = len(returns)
        variance = np.zeros(n)
        
        omega = params[0]
        alphas = params[1:1+self.q]
        gammas = params[1+self.q:1+2*self.q]
        betas = params[1+2*self.q:]
        
        # Initialize with unconditional variance
        var_unconditional = np.var(returns)
        max_lag = max(self.p, self.q)
        variance[:max_lag] = var_unconditional
        
        # Recursive computation
        for t in range(max_lag, n):
            # ARCH term with asymmetry
            epsilon_sq = returns[t-self.q:t][::-1]**2
            indicators = (returns[t-self.q:t][::-1] < 0).astype(float)
            
            arch_term = np.sum((alphas + gammas * indicators) * epsilon_sq)
            
            # GARCH term
            garch_term = np.sum(betas * variance[t-self.p:t][::-1])
            
            variance[t] = omega + arch_term + garch_term
        
        return variance
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for minimization.
        """
        # Parameter constraints
        if params[0] <= 0:  # omega > 0
            return 1e10
        
        alphas = params[1:1+self.q]
        gammas = params[1+self.q:1+2*self.q]
        betas = params[1+2*self.q:]
        
        if np.any(alphas < 0):  # alpha_i >= 0
            return 1e10
        if np.any(gammas < 0):  # gamma_i >= 0 (leverage effect)
            return 1e10
        if np.any(betas < 0):  # beta_j >= 0
            return 1e10
        
        # Stationarity: alpha + 0.5*gamma + beta < 1 (approximate)
        if np.sum(alphas) + 0.5 * np.sum(gammas) + np.sum(betas) >= 0.9999:
            return 1e10
        
        variance = self._compute_variance(params, returns)
        
        # Avoid numerical issues
        variance = np.maximum(variance, 1e-8)
        
        # Log-likelihood
        log_lik = -0.5 * np.sum(np.log(variance) + returns**2 / variance)
        
        if np.isnan(log_lik) or np.isinf(log_lik):
            return 1e10
        
        return -log_lik
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'GJRGARCHModel':
        """
        Estimate GJR-GARCH model via maximum likelihood.
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
        
        for i in range(self.q):
            param_dict[f"gamma[{i+1}]"] = self.params[1+self.q+i]
        
        for j in range(self.p):
            param_dict[f"beta[{j+1}]"] = self.params[1+2*self.q+j]
        
        return param_dict
    
    def leverage_effect(self) -> float:
        """
        Measure leverage effect from gamma parameter.
        
        Positive gamma indicates leverage effect.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        gammas = self.params[1+self.q:1+2*self.q]
        return np.mean(gammas)
    
    def asymmetry_ratio(self) -> float:
        """
        Ratio of negative shock impact to positive shock impact.
        
        Ratio = (alpha + gamma) / alpha
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:1+self.q]
        gammas = self.params[1+self.q:1+2*self.q]
        
        return np.mean((alphas + gammas) / alphas)
    
    def persistence(self) -> float:
        """
        Persistence: alpha + 0.5*gamma + beta (approximate)
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        alphas = self.params[1:1+self.q]
        gammas = self.params[1+self.q:1+2*self.q]
        betas = self.params[1+2*self.q:]
        
        return np.sum(alphas) + 0.5 * np.sum(gammas) + np.sum(betas)
    
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
        
        For GJR-GARCH, we assume E[I_t] = 0.5 for future periods.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega = self.params[0]
        alphas = self.params[1:1+self.q]
        gammas = self.params[1+self.q:1+2*self.q]
        betas = self.params[1+2*self.q:]
        
        # Current variance
        current_variance = self.fitted_variance[-1]
        
        # Last squared returns and indicators
        last_sq_returns = returns[-self.q:][::-1]**2
        last_indicators = (returns[-self.q:][::-1] < 0).astype(float)
        
        forecasts = np.zeros(horizon)
        
        # 1-step ahead
        arch_term = np.sum((alphas + gammas * last_indicators) * last_sq_returns)
        garch_term = np.sum(betas * self.fitted_variance[-self.p:][::-1])
        forecasts[0] = omega + arch_term + garch_term
        
        # Multi-step ahead: assume E[I] = 0.5
        for h in range(1, horizon):
            expected_arch = np.sum((alphas + 0.5 * gammas) * forecasts[max(0, h-self.q):h])
            expected_garch = np.sum(betas * forecasts[max(0, h-self.p):h])
            forecasts[h] = omega + expected_arch + expected_garch
        
        return forecasts
    
    def summary(self) -> str:
        """Print model summary."""
        if self.params is None:
            return "Model not fitted"
        
        params = self.get_params()
        
        summary_str = f"\n{'='*50}\n"
        summary_str += f"GJR-GARCH({self.p},{self.q}) Model Summary\n"
        summary_str += f"{'='*50}\n"
        summary_str += f"Log-Likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"\nParameters:\n"
        for name, value in params.items():
            summary_str += f"  {name:12s}: {value:10.6f}\n"
        summary_str += f"\nLeverage Effect (gamma): {self.leverage_effect():.6f}\n"
        summary_str += f"Asymmetry Ratio: {self.asymmetry_ratio():.4f}\n"
        summary_str += f"Persistence: {self.persistence():.6f}\n"
        summary_str += f"{'='*50}\n"
        
        return summary_str
