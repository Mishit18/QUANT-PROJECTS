"""
Maximum Likelihood Estimation for Hawkes processes.
"""
import numpy as np
from scipy.optimize import minimize
from hawkes.kernels import MultiDimensionalKernel


class HawkesMLEEstimator:
    """MLE estimation for multidimensional Hawkes process with exponential kernels."""
    
    def __init__(self, num_types):
        """
        Parameters
        ----------
        num_types : int
            Number of event types
        """
        self.num_types = num_types
        self.baseline = None
        self.alpha = None
        self.beta = None
    
    def fit(self, events, T, initial_params=None, method='L-BFGS-B', max_iter=100):
        """
        Fit Hawkes process parameters using MLE.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        T : float
            Observation time horizon
        initial_params : dict, optional
            Initial parameter guesses
        method : str
            Optimization method
        max_iter : int
            Maximum iterations
        
        Returns
        -------
        result : dict
            Optimization result with fitted parameters
        """
        n = self.num_types
        
        # Initialize parameters
        if initial_params is None:
            baseline_init = np.ones(n) * 0.5
            alpha_init = np.ones((n, n)) * 0.1
            beta_init = np.ones((n, n)) * 2.0
        else:
            baseline_init = initial_params.get('baseline', np.ones(n) * 0.5)
            alpha_init = initial_params.get('alpha', np.ones((n, n)) * 0.1)
            beta_init = initial_params.get('beta', np.ones((n, n)) * 2.0)
        
        # Pack parameters into vector
        params_init = self._pack_params(baseline_init, alpha_init, beta_init)
        
        # Bounds: all parameters must be positive
        bounds = [(1e-6, None) for _ in range(len(params_init))]
        
        # Optimize
        result = minimize(
            lambda p: -self._log_likelihood(p, events, T),
            params_init,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        # Unpack optimized parameters
        baseline_opt, alpha_opt, beta_opt = self._unpack_params(result.x)
        
        self.baseline = baseline_opt
        self.alpha = alpha_opt
        self.beta = beta_opt
        
        return {
            'baseline': baseline_opt,
            'alpha': alpha_opt,
            'beta': beta_opt,
            'log_likelihood': -result.fun,
            'success': result.success,
            'message': result.message
        }
    
    def _pack_params(self, baseline, alpha, beta):
        """Pack parameters into 1D vector."""
        return np.concatenate([
            baseline.ravel(),
            alpha.ravel(),
            beta.ravel()
        ])
    
    def _unpack_params(self, params):
        """Unpack 1D vector into parameter matrices."""
        n = self.num_types
        baseline = params[:n]
        alpha = params[n:n + n*n].reshape((n, n))
        beta = params[n + n*n:].reshape((n, n))
        return baseline, alpha, beta
    
    def _log_likelihood(self, params, events, T):
        """
        Compute log-likelihood of Hawkes process.
        
        LL = Σᵢ log λᵢ(tᵢ) - Σᵢ ∫₀ᵀ λᵢ(t) dt
        """
        baseline, alpha, beta = self._unpack_params(params)
        
        # Ensure positivity and stationarity
        if np.any(baseline <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return -np.inf
        
        # Check stationarity
        eigenvalues = np.linalg.eigvals(alpha)
        if np.max(np.abs(eigenvalues)) >= 1.0:
            return -np.inf
        
        n = self.num_types
        
        # Separate events by type
        event_times = [[] for _ in range(n)]
        for t, event_type in events:
            event_times[event_type].append(t)
        
        # Convert to arrays
        for i in range(n):
            event_times[i] = np.array(event_times[i]) if event_times[i] else np.array([])
        
        # Compute log-likelihood
        ll = 0.0
        
        # Term 1: Σᵢ log λᵢ(tᵢ)
        for t, event_type in events:
            intensity = baseline[event_type]
            
            for j in range(n):
                if len(event_times[j]) > 0:
                    past_times = event_times[j][event_times[j] < t]
                    if len(past_times) > 0:
                        lags = t - past_times
                        contribution = alpha[event_type, j] * beta[event_type, j] * np.sum(np.exp(-beta[event_type, j] * lags))
                        intensity += contribution
            
            if intensity > 0:
                ll += np.log(intensity)
            else:
                return -np.inf
        
        # Term 2: -Σᵢ ∫₀ᵀ λᵢ(t) dt
        for i in range(n):
            # Baseline contribution
            ll -= baseline[i] * T
            
            # Excitation contribution
            for j in range(n):
                if len(event_times[j]) > 0:
                    # ∫₀ᵀ Σₖ φ(t - tₖ) dt = Σₖ ∫₀^(T-tₖ) φ(s) ds
                    integrals = alpha[i, j] * (1 - np.exp(-beta[i, j] * (T - event_times[j])))
                    ll -= np.sum(integrals)
        
        return ll
    
    def get_kernel(self):
        """Return fitted kernel object."""
        if self.alpha is None:
            raise ValueError("Model not fitted yet")
        return MultiDimensionalKernel(self.alpha, self.beta)
