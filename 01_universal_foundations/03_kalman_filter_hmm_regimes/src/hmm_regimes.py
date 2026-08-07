"""
Hidden Markov Model for regime detection in financial time series.

Implements:
- Gaussian HMM with EM (Baum-Welch) estimation
- Forward-Backward algorithm for regime probability inference
- Viterbi algorithm for most likely regime sequence
- Regime diagnostics and persistence analysis
"""

import numpy as np
from typing import Tuple, Optional, Dict
from src.utils import ensure_positive_definite


def _logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """
    Lightweight NumPy log-sum-exp.

    SciPy's generic implementation carries substantial array API overhead for
    the tiny K-dimensional reductions used inside the HMM recursion.
    """
    a = np.asarray(a, dtype=np.float64)

    if axis is None:
        a_max = np.max(a)
        if not np.isfinite(a_max):
            return a_max
        return np.log(np.sum(np.exp(a - a_max))) + a_max

    a_max = np.max(a, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - safe_max), axis=axis, keepdims=True)) + safe_max

    if not keepdims:
        out = np.squeeze(out, axis=axis)

    return out


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.

    After fitting, regimes are relabeled in ascending emission volatility. This
    makes regime 0 the lowest-volatility state and regime K-1 the highest-
    volatility state, which gives trading code stable semantics across runs.
    
    Latent regime s_t ∈ {1, ..., K}
    Transition: P(s_t | s_{t-1}) = A[s_{t-1}, s_t]
    Emission: P(y_t | s_t) = N(μ_{s_t}, Σ_{s_t})
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 n_iter: int = 50,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None,
                 min_covar: float = 1e-6,
                 covariance_floor_ratio: float = 0.05):
        """
        Initialize Gaussian HMM.
        
        Parameters
        ----------
        n_regimes : int
            Number of hidden regimes
        n_iter : int
            Maximum EM iterations
        tol : float
            Convergence tolerance
        random_state : int, optional
            Random seed
        min_covar : float
            Absolute covariance floor to prevent degenerate Gaussian regimes
        covariance_floor_ratio : float
            Additional floor as a fraction of the full-sample covariance
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.min_covar = min_covar
        self.covariance_floor_ratio = covariance_floor_ratio
        self._covariance_floor = None
        
        # Model parameters
        self.transition_matrix = None  # A: (K, K)
        self.initial_probs = None      # π: (K,)
        self.means = None              # μ: (K, d)
        self.covariances = None        # Σ: (K, d, d)
        
        # Fitted flag
        self.is_fitted = False
        self.converged_ = False
        
        # Convergence history
        self.log_likelihoods = []

    def _covariance_epsilon(self) -> float:
        """Current covariance floor used in initialization and EM updates."""
        if self._covariance_floor is None:
            return self.min_covar
        return max(self.min_covar, self._covariance_floor)
    
    def _initialize_parameters(self, X: np.ndarray):
        """
        Initialize parameters using k-means clustering.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize clusters. For univariate returns, quantile bins are much
        # faster and more stable than KMeans while preserving dispersion across
        # regimes. For multivariate data, fall back to KMeans.
        if n_features == 1:
            thresholds = np.quantile(
                X[:, 0],
                np.linspace(0, 1, self.n_regimes + 1)[1:-1],
            )
            labels = np.searchsorted(thresholds, X[:, 0], side='right')
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=5)
            labels = kmeans.fit_predict(X)
        
        # Initial state probabilities (uniform)
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
        
        # Transition matrix (slight persistence bias)
        self.transition_matrix = np.ones((self.n_regimes, self.n_regimes)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Means and covariances from k-means
        self.means = np.zeros((self.n_regimes, n_features))
        self.covariances = np.zeros((self.n_regimes, n_features, n_features))
        
        for k in range(self.n_regimes):
            mask = labels == k
            if mask.sum() > 0:
                self.means[k] = X[mask].mean(axis=0)
                cov = np.cov(X[mask].T)
                if n_features == 1:
                    cov = cov.reshape(1, 1)
                self.covariances[k] = ensure_positive_definite(cov, epsilon=self._covariance_epsilon())
            else:
                self.means[k] = X[rng.integers(n_samples)]
                self.covariances[k] = np.eye(n_features) * self._covariance_epsilon()
    
    def _log_emission_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized log emission probabilities for all observations/regimes.

        Parameters
        ----------
        X : np.ndarray
            Observations (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Log probabilities with shape (n_samples, n_regimes)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        log_probs = np.empty((len(X), self.n_regimes), dtype=np.float64)

        n_features = X.shape[1]

        for k in range(self.n_regimes):
            mean = self.means[k]
            cov = ensure_positive_definite(self.covariances[k], epsilon=self._covariance_epsilon())

            if n_features == 1:
                variance = max(float(cov[0, 0]), 1e-12)
                diff = X[:, 0] - float(mean[0])
                log_probs[:, k] = -0.5 * (
                    np.log(2 * np.pi * variance) + (diff * diff) / variance
                )
                continue

            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Covariance is not positive definite")
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov = ensure_positive_definite(cov, epsilon=self._covariance_epsilon())
                sign, logdet = np.linalg.slogdet(cov)
                inv_cov = np.linalg.pinv(cov)

            diff = X - mean
            quadratic = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            log_probs[:, k] = -0.5 * (
                n_features * np.log(2 * np.pi) + logdet + quadratic
            )

        return log_probs

    def _forward(self,
                 X: np.ndarray,
                 log_emissions: Optional[np.ndarray] = None,
                 return_log: bool = False) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm: compute filtering probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Observations (n_samples, n_features)
            
        Returns
        -------
        alpha : np.ndarray
            Forward probabilities (n_samples, n_regimes)
        log_likelihood : float
            Log-likelihood of observations
        """
        n_samples = len(X)
        if log_emissions is None:
            log_emissions = self._log_emission_probabilities(X)

        log_alpha = np.zeros((n_samples, self.n_regimes))
        log_transition = np.log(self.transition_matrix + 1e-300)
        
        # Initialize
        log_alpha[0] = np.log(self.initial_probs + 1e-300) + log_emissions[0]
        
        # Forward recursion
        for t in range(1, n_samples):
            log_alpha[t] = (
                log_emissions[t] +
                _logsumexp(log_alpha[t - 1][:, np.newaxis] + log_transition, axis=0)
            )
        
        # Log-likelihood
        log_likelihood = _logsumexp(log_alpha[-1])
        
        # Normalize to probabilities
        alpha = np.exp(log_alpha - _logsumexp(log_alpha, axis=1, keepdims=True))
        
        if return_log:
            return alpha, log_likelihood, log_alpha

        return alpha, log_likelihood
    
    def _backward(self,
                  X: np.ndarray,
                  log_emissions: Optional[np.ndarray] = None,
                  return_log: bool = False) -> np.ndarray:
        """
        Backward algorithm: compute smoothing probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Observations (n_samples, n_features)
            
        Returns
        -------
        beta : np.ndarray
            Backward probabilities (n_samples, n_regimes)
        """
        n_samples = len(X)
        if log_emissions is None:
            log_emissions = self._log_emission_probabilities(X)

        log_beta = np.zeros((n_samples, self.n_regimes))
        log_transition = np.log(self.transition_matrix + 1e-300)
        
        # Initialize (log(1) = 0)
        log_beta[-1, :] = 0
        
        # Backward recursion
        for t in range(n_samples - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_transition + log_emissions[t + 1][np.newaxis, :] + log_beta[t + 1][np.newaxis, :],
                axis=1,
            )
        
        # Normalize
        beta = np.exp(log_beta - _logsumexp(log_beta, axis=1, keepdims=True))
        
        if return_log:
            return beta, log_beta

        return beta
    
    def _log_emission_prob(self, x: np.ndarray, regime: int) -> float:
        """
        Log emission probability for observation x in given regime.
        
        Parameters
        ----------
        x : np.ndarray
            Observation
        regime : int
            Regime index
            
        Returns
        -------
        float
            Log probability
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        mean = self.means[regime]
        cov = ensure_positive_definite(
            self.covariances[regime],
            epsilon=self._covariance_epsilon(),
        )

        if x.size == 1:
            variance = max(float(cov[0, 0]), 1e-12)
            diff = float(x[0] - mean[0])
            return float(-0.5 * (np.log(2 * np.pi * variance) + (diff * diff) / variance))

        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("Covariance is not positive definite")
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov = ensure_positive_definite(cov, epsilon=self._covariance_epsilon())
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.pinv(cov)

        diff = x - mean
        quadratic = float(diff.T @ inv_cov @ diff)
        return float(-0.5 * (x.size * np.log(2 * np.pi) + logdet + quadratic))
    
    def _expectation_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: compute expected sufficient statistics.
        
        PRODUCTION HARDENING:
        - Validates gamma and xi for NaN/Inf
        - Ensures probabilities sum to 1
        - Checks numerical stability
        
        Parameters
        ----------
        X : np.ndarray
            Observations
            
        Returns
        -------
        gamma : np.ndarray
            State probabilities (n_samples, n_regimes)
        xi : np.ndarray
            Transition probabilities (n_samples-1, n_regimes, n_regimes)
        log_likelihood : float
            Log-likelihood
            
        Raises
        ------
        RuntimeError
            If NaN/Inf detected in probabilities
        """
        n_samples = len(X)
        
        # Forward-backward in log space for numerical stability.
        log_emissions = self._log_emission_probabilities(X)
        alpha, log_likelihood, log_alpha = self._forward(
            X, log_emissions=log_emissions, return_log=True
        )
        beta, log_beta = self._backward(
            X, log_emissions=log_emissions, return_log=True
        )
        
        # Validate forward-backward outputs
        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            raise RuntimeError("NaN/Inf detected in forward probabilities (alpha)")
        
        if np.any(np.isnan(beta)) or np.any(np.isinf(beta)):
            raise RuntimeError("NaN/Inf detected in backward probabilities (beta)")
        
        # Gamma: P(s_t = k | y_{1:T})
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        
        # Validate gamma
        if np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
            raise RuntimeError("NaN/Inf detected in state probabilities (gamma)")
        
        # Check gamma sums to 1
        if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-3):
            raise RuntimeError(f"Gamma probabilities don't sum to 1: range [{gamma.sum(axis=1).min():.6f}, {gamma.sum(axis=1).max():.6f}]")
        
        # Xi: P(s_t = i, s_{t+1} = j | y_{1:T})
        xi = np.zeros((n_samples - 1, self.n_regimes, self.n_regimes))
        
        log_transition = np.log(self.transition_matrix + 1e-300)
        for t in range(n_samples - 1):
            log_xi_t = (
                log_alpha[t][:, np.newaxis] +
                log_transition +
                log_emissions[t + 1][np.newaxis, :] +
                log_beta[t + 1][np.newaxis, :]
            )
            log_xi_t = log_xi_t - _logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)
        
        # Validate xi
        if np.any(np.isnan(xi)) or np.any(np.isinf(xi)):
            raise RuntimeError("NaN/Inf detected in transition probabilities (xi)")
        
        return gamma, xi, log_likelihood

    def regime_volatilities(self) -> np.ndarray:
        """
        Return per-regime emission volatility estimates.

        For multivariate emissions, this uses the square root of average
        marginal variance so that regimes can still be ranked by risk.
        """
        if self.covariances is None:
            raise ValueError("Model must be initialized or fitted first")

        vols = np.zeros(self.n_regimes, dtype=np.float64)
        for k in range(self.n_regimes):
            cov = np.asarray(self.covariances[k], dtype=np.float64)
            if cov.ndim == 0:
                variance = cov.item()
            elif cov.ndim == 1:
                variance = float(np.mean(cov))
            else:
                variance = float(np.trace(cov) / cov.shape[0])
            vols[k] = np.sqrt(max(variance, 0.0))

        return vols

    def _sort_regimes_by_volatility(self) -> None:
        """Relabel regimes so labels are stable and economically interpretable."""
        order = np.argsort(self.regime_volatilities())

        if np.array_equal(order, np.arange(self.n_regimes)):
            return

        self.initial_probs = self.initial_probs[order]
        self.means = self.means[order]
        self.covariances = self.covariances[order]
        self.transition_matrix = self.transition_matrix[np.ix_(order, order)]
    
    def _maximization_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """
        M-step: update parameters.
        
        PRODUCTION HARDENING:
        - Validates updated parameters for NaN/Inf
        - Ensures transition matrix is stochastic
        - Enforces covariance positive definiteness
        
        Parameters
        ----------
        X : np.ndarray
            Observations
        gamma : np.ndarray
            State probabilities
        xi : np.ndarray
            Transition probabilities
            
        Raises
        ------
        RuntimeError
            If parameter updates produce invalid values
        """
        n_samples, n_features = X.shape
        
        # Update initial probabilities
        self.initial_probs = gamma[0] / (gamma[0].sum() + 1e-10)
        
        # Validate initial probs
        if np.any(np.isnan(self.initial_probs)) or np.any(np.isinf(self.initial_probs)):
            raise RuntimeError("NaN/Inf in updated initial probabilities")
        
        if not np.isclose(self.initial_probs.sum(), 1.0, atol=1e-3):
            raise RuntimeError(f"Initial probabilities don't sum to 1: {self.initial_probs.sum():.6f}")
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = xi[:, i, j].sum()
                denominator = gamma[:-1, i].sum() + 1e-10
                self.transition_matrix[i, j] = numerator / denominator
        
        # Normalize transition matrix (ensure stochastic)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums < 1e-10):
            raise RuntimeError("Transition matrix row sums too small - numerical instability")
        
        self.transition_matrix = self.transition_matrix / row_sums
        
        # Validate transition matrix
        if np.any(np.isnan(self.transition_matrix)) or np.any(np.isinf(self.transition_matrix)):
            raise RuntimeError("NaN/Inf in updated transition matrix")
        
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0, atol=1e-3):
            raise RuntimeError(f"Transition matrix rows don't sum to 1: {self.transition_matrix.sum(axis=1)}")
        
        if np.any(self.transition_matrix < 0) or np.any(self.transition_matrix > 1):
            raise RuntimeError(f"Transition matrix has invalid probabilities: range [{self.transition_matrix.min():.6f}, {self.transition_matrix.max():.6f}]")
        
        # Update means and covariances
        for k in range(self.n_regimes):
            gamma_k = gamma[:, k]
            gamma_sum = gamma_k.sum()
            
            if gamma_sum < 1e-10:
                raise RuntimeError(f"Regime {k} has insufficient probability mass: {gamma_sum:.2e}")
            
            # Mean
            self.means[k] = (gamma_k[:, np.newaxis] * X).sum(axis=0) / gamma_sum
            
            # Validate mean
            if np.any(np.isnan(self.means[k])) or np.any(np.isinf(self.means[k])):
                raise RuntimeError(f"NaN/Inf in updated mean for regime {k}")
            
            # Covariance
            diff = X - self.means[k]
            self.covariances[k] = (gamma_k[:, np.newaxis, np.newaxis] * 
                                  (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / gamma_sum
            
            # Enforce positive definiteness
            self.covariances[k] = ensure_positive_definite(
                self.covariances[k],
                epsilon=self._covariance_epsilon(),
            )
            
            # Validate covariance
            if np.any(np.isnan(self.covariances[k])) or np.any(np.isinf(self.covariances[k])):
                raise RuntimeError(f"NaN/Inf in updated covariance for regime {k}")
    
    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """
        Fit HMM using EM algorithm.
        
        PRODUCTION HARDENING:
        - Validates input data
        - Monitors convergence
        - Checks for numerical issues
        - Explicit error messages
        
        Parameters
        ----------
        X : np.ndarray
            Training data (n_samples, n_features) or (n_samples,)
            
        Returns
        -------
        self
        
        Raises
        ------
        ValueError
            If input data is invalid
        RuntimeError
            If EM algorithm fails to converge properly
        """
        # Input validation
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X = np.asarray(X, dtype=np.float64)
        
        if np.any(np.isnan(X)):
            raise ValueError(f"Input data contains {np.sum(np.isnan(X))} NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError(f"Input data contains {np.sum(np.isinf(X))} Inf values")
        
        if len(X) < self.n_regimes * 10:
            raise ValueError(f"Insufficient data: {len(X)} samples for {self.n_regimes} regimes (need at least {self.n_regimes * 10})")

        sample_cov = np.cov(X.T)
        if X.shape[1] == 1:
            sample_variance = float(np.asarray(sample_cov).reshape(-1)[0])
            self._covariance_floor = max(self.min_covar, sample_variance * self.covariance_floor_ratio)
        else:
            sample_cov = ensure_positive_definite(sample_cov, epsilon=self.min_covar)
            avg_variance = float(np.trace(sample_cov) / X.shape[1])
            self._covariance_floor = max(self.min_covar, avg_variance * self.covariance_floor_ratio)
        
        # Initialize parameters
        self.log_likelihoods = []
        self.converged_ = False
        self._initialize_parameters(X)
        
        # EM iterations
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.n_iter):
            try:
                # E-step
                gamma, xi, log_likelihood = self._expectation_step(X)
                
                # M-step
                self._maximization_step(X, gamma, xi)
                
                # Store log-likelihood
                self.log_likelihoods.append(log_likelihood)
                
                # Check for NaN in log-likelihood
                if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                    raise RuntimeError(f"Invalid log-likelihood at iteration {iteration + 1}: {log_likelihood}")
                
                # Check for decreasing log-likelihood (should never happen in EM)
                if iteration > 0 and log_likelihood < prev_log_likelihood - 1e-3:
                    raise RuntimeError(f"Log-likelihood decreased at iteration {iteration + 1}: {prev_log_likelihood:.2f} -> {log_likelihood:.2f}")
                
                # Check convergence
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    self.converged_ = True
                    print(f"Converged at iteration {iteration + 1}, log-likelihood: {log_likelihood:.2f}")
                    break
                
                prev_log_likelihood = log_likelihood
                
            except RuntimeError as e:
                raise RuntimeError(f"HMM fitting failed at iteration {iteration + 1}: {e}") from e
        
        else:
            # Max iterations reached without convergence
            print(f"Warning: Max iterations ({self.n_iter}) reached without convergence")
            print(f"Final log-likelihood: {log_likelihood:.2f}")
        
        self._sort_regimes_by_volatility()
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray, method: str = 'filtered') -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features) or (n_samples,)
        method : str
            'filtered' for causal trading probabilities or 'smoothed' for
            offline diagnostics.
            
        Returns
        -------
        probs : np.ndarray
            Regime probabilities (n_samples, n_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if method == 'filtered':
            probs, _ = self._forward(X)
        elif method == 'smoothed':
            alpha, _ = self._forward(X)
            beta = self._backward(X)
            probs = alpha * beta
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown probability method: {method}")
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime sequence using Viterbi algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features) or (n_samples,)
            
        Returns
        -------
        regimes : np.ndarray
            Most likely regime sequence (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(X)
        
        # Viterbi algorithm
        log_delta = np.zeros((n_samples, self.n_regimes))
        psi = np.zeros((n_samples, self.n_regimes), dtype=int)
        
        # Initialize
        for k in range(self.n_regimes):
            log_delta[0, k] = (np.log(self.initial_probs[k] + 1e-10) +
                              self._log_emission_prob(X[0], k))
        
        # Recursion
        for t in range(1, n_samples):
            for k in range(self.n_regimes):
                log_trans = np.log(self.transition_matrix[:, k] + 1e-10)
                candidates = log_delta[t-1] + log_trans
                psi[t, k] = np.argmax(candidates)
                log_delta[t, k] = candidates[psi[t, k]] + self._log_emission_prob(X[t], k)
        
        # Backtrack
        regimes = np.zeros(n_samples, dtype=int)
        regimes[-1] = np.argmax(log_delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            regimes[t] = psi[t + 1, regimes[t + 1]]
        
        return regimes
    
    def get_regime_statistics(self, X: np.ndarray) -> Dict:
        """
        Compute regime statistics and diagnostics.
        
        Parameters
        ----------
        X : np.ndarray
            Data
            
        Returns
        -------
        dict
            Regime statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        probs = self.predict_proba(X, method='smoothed')
        regimes = np.argmax(probs, axis=1)
        
        stats = {
            'transition_matrix': self.transition_matrix,
            'regime_means': self.means,
            'regime_covariances': self.covariances,
            'regime_counts': np.bincount(regimes, minlength=self.n_regimes),
            'regime_probabilities': probs,
            'most_likely_regimes': regimes
        }
        
        # Regime persistence (expected duration)
        persistence = np.diag(self.transition_matrix)
        expected_duration = 1 / (1 - persistence + 1e-10)
        stats['expected_duration'] = expected_duration
        
        # Regime-specific statistics
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        regime_stats = []
        for k in range(self.n_regimes):
            mask = regimes == k
            if mask.sum() > 0:
                regime_stats.append({
                    'mean': X[mask].mean(axis=0),
                    'std': X[mask].std(axis=0),
                    'count': mask.sum(),
                    'frequency': mask.sum() / len(X)
                })
            else:
                regime_stats.append({
                    'mean': np.nan,
                    'std': np.nan,
                    'count': 0,
                    'frequency': 0.0
                })
        
        stats['regime_statistics'] = regime_stats
        
        return stats


if __name__ == '__main__':
    # Test HMM
    from src.data_loader import generate_synthetic_data
    
    print("Testing Gaussian HMM...")
    
    # Generate regime-switching data
    data = generate_synthetic_data(n_samples=500, seed=42)
    returns = data['returns'].iloc[:, 0].values
    
    # Fit HMM
    hmm = GaussianHMM(n_regimes=3, random_state=42)
    hmm.fit(returns)
    
    # Predict regimes
    regimes = hmm.predict(returns)
    probs = hmm.predict_proba(returns)
    
    print(f"\nRegime counts: {np.bincount(regimes)}")
    print(f"\nTransition matrix:\n{hmm.transition_matrix}")
    
    # Statistics
    stats = hmm.get_regime_statistics(returns)
    print(f"\nExpected regime durations: {stats['expected_duration']}")
    
    for k in range(3):
        print(f"\nRegime {k}:")
        print(f"  Mean: {stats['regime_statistics'][k]['mean']}")
        print(f"  Std: {stats['regime_statistics'][k]['std']}")
        print(f"  Frequency: {stats['regime_statistics'][k]['frequency']:.2%}")
