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
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from src.utils import ensure_positive_definite


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.
    
    Latent regime s_t ∈ {1, ..., K}
    Transition: P(s_t | s_{t-1}) = A[s_{t-1}, s_t]
    Emission: P(y_t | s_t) = N(μ_{s_t}, Σ_{s_t})
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 n_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
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
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        
        # Model parameters
        self.transition_matrix = None  # A: (K, K)
        self.initial_probs = None      # π: (K,)
        self.means = None              # μ: (K, d)
        self.covariances = None        # Σ: (K, d, d)
        
        # Fitted flag
        self.is_fitted = False
        
        # Convergence history
        self.log_likelihoods = []
    
    def _initialize_parameters(self, X: np.ndarray):
        """
        Initialize parameters using k-means clustering.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize with k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=10)
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
                self.covariances[k] = ensure_positive_definite(cov)
            else:
                self.means[k] = X[np.random.randint(n_samples)]
                self.covariances[k] = np.eye(n_features)
    
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
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
        log_alpha = np.zeros((n_samples, self.n_regimes))
        
        # Initialize
        for k in range(self.n_regimes):
            log_alpha[0, k] = (np.log(self.initial_probs[k] + 1e-10) +
                              self._log_emission_prob(X[0], k))
        
        # Forward recursion
        for t in range(1, n_samples):
            for k in range(self.n_regimes):
                log_trans = np.log(self.transition_matrix[:, k] + 1e-10)
                log_alpha[t, k] = (logsumexp(log_alpha[t-1] + log_trans) +
                                  self._log_emission_prob(X[t], k))
        
        # Log-likelihood
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Normalize to probabilities
        alpha = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        
        return alpha, log_likelihood
    
    def _backward(self, X: np.ndarray) -> np.ndarray:
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
        log_beta = np.zeros((n_samples, self.n_regimes))
        
        # Initialize (log(1) = 0)
        log_beta[-1, :] = 0
        
        # Backward recursion
        for t in range(n_samples - 2, -1, -1):
            for k in range(self.n_regimes):
                log_trans = np.log(self.transition_matrix[k, :] + 1e-10)
                log_emit = np.array([self._log_emission_prob(X[t+1], j) 
                                    for j in range(self.n_regimes)])
                log_beta[t, k] = logsumexp(log_trans + log_emit + log_beta[t+1])
        
        # Normalize
        beta = np.exp(log_beta - logsumexp(log_beta, axis=1, keepdims=True))
        
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
        mean = self.means[regime]
        cov = self.covariances[regime]
        
        try:
            return multivariate_normal.logpdf(x, mean=mean, cov=cov)
        except:
            return -1e10
    
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
        
        # Forward-backward
        alpha, log_likelihood = self._forward(X)
        beta = self._backward(X)
        
        # Validate forward-backward outputs
        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            raise RuntimeError("NaN/Inf detected in forward probabilities (alpha)")
        
        if np.any(np.isnan(beta)) or np.any(np.isinf(beta)):
            raise RuntimeError("NaN/Inf detected in backward probabilities (beta)")
        
        # Gamma: P(s_t = k | y_{1:T})
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        
        # Check for zero sums before normalization
        if np.any(gamma_sum < 1e-300):
            raise RuntimeError("Numerical underflow in gamma computation - probabilities too small")
        
        gamma = gamma / gamma_sum
        
        # Validate gamma
        if np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
            raise RuntimeError("NaN/Inf detected in state probabilities (gamma)")
        
        # Check gamma sums to 1
        if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-3):
            raise RuntimeError(f"Gamma probabilities don't sum to 1: range [{gamma.sum(axis=1).min():.6f}, {gamma.sum(axis=1).max():.6f}]")
        
        # Xi: P(s_t = i, s_{t+1} = j | y_{1:T})
        xi = np.zeros((n_samples - 1, self.n_regimes, self.n_regimes))
        
        for t in range(n_samples - 1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    xi[t, i, j] = (alpha[t, i] * 
                                  self.transition_matrix[i, j] *
                                  np.exp(self._log_emission_prob(X[t+1], j)) *
                                  beta[t+1, j])
            
            xi_sum = xi[t].sum()
            if xi_sum < 1e-300:
                # Numerical underflow - use uniform distribution
                xi[t] = 1.0 / (self.n_regimes * self.n_regimes)
            else:
                xi[t] = xi[t] / xi_sum
        
        # Validate xi
        if np.any(np.isnan(xi)) or np.any(np.isinf(xi)):
            raise RuntimeError("NaN/Inf detected in transition probabilities (xi)")
        
        return gamma, xi, log_likelihood
    
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
            self.covariances[k] = ensure_positive_definite(self.covariances[k], epsilon=1e-6)
            
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
        
        # Initialize parameters
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
                    print(f"Converged at iteration {iteration + 1}, log-likelihood: {log_likelihood:.2f}")
                    break
                
                prev_log_likelihood = log_likelihood
                
            except RuntimeError as e:
                raise RuntimeError(f"HMM fitting failed at iteration {iteration + 1}: {e}") from e
        
        else:
            # Max iterations reached without convergence
            print(f"Warning: Max iterations ({self.n_iter}) reached without convergence")
            print(f"Final log-likelihood: {log_likelihood:.2f}")
        
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray, method: str = 'smoothed') -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features) or (n_samples,)
        method : str
            'filtered' or 'smoothed'
            
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
        else:  # smoothed
            alpha, _ = self._forward(X)
            beta = self._backward(X)
            probs = alpha * beta
            probs = probs / probs.sum(axis=1, keepdims=True)
        
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
        
        regimes = self.predict(X)
        probs = self.predict_proba(X, method='smoothed')
        
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
