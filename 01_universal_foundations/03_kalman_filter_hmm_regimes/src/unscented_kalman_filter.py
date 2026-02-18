"""
Unscented Kalman Filter (UKF) for nonlinear state-space models.

Uses deterministic sigma-point sampling to capture nonlinear transformations
without requiring Jacobian calculations.

Advantages over EKF:
- No need for derivatives
- Better captures higher-order moments
- More accurate for highly nonlinear systems
"""

import numpy as np
from typing import Callable, Tuple
from src.utils import ensure_positive_definite, matrix_sqrt


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter using unscented transform.
    
    Propagates mean and covariance through nonlinear functions
    using carefully chosen sigma points.
    """
    
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 state_transition_fn: Callable,
                 observation_fn: Callable,
                 process_noise_cov: np.ndarray,
                 observation_noise_cov: np.ndarray,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        """
        Initialize UKF.
        
        Parameters
        ----------
        state_dim : int
            State dimension
        obs_dim : int
            Observation dimension
        state_transition_fn : callable
            f(x, t) -> x_next (nonlinear state transition)
        observation_fn : callable
            h(x, t) -> y (nonlinear observation function)
        process_noise_cov : np.ndarray
            Process noise covariance Q
        observation_noise_cov : np.ndarray
            Observation noise covariance R
        alpha : float
            Spread of sigma points (typically 1e-4 to 1)
        beta : float
            Prior knowledge of distribution (2 optimal for Gaussian)
        kappa : float
            Secondary scaling parameter (typically 0 or 3-n)
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f = state_transition_fn
        self.h = observation_fn
        self.Q = process_noise_cov
        self.R = observation_noise_cov
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived parameters
        self.lambda_ = alpha ** 2 * (state_dim + kappa) - state_dim
        self.n_sigma = 2 * state_dim + 1
        
        # Weights
        self.weights_mean = np.zeros(self.n_sigma)
        self.weights_cov = np.zeros(self.n_sigma)
        
        self.weights_mean[0] = self.lambda_ / (state_dim + self.lambda_)
        self.weights_cov[0] = self.lambda_ / (state_dim + self.lambda_) + (1 - alpha ** 2 + beta)
        
        for i in range(1, self.n_sigma):
            self.weights_mean[i] = 1 / (2 * (state_dim + self.lambda_))
            self.weights_cov[i] = 1 / (2 * (state_dim + self.lambda_))
        
        # Storage
        self.filtered_states = None
        self.filtered_covariances = None
        self.predicted_states = None
        self.predicted_covariances = None
        self.innovations = None
        self.log_likelihood = None
    
    def generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for unscented transform.
        
        Parameters
        ----------
        mean : np.ndarray
            Mean vector (n,)
        cov : np.ndarray
            Covariance matrix (n, n)
            
        Returns
        -------
        sigma_points : np.ndarray
            Sigma points (2n+1, n)
        """
        n = len(mean)
        sigma_points = np.zeros((self.n_sigma, n))
        
        # Central point
        sigma_points[0] = mean
        
        # Matrix square root
        sqrt_cov = matrix_sqrt((n + self.lambda_) * cov)
        
        # Positive deviations
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
        
        # Negative deviations
        for i in range(n):
            sigma_points[n + i + 1] = mean - sqrt_cov[:, i]
        
        return sigma_points
    
    def unscented_transform(self, 
                           sigma_points: np.ndarray,
                           transform_fn: Callable,
                           noise_cov: np.ndarray,
                           t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transform to propagate sigma points.
        
        Parameters
        ----------
        sigma_points : np.ndarray
            Input sigma points (2n+1, n)
        transform_fn : callable
            Nonlinear transformation function
        noise_cov : np.ndarray
            Additive noise covariance
        t : int
            Time index
            
        Returns
        -------
        mean : np.ndarray
            Transformed mean
        cov : np.ndarray
            Transformed covariance
        """
        # Transform sigma points
        transformed = np.array([transform_fn(sp, t) for sp in sigma_points])
        
        # Compute weighted mean
        mean = np.sum(self.weights_mean[:, np.newaxis] * transformed, axis=0)
        
        # Compute weighted covariance
        deviations = transformed - mean
        cov = np.sum(self.weights_cov[:, np.newaxis, np.newaxis] * 
                    (deviations[:, :, np.newaxis] @ deviations[:, np.newaxis, :]),
                    axis=0)
        
        cov = cov + noise_cov
        cov = ensure_positive_definite(cov)
        
        return mean, cov
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run UKF on observation sequence.
        
        Parameters
        ----------
        observations : np.ndarray
            Observation sequence (n_obs, obs_dim) or (n_obs,)
            
        Returns
        -------
        filtered_states : np.ndarray
            Filtered state estimates
        filtered_covariances : np.ndarray
            Filtered state covariances
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        n_obs = len(observations)
        
        # Initialize storage
        self.filtered_states = np.zeros((n_obs, self.state_dim))
        self.filtered_covariances = np.zeros((n_obs, self.state_dim, self.state_dim))
        self.predicted_states = np.zeros((n_obs, self.state_dim))
        self.predicted_covariances = np.zeros((n_obs, self.state_dim, self.state_dim))
        self.innovations = np.zeros((n_obs, self.obs_dim))
        
        # Initialize state
        x_filt = np.zeros(self.state_dim)
        P_filt = np.eye(self.state_dim) * 10.0
        
        log_likelihood = 0.0
        
        for t in range(n_obs):
            # Prediction step
            if t == 0:
                x_pred = x_filt
                P_pred = P_filt
            else:
                # Generate sigma points
                sigma_points = self.generate_sigma_points(x_filt, P_filt)
                
                # Propagate through state transition
                x_pred, P_pred = self.unscented_transform(
                    sigma_points, self.f, self.Q, t
                )
            
            self.predicted_states[t] = x_pred
            self.predicted_covariances[t] = P_pred
            
            # Update step
            y_t = observations[t].reshape(-1, 1) if observations[t].ndim == 0 else observations[t].reshape(-1, 1)
            
            # Generate sigma points for predicted state
            sigma_points_pred = self.generate_sigma_points(x_pred, P_pred)
            
            # Propagate through observation function
            y_pred, S = self.unscented_transform(
                sigma_points_pred, self.h, self.R, t
            )
            
            # Cross-covariance
            transformed_obs = np.array([self.h(sp, t) for sp in sigma_points_pred])
            state_deviations = sigma_points_pred - x_pred
            obs_deviations = transformed_obs - y_pred
            
            P_xy = np.sum(self.weights_cov[:, np.newaxis, np.newaxis] *
                         (state_deviations[:, :, np.newaxis] @ obs_deviations[:, np.newaxis, :]),
                         axis=0)
            
            # Kalman gain
            K = P_xy @ np.linalg.inv(S)
            
            # Innovation
            innovation = y_t.flatten() - y_pred
            self.innovations[t] = innovation
            
            # Update state
            x_filt = x_pred + K @ innovation
            
            # Update covariance
            P_filt = P_pred - K @ S @ K.T
            P_filt = ensure_positive_definite(P_filt)
            
            self.filtered_states[t] = x_filt
            self.filtered_covariances[t] = P_filt
            
            # Log-likelihood
            log_likelihood += self._log_likelihood_contribution(innovation.reshape(-1, 1), S)
        
        self.log_likelihood = log_likelihood
        
        return self.filtered_states, self.filtered_covariances
    
    def _log_likelihood_contribution(self, innovation: np.ndarray,
                                    innovation_cov: np.ndarray) -> float:
        """Calculate log-likelihood contribution."""
        k = len(innovation)
        
        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            logdet = np.log(np.linalg.det(innovation_cov + 1e-10 * np.eye(k)))
        
        mahalanobis = innovation.T @ np.linalg.inv(innovation_cov) @ innovation
        
        ll = -0.5 * (k * np.log(2 * np.pi) + logdet + mahalanobis.item())
        
        return ll


# Example: Nonlinear stochastic volatility model for UKF
def create_ukf_sv_model(persistence: float = 0.95,
                       vol_of_vol: float = 0.1) -> UnscentedKalmanFilter:
    """
    Create UKF for stochastic volatility model.
    
    State: log-volatility h_t
    Observation: log(r_t²)
    
    Parameters
    ----------
    persistence : float
        AR(1) coefficient
    vol_of_vol : float
        Volatility of log-volatility
        
    Returns
    -------
    UnscentedKalmanFilter
    """
    mean_log_vol = 0.0
    
    def state_transition(x, t):
        """AR(1) for log-volatility"""
        return np.array([mean_log_vol + persistence * (x[0] - mean_log_vol)])
    
    def observation(x, t):
        """Nonlinear: log(r²) ≈ h + noise"""
        return x  # Direct observation of log-volatility
    
    Q = np.array([[vol_of_vol ** 2]])
    R = np.array([[np.pi ** 2 / 2]])  # Theoretical variance of log(χ²₁)
    
    return UnscentedKalmanFilter(
        state_dim=1,
        obs_dim=1,
        state_transition_fn=state_transition,
        observation_fn=observation,
        process_noise_cov=Q,
        observation_noise_cov=R,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0
    )


if __name__ == '__main__':
    # Test UKF
    print("Testing Unscented Kalman Filter...")
    
    # Generate nonlinear data (stochastic volatility)
    np.random.seed(42)
    n = 200
    
    # True log-volatility (AR(1))
    log_vol = np.zeros(n)
    for t in range(1, n):
        log_vol[t] = 0.95 * log_vol[t-1] + np.random.randn() * 0.1
    
    # Returns with stochastic volatility
    returns = np.exp(log_vol / 2) * np.random.randn(n)
    
    # Observations: log squared returns
    observations = np.log(returns ** 2 + 1e-8)
    
    # Run UKF
    ukf = create_ukf_sv_model()
    filtered, _ = ukf.filter(observations)
    
    print(f"\nFiltered states shape: {filtered.shape}")
    print(f"Log-likelihood: {ukf.log_likelihood:.2f}")
    print(f"Correlation with true log-vol: {np.corrcoef(filtered.flatten(), log_vol)[0, 1]:.3f}")
