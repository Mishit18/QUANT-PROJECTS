"""
Extended Kalman Filter (EKF) for nonlinear state-space models.

Handles nonlinear dynamics via first-order Taylor approximation:
- State equation: x_t = f(x_{t-1}) + w_t
- Observation equation: y_t = h(x_t) + v_t

Linearization via Jacobian matrices.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from src.utils import ensure_positive_definite


class NonlinearStateSpaceModel:
    """
    Nonlinear state-space model specification for EKF.
    """
    
    def __init__(self, 
                 state_dim: int,
                 obs_dim: int,
                 state_transition_fn: Callable,
                 observation_fn: Callable,
                 state_jacobian_fn: Callable,
                 observation_jacobian_fn: Callable,
                 process_noise_cov: np.ndarray,
                 observation_noise_cov: np.ndarray):
        """
        Initialize nonlinear model.
        
        Parameters
        ----------
        state_dim : int
            State dimension
        obs_dim : int
            Observation dimension
        state_transition_fn : callable
            f(x, t) -> x_next
        observation_fn : callable
            h(x, t) -> y
        state_jacobian_fn : callable
            ∂f/∂x(x, t) -> Jacobian matrix
        observation_jacobian_fn : callable
            ∂h/∂x(x, t) -> Jacobian matrix
        process_noise_cov : np.ndarray
            Process noise covariance Q
        observation_noise_cov : np.ndarray
            Observation noise covariance R
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f = state_transition_fn
        self.h = observation_fn
        self.F_jacobian = state_jacobian_fn
        self.H_jacobian = observation_jacobian_fn
        self.Q = process_noise_cov
        self.R = observation_noise_cov
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state mean and covariance."""
        x0 = np.zeros(self.state_dim)
        P0 = np.eye(self.state_dim) * 10.0
        return x0, P0


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.
    
    Linearizes nonlinear functions around current state estimate
    using first-order Taylor expansion.
    """
    
    def __init__(self, model: NonlinearStateSpaceModel):
        """
        Initialize EKF.
        
        Parameters
        ----------
        model : NonlinearStateSpaceModel
            Nonlinear model specification
        """
        self.model = model
        
        self.filtered_states = None
        self.filtered_covariances = None
        self.predicted_states = None
        self.predicted_covariances = None
        self.innovations = None
        self.log_likelihood = None
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run EKF on observation sequence.
        
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
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim
        
        # Initialize storage
        self.filtered_states = np.zeros((n_obs, state_dim))
        self.filtered_covariances = np.zeros((n_obs, state_dim, state_dim))
        self.predicted_states = np.zeros((n_obs, state_dim))
        self.predicted_covariances = np.zeros((n_obs, state_dim, state_dim))
        self.innovations = np.zeros((n_obs, obs_dim))
        
        # Initialize state
        x_filt, P_filt = self.model.get_initial_state()
        
        log_likelihood = 0.0
        
        for t in range(n_obs):
            # Prediction step (nonlinear)
            if t == 0:
                x_pred = x_filt
                P_pred = P_filt
            else:
                # Propagate state through nonlinear function
                x_pred = self.model.f(x_filt, t)
                
                # Linearize around filtered state
                F = self.model.F_jacobian(x_filt, t)
                
                # Propagate covariance
                P_pred = F @ P_filt @ F.T + self.model.Q
                P_pred = ensure_positive_definite(P_pred)
            
            self.predicted_states[t] = x_pred
            self.predicted_covariances[t] = P_pred
            
            # Update step
            y_t = observations[t].reshape(-1, 1) if observations[t].ndim == 0 else observations[t].reshape(-1, 1)
            
            # Predicted observation (nonlinear)
            y_pred = self.model.h(x_pred, t).reshape(-1, 1)
            
            # Innovation
            innovation = y_t - y_pred
            
            # Linearize observation function
            H = self.model.H_jacobian(x_pred, t)
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.model.R
            S = ensure_positive_definite(S)
            
            self.innovations[t] = innovation.flatten()
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            x_filt = x_pred + (K @ innovation).flatten()
            
            # Update covariance
            P_filt = (np.eye(state_dim) - K @ H) @ P_pred
            P_filt = ensure_positive_definite(P_filt)
            
            self.filtered_states[t] = x_filt
            self.filtered_covariances[t] = P_filt
            
            # Log-likelihood
            log_likelihood += self._log_likelihood_contribution(innovation, S)
        
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


# Example: Nonlinear stochastic volatility model
def create_nonlinear_sv_model(persistence: float = 0.95,
                              vol_of_vol: float = 0.1,
                              leverage: float = -0.5) -> NonlinearStateSpaceModel:
    """
    Create nonlinear stochastic volatility model with leverage effect.
    
    State: log-volatility h_t
    State equation: h_t = μ + φ(h_{t-1} - μ) + σ_h * w_t
    Observation: r_t = exp(h_t/2) * v_t + leverage * σ_h * w_t
    
    Parameters
    ----------
    persistence : float
        AR(1) coefficient
    vol_of_vol : float
        Volatility of log-volatility
    leverage : float
        Leverage effect parameter
        
    Returns
    -------
    NonlinearStateSpaceModel
    """
    mean_log_vol = 0.0
    
    def state_transition(x, t):
        """h_t = μ + φ(h_{t-1} - μ) + noise"""
        return mean_log_vol + persistence * (x - mean_log_vol)
    
    def observation(x, t):
        """Nonlinear: volatility affects returns"""
        # Return mean is zero, but we return the volatility level
        # Actual observation will be r_t / exp(h_t/2)
        return np.array([0.0])
    
    def state_jacobian(x, t):
        """∂f/∂x = φ"""
        return np.array([[persistence]])
    
    def observation_jacobian(x, t):
        """∂h/∂x - linearization of exp(h/2)"""
        return np.array([[0.5 * np.exp(x[0] / 2)]])
    
    Q = np.array([[vol_of_vol ** 2]])
    R = np.array([[1.0]])  # Standardized returns
    
    return NonlinearStateSpaceModel(
        state_dim=1,
        obs_dim=1,
        state_transition_fn=state_transition,
        observation_fn=observation,
        state_jacobian_fn=state_jacobian,
        observation_jacobian_fn=observation_jacobian,
        process_noise_cov=Q,
        observation_noise_cov=R
    )


# Example: Nonlinear mean-reverting model with regime-dependent speed
def create_nonlinear_mean_reversion_model(base_speed: float = 0.1,
                                         speed_sensitivity: float = 0.5) -> NonlinearStateSpaceModel:
    """
    Create nonlinear mean-reversion model where reversion speed depends on distance.
    
    State equation: x_t = x_{t-1} - κ(x_{t-1}) * x_{t-1} + w_t
    where κ(x) = base_speed + speed_sensitivity * |x|
    
    Faster mean reversion when far from equilibrium.
    
    Parameters
    ----------
    base_speed : float
        Base mean-reversion speed
    speed_sensitivity : float
        Additional speed per unit distance
        
    Returns
    -------
    NonlinearStateSpaceModel
    """
    def state_transition(x, t):
        """Nonlinear mean reversion"""
        kappa = base_speed + speed_sensitivity * np.abs(x[0])
        return x * (1 - kappa)
    
    def observation(x, t):
        """Direct observation"""
        return x
    
    def state_jacobian(x, t):
        """Jacobian of state transition"""
        kappa = base_speed + speed_sensitivity * np.abs(x[0])
        dkappa_dx = speed_sensitivity * np.sign(x[0])
        return np.array([[1 - kappa - x[0] * dkappa_dx]])
    
    def observation_jacobian(x, t):
        """Identity for direct observation"""
        return np.array([[1.0]])
    
    Q = np.array([[0.01]])
    R = np.array([[0.1]])
    
    return NonlinearStateSpaceModel(
        state_dim=1,
        obs_dim=1,
        state_transition_fn=state_transition,
        observation_fn=observation,
        state_jacobian_fn=state_jacobian,
        observation_jacobian_fn=observation_jacobian,
        process_noise_cov=Q,
        observation_noise_cov=R
    )


if __name__ == '__main__':
    # Test EKF
    print("Testing Extended Kalman Filter...")
    
    # Generate nonlinear data
    np.random.seed(42)
    n = 200
    
    # Nonlinear mean reversion
    true_state = np.zeros(n)
    observations = np.zeros(n)
    
    for t in range(1, n):
        kappa = 0.1 + 0.5 * np.abs(true_state[t-1])
        true_state[t] = true_state[t-1] * (1 - kappa) + np.random.randn() * 0.1
        observations[t] = true_state[t] + np.random.randn() * 0.3
    
    # Run EKF
    model = create_nonlinear_mean_reversion_model()
    ekf = ExtendedKalmanFilter(model)
    
    filtered, _ = ekf.filter(observations)
    
    print(f"\nFiltered states shape: {filtered.shape}")
    print(f"Log-likelihood: {ekf.log_likelihood:.2f}")
    print(f"Mean absolute error: {np.mean(np.abs(filtered.flatten() - true_state)):.4f}")
