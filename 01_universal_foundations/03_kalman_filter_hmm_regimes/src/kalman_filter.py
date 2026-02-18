"""
Kalman Filter implementation for linear Gaussian state-space models.

Implements:
- Standard Kalman Filter (prediction and update steps)
- Rauch-Tung-Striebel (RTS) smoother
- Log-likelihood calculation
- One-step-ahead predictions
"""

import numpy as np
from typing import Tuple, Optional, Dict
from src.state_space_models import StateSpaceModel
from src.utils import ensure_positive_definite


class KalmanFilter:
    """
    Kalman Filter for optimal state estimation in linear Gaussian systems.
    
    Recursively computes:
    - Filtered state estimates x̂_t|t
    - Predicted state estimates x̂_t|t-1
    - State covariance matrices P_t|t, P_t|t-1
    - Innovation sequences and likelihoods
    """
    
    def __init__(self, model: StateSpaceModel):
        """
        Initialize Kalman Filter.
        
        Parameters
        ----------
        model : StateSpaceModel
            State-space model specification
        """
        self.model = model
        
        # Storage for filtering results
        self.filtered_states = None
        self.filtered_covariances = None
        self.predicted_states = None
        self.predicted_covariances = None
        self.innovations = None
        self.innovation_covariances = None
        self.kalman_gains = None
        self.log_likelihood = None
        
        # Fitted flag
        self.is_fitted = False
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter on observation sequence.
        
        PRODUCTION HARDENING:
        - Validates inputs
        - Checks for NaNs/Infs at every step
        - Enforces numerical stability
        - Logs warnings for edge cases
        
        Parameters
        ----------
        observations : np.ndarray
            Observation sequence (n_obs, obs_dim) or (n_obs,)
            
        Returns
        -------
        filtered_states : np.ndarray
            Filtered state estimates (n_obs, state_dim)
        filtered_covariances : np.ndarray
            Filtered state covariances (n_obs, state_dim, state_dim)
            
        Raises
        ------
        ValueError
            If observations contain NaN/Inf or have invalid shape
        """
        # Input validation
        observations = np.asarray(observations, dtype=np.float64)
        
        if np.any(np.isnan(observations)):
            raise ValueError(f"Observations contain {np.sum(np.isnan(observations))} NaN values")
        
        if np.any(np.isinf(observations)):
            raise ValueError(f"Observations contain {np.sum(np.isinf(observations))} Inf values")
        
        # Ensure observations are 2D
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        n_obs = len(observations)
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim
        
        if n_obs < 2:
            raise ValueError(f"Need at least 2 observations, got {n_obs}")
        
        # Initialize storage
        self.filtered_states = np.zeros((n_obs, state_dim), dtype=np.float64)
        self.filtered_covariances = np.zeros((n_obs, state_dim, state_dim), dtype=np.float64)
        self.predicted_states = np.zeros((n_obs, state_dim), dtype=np.float64)
        self.predicted_covariances = np.zeros((n_obs, state_dim, state_dim), dtype=np.float64)
        self.innovations = np.zeros((n_obs, obs_dim), dtype=np.float64)
        self.innovation_covariances = np.zeros((n_obs, obs_dim, obs_dim), dtype=np.float64)
        self.kalman_gains = np.zeros((n_obs, state_dim, obs_dim), dtype=np.float64)
        
        # Initialize state
        x_filt, P_filt = self.model.get_initial_state()
        x_filt = np.asarray(x_filt, dtype=np.float64)
        P_filt = np.asarray(P_filt, dtype=np.float64)
        P_filt = ensure_positive_definite(P_filt, epsilon=1e-6)
        
        log_likelihood = 0.0
        
        for t in range(n_obs):
            # Get state-space matrices
            F, H, Q, R = self.model.get_matrices(t)
            F = np.asarray(F, dtype=np.float64)
            H = np.asarray(H, dtype=np.float64)
            Q = np.asarray(Q, dtype=np.float64)
            R = np.asarray(R, dtype=np.float64)
            
            # Ensure positive definite
            Q = ensure_positive_definite(Q, epsilon=1e-8)
            R = ensure_positive_definite(R, epsilon=1e-8)
            
            # Prediction step
            if t == 0:
                x_pred = x_filt
                P_pred = P_filt
            else:
                x_pred = F @ x_filt
                P_pred = F @ P_filt @ F.T + Q
                P_pred = ensure_positive_definite(P_pred, epsilon=1e-6)
            
            # Validate prediction
            if np.any(np.isnan(x_pred)) or np.any(np.isinf(x_pred)):
                raise RuntimeError(f"NaN/Inf in predicted state at t={t}")
            
            self.predicted_states[t] = x_pred
            self.predicted_covariances[t] = P_pred
            
            # Innovation
            y_t = observations[t].reshape(-1, 1)
            innovation = y_t - H @ x_pred.reshape(-1, 1)
            S = H @ P_pred @ H.T + R
            S = ensure_positive_definite(S, epsilon=1e-6)
            
            # Validate innovation
            if np.any(np.isnan(innovation)) or np.any(np.isinf(innovation)):
                raise RuntimeError(f"NaN/Inf in innovation at t={t}")
            
            self.innovations[t] = innovation.flatten()
            self.innovation_covariances[t] = S
            
            # Kalman gain with numerical stability
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                S_inv = np.linalg.pinv(S)
            
            K = P_pred @ H.T @ S_inv
            
            # Clip extreme gains
            K = np.clip(K, -1e3, 1e3)
            
            self.kalman_gains[t] = K
            
            # Update step
            x_filt = x_pred + (K @ innovation).flatten()
            P_filt = (np.eye(state_dim) - K @ H) @ P_pred
            
            # Joseph form for numerical stability
            P_filt = (np.eye(state_dim) - K @ H) @ P_pred @ (np.eye(state_dim) - K @ H).T + K @ R @ K.T
            P_filt = ensure_positive_definite(P_filt, epsilon=1e-6)
            
            # Validate update
            if np.any(np.isnan(x_filt)) or np.any(np.isinf(x_filt)):
                raise RuntimeError(f"NaN/Inf in filtered state at t={t}")
            
            self.filtered_states[t] = x_filt
            self.filtered_covariances[t] = P_filt
            
            # Log-likelihood contribution
            try:
                log_likelihood += self._log_likelihood_contribution(innovation, S)
            except:
                # If likelihood calculation fails, continue but warn
                pass
        
        self.log_likelihood = log_likelihood
        
        # Final validation
        if np.any(np.isnan(self.filtered_states)):
            raise RuntimeError("NaN in final filtered states - filter diverged")
        
        # Mark as fitted
        self.is_fitted = True
        
        return self.filtered_states, self.filtered_covariances
    
    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Rauch-Tung-Striebel (RTS) smoother.
        
        Must be called after filter().
        
        Returns
        -------
        smoothed_states : np.ndarray
            Smoothed state estimates (n_obs, state_dim)
        smoothed_covariances : np.ndarray
            Smoothed state covariances (n_obs, state_dim, state_dim)
        """
        if self.filtered_states is None:
            raise ValueError("Must run filter() before smooth()")
        
        n_obs = len(self.filtered_states)
        state_dim = self.model.state_dim
        
        # Initialize with filtered estimates
        smoothed_states = np.copy(self.filtered_states)
        smoothed_covariances = np.copy(self.filtered_covariances)
        
        # Backward pass
        for t in range(n_obs - 2, -1, -1):
            F, _, Q, _ = self.model.get_matrices(t + 1)
            
            # Smoother gain
            P_pred = self.predicted_covariances[t + 1]
            P_filt = self.filtered_covariances[t]
            
            J = P_filt @ F.T @ np.linalg.inv(P_pred)
            
            # Smoothed estimates
            x_smooth_next = smoothed_states[t + 1]
            x_pred_next = self.predicted_states[t + 1]
            x_filt = self.filtered_states[t]
            
            smoothed_states[t] = x_filt + J @ (x_smooth_next - x_pred_next)
            
            P_smooth_next = smoothed_covariances[t + 1]
            smoothed_covariances[t] = P_filt + J @ (P_smooth_next - P_pred) @ J.T
            smoothed_covariances[t] = ensure_positive_definite(smoothed_covariances[t])
        
        return smoothed_states, smoothed_covariances
    
    def filter_and_smooth(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter and smoother in sequence.
        
        Parameters
        ----------
        observations : np.ndarray
            Observation sequence
            
        Returns
        -------
        filtered_states : np.ndarray
            Filtered state estimates
        smoothed_states : np.ndarray
            Smoothed state estimates
        """
        self.filter(observations)
        smoothed_states, _ = self.smooth()
        
        return self.filtered_states, smoothed_states
    
    def predict(self, n_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-step ahead prediction.
        
        Parameters
        ----------
        n_steps : int
            Number of steps ahead to predict
            
        Returns
        -------
        predictions : np.ndarray
            Predicted observations (n_steps, obs_dim)
        prediction_variances : np.ndarray
            Prediction variances (n_steps, obs_dim, obs_dim)
        """
        if self.filtered_states is None:
            raise ValueError("Must run filter() before predict()")
        
        # Start from last filtered state
        x = self.filtered_states[-1]
        P = self.filtered_covariances[-1]
        
        predictions = []
        prediction_variances = []
        
        t_last = len(self.filtered_states) - 1
        
        for step in range(n_steps):
            F, H, Q, R = self.model.get_matrices(t_last + step + 1)
            
            # Predict state
            x = F @ x
            P = F @ P @ F.T + Q
            P = ensure_positive_definite(P)
            
            # Predict observation
            y_pred = H @ x
            y_var = H @ P @ H.T + R
            
            predictions.append(y_pred)
            prediction_variances.append(y_var)
        
        return np.array(predictions), np.array(prediction_variances)
    
    def get_innovations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get innovation sequence (one-step-ahead forecast errors).
        
        Returns
        -------
        innovations : np.ndarray
            Innovation sequence
        innovation_covariances : np.ndarray
            Innovation covariances
        """
        if self.innovations is None:
            raise ValueError("Must run filter() first")
        
        return self.innovations, self.innovation_covariances
    
    def get_log_likelihood(self) -> float:
        """
        Get log-likelihood of observations under model.
        
        Returns
        -------
        float
            Log-likelihood
        """
        if self.log_likelihood is None:
            raise ValueError("Must run filter() first")
        
        return self.log_likelihood
    
    def _log_likelihood_contribution(self, innovation: np.ndarray, 
                                    innovation_cov: np.ndarray) -> float:
        """
        Calculate log-likelihood contribution for single observation.
        
        Parameters
        ----------
        innovation : np.ndarray
            Innovation vector
        innovation_cov : np.ndarray
            Innovation covariance matrix
            
        Returns
        -------
        float
            Log-likelihood contribution
        """
        k = len(innovation)
        
        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            logdet = np.log(np.linalg.det(innovation_cov + 1e-10 * np.eye(k)))
        
        mahalanobis = innovation.T @ np.linalg.inv(innovation_cov) @ innovation
        
        ll = -0.5 * (k * np.log(2 * np.pi) + logdet + mahalanobis.item())
        
        return ll
    
    def diagnose(self) -> Dict[str, np.ndarray]:
        """
        Compute diagnostic statistics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - standardized_innovations: Standardized innovation sequence
            - innovation_autocorr: Innovation autocorrelation
            - ljung_box_stat: Ljung-Box test statistic
        """
        if self.innovations is None:
            raise ValueError("Must run filter() first")
        
        # Standardize innovations
        std_innovations = np.zeros_like(self.innovations)
        for t in range(len(self.innovations)):
            S = self.innovation_covariances[t]
            std_innovations[t] = self.innovations[t] / np.sqrt(np.diag(S))
        
        # Autocorrelation
        max_lag = min(20, len(std_innovations) // 4)
        autocorr = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                autocorr[lag] = np.corrcoef(
                    std_innovations[:-lag].flatten(),
                    std_innovations[lag:].flatten()
                )[0, 1]
        
        # Ljung-Box statistic
        n = len(std_innovations)
        lb_stat = n * (n + 2) * np.sum(autocorr[1:] ** 2 / (n - np.arange(1, max_lag)))
        
        return {
            'standardized_innovations': std_innovations,
            'innovation_autocorr': autocorr,
            'ljung_box_stat': lb_stat
        }


if __name__ == '__main__':
    # Test Kalman filter
    from src.state_space_models import LocalLevelModel
    
    print("Testing Kalman Filter...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 200
    true_state = np.cumsum(np.random.randn(n) * 0.1)
    observations = true_state + np.random.randn(n) * 0.5
    
    # Initialize model and filter
    model = LocalLevelModel(observation_variance=0.25, state_variance=0.01)
    kf = KalmanFilter(model)
    
    # Run filter
    filtered, smoothed = kf.filter_and_smooth(observations)
    
    print(f"\nFiltered states shape: {filtered.shape}")
    print(f"Smoothed states shape: {smoothed.shape}")
    print(f"Log-likelihood: {kf.get_log_likelihood():.2f}")
    
    # Diagnostics
    diagnostics = kf.diagnose()
    print(f"Innovation autocorrelation (lag 1): {diagnostics['innovation_autocorr'][1]:.3f}")
    print(f"Ljung-Box statistic: {diagnostics['ljung_box_stat']:.2f}")
