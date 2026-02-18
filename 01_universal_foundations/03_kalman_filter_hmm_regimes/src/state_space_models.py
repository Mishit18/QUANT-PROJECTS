"""
State-space model specifications for financial time series.

Defines various state-space representations used in Kalman filtering:
- Local level model (random walk + noise)
- Local linear trend model
- Dynamic regression (time-varying coefficients)
- Stochastic volatility approximation
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class StateSpaceModel:
    """
    Base class for state-space model specification.
    
    State equation: x_t = F_t @ x_{t-1} + w_t,  w_t ~ N(0, Q_t)
    Observation equation: y_t = H_t @ x_t + v_t,  v_t ~ N(0, R_t)
    """
    
    def __init__(self, state_dim: int, obs_dim: int):
        """
        Initialize state-space model.
        
        Parameters
        ----------
        state_dim : int
            Dimension of state vector
        obs_dim : int
            Dimension of observation vector
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get state-space matrices at time t.
        
        Returns
        -------
        F : np.ndarray
            State transition matrix (state_dim x state_dim)
        H : np.ndarray
            Observation matrix (obs_dim x state_dim)
        Q : np.ndarray
            State noise covariance (state_dim x state_dim)
        R : np.ndarray
            Observation noise covariance (obs_dim x obs_dim)
        """
        raise NotImplementedError
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get initial state mean and covariance.
        
        Returns
        -------
        x0 : np.ndarray
            Initial state mean (state_dim,)
        P0 : np.ndarray
            Initial state covariance (state_dim x state_dim)
        """
        raise NotImplementedError


class LocalLevelModel(StateSpaceModel):
    """
    Local level model: random walk with observation noise.
    
    State equation: μ_t = μ_{t-1} + w_t,  w_t ~ N(0, σ²_state)
    Observation equation: y_t = μ_t + v_t,  v_t ~ N(0, σ²_obs)
    
    Used for trend extraction and smoothing.
    """
    
    def __init__(self, observation_variance: float = 1.0, 
                 state_variance: float = 0.1,
                 initial_state_variance: float = 10.0):
        """
        Initialize local level model.
        
        Parameters
        ----------
        observation_variance : float
            Observation noise variance (σ²_obs)
        state_variance : float
            State evolution noise variance (σ²_state)
        initial_state_variance : float
            Initial state uncertainty
        """
        super().__init__(state_dim=1, obs_dim=1)
        self.observation_variance = observation_variance
        self.state_variance = state_variance
        self.initial_state_variance = initial_state_variance
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state-space matrices."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[self.state_variance]])
        R = np.array([[self.observation_variance]])
        
        return F, H, Q, R
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        x0 = np.array([0.0])
        P0 = np.array([[self.initial_state_variance]])
        
        return x0, P0


class LocalLinearTrendModel(StateSpaceModel):
    """
    Local linear trend model: random walk with stochastic slope.
    
    State equation:
        μ_t = μ_{t-1} + β_{t-1} + w1_t
        β_t = β_{t-1} + w2_t
    
    Observation equation:
        y_t = μ_t + v_t
    
    Captures both level and trend dynamics.
    """
    
    def __init__(self, 
                 observation_variance: float = 1.0,
                 level_variance: float = 0.1,
                 slope_variance: float = 0.01,
                 initial_state_variance: float = 10.0):
        """
        Initialize local linear trend model.
        
        Parameters
        ----------
        observation_variance : float
            Observation noise variance
        level_variance : float
            Level evolution noise variance
        slope_variance : float
            Slope evolution noise variance
        initial_state_variance : float
            Initial state uncertainty
        """
        super().__init__(state_dim=2, obs_dim=1)
        self.observation_variance = observation_variance
        self.level_variance = level_variance
        self.slope_variance = slope_variance
        self.initial_state_variance = initial_state_variance
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state-space matrices."""
        F = np.array([[1.0, 1.0],
                     [0.0, 1.0]])
        
        H = np.array([[1.0, 0.0]])
        
        Q = np.array([[self.level_variance, 0.0],
                     [0.0, self.slope_variance]])
        
        R = np.array([[self.observation_variance]])
        
        return F, H, Q, R
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * self.initial_state_variance
        
        return x0, P0


class DynamicRegressionModel(StateSpaceModel):
    """
    Dynamic regression with time-varying coefficients.
    
    State equation: β_t = β_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation equation: y_t = X_t @ β_t + v_t,  v_t ~ N(0, R)
    
    Used for time-varying beta estimation (e.g., rolling CAPM beta).
    """
    
    def __init__(self, 
                 n_regressors: int,
                 observation_variance: float = 1.0,
                 coefficient_variance: float = 0.01,
                 initial_state_variance: float = 10.0):
        """
        Initialize dynamic regression model.
        
        Parameters
        ----------
        n_regressors : int
            Number of regressors (including intercept if desired)
        observation_variance : float
            Observation noise variance
        coefficient_variance : float
            Coefficient evolution noise variance
        initial_state_variance : float
            Initial coefficient uncertainty
        """
        super().__init__(state_dim=n_regressors, obs_dim=1)
        self.n_regressors = n_regressors
        self.observation_variance = observation_variance
        self.coefficient_variance = coefficient_variance
        self.initial_state_variance = initial_state_variance
        
        self.X = None  # Will be set externally
    
    def set_regressors(self, X: np.ndarray):
        """
        Set regressor matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Regressor matrix (n_obs x n_regressors)
        """
        self.X = X
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state-space matrices."""
        F = np.eye(self.n_regressors)
        
        if self.X is not None and t < len(self.X):
            H = self.X[t:t+1, :]  # Shape (1, n_regressors)
        else:
            H = np.ones((1, self.n_regressors))
        
        Q = np.eye(self.n_regressors) * self.coefficient_variance
        R = np.array([[self.observation_variance]])
        
        return F, H, Q, R
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        x0 = np.zeros(self.n_regressors)
        P0 = np.eye(self.n_regressors) * self.initial_state_variance
        
        return x0, P0


class StochasticVolatilityModel(StateSpaceModel):
    """
    Linearized stochastic volatility model.
    
    Log-volatility follows AR(1):
        h_t = μ + φ(h_{t-1} - μ) + w_t,  w_t ~ N(0, σ²_h)
    
    Observation (log squared returns):
        log(y_t²) ≈ h_t + v_t,  v_t ~ N(0, σ²_v)
    
    Approximation to true SV model for Kalman filtering.
    """
    
    def __init__(self,
                 persistence: float = 0.95,
                 volatility_of_volatility: float = 0.1,
                 observation_variance: float = 1.0,
                 mean_log_volatility: float = 0.0,
                 initial_state_variance: float = 1.0):
        """
        Initialize stochastic volatility model.
        
        Parameters
        ----------
        persistence : float
            AR(1) coefficient φ (typically 0.9-0.99)
        volatility_of_volatility : float
            Volatility of log-volatility σ_h
        observation_variance : float
            Observation noise variance σ²_v
        mean_log_volatility : float
            Long-run mean of log-volatility μ
        initial_state_variance : float
            Initial state uncertainty
        """
        super().__init__(state_dim=1, obs_dim=1)
        self.persistence = persistence
        self.volatility_of_volatility = volatility_of_volatility
        self.observation_variance = observation_variance
        self.mean_log_volatility = mean_log_volatility
        self.initial_state_variance = initial_state_variance
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state-space matrices."""
        F = np.array([[self.persistence]])
        H = np.array([[1.0]])
        Q = np.array([[self.volatility_of_volatility ** 2]])
        R = np.array([[self.observation_variance]])
        
        return F, H, Q, R
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        x0 = np.array([self.mean_log_volatility])
        P0 = np.array([[self.initial_state_variance]])
        
        return x0, P0
    
    def transform_observations(self, returns: np.ndarray, 
                              offset: float = 1e-8) -> np.ndarray:
        """
        Transform returns to log squared returns for observation equation.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
        offset : float
            Small constant to avoid log(0)
            
        Returns
        -------
        np.ndarray
            Log squared returns
        """
        return np.log(returns ** 2 + offset)
    
    def inverse_transform(self, log_volatility: np.ndarray) -> np.ndarray:
        """
        Convert log-volatility to volatility.
        
        Parameters
        ----------
        log_volatility : np.ndarray
            Log-volatility series
            
        Returns
        -------
        np.ndarray
            Volatility series
        """
        return np.exp(log_volatility / 2)


class ARModel(StateSpaceModel):
    """
    AR(p) model in state-space form.
    
    State equation: x_t = F @ x_{t-1} + w_t
    where x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]'
    
    Observation equation: y_t = [1, 0, ..., 0] @ x_t
    """
    
    def __init__(self, 
                 ar_coefficients: np.ndarray,
                 innovation_variance: float = 1.0,
                 initial_state_variance: float = 10.0):
        """
        Initialize AR model.
        
        Parameters
        ----------
        ar_coefficients : np.ndarray
            AR coefficients [φ_1, φ_2, ..., φ_p]
        innovation_variance : float
            Innovation variance σ²
        initial_state_variance : float
            Initial state uncertainty
        """
        p = len(ar_coefficients)
        super().__init__(state_dim=p, obs_dim=1)
        
        self.ar_coefficients = ar_coefficients
        self.innovation_variance = innovation_variance
        self.initial_state_variance = initial_state_variance
    
    def get_matrices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state-space matrices."""
        p = self.state_dim
        
        # Companion form
        F = np.zeros((p, p))
        F[0, :] = self.ar_coefficients
        if p > 1:
            F[1:, :-1] = np.eye(p - 1)
        
        H = np.zeros((1, p))
        H[0, 0] = 1.0
        
        Q = np.zeros((p, p))
        Q[0, 0] = self.innovation_variance
        
        R = np.array([[1e-10]])  # Minimal observation noise
        
        return F, H, Q, R
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        x0 = np.zeros(self.state_dim)
        P0 = np.eye(self.state_dim) * self.initial_state_variance
        
        return x0, P0


if __name__ == '__main__':
    # Test model specifications
    print("Testing state-space models...")
    
    # Local level model
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    F, H, Q, R = model.get_matrices(0)
    print(f"\nLocal Level Model:")
    print(f"F shape: {F.shape}, H shape: {H.shape}")
    print(f"Q: {Q}, R: {R}")
    
    # Local linear trend
    model = LocalLinearTrendModel()
    F, H, Q, R = model.get_matrices(0)
    print(f"\nLocal Linear Trend Model:")
    print(f"F:\n{F}")
    print(f"H: {H}")
    
    # Dynamic regression
    model = DynamicRegressionModel(n_regressors=2)
    X = np.random.randn(100, 2)
    model.set_regressors(X)
    F, H, Q, R = model.get_matrices(0)
    print(f"\nDynamic Regression Model:")
    print(f"State dim: {model.state_dim}")
    print(f"H shape: {H.shape}")
