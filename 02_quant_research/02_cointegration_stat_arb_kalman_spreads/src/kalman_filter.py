"""
Kalman Filter for adaptive spread modeling.

Implements state-space model for time-varying hedge ratios and equilibrium levels.
Provides superior performance over static cointegration approaches in non-stationary
market regimes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pykalman import KalmanFilter
import logging

from utils import setup_logging


logger = setup_logging()


class KalmanSpreadModel:
    """
    Kalman Filter model for adaptive spread construction.
    
    State-space formulation:
    - State: [hedge_ratio, equilibrium_level]
    - Observation: y_t = hedge_ratio_t * x_t + equilibrium_level_t + noise
    
    The model adapts to time-varying relationships between securities.
    """
    
    def __init__(self,
                 transition_covariance: float = 1e-5,
                 observation_covariance: float = 1e-3,
                 initial_state_mean: Optional[np.ndarray] = None,
                 initial_state_covariance: Optional[np.ndarray] = None):
        """
        Initialize Kalman Filter spread model.
        
        Args:
            transition_covariance: Process noise (how much hedge ratio can change)
            observation_covariance: Measurement noise
            initial_state_mean: Initial state estimate [hedge_ratio, equilibrium]
            initial_state_covariance: Initial state covariance matrix
        """
        self.transition_covariance = float(transition_covariance)
        self.observation_covariance = float(observation_covariance)
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        
        self.kf = None
        self.state_means = None
        self.state_covariances = None
    
    def _setup_kalman_filter(self, n_obs: int) -> KalmanFilter:
        """
        Setup Kalman Filter with state-space matrices.
        
        State equation: s_t = F * s_{t-1} + w_t, w_t ~ N(0, Q)
        Observation equation: y_t = H_t * s_t + v_t, v_t ~ N(0, R)
        
        Where:
        - s_t = [beta_t, alpha_t]' (hedge ratio and equilibrium level)
        - F = I (random walk for both states)
        - Q = transition_covariance * I
        - H_t = [x_t, 1] (observation matrix depends on x_t)
        - R = observation_covariance
        
        Args:
            n_obs: Number of observations
            
        Returns:
            Configured KalmanFilter object
        """
        # State transition matrix (random walk)
        transition_matrix = np.eye(2)
        
        # Process noise covariance
        transition_cov = np.eye(2) * self.transition_covariance
        
        # Observation noise covariance
        observation_cov = self.observation_covariance
        
        # Initial state
        if self.initial_state_mean is None:
            initial_state_mean = np.array([1.0, 0.0])  # [beta, alpha]
        else:
            initial_state_mean = self.initial_state_mean
        
        if self.initial_state_covariance is None:
            initial_state_cov = np.eye(2)
        else:
            initial_state_cov = self.initial_state_covariance
        
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_covariance=observation_cov,
            transition_covariance=transition_cov,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_cov,
            n_dim_state=2,
            n_dim_obs=1
        )
        
        return kf
    
    def fit(self,
           y: pd.Series,
           x: pd.Series,
           em_iterations: int = 10) -> 'KalmanSpreadModel':
        """
        Fit Kalman Filter to data using Expectation-Maximization.
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            em_iterations: Number of EM iterations for parameter estimation
            
        Returns:
            Self (fitted model)
        """
        logger.info("Fitting Kalman Filter spread model")
        
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < 30:
            raise ValueError("Insufficient data for Kalman Filter")
        
        y_values = df['y'].values
        x_values = df['x'].values
        n_obs = len(df)
        
        # Setup Kalman Filter
        self.kf = self._setup_kalman_filter(n_obs)
        
        # Observation matrix varies with x_t: H_t = [x_t, 1]
        observation_matrices = np.zeros((n_obs, 1, 2))
        observation_matrices[:, 0, 0] = x_values
        observation_matrices[:, 0, 1] = 1.0
        
        # Update Kalman Filter with time-varying observation matrix
        self.kf.observation_matrices = observation_matrices
        
        # Fit using EM algorithm to optimize parameters
        if em_iterations > 0:
            logger.info(f"Running EM algorithm with {em_iterations} iterations")
            try:
                self.kf = self.kf.em(
                    y_values.reshape(-1, 1),
                    n_iter=em_iterations
                )
                
                # Update observation matrices after EM
                self.kf.observation_matrices = observation_matrices
                logger.info("EM optimization complete")
            except Exception as e:
                logger.warning(f"EM algorithm failed ({e}), using initial parameters")
                # Continue with initial parameters
        
        # Filter to get state estimates
        self.state_means, self.state_covariances = self.kf.filter(
            y_values.reshape(-1, 1)
        )
        
        # Store index for later use
        self.index = df.index
        
        logger.info("Kalman Filter fitted successfully")
        
        return self
    
    def get_hedge_ratio(self) -> pd.Series:
        """
        Extract time-varying hedge ratio from filtered states.
        
        Returns:
            Series with hedge ratio over time
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hedge_ratio = pd.Series(
            self.state_means[:, 0],
            index=self.index,
            name='hedge_ratio'
        )
        
        return hedge_ratio
    
    def get_equilibrium_level(self) -> pd.Series:
        """
        Extract time-varying equilibrium level from filtered states.
        
        Returns:
            Series with equilibrium level over time
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        equilibrium = pd.Series(
            self.state_means[:, 1],
            index=self.index,
            name='equilibrium_level'
        )
        
        return equilibrium
    
    def get_spread(self,
                  y: pd.Series,
                  x: pd.Series) -> pd.Series:
        """
        Compute spread using time-varying hedge ratio.
        
        Spread = y_t - beta_t * x_t - alpha_t
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            
        Returns:
            Spread series
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        df = df.loc[self.index]
        
        hedge_ratio = self.state_means[:, 0]
        equilibrium = self.state_means[:, 1]
        
        spread = df['y'].values - hedge_ratio * df['x'].values - equilibrium
        
        spread_series = pd.Series(
            spread,
            index=self.index,
            name='spread'
        )
        
        return spread_series
    
    def fit_transform(self,
                     y: pd.Series,
                     x: pd.Series,
                     em_iterations: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Fit model and return spread and hedge ratio.
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            em_iterations: Number of EM iterations
            
        Returns:
            Tuple of (spread, hedge_ratio)
        """
        self.fit(y, x, em_iterations)
        spread = self.get_spread(y, x)
        hedge_ratio = self.get_hedge_ratio()
        
        return spread, hedge_ratio
    
    def get_state_uncertainty(self) -> pd.DataFrame:
        """
        Extract state uncertainty (standard deviations) over time.
        
        Returns:
            DataFrame with hedge ratio and equilibrium uncertainties
        """
        if self.state_covariances is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract diagonal elements (variances)
        hedge_ratio_var = self.state_covariances[:, 0, 0]
        equilibrium_var = self.state_covariances[:, 1, 1]
        
        uncertainty = pd.DataFrame({
            'hedge_ratio_std': np.sqrt(hedge_ratio_var),
            'equilibrium_std': np.sqrt(equilibrium_var)
        }, index=self.index)
        
        return uncertainty
    
    def predict_next(self,
                    x_next: float) -> Tuple[float, float]:
        """
        Predict next observation and its uncertainty.
        
        Args:
            x_next: Next value of independent variable
            
        Returns:
            Tuple of (predicted_y, prediction_std)
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use last filtered state
        last_state_mean = self.state_means[-1]
        last_state_cov = self.state_covariances[-1]
        
        # Predict next state (random walk)
        predicted_state_mean = last_state_mean
        predicted_state_cov = last_state_cov + self.kf.transition_covariance
        
        # Predict observation
        beta, alpha = predicted_state_mean
        predicted_y = beta * x_next + alpha
        
        # Prediction uncertainty
        H = np.array([[x_next, 1.0]])
        prediction_var = (
            H @ predicted_state_cov @ H.T +
            self.kf.observation_covariance
        )
        prediction_std = np.sqrt(prediction_var[0, 0])
        
        return predicted_y, prediction_std
    
    def get_diagnostics(self) -> Dict:
        """
        Compute diagnostic statistics for the fitted model.
        
        Returns:
            Dictionary with diagnostic information
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hedge_ratio = self.state_means[:, 0]
        equilibrium = self.state_means[:, 1]
        
        diagnostics = {
            'mean_hedge_ratio': np.mean(hedge_ratio),
            'std_hedge_ratio': np.std(hedge_ratio),
            'min_hedge_ratio': np.min(hedge_ratio),
            'max_hedge_ratio': np.max(hedge_ratio),
            'mean_equilibrium': np.mean(equilibrium),
            'std_equilibrium': np.std(equilibrium),
            'transition_covariance': self.transition_covariance,
            'observation_covariance': self.observation_covariance,
            'n_observations': len(self.state_means)
        }
        
        return diagnostics


class StaticHedgeRatioModel:
    """
    Static hedge ratio model for comparison with Kalman Filter.
    
    Uses OLS regression to estimate a constant hedge ratio.
    """
    
    def __init__(self):
        """Initialize static hedge ratio model."""
        self.beta = None
        self.alpha = None
        self.index = None
    
    def fit(self, y: pd.Series, x: pd.Series) -> 'StaticHedgeRatioModel':
        """
        Fit static hedge ratio using OLS.
        
        Args:
            y: Dependent variable
            x: Independent variable
            
        Returns:
            Self (fitted model)
        """
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        # OLS regression
        X = df['x'].values.reshape(-1, 1)
        Y = df['y'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([X, np.ones(len(X))])
        
        # Solve normal equations
        coeffs = np.linalg.lstsq(X_with_intercept, Y, rcond=None)[0]
        
        self.beta = coeffs[0]
        self.alpha = coeffs[1]
        self.index = df.index
        
        return self
    
    def get_hedge_ratio(self) -> pd.Series:
        """Get constant hedge ratio."""
        if self.beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hedge_ratio = pd.Series(
            [self.beta] * len(self.index),
            index=self.index,
            name='hedge_ratio'
        )
        
        return hedge_ratio
    
    def get_spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Compute spread using static hedge ratio."""
        if self.beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        df = df.loc[self.index]
        
        spread = df['y'].values - self.beta * df['x'].values - self.alpha
        
        spread_series = pd.Series(
            spread,
            index=self.index,
            name='spread'
        )
        
        return spread_series
