"""
Kalman filter for dynamic hedge ratio estimation.
Minimal implementation - no over-engineering.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class KalmanHedge:
    """
    Kalman filter for time-varying hedge ratio.
    
    State: beta_t (hedge ratio)
    Observation: y_t = beta_t * x_t + noise
    """
    
    def __init__(self, transition_cov: float = 1e-4, observation_cov: float = 1e-2):
        """
        Args:
            transition_cov: Process noise (Q) - how much beta changes
            observation_cov: Measurement noise (R) - spread noise
        """
        # PRODUCTION FIX: Defensive type casting (config hygiene)
        # YAML can parse scientific notation as strings
        # Enforce numeric types at boundary
        try:
            self.Q = float(transition_cov)
            self.R = float(observation_cov)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Kalman filter parameters must be numeric. "
                f"Got Q={transition_cov} (type: {type(transition_cov)}), "
                f"R={observation_cov} (type: {type(observation_cov)}). "
                f"Error: {e}"
            )
        
        # Validation: Must be positive
        if self.Q <= 0 or self.R <= 0:
            raise ValueError(
                f"Kalman filter parameters must be positive. "
                f"Got Q={self.Q}, R={self.R}"
            )
        
        self.beta_history = []
        self.P_history = []
    
    def filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Run Kalman filter to estimate dynamic hedge ratio.
        
        Args:
            y: Asset 1 prices
            x: Asset 2 prices
        
        Returns:
            DataFrame with beta estimates
        """
        # Align data
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        # Initialize with OLS estimate
        initial_beta = np.cov(df['y'], df['x'])[0, 1] / np.var(df['x'])
        
        beta = initial_beta
        P = 1.0  # Initial uncertainty
        
        self.beta_history = [beta]
        self.P_history = [P]
        
        # Filter through data
        for i in range(len(df)):
            y_t = df['y'].iloc[i]
            x_t = df['x'].iloc[i]
            
            # Predict
            beta_pred = beta
            P_pred = P + self.Q
            
            # Update
            innovation = y_t - beta_pred * x_t
            S = x_t**2 * P_pred + self.R
            K = P_pred * x_t / S
            
            beta = beta_pred + K * innovation
            P = P_pred - K * x_t * P_pred
            
            self.beta_history.append(beta)
            self.P_history.append(P)
        
        # Create results
        results = pd.DataFrame({
            'beta': self.beta_history[1:],
            'beta_std': np.sqrt(self.P_history[1:])
        }, index=df.index)
        
        return results
    
    def generate_spread(self, y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
        """
        Generate spread using dynamic hedge ratio.
        
        Args:
            y: Asset 1 prices
            x: Asset 2 prices
            beta: Dynamic hedge ratios
        
        Returns:
            Spread series
        """
        df = pd.DataFrame({'y': y, 'x': x, 'beta': beta}).dropna()
        spread = df['y'] - df['beta'] * df['x']
        return spread
