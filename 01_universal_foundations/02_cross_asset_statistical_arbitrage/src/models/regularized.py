import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from typing import Optional


class RegularizedModel:
    """Ridge/Lasso/ElasticNet with cross-sectional training."""
    
    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0, l1_ratio: float = 0.5):
        self.model_type = model_type
        self.alpha = alpha
        self.scaler = StandardScaler()
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=2000)
        elif model_type == 'elastic':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit with standardization."""
        X_filled = X.fillna(0)
        valid = y.notna()
        if valid.sum() < 20:
            return
        
        X_scaled = self.scaler.fit_transform(X_filled[valid])
        self.model.fit(X_scaled, y[valid])
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with standardization."""
        X_filled = X.fillna(0)
        predictions = pd.Series(np.nan, index=X.index)
        X_scaled = self.scaler.transform(X_filled)
        predictions[:] = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self) -> pd.Series:
        """Coefficient magnitudes."""
        return pd.Series(np.abs(self.model.coef_))


class AdaptiveLasso:
    """Adaptive Lasso with weighted penalties."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.ridge = Ridge(alpha=0.1)
        self.lasso = Lasso(alpha=alpha, max_iter=2000)
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Two-stage adaptive fitting."""
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 20:
            return
        
        X_scaled = self.scaler.fit_transform(X[valid])
        
        self.ridge.fit(X_scaled, y[valid])
        ridge_coefs = np.abs(self.ridge.coef_)
        weights = 1.0 / (ridge_coefs + 1e-8) ** self.gamma
        
        X_weighted = X_scaled / weights
        self.lasso.fit(X_weighted, y[valid])
        self.weights = weights
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with adaptive weights."""
        valid = X.notna().all(axis=1)
        predictions = pd.Series(np.nan, index=X.index)
        
        if valid.sum() > 0:
            X_scaled = self.scaler.transform(X[valid])
            X_weighted = X_scaled / self.weights
            predictions[valid] = self.lasso.predict(X_weighted)
        
        return predictions
