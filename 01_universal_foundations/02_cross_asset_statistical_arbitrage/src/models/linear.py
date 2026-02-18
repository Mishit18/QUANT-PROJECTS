import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple


class CrossSectionalOLS:
    """Cross-sectional OLS baseline."""
    
    def __init__(self):
        self.model = LinearRegression(fit_intercept=True)
        self.coefs = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit on single cross-section."""
        X_filled = X.fillna(0)
        valid = y.notna()
        if valid.sum() < 10:
            return
        
        self.model.fit(X_filled[valid], y[valid])
        self.coefs.append(self.model.coef_)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict on cross-section."""
        X_filled = X.fillna(0)
        predictions = pd.Series(np.nan, index=X.index)
        predictions[:] = self.model.predict(X_filled)
        return predictions
    
    def get_feature_importance(self) -> pd.Series:
        """Average coefficient magnitudes."""
        if not self.coefs:
            return pd.Series()
        coef_array = np.array(self.coefs)
        return pd.Series(np.abs(coef_array).mean(axis=0))


class RollingOLS:
    """Rolling window OLS for time-series prediction."""
    
    def __init__(self, window: int = 252):
        self.window = window
        self.model = LinearRegression()
        
    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Walk-forward fit and predict."""
        predictions = pd.Series(np.nan, index=y.index)
        
        for i in range(self.window, len(X)):
            X_train = X.iloc[i - self.window:i]
            y_train = y.iloc[i - self.window:i]
            
            valid = X_train.notna().all(axis=1) & y_train.notna()
            if valid.sum() < self.window // 2:
                continue
            
            self.model.fit(X_train[valid], y_train[valid])
            
            X_test = X.iloc[i:i+1]
            if X_test.notna().all(axis=1).iloc[0]:
                predictions.iloc[i] = self.model.predict(X_test)[0]
        
        return predictions
