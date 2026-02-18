import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Optional, Dict


class XGBoostModel:
    """XGBoost for cross-sectional prediction."""
    
    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 100,
            'random_state': 42
        }
        self.params = params if params else default_params
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit XGBoost model."""
        X_filled = X.fillna(0)
        valid = y.notna()
        if valid.sum() < 50:
            return
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_filled[valid], y[valid], verbose=False)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with XGBoost."""
        if self.model is None:
            return pd.Series(np.nan, index=X.index)
        
        X_filled = X.fillna(0)
        predictions = pd.Series(np.nan, index=X.index)
        predictions[:] = self.model.predict(X_filled)
        return predictions
    
    def get_feature_importance(self) -> pd.Series:
        """Feature importance from gain."""
        if self.model is None:
            return pd.Series()
        importance = self.model.feature_importances_
        return pd.Series(importance)


class RankXGBoost:
    """XGBoost with ranking objective."""
    
    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'objective': 'rank:pairwise',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'n_estimators': 100,
            'random_state': 42
        }
        self.params = params if params else default_params
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit with ranking objective."""
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 50:
            return
        
        group = [valid.sum()]
        
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X[valid], y[valid], group=group, verbose=False)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict rankings."""
        if self.model is None:
            return pd.Series(np.nan, index=X.index)
        
        valid = X.notna().all(axis=1)
        predictions = pd.Series(np.nan, index=X.index)
        
        if valid.sum() > 0:
            predictions[valid] = self.model.predict(X[valid])
        
        return predictions
