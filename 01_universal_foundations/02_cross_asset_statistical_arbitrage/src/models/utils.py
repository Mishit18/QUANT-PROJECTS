import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Time series cross-validation splits."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))


def compute_oos_predictions(model, X: pd.DataFrame, y: pd.Series, 
                           train_window: int, test_window: int) -> pd.DataFrame:
    """Walk-forward out-of-sample predictions."""
    predictions = []
    
    for i in range(train_window, len(X), test_window):
        X_train = X.iloc[max(0, i - train_window):i]
        y_train = y.iloc[max(0, i - train_window):i]
        
        model.fit(X_train, y_train)
        
        X_test = X.iloc[i:min(i + test_window, len(X))]
        pred = model.predict(X_test)
        predictions.append(pred)
    
    return pd.concat(predictions)


def ensemble_predictions(predictions: Dict[str, pd.Series], weights: Dict[str, float] = None) -> pd.Series:
    """Weighted ensemble of model predictions."""
    if weights is None:
        weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
    
    ensemble = pd.Series(0.0, index=predictions[list(predictions.keys())[0]].index)
    
    for name, pred in predictions.items():
        ensemble += weights[name] * pred
    
    return ensemble


def rank_ensemble(predictions: Dict[str, pd.Series]) -> pd.Series:
    """Ensemble via average rank."""
    ranked = {}
    for name, pred in predictions.items():
        ranked[name] = pred.rank(pct=True)
    
    return pd.DataFrame(ranked).mean(axis=1)


def detect_overfitting(train_ic: pd.Series, test_ic: pd.Series, threshold: float = 0.5) -> bool:
    """Check for overfitting via IC degradation."""
    train_mean = train_ic.mean()
    test_mean = test_ic.mean()
    
    if train_mean <= 0:
        return False
    
    degradation = (train_mean - test_mean) / train_mean
    return degradation > threshold
