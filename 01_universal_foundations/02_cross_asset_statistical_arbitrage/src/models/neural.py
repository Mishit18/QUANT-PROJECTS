import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class FeedForwardNN:
    """Simple feed-forward neural network."""
    
    def __init__(self, hidden_layers: tuple = (64, 32), learning_rate: float = 0.001):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit neural network."""
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 100:
            return
        
        X_scaled = self.scaler.fit_transform(X[valid])
        self.model.fit(X_scaled, y[valid])
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with neural network."""
        valid = X.notna().all(axis=1)
        predictions = pd.Series(np.nan, index=X.index)
        
        if valid.sum() > 0:
            X_scaled = self.scaler.transform(X[valid])
            predictions[valid] = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> pd.Series:
        """Approximate importance via input layer weights."""
        if not hasattr(self.model, 'coefs_'):
            return pd.Series()
        
        input_weights = self.model.coefs_[0]
        importance = np.abs(input_weights).mean(axis=1)
        return pd.Series(importance)
