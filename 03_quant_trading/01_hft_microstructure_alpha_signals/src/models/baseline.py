"""
Baseline logistic regression model.

Provides interpretable linear baseline for comparison with tree models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import joblib


class BaselineModel:
    """
    Logistic regression baseline for HFT alpha prediction.
    
    Advantages:
    - Fast training and prediction
    - Interpretable coefficients
    - Stable across regimes
    
    Limitations:
    - Linear decision boundaries
    - No feature interactions
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        
        self.config = config
        self.model = LogisticRegression(
            C=config['C'],
            max_iter=config['max_iter'],
            class_weight=config['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train logistic regression model.
        
        Args:
            X: Feature matrix
            y: Labels (-1, 0, 1)
        """
        self.feature_names = X.columns.tolist()
        
        # Remove NaN and inf
        X_clean, y_clean = self._clean_data(X, y)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train model
        print(f"Training logistic regression on {len(X_clean)} samples...")
        self.model.fit(X_scaled, y_clean)
        
        print(f"Training complete. Classes: {self.model.classes_}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        X_clean = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_clean = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model coefficients.
        
        Returns DataFrame with feature names and importance scores.
        """
        if self.model.coef_.shape[0] == 1:
            # Binary classification
            importance = np.abs(self.model.coef_[0])
        else:
            # Multi-class: average absolute coefficients
            importance = np.abs(self.model.coef_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove NaN and inf values."""
        # Replace inf with NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN
        valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
        
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"Cleaned data: {len(X_clean)} / {len(X)} samples retained")
        
        return X_clean, y_clean
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction (handle NaN/inf)."""
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)  # simple imputation
        return X_clean
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.config = data['config']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test baseline model
    from src.features.base_features import compute_all_base_features
    from src.labels.future_ticks import create_labels_all_horizons
    
    # Load data
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    # Compute features
    features = compute_all_base_features(df)
    
    # Create labels
    labels = create_labels_all_horizons(df, horizons=[5])
    
    # Train/test split (time-respecting)
    split_idx = int(len(df) * 0.8)
    
    X_train = features.iloc[:split_idx]
    y_train = labels['label_5'].iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = labels['label_5'].iloc[split_idx:]
    
    # Train model
    model = BaselineModel()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test.values).mean()
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Feature importance
    print("\nTop 10 features:")
    print(model.get_feature_importance().head(10))
