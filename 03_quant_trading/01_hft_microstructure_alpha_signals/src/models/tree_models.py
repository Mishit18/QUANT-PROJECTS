"""
Tree-based models (XGBoost) for HFT alpha prediction.

Tree models capture non-linear relationships and feature interactions
that are common in microstructure data.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple
import joblib


class XGBoostModel:
    """
    XGBoost classifier for HFT alpha prediction.
    
    Advantages:
    - Captures non-linear patterns
    - Handles feature interactions
    - Robust to outliers
    - Built-in regularization
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
        
        self.config = config
        self.model = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            min_child_weight=config['min_child_weight'],
            gamma=config['gamma'],
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Train XGBoost model with optional early stopping.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        self.feature_names = X.columns.tolist()
        
        # Clean data
        X_clean, y_clean = self._clean_data(X, y)
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_clean, y_val_clean = self._clean_data(X_val, y_val)
            eval_set = [(X_val_clean, y_val_clean)]
        
        # Train model
        print(f"Training XGBoost on {len(X_clean)} samples...")
        self.model.fit(
            X_clean, y_clean,
            eval_set=eval_set,
            verbose=False
        )
        
        print(f"Training complete. Classes: {self.model.classes_}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        X_clean = self._prepare_features(X)
        return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_clean = self._prepare_features(X)
        return self.model.predict_proba(X_clean)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from XGBoost.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            
        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove NaN and inf values."""
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN
        valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
        
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"Cleaned data: {len(X_clean)} / {len(X)} samples retained")
        
        return X_clean, y_clean
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)
        return X_clean
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.config = data['config']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from src.features.base_features import compute_all_base_features
    from src.features.ofi import compute_ofi_features
    from src.labels.future_ticks import create_labels_all_horizons
    
    # Load data
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    # Compute features
    base_features = compute_all_base_features(df)
    ofi_features = compute_ofi_features(df)
    features = pd.concat([base_features, ofi_features], axis=1)
    
    # Create labels
    labels = create_labels_all_horizons(df, horizons=[5])
    
    # Train/val/test split
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train = features.iloc[:train_end]
    y_train = labels['label_5'].iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    y_val = labels['label_5'].iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    y_test = labels['label_5'].iloc[val_end:]
    
    # Train model
    model = XGBoostModel()
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test.values).mean()
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Feature importance
    print("\nTop 10 features:")
    print(model.get_feature_importance().head(10))
