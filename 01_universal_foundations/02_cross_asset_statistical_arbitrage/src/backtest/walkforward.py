import pandas as pd
import numpy as np
from typing import Dict, List
from ..models.linear import CrossSectionalOLS
from ..models.regularized import RegularizedModel
from ..models.trees import XGBoostModel
from ..evaluation.ic import compute_ic_series


class WalkForwardValidator:
    """Walk-forward model training and validation."""
    
    def __init__(self, config: dict):
        self.train_window = config['models']['train_window']
        self.test_window = config['models']['test_window']
        self.retrain_freq = config['models']['retrain_freq']
        self.embargo = config['models'].get('embargo', config.get('targets', {}).get('horizon', 0))
        self.min_train_samples = config['models'].get('min_train_samples', 50)

    @staticmethod
    def _coerce_targets(targets: pd.DataFrame) -> pd.Series:
        if isinstance(targets, pd.Series):
            return targets
        if targets.shape[1] == 1:
            return targets.iloc[:, 0]
        if targets.index.nlevels == 1:
            stacked = targets.stack()
            stacked.name = 'target'
            return stacked
        raise ValueError("Multi-column targets must be wide date x asset data.")
        
    def validate(self, features: pd.DataFrame, targets: pd.DataFrame, 
                model_type: str = 'xgboost') -> Dict[str, pd.DataFrame]:
        """Walk-forward validation."""
        
        if model_type == 'ols':
            model = CrossSectionalOLS()
        elif model_type == 'ridge':
            model = RegularizedModel('ridge', alpha=1.0)
        elif model_type == 'xgboost':
            model = XGBoostModel()
        else:
            raise ValueError(f"Unknown model: {model_type}")
        
        predictions = []
        test_ics = []
        
        targets = self._coerce_targets(targets)
        
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx].replace([np.inf, -np.inf], np.nan)
        targets = targets.loc[common_idx]
        
        dates = features.index.get_level_values(0).unique() if features.index.nlevels > 1 else features.index.unique()
        
        for i in range(self.train_window, len(dates), self.retrain_freq):
            train_end = max(0, i - self.embargo)
            train_dates = dates[max(0, train_end - self.train_window):train_end]
            test_dates = dates[i:min(i + self.test_window, len(dates))]

            if len(train_dates) == 0 or len(test_dates) == 0:
                continue
            
            if features.index.nlevels > 1:
                X_train = features.loc[train_dates]
                y_train = targets.loc[train_dates]
                X_test = features.loc[test_dates]
                y_test = targets.loc[test_dates]
            else:
                X_train = features.loc[train_dates]
                y_train = targets.loc[train_dates]
                X_test = features.loc[test_dates]
                y_test = targets.loc[test_dates]
            
            # Require at least 20% of features to be non-NaN
            valid_train = (X_train.notna().sum(axis=1) > X_train.shape[1] * 0.2) & y_train.notna()
            
            if valid_train.sum() < self.min_train_samples:
                continue
            
            model.fit(X_train[valid_train], y_train[valid_train])
            test_pred = model.predict(X_test)
            predictions.append(test_pred)
            
            # Compute IC for each test period
            for test_date in test_dates:
                if test_date in test_pred.index and test_date in y_test.index:
                    pred_slice = test_pred.loc[test_date]
                    target_slice = y_test.loc[test_date]
                    
                    valid = pred_slice.notna() & target_slice.notna()
                    if valid.sum() > 10:
                        ic = pred_slice[valid].corr(target_slice[valid], method='spearman')
                        test_ics.append((test_date, ic))
        
        if not predictions:
            return {
                'predictions': pd.Series(dtype=float),
                'train_ic': pd.Series(dtype=float),
                'test_ic': pd.Series(dtype=float)
            }
        
        all_predictions = pd.concat(predictions)
        test_ic_series = pd.Series(dict(test_ics)) if test_ics else pd.Series(dtype=float)
        
        return {
            'predictions': all_predictions,
            'train_ic': pd.Series(dtype=float),
            'test_ic': test_ic_series
        }
