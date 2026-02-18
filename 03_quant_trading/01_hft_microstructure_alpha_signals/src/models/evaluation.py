"""
Model evaluation metrics and utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Multi-class metrics (macro average)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    for label in np.unique(y_true):
        label_mask = (y_true == label)
        if label_mask.sum() > 0:
            metrics[f'precision_class_{label}'] = precision_score(
                y_true == label, y_pred == label, zero_division=0
            )
            metrics[f'recall_class_{label}'] = recall_score(
                y_true == label, y_pred == label, zero_division=0
            )
    
    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        try:
            # For multi-class, use OvR strategy
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
        except:
            metrics['roc_auc'] = np.nan
    
    return metrics


def compute_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute hit rate (accuracy for directional prediction).
    
    Ignores flat predictions (label = 0).
    """
    # Only consider non-flat predictions
    non_flat_mask = (y_true != 0) & (y_pred != 0)
    
    if non_flat_mask.sum() == 0:
        return np.nan
    
    hit_rate = (y_true[non_flat_mask] == y_pred[non_flat_mask]).mean()
    
    return hit_rate


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: str = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_evaluation_report(metrics: Dict[str, float], title: str = "Evaluation"):
    """Print formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric:30s}: {value:.4f}")


def evaluate_model_all_horizons(model: Any, features: pd.DataFrame, 
                                labels: pd.DataFrame, 
                                horizons: list) -> pd.DataFrame:
    """
    Evaluate model across multiple prediction horizons.
    
    Returns DataFrame with metrics for each horizon.
    """
    results = []
    
    for horizon in horizons:
        label_col = f'label_{horizon}'
        
        if label_col not in labels.columns:
            continue
        
        # Remove NaN labels
        valid_mask = labels[label_col].notna()
        X = features[valid_mask]
        y = labels[label_col][valid_mask]
        
        # Predict
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Evaluate
        metrics = evaluate_classification(y.values, y_pred, y_proba)
        metrics['horizon'] = horizon
        metrics['hit_rate'] = compute_hit_rate(y.values, y_pred)
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    return results_df


if __name__ == "__main__":
    # Test evaluation
    np.random.seed(42)
    
    # Simulate predictions (0=down, 1=flat, 2=up)
    y_true = np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.4, 0.3])
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(len(y_pred), size=200, replace=False)
    y_pred[noise_idx] = np.random.choice([0, 1, 2], size=200)
    
    # Evaluate
    metrics = evaluate_classification(y_true, y_pred)
    print_evaluation_report(metrics, "Test Evaluation")
    
    # Hit rate
    hit_rate = compute_hit_rate(y_true, y_pred)
    print(f"\nHit rate: {hit_rate:.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
