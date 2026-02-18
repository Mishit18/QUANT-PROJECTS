import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def compute_alpha_correlations(alphas: dict) -> pd.DataFrame:
    """Correlation matrix between alpha signals."""
    alpha_df = pd.DataFrame(alphas)
    return alpha_df.corr(method='spearman')


def compute_rolling_correlation(alpha1: pd.Series, alpha2: pd.Series, window: int = 60) -> pd.Series:
    """Rolling correlation between two alphas."""
    return alpha1.rolling(window).corr(alpha2)


def cluster_alphas(alphas: dict, method: str = 'ward') -> np.ndarray:
    """Hierarchical clustering of alpha signals."""
    alpha_df = pd.DataFrame(alphas)
    corr_matrix = alpha_df.corr(method='spearman')
    
    distance_matrix = 1 - corr_matrix.abs()
    condensed_dist = squareform(distance_matrix)
    
    return linkage(condensed_dist, method=method)


def compute_diversification_ratio(alphas: dict) -> float:
    """Portfolio diversification via eigenvalue concentration."""
    alpha_df = pd.DataFrame(alphas)
    corr_matrix = alpha_df.corr()
    
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    return eigenvalues.max() / eigenvalues.sum()


def compute_orthogonalized_alpha(target_alpha: pd.Series, reference_alphas: pd.DataFrame) -> pd.Series:
    """Orthogonalize target alpha against reference set."""
    valid = target_alpha.notna() & reference_alphas.notna().all(axis=1)
    
    if valid.sum() < 50:
        return target_alpha
    
    X = reference_alphas[valid].values
    y = target_alpha[valid].values
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    
    result = target_alpha.copy()
    result[valid] = residual
    
    return result
