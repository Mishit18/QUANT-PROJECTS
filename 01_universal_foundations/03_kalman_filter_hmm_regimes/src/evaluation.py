"""
Model evaluation and diagnostic tools.

Provides comprehensive evaluation metrics for:
- Kalman filter performance
- HMM regime detection quality
- Trading strategy performance
- Model comparison and selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from src.utils import sharpe_ratio, sortino_ratio, max_drawdown, information_ratio


class KalmanFilterEvaluator:
    """
    Evaluation metrics for Kalman filter performance.
    """
    
    @staticmethod
    def innovation_diagnostics(innovations: np.ndarray,
                               innovation_cov: np.ndarray) -> Dict:
        """
        Test innovation sequence for whiteness and normality.
        
        Parameters
        ----------
        innovations : np.ndarray
            Innovation sequence
        innovation_cov : np.ndarray
            Innovation covariances
            
        Returns
        -------
        dict
            Diagnostic statistics
        """
        # Standardize innovations
        std_innovations = np.zeros_like(innovations)
        for t in range(len(innovations)):
            if innovation_cov[t].ndim == 2:
                std = np.sqrt(np.diag(innovation_cov[t]))
            else:
                std = np.sqrt(innovation_cov[t])
            std_innovations[t] = innovations[t] / (std + 1e-10)
        
        # Normality test (Jarque-Bera)
        jb_stat, jb_pval = stats.jarque_bera(std_innovations.flatten())
        
        # Autocorrelation test (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(std_innovations.flatten(), lags=20, return_df=True)
        
        # Mean and variance
        mean = np.mean(std_innovations)
        variance = np.var(std_innovations)
        
        return {
            'mean': mean,
            'variance': variance,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'ljung_box_stats': lb_result['lb_stat'].values,
            'ljung_box_pvals': lb_result['lb_pvalue'].values,
            'is_white_noise': (jb_pval > 0.05) and (lb_result['lb_pvalue'].iloc[-1] > 0.05)
        }
    
    @staticmethod
    def forecast_accuracy(predictions: np.ndarray,
                         actuals: np.ndarray) -> Dict:
        """
        Calculate forecast accuracy metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        actuals : np.ndarray
            Actual values
            
        Returns
        -------
        dict
            Accuracy metrics
        """
        errors = actuals - predictions
        
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors / (actuals + 1e-10))) * 100
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'r_squared': r_squared
        }
    
    @staticmethod
    def state_estimation_quality(filtered_states: np.ndarray,
                                 smoothed_states: np.ndarray,
                                 true_states: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate state estimation quality.
        
        Parameters
        ----------
        filtered_states : np.ndarray
            Filtered state estimates
        smoothed_states : np.ndarray
            Smoothed state estimates
        true_states : np.ndarray, optional
            True states (if available)
            
        Returns
        -------
        dict
            Quality metrics
        """
        metrics = {}
        
        # Smoothing gain (reduction in uncertainty)
        filter_smooth_diff = np.mean((filtered_states - smoothed_states) ** 2)
        metrics['smoothing_gain'] = filter_smooth_diff
        
        if true_states is not None:
            # Filtering error
            filter_error = np.mean((filtered_states - true_states) ** 2)
            metrics['filter_mse'] = filter_error
            
            # Smoothing error
            smooth_error = np.mean((smoothed_states - true_states) ** 2)
            metrics['smooth_mse'] = smooth_error
            
            # Correlation with true states
            if filtered_states.ndim == 1:
                metrics['filter_correlation'] = np.corrcoef(filtered_states, true_states)[0, 1]
                metrics['smooth_correlation'] = np.corrcoef(smoothed_states, true_states)[0, 1]
            else:
                metrics['filter_correlation'] = np.corrcoef(filtered_states[:, 0], true_states[:, 0])[0, 1]
                metrics['smooth_correlation'] = np.corrcoef(smoothed_states[:, 0], true_states[:, 0])[0, 1]
        
        return metrics


class HMMEvaluator:
    """
    Evaluation metrics for HMM regime detection.
    """
    
    @staticmethod
    def regime_persistence(transition_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate expected regime duration.
        
        Parameters
        ----------
        transition_matrix : np.ndarray
            Transition probability matrix
            
        Returns
        -------
        np.ndarray
            Expected duration for each regime
        """
        persistence = np.diag(transition_matrix)
        expected_duration = 1 / (1 - persistence + 1e-10)
        return expected_duration
    
    @staticmethod
    def regime_separation(means: np.ndarray,
                         covariances: np.ndarray) -> float:
        """
        Measure separation between regimes (Mahalanobis distance).
        
        Parameters
        ----------
        means : np.ndarray
            Regime means
        covariances : np.ndarray
            Regime covariances
            
        Returns
        -------
        float
            Average pairwise Mahalanobis distance
        """
        n_regimes = len(means)
        distances = []
        
        for i in range(n_regimes):
            for j in range(i + 1, n_regimes):
                diff = means[i] - means[j]
                avg_cov = (covariances[i] + covariances[j]) / 2
                
                try:
                    mahal = np.sqrt(diff.T @ np.linalg.inv(avg_cov) @ diff)
                    distances.append(mahal)
                except:
                    pass
        
        return np.mean(distances) if distances else 0.0
    
    @staticmethod
    def regime_classification_quality(regime_probs: np.ndarray) -> Dict:
        """
        Evaluate regime classification quality.
        
        Parameters
        ----------
        regime_probs : np.ndarray
            Regime probabilities
            
        Returns
        -------
        dict
            Quality metrics
        """
        # Entropy (uncertainty)
        epsilon = 1e-10
        entropy = -np.sum(regime_probs * np.log(regime_probs + epsilon), axis=1)
        
        # Maximum probability (confidence)
        max_prob = np.max(regime_probs, axis=1)
        
        # Regime switches
        dominant_regime = np.argmax(regime_probs, axis=1)
        switches = np.sum(np.diff(dominant_regime) != 0)
        
        return {
            'mean_entropy': np.mean(entropy),
            'mean_confidence': np.mean(max_prob),
            'high_confidence_pct': np.mean(max_prob > 0.7),
            'n_switches': switches,
            'switch_frequency': switches / len(regime_probs)
        }
    
    @staticmethod
    def compare_with_true_regimes(predicted_regimes: np.ndarray,
                                 true_regimes: np.ndarray) -> Dict:
        """
        Compare predicted regimes with true regimes (if available).
        
        Parameters
        ----------
        predicted_regimes : np.ndarray
            Predicted regime labels
        true_regimes : np.ndarray
            True regime labels
            
        Returns
        -------
        dict
            Comparison metrics
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Adjusted Rand Index
        ari = adjusted_rand_score(true_regimes, predicted_regimes)
        
        # Normalized Mutual Information
        nmi = normalized_mutual_info_score(true_regimes, predicted_regimes)
        
        # Accuracy (with optimal label permutation)
        from scipy.optimize import linear_sum_assignment
        n_regimes = len(np.unique(true_regimes))
        confusion = np.zeros((n_regimes, n_regimes))
        
        for i in range(n_regimes):
            for j in range(n_regimes):
                confusion[i, j] = np.sum((true_regimes == i) & (predicted_regimes == j))
        
        row_ind, col_ind = linear_sum_assignment(-confusion)
        accuracy = confusion[row_ind, col_ind].sum() / len(true_regimes)
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'accuracy': accuracy
        }


class StrategyEvaluator:
    """
    Comprehensive strategy evaluation.
    """
    
    @staticmethod
    def performance_metrics(returns: np.ndarray,
                           benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray, optional
            Benchmark returns
            
        Returns
        -------
        dict
            Performance metrics
        """
        metrics = {}
        
        # Return statistics
        metrics['mean_return'] = np.mean(returns) * 252
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = sharpe_ratio(returns)
        metrics['sortino_ratio'] = sortino_ratio(returns)
        
        # Drawdown
        max_dd, dd_start, dd_end = max_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['calmar_ratio'] = metrics['mean_return'] / (max_dd + 1e-10)
        
        # Win statistics
        metrics['win_rate'] = np.mean(returns > 0)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        metrics['avg_win'] = np.mean(wins) if len(wins) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) if len(losses) > 0 else 0
        metrics['profit_factor'] = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else np.inf
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['information_ratio'] = information_ratio(returns, benchmark_returns)
            metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            metrics['alpha'] = metrics['mean_return'] - metrics['beta'] * np.mean(benchmark_returns) * 252
        
        return metrics
    
    @staticmethod
    def rolling_performance(returns: np.ndarray,
                           window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        window : int
            Rolling window size
            
        Returns
        -------
        pd.DataFrame
            Rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=range(len(returns)))
        
        returns_series = pd.Series(returns)
        
        # Rolling Sharpe
        rolling_mean = returns_series.rolling(window).mean()
        rolling_std = returns_series.rolling(window).std()
        rolling_metrics['sharpe'] = (rolling_mean / rolling_std) * np.sqrt(252)
        
        # Rolling volatility
        rolling_metrics['volatility'] = rolling_std * np.sqrt(252)
        
        # Rolling drawdown
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        rolling_metrics['drawdown'] = (cumulative - rolling_max) / rolling_max
        
        return rolling_metrics
    
    @staticmethod
    def tail_risk_metrics(returns: np.ndarray,
                         confidence_level: float = 0.95) -> Dict:
        """
        Calculate tail risk metrics (VaR, CVaR).
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        confidence_level : float
            Confidence level for VaR/CVaR
            
        Returns
        -------
        dict
            Tail risk metrics
        """
        # Value at Risk
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar = np.mean(returns[returns <= var])
        
        # Tail ratio (right tail / left tail)
        right_tail = np.percentile(returns, 95)
        left_tail = np.percentile(returns, 5)
        tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else np.inf
        
        return {
            'var_95': var,
            'cvar_95': cvar,
            'tail_ratio': tail_ratio
        }


def model_comparison_table(models: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table for multiple models.
    
    Parameters
    ----------
    models : dict
        Dictionary of model_name -> metrics_dict
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    df = pd.DataFrame(models).T
    return df


if __name__ == '__main__':
    # Test evaluation
    from src.data_loader import generate_synthetic_data
    from src.state_space_models import LocalLevelModel
    from src.kalman_filter import KalmanFilter
    
    print("Testing evaluation metrics...")
    
    # Generate data
    data = generate_synthetic_data(n_samples=500, seed=42)
    returns = data['returns'].iloc[:, 0].values
    
    # Fit Kalman filter
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    kf = KalmanFilter(model)
    filtered, smoothed = kf.filter_and_smooth(returns)
    
    # Innovation diagnostics
    innovations, innovation_cov = kf.get_innovations()
    diagnostics = KalmanFilterEvaluator.innovation_diagnostics(innovations, innovation_cov)
    
    print("\nInnovation Diagnostics:")
    print(f"Mean: {diagnostics['mean']:.4f}")
    print(f"Variance: {diagnostics['variance']:.4f}")
    print(f"Jarque-Bera p-value: {diagnostics['jarque_bera_pval']:.4f}")
    print(f"Is white noise: {diagnostics['is_white_noise']}")
    
    # Strategy performance
    synthetic_returns = np.random.randn(500) * 0.01 + 0.0005
    perf = StrategyEvaluator.performance_metrics(synthetic_returns)
    
    print("\nStrategy Performance:")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"Win Rate: {perf['win_rate']:.2%}")
