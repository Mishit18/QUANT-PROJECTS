"""
Unit tests for Kalman filter and HMM implementations.

Tests cover:
- State-space model specifications
- Kalman filter correctness
- HMM parameter estimation
- Signal generation
- Backtesting logic
"""

import unittest
import numpy as np
import sys
sys.path.append('..')

from src.state_space_models import LocalLevelModel, DynamicRegressionModel
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.signals import KalmanTrendSignal, RegimeAwareSignal
from src.backtest import Backtest
from src.data_loader import generate_synthetic_data


class TestStateSpaceModels(unittest.TestCase):
    """Test state-space model specifications."""
    
    def test_local_level_model(self):
        """Test local level model matrices."""
        model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
        F, H, Q, R = model.get_matrices(0)
        
        self.assertEqual(F.shape, (1, 1))
        self.assertEqual(H.shape, (1, 1))
        self.assertEqual(Q.shape, (1, 1))
        self.assertEqual(R.shape, (1, 1))
        
        self.assertAlmostEqual(F[0, 0], 1.0)
        self.assertAlmostEqual(H[0, 0], 1.0)
    
    def test_dynamic_regression_model(self):
        """Test dynamic regression model."""
        model = DynamicRegressionModel(n_regressors=2)
        X = np.random.randn(100, 2)
        model.set_regressors(X)
        
        F, H, Q, R = model.get_matrices(0)
        
        self.assertEqual(F.shape, (2, 2))
        self.assertEqual(H.shape, (1, 2))
        self.assertTrue(np.allclose(F, np.eye(2)))


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman filter implementation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 200
        self.true_state = np.cumsum(np.random.randn(self.n) * 0.1)
        self.observations = self.true_state + np.random.randn(self.n) * 0.5
    
    def test_filter_dimensions(self):
        """Test output dimensions."""
        model = LocalLevelModel(observation_variance=0.25, state_variance=0.01)
        kf = KalmanFilter(model)
        
        filtered, smoothed = kf.filter_and_smooth(self.observations)
        
        self.assertEqual(filtered.shape, (self.n, 1))
        self.assertEqual(smoothed.shape, (self.n, 1))
    
    def test_filter_accuracy(self):
        """Test filtering accuracy."""
        model = LocalLevelModel(observation_variance=0.25, state_variance=0.01)
        kf = KalmanFilter(model)
        
        filtered, _ = kf.filter_and_smooth(self.observations)
        
        # Filtered state should be closer to true state than observations
        filter_error = np.mean((filtered.flatten() - self.true_state) ** 2)
        obs_error = np.mean((self.observations - self.true_state) ** 2)
        
        self.assertLess(filter_error, obs_error)
    
    def test_smoothing_improvement(self):
        """Test that smoothing improves over filtering."""
        model = LocalLevelModel(observation_variance=0.25, state_variance=0.01)
        kf = KalmanFilter(model)
        
        filtered, smoothed = kf.filter_and_smooth(self.observations)
        
        filter_error = np.mean((filtered.flatten() - self.true_state) ** 2)
        smooth_error = np.mean((smoothed.flatten() - self.true_state) ** 2)
        
        self.assertLessEqual(smooth_error, filter_error)
    
    def test_log_likelihood(self):
        """Test log-likelihood calculation."""
        model = LocalLevelModel(observation_variance=0.25, state_variance=0.01)
        kf = KalmanFilter(model)
        
        kf.filter(self.observations)
        ll = kf.get_log_likelihood()
        
        self.assertIsInstance(ll, float)
        self.assertLess(ll, 0)  # Log-likelihood should be negative


class TestHMM(unittest.TestCase):
    """Test HMM implementation."""
    
    def setUp(self):
        """Set up test data."""
        data = generate_synthetic_data(n_samples=500, seed=42)
        self.returns = data['returns'].iloc[:, 0].values
        self.true_regimes = data['regimes']
    
    def test_hmm_fitting(self):
        """Test HMM parameter estimation."""
        hmm = GaussianHMM(n_regimes=3, n_iter=50, random_state=42)
        hmm.fit(self.returns)
        
        self.assertTrue(hmm.is_fitted)
        self.assertEqual(hmm.transition_matrix.shape, (3, 3))
        self.assertEqual(hmm.means.shape, (3, 1))
        self.assertEqual(hmm.covariances.shape, (3, 1, 1))
    
    def test_regime_probabilities(self):
        """Test regime probability inference."""
        hmm = GaussianHMM(n_regimes=3, random_state=42)
        hmm.fit(self.returns)
        
        probs = hmm.predict_proba(self.returns)
        
        self.assertEqual(probs.shape, (len(self.returns), 3))
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
    
    def test_viterbi_path(self):
        """Test Viterbi algorithm."""
        hmm = GaussianHMM(n_regimes=3, random_state=42)
        hmm.fit(self.returns)
        
        regimes = hmm.predict(self.returns)
        
        self.assertEqual(len(regimes), len(self.returns))
        self.assertTrue(np.all(regimes >= 0))
        self.assertTrue(np.all(regimes < 3))
    
    def test_transition_matrix_validity(self):
        """Test transition matrix properties."""
        hmm = GaussianHMM(n_regimes=3, random_state=42)
        hmm.fit(self.returns)
        
        A = hmm.transition_matrix
        
        # Each row should sum to 1
        self.assertTrue(np.allclose(A.sum(axis=1), 1.0))
        
        # All probabilities should be non-negative
        self.assertTrue(np.all(A >= 0))


class TestSignals(unittest.TestCase):
    """Test signal generation."""
    
    def setUp(self):
        """Set up test data."""
        data = generate_synthetic_data(n_samples=500, seed=42)
        self.returns = data['returns'].iloc[:, 0].values
        
        # Fit models
        model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
        self.kf = KalmanFilter(model)
        self.kf.filter(self.returns)
        
        self.hmm = GaussianHMM(n_regimes=3, random_state=42)
        self.hmm.fit(self.returns)
    
    def test_kalman_trend_signal(self):
        """Test Kalman trend signal generation."""
        signal_gen = KalmanTrendSignal(self.kf)
        signals = signal_gen.generate(self.returns)
        
        self.assertEqual(len(signals), len(self.returns))
        self.assertTrue(np.all(signals >= -1))
        self.assertTrue(np.all(signals <= 1))
    
    def test_regime_aware_signal(self):
        """Test regime-aware signal generation."""
        signal_gen = RegimeAwareSignal(self.kf, self.hmm)
        signals = signal_gen.generate(self.returns)
        
        self.assertEqual(len(signals), len(self.returns))
        self.assertTrue(np.all(signals >= -1))
        self.assertTrue(np.all(signals <= 1))


class TestBacktest(unittest.TestCase):
    """Test backtesting engine."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = np.random.randn(500) * 0.01
        self.signals = np.random.choice([-1, 0, 1], size=500)
    
    def test_backtest_execution(self):
        """Test backtest runs without errors."""
        bt = Backtest(self.signals, self.returns, transaction_cost=0.0005)
        results = bt.run()
        
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
    
    def test_transaction_costs(self):
        """Test transaction cost calculation."""
        bt = Backtest(self.signals, self.returns, transaction_cost=0.001)
        results = bt.run()
        
        self.assertGreater(results['total_costs'], 0)
    
    def test_no_lookahead_bias(self):
        """Test that positions are lagged."""
        signals = np.ones(len(self.returns))
        bt = Backtest(signals, self.returns)
        bt.run()
        
        # First position should be zero (no lookahead)
        self.assertEqual(bt.positions[0], 0)
    
    def test_equity_curve_monotonicity(self):
        """Test equity curve properties."""
        bt = Backtest(self.signals, self.returns)
        bt.run()
        
        equity = bt.get_equity_curve().values
        
        # Equity should start at initial capital
        self.assertAlmostEqual(equity[0], bt.initial_capital)
        
        # Equity should be positive
        self.assertTrue(np.all(equity > 0))


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete workflow."""
        # Generate data
        data = generate_synthetic_data(n_samples=500, seed=42)
        returns = data['returns'].iloc[:, 0].values
        
        # Fit Kalman filter
        model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
        kf = KalmanFilter(model)
        kf.filter(returns)
        
        # Fit HMM
        hmm = GaussianHMM(n_regimes=3, random_state=42)
        hmm.fit(returns)
        
        # Generate signals
        signal_gen = RegimeAwareSignal(kf, hmm)
        signals = signal_gen.generate(returns)
        
        # Backtest
        bt = Backtest(signals, returns, transaction_cost=0.0005)
        results = bt.run()
        
        # Verify results
        self.assertIsInstance(results['sharpe_ratio'], float)
        self.assertGreater(results['sharpe_ratio'], -5)  # Sanity check
        self.assertLess(results['sharpe_ratio'], 10)


if __name__ == '__main__':
    unittest.main()
