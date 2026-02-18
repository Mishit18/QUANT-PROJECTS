"""
Unit tests for rBergomi model implementation.
"""

import pytest
import numpy as np
from src.models.rbergomi import RBergomiModel, RBergomiParameters
from src.simulation.fbm import FractionalBrownianMotion, HybridFBMGenerator
from src.simulation.hybrid_scheme import HybridScheme
from src.pricing.monte_carlo_pricer import MonteCarloOptionPricer, black_scholes_call
from src.pricing.implied_vol import implied_volatility


class TestRBergomiParameters:
    """Test parameter validation."""

    def test_valid_parameters(self):
        params = RBergomiParameters(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
        assert params.H == 0.1
        assert params.eta == 1.5
        assert params.rho == -0.7
        assert params.xi0 == 0.04

    def test_invalid_hurst(self):
        with pytest.raises(ValueError):
            RBergomiParameters(H=0.6, eta=1.5, rho=-0.7, xi0=0.04)

        with pytest.raises(ValueError):
            RBergomiParameters(H=-0.1, eta=1.5, rho=-0.7, xi0=0.04)

    def test_invalid_eta(self):
        with pytest.raises(ValueError):
            RBergomiParameters(H=0.1, eta=-1.0, rho=-0.7, xi0=0.04)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            RBergomiParameters(H=0.1, eta=1.5, rho=-1.5, xi0=0.04)

    def test_invalid_xi0(self):
        with pytest.raises(ValueError):
            RBergomiParameters(H=0.1, eta=1.5, rho=-0.7, xi0=-0.01)


class TestFractionalBrownianMotion:
    """Test fBM simulation."""

    def test_fbm_initialization(self):
        fbm = FractionalBrownianMotion(H=0.1, n_steps=100, T=1.0)
        assert fbm.H == 0.1
        assert fbm.n_steps == 100
        assert fbm.T == 1.0

    def test_fbm_simulation_shape(self):
        fbm = FractionalBrownianMotion(H=0.1, n_steps=100, T=1.0)
        paths = fbm.simulate(n_paths=1000, rng=np.random.default_rng(42))
        assert paths.shape == (1000, 101)
        assert np.all(paths[:, 0] == 0)  # Start at zero

    def test_fbm_mean(self):
        fbm = FractionalBrownianMotion(H=0.1, n_steps=100, T=1.0)
        paths = fbm.simulate(n_paths=5000, rng=np.random.default_rng(42))
        mean_terminal = np.mean(paths[:, -1])
        assert abs(mean_terminal) < 0.1  # Should be close to zero

    def test_hybrid_fbm_generator(self):
        gen = HybridFBMGenerator(H=0.1, n_steps=100, T=1.0, cutoff=20)
        paths = gen.simulate(n_paths=1000, rng=np.random.default_rng(42))
        assert paths.shape == (1000, 101)


class TestRBergomiModel:
    """Test rBergomi model."""

    def test_model_initialization(self):
        model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
        assert model.H == 0.1
        assert model.eta == 1.5
        assert model.rho == -0.7
        assert model.xi0 == 0.04

    def test_variance_process(self):
        model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)

        n_paths = 100
        n_steps = 50
        T = 1.0
        times = np.linspace(0, T, n_steps + 1)

        # Generate fBM
        fbm = FractionalBrownianMotion(H=0.1, n_steps=n_steps, T=T)
        Y = fbm.simulate(n_paths, rng=np.random.default_rng(42))

        # Compute variance
        v = model.variance_process(Y, times)

        assert v.shape == (n_paths, n_steps + 1)
        assert np.all(v > 0)  # Variance must be positive

    def test_parameter_dict(self):
        model = RBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
        params = model.get_parameter_dict()

        assert params['H'] == 0.1
        assert params['eta'] == 1.5
        assert params['rho'] == -0.7
        assert params['xi0'] == 0.04


class TestHybridScheme:
    """Test hybrid simulation scheme."""

    def test_rbergomi_simulation(self):
        scheme = HybridScheme(n_steps=100, n_paths=1000, seed=42)

        S_paths, v_paths = scheme.simulate_rbergomi(
            H=0.1, eta=1.5, rho=-0.7, xi0=0.04, T=1.0, S0=100.0
        )

        assert S_paths.shape == (1000, 101)
        assert v_paths.shape == (1000, 101)
        assert np.all(S_paths[:, 0] == 100.0)
        assert np.all(S_paths > 0)
        assert np.all(v_paths > 0)

    def test_terminal_distribution(self):
        scheme = HybridScheme(n_steps=100, n_paths=5000, seed=42)

        S_paths, v_paths = scheme.simulate_rbergomi(
            H=0.1, eta=1.5, rho=-0.7, xi0=0.04, T=1.0, S0=100.0
        )

        # Check that terminal spot is reasonable
        S_T = S_paths[:, -1]
        assert 50 < np.mean(S_T) < 150
        assert np.std(S_T) > 0


class TestMonteCarloOptionPricer:
    """Test option pricing."""

    def test_call_pricing(self):
        # Generate simple paths
        n_paths = 10000
        S_T = 100 + np.random.randn(n_paths, 1) * 20
        S_paths = np.hstack([np.ones((n_paths, 1)) * 100, S_T])

        pricer = MonteCarloOptionPricer(n_paths, seed=42)
        price, std_err = pricer.price_european_call(S_paths, K=100, r=0.0, T=1.0)

        assert price > 0
        assert std_err > 0
        assert std_err < price  # Standard error should be smaller than price

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K."""
        n_paths = 10000
        S0 = 100
        K = 100
        T = 1.0

        # Generate lognormal paths
        sigma = 0.2
        Z = np.random.randn(n_paths)
        S_T = S0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * Z)
        S_paths = np.hstack([np.ones((n_paths, 1)) * S0, S_T.reshape(-1, 1)])

        pricer = MonteCarloOptionPricer(n_paths, seed=42, use_control_variate=False)
        call_price, _ = pricer.price_european_call(S_paths, K, 0.0, T)
        put_price, _ = pricer.price_european_put(S_paths, K, 0.0, T)

        # Put-call parity (approximately)
        parity_diff = abs((call_price - put_price) - (S0 - K))
        assert parity_diff < 1.0  # Should be close


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def test_atm_call(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.0
        sigma_true = 0.2

        # Compute BS price
        price = black_scholes_call(S, K, T, r, sigma_true)

        # Recover implied vol
        iv = implied_volatility(price, S, K, T, r, 'call')

        assert abs(iv - sigma_true) < 1e-4

    def test_otm_call(self):
        S = 100
        K = 110
        T = 1.0
        r = 0.0
        sigma_true = 0.25

        price = black_scholes_call(S, K, T, r, sigma_true)
        iv = implied_volatility(price, S, K, T, r, 'call')

        assert abs(iv - sigma_true) < 1e-4

    def test_invalid_price(self):
        """Test that invalid prices return NaN."""
        iv = implied_volatility(price=-1.0, S=100, K=100, T=1.0, r=0.0, option_type='call')
        assert np.isnan(iv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
