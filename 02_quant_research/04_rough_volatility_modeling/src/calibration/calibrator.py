"""
Model calibration framework for rBergomi, Heston, and SABR.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass

from ..models.rbergomi import RBergomiModel
from ..models.heston import HestonModel
from ..models.sabr import SABRModel
from ..simulation.hybrid_scheme import HybridScheme
from ..pricing.monte_carlo_pricer import MonteCarloOptionPricer
from ..pricing.implied_vol import implied_volatility


@dataclass
class MarketData:
    """Market data for calibration."""
    spot: float
    strikes: np.ndarray
    maturities: np.ndarray
    implied_vols: np.ndarray  # Shape: (n_maturities, n_strikes)
    option_type: str = 'call'
    risk_free_rate: float = 0.0


class ModelCalibrator:
    """
    Calibrate volatility models to market implied volatility surface.
    """

    def __init__(
        self,
        market_data: MarketData,
        n_paths: int = 5000,
        n_steps: int = 100,
        seed: Optional[int] = 42
    ):
        """
        Args:
            market_data: Market implied volatility data
            n_paths: Monte Carlo paths for pricing
            n_steps: Time steps for simulation
            seed: Random seed
        """
        self.market_data = market_data
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def calibrate_rbergomi(
        self,
        initial_params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'differential_evolution'
    ) -> Tuple[RBergomiModel, Dict]:
        """
        Calibrate rBergomi model to market data.

        Args:
            initial_params: Initial parameter guess {'H', 'eta', 'rho', 'xi0'}
            bounds: Parameter bounds
            method: 'differential_evolution' or 'nelder-mead'

        Returns:
            (calibrated_model, diagnostics)
        """
        if initial_params is None:
            initial_params = {'H': 0.1, 'eta': 1.5, 'rho': -0.7, 'xi0': 0.04}

        if bounds is None:
            bounds = {
                'H': (0.01, 0.49),
                'eta': (0.1, 3.0),
                'rho': (-0.99, 0.99),
                'xi0': (0.001, 0.5)
            }

        def objective(params):
            H, eta, rho, xi0 = params

            # Parameter validation
            if not (0 < H < 0.5):
                return 1e10
            if eta <= 0 or xi0 <= 0:
                return 1e10
            if abs(rho) >= 1:
                return 1e10

            try:
                model = RBergomiModel(H, eta, rho, xi0)
                loss = self._compute_calibration_loss(model, 'rbergomi')

                # Regularization to avoid extreme parameters
                reg = 0.01 * (eta**2 + 100 * (H - 0.1)**2)

                return loss + reg
            except Exception:
                return 1e10

        # Optimization
        x0 = [initial_params['H'], initial_params['eta'],
              initial_params['rho'], initial_params['xi0']]
        bounds_list = [bounds['H'], bounds['eta'], bounds['rho'], bounds['xi0']]

        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds_list,
                seed=self.seed,
                maxiter=50,
                popsize=10,
                atol=1e-3,
                tol=1e-3
            )
        else:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4}
            )

        # Extract optimal parameters
        H_opt, eta_opt, rho_opt, xi0_opt = result.x
        calibrated_model = RBergomiModel(H_opt, eta_opt, rho_opt, xi0_opt)

        diagnostics = {
            'success': result.success,
            'loss': result.fun,
            'n_iterations': result.nit if hasattr(result, 'nit') else None,
            'parameters': calibrated_model.get_parameter_dict()
        }

        return calibrated_model, diagnostics

    def calibrate_heston(
        self,
        initial_params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[HestonModel, Dict]:
        """
        Calibrate Heston model to market data.

        Args:
            initial_params: Initial guess {'kappa', 'theta', 'sigma', 'rho', 'v0'}
            bounds: Parameter bounds

        Returns:
            (calibrated_model, diagnostics)
        """
        if initial_params is None:
            initial_params = {
                'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3,
                'rho': -0.7, 'v0': 0.04
            }

        if bounds is None:
            bounds = {
                'kappa': (0.1, 10.0),
                'theta': (0.001, 0.5),
                'sigma': (0.01, 2.0),
                'rho': (-0.99, 0.99),
                'v0': (0.001, 0.5)
            }

        def objective(params):
            kappa, theta, sigma, rho, v0 = params

            # Feller condition check
            if 2 * kappa * theta < sigma**2:
                return 1e10

            try:
                model = HestonModel(kappa, theta, sigma, rho, v0)
                loss = self._compute_calibration_loss(model, 'heston')
                return loss
            except Exception:
                return 1e10

        x0 = [initial_params['kappa'], initial_params['theta'],
              initial_params['sigma'], initial_params['rho'], initial_params['v0']]
        bounds_list = [bounds['kappa'], bounds['theta'], bounds['sigma'],
                      bounds['rho'], bounds['v0']]

        result = differential_evolution(
            objective,
            bounds_list,
            seed=self.seed,
            maxiter=50,
            popsize=10
        )

        kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt = result.x
        calibrated_model = HestonModel(kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt)

        diagnostics = {
            'success': result.success,
            'loss': result.fun,
            'parameters': calibrated_model.get_parameter_dict()
        }

        return calibrated_model, diagnostics

    def calibrate_sabr(
        self,
        beta: float = 0.7,
        maturity_idx: int = 0
    ) -> Tuple[SABRModel, Dict]:
        """
        Calibrate SABR model to a single maturity slice.

        Args:
            beta: Fixed CEV exponent
            maturity_idx: Index of maturity to calibrate

        Returns:
            (calibrated_model, diagnostics)
        """
        T = self.market_data.maturities[maturity_idx]
        strikes = self.market_data.strikes
        market_vols = self.market_data.implied_vols[maturity_idx]

        # Remove NaN values
        valid_mask = ~np.isnan(market_vols)
        strikes = strikes[valid_mask]
        market_vols = market_vols[valid_mask]

        def objective(params):
            alpha, rho, nu = params

            if alpha <= 0 or nu < 0 or abs(rho) >= 1:
                return 1e10

            try:
                model = SABRModel(alpha, beta, rho, nu)
                model_vols = np.array([
                    model.implied_volatility_hagan(self.market_data.spot, K, T)
                    for K in strikes
                ])

                return np.sum((model_vols - market_vols)**2)
            except Exception:
                return 1e10

        # Initial guess
        x0 = [0.2, -0.3, 0.3]
        bounds = [(1e-4, 2.0), (-0.99, 0.99), (1e-4, 2.0)]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        alpha_opt, rho_opt, nu_opt = result.x
        calibrated_model = SABRModel(alpha_opt, beta, rho_opt, nu_opt)

        diagnostics = {
            'success': result.success,
            'loss': result.fun,
            'maturity': T,
            'parameters': calibrated_model.get_parameter_dict()
        }

        return calibrated_model, diagnostics

    def _compute_calibration_loss(
        self,
        model,
        model_type: str
    ) -> float:
        """
        Compute calibration loss (RMSE of implied volatilities).

        Args:
            model: Model instance (rBergomi or Heston)
            model_type: 'rbergomi' or 'heston'

        Returns:
            Root mean squared error
        """
        total_error = 0.0
        n_points = 0

        for i, T in enumerate(self.market_data.maturities):
            for j, K in enumerate(self.market_data.strikes):
                market_iv = self.market_data.implied_vols[i, j]

                if np.isnan(market_iv):
                    continue

                # Simulate and price
                try:
                    if model_type == 'rbergomi':
                        scheme = HybridScheme(self.n_steps, self.n_paths, self.seed)
                        S_paths, _ = scheme.simulate_rbergomi(
                            model.H, model.eta, model.rho, model.xi0,
                            T, self.market_data.spot
                        )
                    elif model_type == 'heston':
                        S_paths, _ = model.simulate(
                            self.market_data.spot, T, self.n_steps,
                            self.n_paths, scheme='euler'
                        )
                    else:
                        continue

                    # Price option
                    pricer = MonteCarloOptionPricer(self.n_paths, self.seed)
                    price, _ = pricer.price_european_call(
                        S_paths, K, self.market_data.risk_free_rate, T
                    )

                    # Compute implied vol
                    model_iv = implied_volatility(
                        price, self.market_data.spot, K, T,
                        self.market_data.risk_free_rate, 'call'
                    )

                    if not np.isnan(model_iv):
                        total_error += (model_iv - market_iv)**2
                        n_points += 1

                except Exception:
                    continue

        if n_points == 0:
            return 1e10

        return np.sqrt(total_error / n_points)

    def compare_models(
        self,
        rbergomi_model: RBergomiModel,
        heston_model: HestonModel,
        sabr_model: SABRModel
    ) -> Dict:
        """
        Compare calibration quality across models.

        Returns:
            Dictionary with comparison metrics
        """
        results = {
            'rbergomi': self._compute_calibration_loss(rbergomi_model, 'rbergomi'),
            'heston': self._compute_calibration_loss(heston_model, 'heston'),
            'sabr': None  # SABR uses analytical formula
        }

        return results
