"""
Microstructure Diagnostics

Diagnostic tools for understanding market microstructure effects.
No optimization - purely diagnostic.
"""

import numpy as np
from typing import Dict, List, Tuple


class MicrostructureDiagnostics:
    """
    Diagnostic tools for microstructure analysis.
    
    Focuses on:
    - Mid-price vs microprice comparison
    - Queue position effects
    - Fill probability analysis
    - Adverse selection measurement
    
    All methods are diagnostic, not prescriptive.
    """
    
    @staticmethod
    def compute_effective_spread(
        execution_price: float,
        mid_price: float,
        side: str
    ) -> float:
        """
        Compute effective spread.
        
        Effective spread = 2 * |execution_price - mid_price|
        
        Args:
            execution_price: Actual execution price
            mid_price: Mid-price at execution
            side: 'buy' or 'sell'
        
        Returns:
            Effective spread
        """
        return 2 * abs(execution_price - mid_price)
    
    @staticmethod
    def compute_realized_spread(
        execution_price: float,
        mid_price_at_execution: float,
        mid_price_after: float,
        side: str
    ) -> float:
        """
        Compute realized spread.
        
        Realized spread = effective spread - price impact
        
        Args:
            execution_price: Actual execution price
            mid_price_at_execution: Mid-price at execution
            mid_price_after: Mid-price after execution
            side: 'buy' or 'sell'
        
        Returns:
            Realized spread
        """
        effective = MicrostructureDiagnostics.compute_effective_spread(
            execution_price, mid_price_at_execution, side
        )
        
        # Price impact
        if side == 'buy':
            impact = 2 * (mid_price_after - mid_price_at_execution)
        else:
            impact = 2 * (mid_price_at_execution - mid_price_after)
        
        return effective - impact
    
    @staticmethod
    def microprice_forecast_error(
        microprice_history: np.ndarray,
        mid_price_history: np.ndarray,
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Compare microprice vs mid-price as forecast of future mid-price.
        
        Microprice should be a better predictor if it captures order flow information.
        
        Args:
            microprice_history: Array of microprice estimates
            mid_price_history: Array of mid-prices
            horizon: Forecast horizon (steps ahead)
        
        Returns:
            Dictionary with forecast errors
        """
        if len(microprice_history) < horizon + 1:
            return {'micro_mse': 0.0, 'mid_mse': 0.0, 'improvement': 0.0}
        
        # Future mid-prices
        future_mid = mid_price_history[horizon:]
        
        # Forecasts
        micro_forecast = microprice_history[:-horizon]
        mid_forecast = mid_price_history[:-horizon]
        
        # Mean squared errors
        micro_mse = np.mean((future_mid - micro_forecast) ** 2)
        mid_mse = np.mean((future_mid - mid_forecast) ** 2)
        
        improvement = (mid_mse - micro_mse) / mid_mse if mid_mse > 0 else 0.0
        
        return {
            'micro_mse': float(micro_mse),
            'mid_mse': float(mid_mse),
            'improvement_pct': float(improvement * 100)
        }
    
    @staticmethod
    def queue_position_analysis(
        queue_positions: List[int],
        fill_indicators: List[bool]
    ) -> Dict[str, float]:
        """
        Analyze relationship between queue position and fill probability.
        
        Args:
            queue_positions: List of queue positions
            fill_indicators: List of whether order was filled
        
        Returns:
            Dictionary with queue statistics
        """
        if not queue_positions:
            return {'avg_queue_pos': 0.0, 'fill_rate': 0.0}
        
        queue_array = np.array(queue_positions)
        fill_array = np.array(fill_indicators, dtype=float)
        
        # Overall statistics
        avg_queue_pos = np.mean(queue_array)
        fill_rate = np.mean(fill_array)
        
        # Fill rate by queue position
        unique_positions = np.unique(queue_array)
        fill_by_position = {}
        
        for pos in unique_positions:
            mask = queue_array == pos
            if np.sum(mask) > 0:
                fill_by_position[int(pos)] = float(np.mean(fill_array[mask]))
        
        return {
            'avg_queue_position': float(avg_queue_pos),
            'overall_fill_rate': float(fill_rate),
            'fill_by_position': fill_by_position
        }
    
    @staticmethod
    def adverse_selection_by_spread(
        spreads: np.ndarray,
        adverse_selection_costs: np.ndarray,
        n_bins: int = 5
    ) -> Dict[str, List]:
        """
        Analyze adverse selection as function of spread.
        
        Wider spreads should have less adverse selection.
        
        Args:
            spreads: Array of quoted spreads
            adverse_selection_costs: Array of adverse selection costs
            n_bins: Number of bins for analysis
        
        Returns:
            Dictionary with binned analysis
        """
        if len(spreads) == 0:
            return {'spread_bins': [], 'avg_adverse_selection': []}
        
        # Create bins
        spread_bins = np.linspace(np.min(spreads), np.max(spreads), n_bins + 1)
        bin_centers = (spread_bins[:-1] + spread_bins[1:]) / 2
        
        # Average adverse selection per bin
        avg_as = []
        for i in range(n_bins):
            mask = (spreads >= spread_bins[i]) & (spreads < spread_bins[i + 1])
            if np.sum(mask) > 0:
                avg_as.append(float(np.mean(adverse_selection_costs[mask])))
            else:
                avg_as.append(0.0)
        
        return {
            'spread_bins': bin_centers.tolist(),
            'avg_adverse_selection': avg_as
        }
