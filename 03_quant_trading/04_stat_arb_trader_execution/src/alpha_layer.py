"""
Alpha generation layer.

Core hypothesis: Mean reversion strength varies over time.
Trade more aggressively when OU parameters are strong.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class AlphaLayer:
    """
    Generates alpha signals based on mean reversion quality.
    
    Key features:
    1. Asymmetric entry/exit (capture convexity)
    2. OU-quality-based position sizing
    3. Time-based exits (kill stale trades)
    4. Z-score velocity filter (momentum veto)
    """
    
    def __init__(self, 
                 entry_z: float = 2.0,
                 exit_z: float = 0.3,
                 stop_loss_z: float = 4.0,
                 max_hold_days: int = 30,
                 velocity_threshold: float = 0.5,
                 half_life_multiplier: float = 1.5):
        """
        Args:
            entry_z: Entry threshold (higher = more selective)
            exit_z: Exit threshold (lower = take profits faster)
            stop_loss_z: Stop loss threshold
            max_hold_days: Maximum holding period (kill stale trades)
            velocity_threshold: Z-score velocity filter (prevent counter-trend)
            half_life_multiplier: Time-to-reversion consistency check multiplier
                                 Exit if position doesn't revert within HL * multiplier
                                 This enforces OU model validity, not optimization
        """
        # PRODUCTION FIX: Defensive type casting
        try:
            self.entry_z = float(entry_z)
            self.exit_z = float(exit_z)
            self.stop_loss_z = float(stop_loss_z)
            self.max_hold_days = int(max_hold_days)
            self.velocity_threshold = float(velocity_threshold)
            self.half_life_multiplier = float(half_life_multiplier)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"AlphaLayer parameters must be numeric. Error: {e}"
            )
        
        # Validation
        if self.entry_z <= self.exit_z:
            raise ValueError("entry_z must be > exit_z")
        if self.max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")
        if self.half_life_multiplier <= 0:
            raise ValueError("half_life_multiplier must be positive")
    
    def generate_signals(self, 
                        z_scores: pd.Series,
                        ou_quality: float,
                        regime_gate: pd.Series = None,
                        ou_half_life: float = None) -> pd.DataFrame:
        """
        Generate trading signals with alpha enhancements.
        
        Args:
            z_scores: Spread z-scores
            ou_quality: OU model quality multiplier (0-1.5)
            regime_gate: Binary gate (0 = block new positions, 1 = allow)
            ou_half_life: OU half-life for time-to-reversion check
        
        Returns:
            DataFrame with signals and metadata
        """
        signals = pd.DataFrame(index=z_scores.index)
        signals['z_score'] = z_scores
        signals['raw_signal'] = 0
        
        # Z-score velocity (rate of change)
        signals['z_velocity'] = z_scores.diff()
        
        # Entry logic with velocity filter
        # Long: spread too low AND not falling too fast
        long_entry = (z_scores < -self.entry_z) & (signals['z_velocity'] > -self.velocity_threshold)
        
        # Short: spread too high AND not rising too fast
        short_entry = (z_scores > self.entry_z) & (signals['z_velocity'] < self.velocity_threshold)
        
        signals.loc[long_entry, 'raw_signal'] = 1
        signals.loc[short_entry, 'raw_signal'] = -1
        
        # Exit logic (asymmetric - take profits faster than entry)
        exit_condition = abs(z_scores) < self.exit_z
        signals.loc[exit_condition, 'raw_signal'] = 0
        
        # Convert to positions with holding period tracking and time-to-reversion check
        signals['position'] = self._apply_position_logic(
            signals['raw_signal'], 
            z_scores,
            ou_half_life
        )
        
        # Scale by OU quality
        signals['position_scaled'] = signals['position'] * ou_quality
        
        # Apply regime gate if provided
        # Rationale: Binary gate enforces OU stationarity assumptions
        # Volatile regimes violate model validity
        if regime_gate is not None:
            signals['position_final'] = self._apply_regime_gate(
                signals['position_scaled'],
                regime_gate
            )
        else:
            signals['position_final'] = signals['position_scaled']
        
        # Clip to reasonable bounds
        signals['position_final'] = signals['position_final'].clip(-1.5, 1.5)
        
        return signals
    
    def _apply_position_logic(self, raw_signals: pd.Series, z_scores: pd.Series, 
                             ou_half_life: float = None) -> pd.Series:
        """
        Convert raw signals to positions with:
        - Entry/exit logic
        - Stop losses
        - Time-based exits
        - Time-to-reversion consistency check (enforces OU model validity)
        
        Args:
            raw_signals: Raw entry/exit signals
            z_scores: Spread z-scores
            ou_half_life: OU half-life for time-to-reversion check
        """
        positions = pd.Series(0, index=raw_signals.index)
        
        current_position = 0
        entry_idx = None
        hold_days = 0
        
        for i in range(len(raw_signals)):
            signal = raw_signals.iloc[i]
            z = z_scores.iloc[i]
            
            # Check stop loss
            if current_position != 0:
                if (current_position == 1 and z < -self.stop_loss_z) or \
                   (current_position == -1 and z > self.stop_loss_z):
                    # Stop loss hit
                    current_position = 0
                    entry_idx = None
                    hold_days = 0
            
            # Check time-based exit
            if current_position != 0:
                hold_days += 1
                if hold_days >= self.max_hold_days:
                    # Stale trade - exit
                    current_position = 0
                    entry_idx = None
                    hold_days = 0
            
            # Time-to-reversion consistency check
            # Rationale: If position doesn't revert within 1.5x OU half-life,
            # the OU model estimate is likely invalid for this regime.
            # This enforces model validity, not return optimization.
            if current_position != 0 and ou_half_life is not None:
                expected_reversion_time = ou_half_life * self.half_life_multiplier
                if hold_days >= expected_reversion_time:
                    # Model validity violated - force exit
                    current_position = 0
                    entry_idx = None
                    hold_days = 0
            
            # New signal
            if signal != 0 and current_position == 0:
                # Enter new position
                current_position = signal
                entry_idx = i
                hold_days = 0
            elif signal == 0 and current_position != 0:
                # Exit signal
                current_position = 0
                entry_idx = None
                hold_days = 0
            
            positions.iloc[i] = current_position
        
        return positions
    
    def _apply_regime_gate(self, positions: pd.Series, regime_gate: pd.Series) -> pd.Series:
        """
        Apply binary regime gate to positions.
        
        Rationale: Enforce OU stationarity assumptions.
        - Gate = 0 (volatile regime): Block new positions, allow exits
        - Gate = 1 (calm regime): Allow all operations
        
        This is NOT return optimization. Volatile regimes violate the
        stationarity assumption underlying OU parameter estimation.
        
        Args:
            positions: Position series
            regime_gate: Binary gate (0 = block new, 1 = allow)
        
        Returns:
            Gated positions
        """
        gated_positions = pd.Series(0.0, index=positions.index)
        current_position = 0.0
        
        for i in range(len(positions)):
            desired_position = positions.iloc[i]
            gate = regime_gate.iloc[i]
            
            if gate == 1:
                # Calm regime - allow all operations
                current_position = desired_position
            else:
                # Volatile regime - block new positions, allow exits
                if current_position == 0:
                    # No position - stay flat
                    current_position = 0
                elif desired_position == 0:
                    # Exit signal - allow exit
                    current_position = 0
                else:
                    # Hold existing position (no new entries)
                    pass  # current_position unchanged
            
            gated_positions.iloc[i] = current_position
        
        return gated_positions
    
    def get_signal_quality_metrics(self, signals: pd.DataFrame) -> dict:
        """
        Analyze signal quality (not just backtest metrics).
        
        Returns metrics that indicate edge quality.
        """
        positions = signals['position_final']
        z_scores = signals['z_score']
        
        # Entry z-score distribution (should be extreme)
        entries = positions.diff() != 0
        entry_z = z_scores[entries].abs()
        
        # Exit z-score distribution (should be near zero)
        exits = (positions.diff() != 0) & (positions == 0)
        exit_z = z_scores[exits].abs()
        
        # Position changes
        trades = (positions.diff() != 0).sum()
        
        # Time in market
        time_in_market = (positions != 0).sum() / len(positions)
        
        return {
            'n_trades': trades,
            'time_in_market': time_in_market,
            'mean_entry_z': entry_z.mean() if len(entry_z) > 0 else 0,
            'mean_exit_z': exit_z.mean() if len(exit_z) > 0 else 0,
            'entry_exit_asymmetry': (entry_z.mean() - exit_z.mean()) if len(entry_z) > 0 and len(exit_z) > 0 else 0
        }
