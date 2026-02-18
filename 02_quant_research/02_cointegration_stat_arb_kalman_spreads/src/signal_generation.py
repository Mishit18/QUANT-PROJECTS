"""
Trading signal generation module.

Implements entry/exit logic based on z-score thresholds, volatility filters,
and risk management rules for mean-reversion strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

from utils import setup_logging, compute_zscore


logger = setup_logging()


class SignalGenerator:
    """
    Generates trading signals for statistical arbitrage.
    
    Signal types:
    - 1: Long spread (buy Y, sell X)
    - -1: Short spread (sell Y, buy X)
    - 0: No position / exit
    """
    
    def __init__(self,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.5,
                 lookback_window: int = 60,
                 min_holding_period: int = 1,
                 max_holding_period: int = 60,
                 volatility_filter: bool = True,
                 vol_lookback: int = 20,
                 vol_percentile: float = 90):
        """
        Initialize signal generator.
        
        Args:
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop-loss
            lookback_window: Window for z-score calculation
            min_holding_period: Minimum days to hold position
            max_holding_period: Maximum days to hold position
            volatility_filter: Whether to filter high volatility periods
            vol_lookback: Lookback for volatility calculation
            vol_percentile: Volatility percentile threshold
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.lookback_window = lookback_window
        self.min_holding_period = min_holding_period
        self.max_holding_period = max_holding_period
        self.volatility_filter = volatility_filter
        self.vol_lookback = vol_lookback
        self.vol_percentile = vol_percentile
    
    def compute_volatility_filter(self, spread: pd.Series) -> pd.Series:
        """
        Compute volatility filter to avoid trading in high volatility regimes.
        
        Args:
            spread: Spread time series
            
        Returns:
            Boolean series (True = allow trading, False = filter out)
        """
        # Compute rolling volatility
        rolling_vol = spread.rolling(window=self.vol_lookback).std()
        
        # Compute percentile threshold
        vol_threshold = rolling_vol.rolling(
            window=self.lookback_window
        ).quantile(self.vol_percentile / 100)
        
        # Allow trading when volatility is below threshold
        filter_series = rolling_vol <= vol_threshold
        
        return filter_series
    
    def generate_raw_signals(self,
                           spread: pd.Series,
                           zscore: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate raw entry/exit signals based on z-score thresholds.
        
        Args:
            spread: Spread time series
            zscore: Pre-computed z-score (computed if None)
            
        Returns:
            DataFrame with signals and z-scores
        """
        logger.info("Generating raw trading signals")
        
        # Compute z-score if not provided
        if zscore is None:
            zscore = compute_zscore(spread, self.lookback_window)
        
        # Initialize signals
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['raw_signal'] = 0
        
        # Entry signals
        # Long spread when z-score < -entry_threshold (spread too low)
        signals.loc[zscore < -self.entry_threshold, 'raw_signal'] = 1
        
        # Short spread when z-score > entry_threshold (spread too high)
        signals.loc[zscore > self.entry_threshold, 'raw_signal'] = -1
        
        # Exit signals (when spread reverts to mean)
        signals.loc[zscore.abs() < self.exit_threshold, 'raw_signal'] = 0
        
        # Apply volatility filter
        if self.volatility_filter:
            vol_filter = self.compute_volatility_filter(spread)
            signals['vol_filter'] = vol_filter
            
            # Set signal to 0 when volatility is too high
            signals.loc[~vol_filter, 'raw_signal'] = 0
        else:
            signals['vol_filter'] = True
        
        return signals
    
    def apply_position_logic(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position management logic including holding periods and stop-loss.
        
        Args:
            signals: DataFrame with raw signals
            
        Returns:
            DataFrame with final positions
        """
        logger.info("Applying position management logic")
        
        signals = signals.copy()
        signals['position'] = 0
        signals['days_in_position'] = 0
        signals['stop_loss_triggered'] = False
        
        current_position = 0
        days_held = 0
        entry_zscore = 0
        
        for i in range(len(signals)):
            zscore = signals['zscore'].iloc[i]
            raw_signal = signals['raw_signal'].iloc[i]
            
            # Check stop-loss
            if current_position != 0:
                # Stop-loss: exit if z-score moves too far against us
                if current_position == 1 and zscore < -self.stop_loss_threshold:
                    # Long position, spread moved even lower
                    signals['stop_loss_triggered'].iloc[i] = True
                    current_position = 0
                    days_held = 0
                elif current_position == -1 and zscore > self.stop_loss_threshold:
                    # Short position, spread moved even higher
                    signals['stop_loss_triggered'].iloc[i] = True
                    current_position = 0
                    days_held = 0
            
            # Update days held
            if current_position != 0:
                days_held += 1
            
            # Position management
            if current_position == 0:
                # No position: check for entry
                if raw_signal != 0:
                    current_position = raw_signal
                    days_held = 1
                    entry_zscore = zscore
            
            else:
                # In position: check for exit
                
                # Exit if minimum holding period met and exit signal
                if days_held >= self.min_holding_period:
                    # Exit on mean reversion
                    if abs(zscore) < self.exit_threshold:
                        current_position = 0
                        days_held = 0
                    
                    # Exit on opposite signal
                    elif (current_position == 1 and raw_signal == -1) or \
                         (current_position == -1 and raw_signal == 1):
                        current_position = 0
                        days_held = 0
                
                # Force exit if maximum holding period reached
                if days_held >= self.max_holding_period:
                    current_position = 0
                    days_held = 0
            
            signals['position'].iloc[i] = current_position
            signals['days_in_position'].iloc[i] = days_held
        
        return signals
    
    def generate_signals(self,
                        spread: pd.Series,
                        zscore: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate complete trading signals with position management.
        
        Args:
            spread: Spread time series
            zscore: Pre-computed z-score (computed if None)
            
        Returns:
            DataFrame with signals and positions
        """
        # Generate raw signals
        signals = self.generate_raw_signals(spread, zscore)
        
        # Apply position logic
        signals = self.apply_position_logic(signals)
        
        # Compute position changes (for transaction cost calculation)
        signals['position_change'] = signals['position'].diff().fillna(0)
        
        logger.info(
            f"Generated signals: {(signals['position'] != 0).sum()} days in position, "
            f"{(signals['position_change'] != 0).sum()} trades"
        )
        
        return signals
    
    def get_trade_summary(self, signals: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for generated signals.
        
        Args:
            signals: DataFrame with signals
            
        Returns:
            Dictionary with trade statistics
        """
        # Count trades
        n_trades = (signals['position_change'] != 0).sum()
        n_long_entries = (signals['position_change'] == 1).sum()
        n_short_entries = (signals['position_change'] == -1).sum()
        
        # Days in position
        days_in_position = (signals['position'] != 0).sum()
        total_days = len(signals)
        utilization = days_in_position / total_days if total_days > 0 else 0
        
        # Average holding period
        position_changes = signals['position_change'].abs()
        if position_changes.sum() > 0:
            avg_holding_period = days_in_position / (position_changes.sum() / 2)
        else:
            avg_holding_period = 0
        
        # Stop-loss frequency
        n_stop_losses = signals['stop_loss_triggered'].sum()
        
        summary = {
            'n_trades': n_trades,
            'n_long_entries': n_long_entries,
            'n_short_entries': n_short_entries,
            'days_in_position': days_in_position,
            'total_days': total_days,
            'utilization': utilization,
            'avg_holding_period': avg_holding_period,
            'n_stop_losses': n_stop_losses,
            'stop_loss_rate': n_stop_losses / n_trades if n_trades > 0 else 0
        }
        
        return summary
    
    def plot_signals(self,
                    signals: pd.DataFrame,
                    save_path: Optional[str] = None):
        """
        Plot spread, z-score, and trading signals.
        
        Args:
            signals: DataFrame with signals
            save_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot spread
        axes[0].plot(signals.index, signals['spread'], label='Spread', color='black', linewidth=1)
        axes[0].set_ylabel('Spread')
        axes[0].set_title('Spread and Trading Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot z-score with thresholds
        axes[1].plot(signals.index, signals['zscore'], label='Z-Score', color='blue', linewidth=1)
        axes[1].axhline(y=self.entry_threshold, color='red', linestyle='--', label='Entry Threshold')
        axes[1].axhline(y=-self.entry_threshold, color='red', linestyle='--')
        axes[1].axhline(y=self.exit_threshold, color='green', linestyle='--', label='Exit Threshold')
        axes[1].axhline(y=-self.exit_threshold, color='green', linestyle='--')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot positions
        axes[2].fill_between(signals.index, 0, signals['position'], 
                            where=(signals['position'] > 0), 
                            color='green', alpha=0.3, label='Long Spread')
        axes[2].fill_between(signals.index, 0, signals['position'], 
                            where=(signals['position'] < 0), 
                            color='red', alpha=0.3, label='Short Spread')
        
        # Mark stop-losses
        stop_loss_dates = signals[signals['stop_loss_triggered']].index
        if len(stop_loss_dates) > 0:
            axes[2].scatter(stop_loss_dates, 
                          [0] * len(stop_loss_dates),
                          color='orange', marker='x', s=100, 
                          label='Stop-Loss', zorder=5)
        
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved signal plot to {save_path}")
        
        plt.close()
