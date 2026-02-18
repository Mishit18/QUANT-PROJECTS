"""
Multi-Pair Portfolio Backtesting Engine.

Upgrades from single-pair to portfolio-level backtesting with:
- Volatility-targeted position sizing
- Risk-parity allocation
- Regime filters
- Capital recycling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from portfolio_manager import PortfolioManager, PairPosition
from signal_generation import SignalGenerator
from spread_metrics import SpreadAnalyzer
from utils import setup_logging, load_config, ensure_dir

logger = setup_logging()


class MultiPairBacktestEngine:
    """
    Portfolio-level backtest engine for statistical arbitrage.
    
    Key Improvements:
    - Trades multiple pairs simultaneously
    - Volatility-targeted position sizing
    - Portfolio-level risk management
    - Regime-aware trading
    """
    
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        """Initialize multi-pair backtest engine."""
        self.config = load_config(config_path)
        
        # Portfolio manager
        self.portfolio = PortfolioManager(
            initial_capital=self.config['backtest']['initial_capital'],
            target_portfolio_vol=self.config['portfolio']['target_volatility'],
            max_pairs=self.config['portfolio']['max_pairs'],
            max_pair_weight=self.config['portfolio']['max_pair_weight'],
            vol_lookback=self.config['portfolio']['vol_lookback']
        )
        
        # Signal generator
        self.signal_generator = SignalGenerator(
            entry_threshold=self.config['signals']['entry_threshold'],
            exit_threshold=self.config['signals']['exit_threshold'],
            stop_loss_threshold=self.config['signals']['stop_loss_threshold'],
            lookback_window=self.config['spread']['lookback_window']
        )
        
        # Spread analyzer
        self.spread_analyzer = SpreadAnalyzer()
        
        # Regime filters
        self.regime_filters_enabled = self.config['portfolio'].get('regime_filters', True)
        self.vol_explosion_threshold = self.config['portfolio'].get('vol_explosion_threshold', 3.0)
        
        logger.info("Multi-pair backtest engine initialized")
    
    def run_portfolio_backtest(self,
                              pairs_data: Dict[Tuple[str, str], Dict],
                              test_prices: pd.DataFrame) -> Dict:
        """
        Run portfolio backtest across multiple pairs.
        
        Args:
            pairs_data: Dictionary mapping pair -> {'spread', 'hedge_ratio', 'signals'}
            test_prices: Test period price data
            
        Returns:
            Portfolio backtest results
        """
        logger.info(f"Running portfolio backtest on {len(pairs_data)} pairs")
        
        # Get common date index
        dates = test_prices.index
        
        # Initialize tracking
        equity_curve = []
        portfolio_positions = []
        all_trades = []
        pair_contributions = {pair: [] for pair in pairs_data.keys()}
        
        # Daily loop
        for i, date in enumerate(dates):
            daily_stats = {
                'date': date,
                'equity': self.portfolio.capital,
                'num_positions': len(self.portfolio.positions),
                'capital_allocated': sum(self.portfolio.pair_allocations.values())
            }
            
            # Check each pair
            for pair, data in pairs_data.items():
                ticker_y, ticker_x = pair
                
                # Get current data
                if date not in data['spread'].index:
                    continue
                
                spread_history = data['spread'].loc[:date]
                signals = data['signals'].loc[:date]
                
                if len(spread_history) < self.config['spread']['lookback_window']:
                    continue
                
                current_signal = signals.iloc[-1]
                current_spread = spread_history.iloc[-1]
                price_y = test_prices.loc[date, ticker_y]
                price_x = test_prices.loc[date, ticker_x]
                
                # Get hedge ratio (use last value or constant)
                if isinstance(data['hedge_ratio'], pd.Series):
                    hedge_ratio = data['hedge_ratio'].loc[date]
                else:
                    hedge_ratio = data['hedge_ratio']
                
                # Regime filter
                if self.regime_filters_enabled:
                    if not self.portfolio.check_regime_stability(
                        spread_history,
                        self.vol_explosion_threshold
                    ):
                        # Close position if in unstable regime
                        if pair in self.portfolio.positions:
                            pnl = self.portfolio.close_position(
                                pair, date, current_signal['zscore'],
                                price_y, price_x, reason="regime_unstable"
                            )
                            if pnl is not None:
                                all_trades.append({
                                    'pair': pair,
                                    'exit_date': date,
                                    'pnl': pnl,
                                    'reason': 'regime_unstable'
                                })
                        continue
                
                # Position management
                if pair in self.portfolio.positions:
                    # Check exit conditions
                    position = self.portfolio.positions[pair]
                    
                    # Exit signal
                    if current_signal['position'] == 0:
                        pnl = self.portfolio.close_position(
                            pair, date, current_signal['zscore'],
                            price_y, price_x, reason="exit_signal"
                        )
                        if pnl is not None:
                            all_trades.append({
                                'pair': pair,
                                'entry_date': position.entry_date,
                                'exit_date': date,
                                'direction': 'LONG' if position.direction == 1 else 'SHORT',
                                'entry_zscore': position.entry_zscore,
                                'exit_zscore': current_signal['zscore'],
                                'pnl': pnl,
                                'notional': position.notional,
                                'reason': 'exit_signal'
                            })
                            pair_contributions[pair].append(pnl)
                    
                    # Stop loss
                    elif abs(current_signal['zscore']) > self.config['signals']['stop_loss_threshold']:
                        pnl = self.portfolio.close_position(
                            pair, date, current_signal['zscore'],
                            price_y, price_x, reason="stop_loss"
                        )
                        if pnl is not None:
                            all_trades.append({
                                'pair': pair,
                                'entry_date': position.entry_date,
                                'exit_date': date,
                                'direction': 'LONG' if position.direction == 1 else 'SHORT',
                                'entry_zscore': position.entry_zscore,
                                'exit_zscore': current_signal['zscore'],
                                'pnl': pnl,
                                'notional': position.notional,
                                'reason': 'stop_loss'
                            })
                            pair_contributions[pair].append(pnl)
                
                else:
                    # Check entry conditions
                    if current_signal['position'] != 0:
                        direction = int(current_signal['position'])
                        
                        position = self.portfolio.open_position(
                            pair=pair,
                            direction=direction,
                            entry_date=date,
                            entry_zscore=current_signal['zscore'],
                            spread=spread_history,
                            price_y=price_y,
                            price_x=price_x,
                            hedge_ratio=hedge_ratio
                        )
            
            # Record daily state
            equity_curve.append(daily_stats)
            portfolio_positions.append({
                'date': date,
                'positions': list(self.portfolio.positions.keys()),
                'num_positions': len(self.portfolio.positions)
            })
        
        # Close all remaining positions
        final_date = dates[-1]
        for pair in list(self.portfolio.positions.keys()):
            ticker_y, ticker_x = pair
            price_y = test_prices.loc[final_date, ticker_y]
            price_x = test_prices.loc[final_date, ticker_x]
            
            pnl = self.portfolio.close_position(
                pair, final_date, 0.0,
                price_y, price_x, reason="end_of_period"
            )
            if pnl is not None:
                all_trades.append({
                    'pair': pair,
                    'exit_date': final_date,
                    'pnl': pnl,
                    'reason': 'end_of_period'
                })
        
        # Compile results
        results = {
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'trades': pd.DataFrame(all_trades) if all_trades else pd.DataFrame(),
            'portfolio_positions': pd.DataFrame(portfolio_positions).set_index('date'),
            'pair_contributions': pair_contributions,
            'final_capital': self.portfolio.capital,
            'initial_capital': self.portfolio.initial_capital,
            'total_return': (self.portfolio.capital / self.portfolio.initial_capital - 1),
            'num_trades': len(all_trades)
        }
        
        logger.info(
            f"Portfolio backtest complete: "
            f"return={results['total_return']:.2%}, "
            f"trades={results['num_trades']}"
        )
        
        return results
    
    def generate_report(self, results: Dict, output_dir: str = "results/portfolio"):
        """
        Generate portfolio backtest report.
        
        Args:
            results: Backtest results dictionary
            output_dir: Output directory
        """
        ensure_dir(output_dir)
        
        # Save equity curve
        results['equity_curve'].to_csv(f"{output_dir}/portfolio_equity_curve.csv")
        
        # Save trades
        if not results['trades'].empty:
            results['trades'].to_csv(f"{output_dir}/portfolio_trades.csv", index=False)
        
        # Save pair contributions
        contributions_df = pd.DataFrame([
            {'pair': str(pair), 'total_pnl': sum(pnls), 'num_trades': len(pnls)}
            for pair, pnls in results['pair_contributions'].items()
            if pnls
        ])
        if not contributions_df.empty:
            contributions_df.to_csv(f"{output_dir}/pair_contributions.csv", index=False)
        
        logger.info(f"Portfolio report saved to {output_dir}/")
