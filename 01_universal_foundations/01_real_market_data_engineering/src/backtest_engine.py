# src/backtest_engine.py
"""
Core backtesting engine with walk-forward validation.
Implements transaction costs, slippage, and performance metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Production-grade backtesting engine.
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_bps: float = 5.0):
        """
        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction (0.001 = 10 bps = 0.1%)
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        
        self.trades = []
        self.equity_curve = []
        
    def apply_transaction_costs(self, 
                                price: float, 
                                quantity: float, 
                                side: str) -> Tuple[float, float]:
        """
        Apply commission and slippage to a trade.
        
        Args:
            price: Execution price
            quantity: Trade quantity (positive)
            side: 'buy' or 'sell'
        
        Returns:
            (effective_price, total_cost)
        """
        # Slippage: buy higher, sell lower
        slippage_factor = self.slippage_bps / 10000.0
        if side == 'buy':
            effective_price = price * (1 + slippage_factor)
        else:
            effective_price = price * (1 - slippage_factor)
        
        # Commission
        notional = effective_price * quantity
        commission = notional * self.commission_rate
        
        # Total cost
        if side == 'buy':
            total_cost = notional + commission
        else:
            total_cost = notional - commission
        
        return effective_price, total_cost
    
    def simulate_strategy(self,
                         df: pd.DataFrame,
                         signals: pd.Series,
                         price_col: str = 'close',
                         execution_lag: int = 1) -> pd.DataFrame:
        """
        Simulate a trading strategy given signals using return attribution.

        Signals generated from bar ``t`` are shifted by ``execution_lag`` before
        they earn PnL. The default one-bar delay prevents close-to-close
        look-ahead bias when features use the current close.
        
        Args:
            df: DataFrame with price data
            signals: Series with trading signals (-1, 0, 1)
                    -1 = short, 0 = flat, 1 = long
            price_col: Column name for execution price
        
        Returns:
            DataFrame with backtest results
        """
        df = df.copy()
        signal = signals.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
        price = pd.to_numeric(df[price_col], errors='coerce')
        asset_returns = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        position = signal.shift(execution_lag).fillna(0.0)
        turnover = position.diff().abs().fillna(position.abs())
        cost_rate = self.commission_rate + (self.slippage_bps / 10000.0)

        gross_returns = position * asset_returns
        costs = turnover * cost_rate
        net_returns = gross_returns - costs
        equity = self.initial_capital * (1.0 + net_returns).cumprod()

        df['signal'] = signal
        df['position'] = position
        df['asset_returns'] = asset_returns
        df['turnover'] = turnover
        df['transaction_costs'] = costs
        df['gross_returns'] = gross_returns
        df['returns'] = net_returns
        df['equity'] = equity
        df['cash'] = np.nan
        df['holdings'] = np.nan

        return df
    
    @staticmethod
    def infer_periods_per_year(df: pd.DataFrame, default: int = 252 * 375) -> int:
        """
        Infer annualization factor from observed bars per trading day.
        """
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
            return default
        counts = df.groupby(df.index.normalize()).size()
        if counts.empty:
            return default
        median_bars_per_day = int(counts.median())
        return max(1, median_bars_per_day * 252)
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        """
        returns = df['returns'].dropna()
        equity = df['equity'].dropna()
        
        if len(returns) == 0 or len(equity) == 0:
            return {'error': 'No valid returns'}
        
        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized return from observed market bars.
        n_periods = len(returns)
        periods_per_year = self.infer_periods_per_year(df)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Volatility
        returns_std = returns.std()
        annualized_vol = returns_std * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        annualized_downside = downside_std * np.sqrt(periods_per_year)
        sortino = annualized_return / annualized_downside if annualized_downside > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Hit rate and turnover
        active_returns = returns[df.get('position', pd.Series(index=df.index, dtype=float)).abs() > 0]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0
        total_trades = int((df.get('turnover', pd.Series(index=df.index, dtype=float)).fillna(0) > 0).sum())
        avg_exposure = float(df.get('position', pd.Series(index=df.index, dtype=float)).abs().mean())
        total_turnover = float(df.get('turnover', pd.Series(index=df.index, dtype=float)).fillna(0).sum())
        gross_return = float(df.get('gross_returns', pd.Series(index=df.index, dtype=float)).fillna(0).sum())
        total_costs = float(df.get('transaction_costs', pd.Series(index=df.index, dtype=float)).fillna(0).sum())
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'total_trades': int(total_trades),
            'avg_abs_exposure': avg_exposure,
            'total_turnover': total_turnover,
            'gross_return_sum': gross_return,
            'total_cost_sum': total_costs,
            'periods_per_year': int(periods_per_year),
            'n_periods': n_periods
        }
        
        return metrics
    
    def walk_forward_validation(self,
                                df: pd.DataFrame,
                                train_days: int = 60,
                                test_days: int = 20,
                                step_days: int = 10) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            df: DataFrame with features and price data
            train_days: Training period in days
            test_days: Testing period in days
            step_days: Step size in days
        
        Returns:
            Dictionary with validation results
        """
        logger.info(f"\nWalk-forward validation:")
        logger.info(f"  Train: {train_days} days, Test: {test_days} days, Step: {step_days} days")
        
        # Convert days to periods (assuming minute data with ~390 minutes/day)
        minutes_per_day = 390
        train_periods = train_days * minutes_per_day
        test_periods = test_days * minutes_per_day
        step_periods = step_days * minutes_per_day
        
        results = []
        start_idx = 0
        
        while start_idx + train_periods + test_periods < len(df):
            train_end = start_idx + train_periods
            test_end = train_end + test_periods
            
            train_data = df.iloc[start_idx:train_end]
            test_data = df.iloc[train_end:test_end]
            
            logger.info(f"\n  Window: train {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"          test  {test_data.index[0]} to {test_data.index[-1]}")
            
            # Here you would train your model on train_data
            # For now, we'll use a simple momentum strategy as placeholder
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
            
            start_idx += step_periods
        
        logger.info(f"\nCompleted {len(results)} walk-forward windows")
        
        return {'windows': results, 'n_windows': len(results)}


def simple_momentum_strategy(df: pd.DataFrame, 
                             lookback: int = 60,
                             threshold: float = 0.0) -> pd.Series:
    """
    Simple momentum strategy for demonstration.
    
    Args:
        df: DataFrame with 'close' column
        lookback: Lookback period for momentum
        threshold: Threshold for signal generation
    
    Returns:
        Series with signals (-1, 0, 1)
    """
    momentum = df['close'].pct_change(lookback)
    
    signals = pd.Series(0, index=df.index)
    signals[momentum > threshold] = 1
    signals[momentum < -threshold] = -1
    
    return signals


def volatility_scaled_momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 60,
    vol_window: int = 240,
    threshold: float = 0.75,
) -> pd.Series:
    """
    Momentum signal normalized by recent realized volatility.

    This is still a diagnostic baseline, not an optimized alpha model. The goal
    is to produce a cleaner, lower-turnover sanity check than threshold-zero
    raw momentum.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    log_price = np.log(df['close'].where(df['close'] > 0))
    momentum = log_price.diff(lookback)
    realized_vol = log_price.diff().rolling(vol_window).std() * np.sqrt(max(lookback, 1))
    score = momentum / realized_vol.replace(0, np.nan)

    signals = pd.Series(0.0, index=df.index)
    signals[score > threshold] = 1.0
    signals[score < -threshold] = -1.0
    return signals


if __name__ == "__main__":
    from config import FEATURES_PARQUET, REPORTS_DIR
    
    logger.info("Loading features...")
    df = pd.read_parquet(FEATURES_PARQUET)
    
    # Generate simple signals
    signals = volatility_scaled_momentum_strategy(df, lookback=60, vol_window=240, threshold=0.75)
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_bps=5.0
    )
    
    results = engine.simulate_strategy(df, signals, execution_lag=1)
    metrics = engine.calculate_metrics(results)
    
    # Print metrics
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key:.<30} {value:.4f}")
        else:
            logger.info(f"{key:.<30} {value}")
    logger.info("="*60)
    
    # Save results
    backtest_cols = [
        'close', 'signal', 'position', 'asset_returns', 'turnover',
        'transaction_costs', 'gross_returns', 'returns', 'equity'
    ]
    results[backtest_cols].to_parquet(REPORTS_DIR / 'backtest_results.parquet', compression='zstd')
    logger.info(f"\nSaved backtest results to {REPORTS_DIR / 'backtest_results.parquet'}")
