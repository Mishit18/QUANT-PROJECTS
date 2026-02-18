"""
Performance metrics for backtesting.
"""
import numpy as np


def compute_sharpe_ratio(returns, periods_per_year=252*6.5*3600):
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : array-like
        Return series
    periods_per_year : float
        Number of periods per year for annualization
    
    Returns
    -------
    sharpe : float
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return sharpe


def compute_max_drawdown(pnl_series):
    """
    Compute maximum drawdown.
    
    Parameters
    ----------
    pnl_series : array-like
        PnL time series
    
    Returns
    -------
    max_dd : float
        Maximum drawdown
    max_dd_pct : float
        Maximum drawdown percentage
    """
    pnl = np.asarray(pnl_series)
    if len(pnl) == 0:
        return 0.0, 0.0
    
    running_max = np.maximum.accumulate(pnl)
    drawdowns = running_max - pnl
    max_dd = np.max(drawdowns)
    
    # Percentage drawdown
    max_dd_pct = max_dd / running_max[np.argmax(drawdowns)] if running_max[np.argmax(drawdowns)] > 0 else 0.0
    
    return max_dd, max_dd_pct


def compute_spread_capture(trades, tick_size):
    """
    Compute average spread capture per trade.
    
    Parameters
    ----------
    trades : list of dict
        Trade records with 'side' and 'price'
    tick_size : float
        Price tick size
    
    Returns
    -------
    avg_capture : float
        Average spread capture in ticks
    """
    if len(trades) < 2:
        return 0.0
    
    # Pair buy and sell trades
    buys = [t for t in trades if t['side'].name == 'BUY']
    sells = [t for t in trades if t['side'].name == 'SELL']
    
    if len(buys) == 0 or len(sells) == 0:
        return 0.0
    
    # Simple average spread
    avg_buy_price = np.mean([t['price'] for t in buys])
    avg_sell_price = np.mean([t['price'] for t in sells])
    
    spread_capture = (avg_sell_price - avg_buy_price) / tick_size
    
    return spread_capture


def compute_adverse_selection_loss(trades, mid_prices):
    """
    Estimate adverse selection loss.
    
    Parameters
    ----------
    trades : list of dict
        Trade records
    mid_prices : dict
        Timestamp -> mid price mapping
    
    Returns
    -------
    avg_loss : float
        Average adverse selection loss per trade
    """
    if len(trades) == 0:
        return 0.0
    
    losses = []
    
    for trade in trades:
        timestamp = trade['timestamp']
        price = trade['price']
        side = trade['side']
        
        # Find closest mid price
        mid = mid_prices.get(timestamp, price)
        
        # Loss is difference between trade price and mid
        if side.name == 'BUY':
            loss = price - mid  # Paid more than mid
        else:
            loss = mid - price  # Received less than mid
        
        losses.append(loss)
    
    return np.mean(losses) if losses else 0.0


def compute_inventory_turnover(inventory_history, total_time):
    """
    Compute inventory turnover rate.
    
    Parameters
    ----------
    inventory_history : list of tuples
        (timestamp, inventory) pairs
    total_time : float
        Total simulation time
    
    Returns
    -------
    turnover : float
        Average absolute inventory change per unit time
    """
    if len(inventory_history) < 2:
        return 0.0
    
    inventories = [inv for _, inv, _ in inventory_history]
    changes = np.abs(np.diff(inventories))
    
    return np.sum(changes) / total_time


def generate_performance_report(agent, total_time):
    """
    Generate comprehensive performance report.
    
    Parameters
    ----------
    agent : MarketMaker
        Market-making agent
    total_time : float
        Total simulation time
    
    Returns
    -------
    report : dict
        Performance metrics
    """
    if len(agent.pnl_history) == 0:
        return {}
    
    times, pnls, inventories = zip(*agent.pnl_history)
    pnls = np.array(pnls)
    
    # PnL metrics
    final_pnl = pnls[-1]
    returns = np.diff(pnls) if len(pnls) > 1 else np.array([])
    
    sharpe = compute_sharpe_ratio(returns)
    max_dd, max_dd_pct = compute_max_drawdown(pnls)
    
    # Trade metrics
    num_trades = len(agent.trades)
    spread_capture = compute_spread_capture(agent.trades, agent.tick_size)
    
    # Inventory metrics
    inv_stats = agent.inventory_controller.get_inventory_stats()
    turnover = compute_inventory_turnover(
        agent.inventory_controller.inventory_history,
        total_time
    )
    
    report = {
        'pnl': {
            'final': final_pnl,
            'max': np.max(pnls),
            'min': np.min(pnls),
            'mean': np.mean(pnls),
            'std': np.std(pnls)
        },
        'risk': {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct * 100
        },
        'trading': {
            'num_trades': num_trades,
            'spread_capture_ticks': spread_capture,
            'trades_per_second': num_trades / total_time if total_time > 0 else 0
        },
        'inventory': {
            'final': inv_stats.get('current', 0),
            'max': inv_stats.get('max', 0),
            'min': inv_stats.get('min', 0),
            'mean': inv_stats.get('mean', 0),
            'std': inv_stats.get('std', 0),
            'turnover': turnover
        }
    }
    
    return report
