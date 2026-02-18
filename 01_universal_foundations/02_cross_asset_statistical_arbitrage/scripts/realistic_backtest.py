import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.neutralization.market import neutralize_market_beta
from src.neutralization.sector import neutralize_sector
from src.utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown


def create_synthetic_sectors(tickers: list, n_sectors: int = 10) -> dict:
    """Create synthetic sector assignments."""
    np.random.seed(42)
    sectors = [f'SECTOR_{i%n_sectors}' for i in range(len(tickers))]
    return dict(zip(tickers, sectors))


def construct_quantile_portfolio(alpha: pd.Series, top_pct: float = 0.2, 
                                 bottom_pct: float = 0.2) -> pd.Series:
    """Long top quantile, short bottom quantile."""
    valid = alpha.notna()
    if valid.sum() < 10:
        return pd.Series(0.0, index=alpha.index)
    
    alpha_valid = alpha[valid]
    n_assets = len(alpha_valid)
    
    top_threshold = alpha_valid.quantile(1 - top_pct)
    bottom_threshold = alpha_valid.quantile(bottom_pct)
    
    weights = pd.Series(0.0, index=alpha.index)
    
    # Long top quantile
    long_mask = alpha >= top_threshold
    n_long = long_mask.sum()
    if n_long > 0:
        weights[long_mask] = 1.0 / n_long
    
    # Short bottom quantile
    short_mask = alpha <= bottom_threshold
    n_short = short_mask.sum()
    if n_short > 0:
        weights[short_mask] = -1.0 / n_short
    
    return weights


def apply_volatility_targeting(weights: pd.Series, returns: pd.DataFrame, 
                               date: pd.Timestamp, window: int = 60,
                               target_vol: float = 0.10) -> pd.Series:
    """Scale portfolio to target volatility."""
    if date not in returns.index:
        return weights
    
    date_idx = returns.index.get_loc(date)
    if date_idx < window:
        return weights
    
    hist_returns = returns.iloc[date_idx - window:date_idx]
    
    # Compute portfolio volatility
    common = weights.index.intersection(hist_returns.columns)
    if len(common) < 5:
        return weights
    
    w = weights[common].fillna(0)
    ret_hist = hist_returns[common].fillna(0)
    
    port_returns = (ret_hist * w).sum(axis=1)
    port_vol = port_returns.std() * np.sqrt(252)
    
    if port_vol > 0.01:
        scale = target_vol / port_vol
        scale = min(scale, 2.0)  # Cap at 2x leverage
        weights = weights * scale
    
    return weights


def apply_position_limits(weights: pd.Series, max_weight: float = 0.05) -> pd.Series:
    """Cap individual position sizes."""
    return weights.clip(-max_weight, max_weight)


def compute_turnover(weights_prev: pd.Series, weights_curr: pd.Series) -> float:
    """One-way turnover."""
    common = weights_prev.index.union(weights_curr.index)
    w_prev = weights_prev.reindex(common, fill_value=0)
    w_curr = weights_curr.reindex(common, fill_value=0)
    return (w_prev - w_curr).abs().sum() / 2


def compute_transaction_costs(turnover: float, tcost_bps: float = 7.5) -> float:
    """Linear transaction cost."""
    return turnover * tcost_bps / 10000


def run_realistic_backtest(predictions: pd.DataFrame, returns: pd.DataFrame,
                           config: dict, neutralize_risks: bool = True) -> dict:
    """Execute realistic backtest with all constraints."""
    
    # Parameters
    top_pct = 0.2
    bottom_pct = 0.2
    target_vol = config['backtest']['vol_target']
    max_weight = 0.05
    tcost_bps = config['backtest']['tcost_bps']
    rebalance_freq = config['backtest']['rebalance_freq']
    
    # Neutralization setup
    if neutralize_risks:
        market_returns = returns.mean(axis=1)
        sector_map = create_synthetic_sectors(returns.columns.tolist())
        predictions = neutralize_market_beta(predictions, returns, market_returns, window=252)
        predictions = neutralize_sector(predictions, sector_map)
    
    # Align dates - predictions are for forward returns
    # So we need to shift returns forward by 1 to align
    common_dates = predictions.index.intersection(returns.index)
    predictions = predictions.loc[common_dates]
    
    # Initialize
    prev_weights = pd.Series(0.0, index=predictions.columns)
    portfolio_returns = []
    gross_returns = []
    costs_series = []
    turnover_series = []
    weights_history = []
    
    rebalance_dates = common_dates[::rebalance_freq]
    
    for i, date in enumerate(common_dates):
        # Rebalance logic
        if date in rebalance_dates:
            alpha = predictions.loc[date]
            
            # Construct quantile portfolio
            target_weights = construct_quantile_portfolio(alpha, top_pct, bottom_pct)
            
            # Apply position limits
            target_weights = apply_position_limits(target_weights, max_weight)
            
            # Volatility targeting
            target_weights = apply_volatility_targeting(target_weights, returns, 
                                                       date, window=60, target_vol=target_vol)
            
            # Compute turnover and costs
            turnover = compute_turnover(prev_weights, target_weights)
            costs = compute_transaction_costs(turnover, tcost_bps)
            
            turnover_series.append((date, turnover))
            costs_series.append((date, costs))
            weights_history.append((date, target_weights))
            
            prev_weights = target_weights
        else:
            costs = 0.0
        
        # Compute returns - use NEXT period's returns
        date_idx = returns.index.get_loc(date)
        if date_idx + 1 < len(returns):
            next_date = returns.index[date_idx + 1]
            forward_ret = returns.loc[next_date]
        else:
            forward_ret = pd.Series(0.0, index=prev_weights.index)
        
        common_assets = prev_weights.index.intersection(forward_ret.index)
        
        gross_ret = (prev_weights[common_assets] * forward_ret[common_assets]).sum()
        net_ret = gross_ret - costs
        
        portfolio_returns.append((date, net_ret))
        gross_returns.append((date, gross_ret))
    
    return {
        'returns': pd.Series(dict(portfolio_returns)),
        'gross_returns': pd.Series(dict(gross_returns)),
        'costs': pd.Series(dict(costs_series)),
        'turnover': pd.Series(dict(turnover_series)),
        'weights': pd.DataFrame([w for _, w in weights_history],
                               index=[d for d, _ in weights_history])
    }


def analyze_subperiods(returns: pd.Series, n_periods: int = 4) -> pd.DataFrame:
    """Subperiod performance analysis."""
    period_length = len(returns) // n_periods
    results = []
    
    for i in range(n_periods):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < n_periods - 1 else len(returns)
        
        period_returns = returns.iloc[start_idx:end_idx]
        
        if len(period_returns) > 0:
            results.append({
                'period': i + 1,
                'start': period_returns.index[0],
                'end': period_returns.index[-1],
                'sharpe': sharpe_ratio(period_returns),
                'return': (1 + period_returns).prod() - 1,
                'volatility': period_returns.std() * np.sqrt(252),
                'max_dd': max_drawdown((1 + period_returns).cumprod())
            })
    
    return pd.DataFrame(results)


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    predictions = pd.read_parquet('data/processed/predictions_xgboost.parquet')
    returns = pd.read_parquet('data/processed/returns.parquet')
    
    # Unstack if needed
    if predictions.index.nlevels > 1:
        predictions = predictions['prediction'].unstack() if 'prediction' in predictions.columns else predictions.unstack()
    
    print("=" * 70)
    print("REALISTIC BACKTEST: ALPHA TO PNL TRANSLATION")
    print("=" * 70)
    print(f"\nData: {len(predictions)} dates, {predictions.shape[1]} assets")
    print(f"Rebalance frequency: {config['backtest']['rebalance_freq']} days")
    print(f"Transaction costs: {config['backtest']['tcost_bps']} bps")
    print(f"Vol target: {config['backtest']['vol_target']:.1%}")
    
    # Test 1: No neutralization, no costs
    print("\n" + "=" * 70)
    print("[1] BASELINE: No Neutralization, No Costs")
    print("=" * 70)
    
    config_nocost = config.copy()
    config_nocost['backtest']['tcost_bps'] = 0.0
    
    results_baseline = run_realistic_backtest(predictions, returns, config_nocost, 
                                              neutralize_risks=False)
    
    ret_baseline = results_baseline['returns']
    sharpe_baseline = sharpe_ratio(ret_baseline)
    sortino_baseline = sortino_ratio(ret_baseline)
    mdd_baseline = max_drawdown((1 + ret_baseline).cumprod())
    total_ret_baseline = (1 + ret_baseline).prod() - 1
    turnover_baseline = results_baseline['turnover'].mean()
    
    print(f"Sharpe:   {sharpe_baseline:6.2f}")
    print(f"Sortino:  {sortino_baseline:6.2f}")
    print(f"Return:   {total_ret_baseline:6.2%}")
    print(f"Max DD:   {mdd_baseline:6.2%}")
    print(f"Turnover: {turnover_baseline:6.2%}")
    
    # Test 2: With neutralization, no costs
    print("\n" + "=" * 70)
    print("[2] NEUTRALIZED: Market + Sector, No Costs")
    print("=" * 70)
    
    results_neutral = run_realistic_backtest(predictions, returns, config_nocost,
                                            neutralize_risks=True)
    
    ret_neutral = results_neutral['returns']
    sharpe_neutral = sharpe_ratio(ret_neutral)
    sortino_neutral = sortino_ratio(ret_neutral)
    mdd_neutral = max_drawdown((1 + ret_neutral).cumprod())
    total_ret_neutral = (1 + ret_neutral).prod() - 1
    turnover_neutral = results_neutral['turnover'].mean()
    
    print(f"Sharpe:   {sharpe_neutral:6.2f}")
    print(f"Sortino:  {sortino_neutral:6.2f}")
    print(f"Return:   {total_ret_neutral:6.2%}")
    print(f"Max DD:   {mdd_neutral:6.2%}")
    print(f"Turnover: {turnover_neutral:6.2%}")
    
    # Test 3: With neutralization and costs
    print("\n" + "=" * 70)
    print("[3] REALISTIC: Neutralized + Transaction Costs")
    print("=" * 70)
    
    results_realistic = run_realistic_backtest(predictions, returns, config,
                                               neutralize_risks=True)
    
    ret_realistic = results_realistic['returns']
    ret_gross = results_realistic['gross_returns']
    costs = results_realistic['costs']
    
    sharpe_realistic = sharpe_ratio(ret_realistic)
    sortino_realistic = sortino_ratio(ret_realistic)
    mdd_realistic = max_drawdown((1 + ret_realistic).cumprod())
    total_ret_realistic = (1 + ret_realistic).prod() - 1
    total_ret_gross = (1 + ret_gross).prod() - 1
    turnover_realistic = results_realistic['turnover'].mean()
    total_costs = costs.sum()
    
    print(f"Sharpe:        {sharpe_realistic:6.2f}")
    print(f"Sortino:       {sortino_realistic:6.2f}")
    print(f"Return (net):  {total_ret_realistic:6.2%}")
    print(f"Return (gross):{total_ret_gross:6.2%}")
    print(f"Max DD:        {mdd_realistic:6.2%}")
    print(f"Turnover:      {turnover_realistic:6.2%}")
    print(f"Total costs:   {total_costs:6.2%}")
    print(f"Cost drag:     {(total_ret_gross - total_ret_realistic):6.2%}")
    
    # Subperiod analysis
    print("\n" + "=" * 70)
    print("[4] SUBPERIOD ROBUSTNESS")
    print("=" * 70)
    
    subperiods = analyze_subperiods(ret_realistic, n_periods=4)
    print("\n" + subperiods.to_string(index=False))
    
    # Capacity estimate
    print("\n" + "=" * 70)
    print("[5] CAPACITY ESTIMATE")
    print("=" * 70)
    
    avg_gross_exposure = results_realistic['weights'].abs().sum(axis=1).mean()
    avg_position_size = results_realistic['weights'].abs().mean().mean()
    
    print(f"Avg gross exposure: {avg_gross_exposure:6.2f}")
    print(f"Avg position size:  {avg_position_size:6.4f}")
    print(f"Avg turnover:       {turnover_realistic:6.2%}")
    
    # Rough capacity estimate (assuming $1M ADV per asset)
    capacity_per_asset = 1_000_000 * 0.1  # 10% participation
    implied_capacity = capacity_per_asset / avg_position_size if avg_position_size > 0 else 0
    
    print(f"\nEstimated capacity: ${implied_capacity/1e6:.1f}M")
    print("(Assumes $1M ADV per asset, 10% participation)")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: ALPHA TO PNL TRANSLATION")
    print("=" * 70)
    
    print(f"\n{'Scenario':<30s} {'Sharpe':>8s} {'Return':>8s} {'MaxDD':>8s}")
    print("-" * 70)
    print(f"{'Baseline (no neut, no cost)':<30s} {sharpe_baseline:8.2f} {total_ret_baseline:7.2%} {mdd_baseline:7.2%}")
    print(f"{'Neutralized (no cost)':<30s} {sharpe_neutral:8.2f} {total_ret_neutral:7.2%} {mdd_neutral:7.2%}")
    print(f"{'Realistic (neut + cost)':<30s} {sharpe_realistic:8.2f} {total_ret_realistic:7.2%} {mdd_realistic:7.2%}")
    
    # Degradation analysis
    sharpe_retention = (sharpe_realistic / sharpe_neutral * 100) if sharpe_neutral != 0 else 0
    return_retention = (total_ret_realistic / total_ret_neutral * 100) if total_ret_neutral != 0 else 0
    
    print(f"\nSharpe retention:  {sharpe_retention:5.1f}%")
    print(f"Return retention:  {return_retention:5.1f}%")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    if sharpe_realistic > 1.0:
        print("✓ SIGNAL IS TRADABLE: Sharpe > 1.0 after costs")
        print("  → Alpha survives realistic trading constraints")
        print("  → Suitable for live trading")
    elif sharpe_realistic > 0.5:
        print("⚠ SIGNAL IS MARGINAL: Sharpe 0.5-1.0 after costs")
        print("  → Alpha partially survives constraints")
        print("  → May be tradable with optimization")
    else:
        print("✗ SIGNAL IS NOT TRADABLE: Sharpe < 0.5 after costs")
        print("  → Alpha does not survive constraints")
        print("  → Requires fundamental redesign")
    
    if sharpe_retention > 80:
        print(f"\n✓ LOW DEGRADATION: {sharpe_retention:.0f}% Sharpe retention")
    elif sharpe_retention > 50:
        print(f"\n⚠ MODERATE DEGRADATION: {sharpe_retention:.0f}% Sharpe retention")
    else:
        print(f"\n✗ HIGH DEGRADATION: {sharpe_retention:.0f}% Sharpe retention")
    
    print("\n" + "=" * 70)
    
    # Save results
    results_realistic['returns'].to_csv('data/processed/backtest_returns_realistic.csv')
    results_realistic['weights'].to_parquet('data/processed/backtest_weights_realistic.parquet')


if __name__ == '__main__':
    main()
