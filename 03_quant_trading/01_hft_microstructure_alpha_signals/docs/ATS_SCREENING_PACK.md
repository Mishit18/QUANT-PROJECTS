# ATS Screening Pack

## Best-Fit Roles

This project is strongest for:

- Quant Trader
- HFT Research
- Market Microstructure Research
- Execution Research
- Quant Research roles that value trading realism

It should be used as the second Quant Trader anchor after the Avellaneda-Stoikov project.

## ATS Keyword Coverage

| Area | Keywords |
|---|---|
| Microstructure | order flow imbalance, queue imbalance, microprice, spread dynamics, limit order book |
| Modeling | XGBoost, logistic regression baseline, walk-forward validation, alpha decay, feature importance |
| Execution | event-driven backtesting, market orders, limit orders, EV filter, queue-aware execution |
| Trading Risk | transaction costs, adverse selection, fill rate, latency, passive fills |
| Interview Signal | real FI-2010 benchmark, chronological holdout, class imbalance, cost-aware signal rejection |

## Resume Bullets

- Built 57 LOB features on 394,337 real FI-2010 observations across five Nasdaq Nordic stocks, ten levels, and ten trading days.
- Used a chronological days 1-7/day 8/days 9-10 split; class-balanced XGBoost reached 0.468 holdout balanced accuracy and 0.419 macro-F1.
- Added event-driven execution tests showing market-order PnL -$58.47 and limit-filtered PnL -$3.30, proving that spread costs and adverse selection can erase high predictive accuracy.
- Implemented EV-based execution filter rejecting 94.6% of candidate signals as uneconomic after costs; final +$0.41 result came from only 1 trade and is treated as rejection discipline, not a robust profit claim.

## 30-Second Interview Pitch

I rebuilt the predictive benchmark on 394,337 real FI-2010 ten-level snapshots. A chronological holdout and class-balanced objective improved balanced accuracy from a 0.333 majority baseline to 0.468. I keep the old synthetic simulator only as a controlled execution-friction study because FI-2010 does not contain queue position, fees, or fill outcomes.

## Interview Defense

### Why is one profitable EV trade not enough?

One trade is not statistical evidence. The value of the EV filter is that it rejected most uneconomic signals after costs, not that it proved profitability.

### Why retain any synthetic LOB data?

The synthetic component isolates execution assumptions and is no longer the predictive benchmark. FI-2010 supplies real book states, while the simulator tests controlled spread, fill, and adverse-selection scenarios.

### What would improve the project?

Use real LOB data, calibrate fill probabilities, model latency and queue priority, include venue-specific rebates/fees, and test market-making or cross-asset execution overlays.

## Claims To Avoid

- Do not call this profitable HFT.
- Do not say the model is production-ready.
- Do not imply FI-2010 is modern, proprietary, or sufficient for live execution claims.
- Do not overstate the +$0.41 result.
- Do not say 90.8% accuracy is enough to trade.
