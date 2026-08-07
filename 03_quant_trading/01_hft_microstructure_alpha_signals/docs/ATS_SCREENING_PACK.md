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
| Interview Signal | accuracy does not imply profitability, cost-aware signal rejection, synthetic data caveat |

## Resume Bullets

- Built a 59-feature synthetic LOB microstructure pipeline using OFI, queue imbalance, microprice, spread dynamics, and event-time features; XGBoost reached 90.8% 5-tick directional accuracy versus 62.5% logistic baseline.
- Added event-driven execution tests showing market-order PnL -$58.47 and limit-filtered PnL -$3.30, proving that spread costs and adverse selection can erase high predictive accuracy.
- Implemented EV-based execution filter rejecting 94.6% of candidate signals as uneconomic after costs; final +$0.41 result came from only 1 trade and is treated as rejection discipline, not a robust profit claim.

## 30-Second Interview Pitch

I built a synthetic limit order book microstructure project to test whether short-horizon price prediction survives realistic execution. The XGBoost model reached 90.8% 5-tick accuracy, but market orders lost money because crossing the spread consumed the edge. Limit orders helped but still lost money due to low fills and adverse selection. The EV filter rejected 94.6% of signals and left only one trade. The main lesson is that HFT needs execution realism; accuracy alone is not a trading edge.

## Interview Defense

### Why is one profitable EV trade not enough?

One trade is not statistical evidence. The value of the EV filter is that it rejected most uneconomic signals after costs, not that it proved profitability.

### Why use synthetic LOB data?

Synthetic data makes the project reproducible and shareable. The limitation is that it misses real hidden orders, queue priority, toxic flow, participant behavior, and venue-specific mechanics.

### What would improve the project?

Use real LOB data, calibrate fill probabilities, model latency and queue priority, include venue-specific rebates/fees, and test market-making or cross-asset execution overlays.

## Claims To Avoid

- Do not call this profitable HFT.
- Do not say the model is production-ready.
- Do not claim real exchange data.
- Do not overstate the +$0.41 result.
- Do not say 90.8% accuracy is enough to trade.
