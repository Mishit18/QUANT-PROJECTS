# FI-2010 Real LOB Benchmark

This experiment replaces the primary synthetic LOB benchmark with the real FI-2010 academic dataset: five Nasdaq Nordic stocks, ten levels, and ten trading days. The archive contains **394,337 observations**. Only the 40 price/volume book variables and derived state features are used; the authors' engineered features are excluded to reduce leakage risk.

## Chronological protocol

- Train: days 1-7 (254,750 observations)
- Validation: day 8 (55,478 observations)
- Untouched holdout: days 9-10 (84,109 observations)
- Target: published 10-event mid-price direction label

## Holdout results

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority baseline | 0.6898 | 0.3333 | 0.2721 |
| XGBoost | 0.4885 | 0.4684 | 0.4191 |

## Interpretation and limits

FI-2010 is real historical equity LOB data, but it is an academic benchmark from June 2010 and is decimal-precision normalized. The classification result is not a PnL claim. Queue position, fees, latency, hidden liquidity, and current-market transfer still require separate evaluation.
