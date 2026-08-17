# Deploy / Reject Memo: Cross-Asset Statistical Arbitrage

## Research Question

Can volatility-scaled cross-sectional equity targets produce a deployable alpha signal after embargoed walk-forward validation, market/sector neutralization, turnover control, and realistic transaction costs?

## Decision

**Reject for deployment.** The research framework is valid, but the signal does not survive the full acceptance gate.

## Acceptance Gates

| Gate | Result | Decision |
|---|---:|---|
| Point-in-time feature and target construction | Passed | Continue |
| Embargoed walk-forward validation | Passed | Continue |
| Mean IC positive and stable | Weak | Caution |
| IC-IR materially positive | Failed | Reject |
| Market + sector neutralization | Gross improvement | Continue |
| Transaction-cost adjusted performance | Failed | Reject |
| Turnover within realistic capacity | Weak | Caution |

## Key Metrics

| Metric | Value |
|---|---:|
| Horizon | 5 days |
| Mean valid targets/date | 76.2 |
| XGBoost mean IC | 0.0051 |
| XGBoost IC-IR | 0.05 |
| XGBoost hit rate | 52.41% |
| Market + sector neutral gross return | 43.43% |
| Market + sector neutral net return | -8.24% |
| Cost drag | 51.68% |
| Average turnover | 126.73% |
| Estimated capacity | $4.9M |

## Why The Signal Is Rejected

The signal has small positive rank correlation but weak stability. After market/sector neutralization and realistic costs, turnover consumes the edge. The research process correctly avoids turning a weak gross result into an overclaimed strategy.

## What Worked

- Corrected volatility-scaled target design.
- Added embargo to avoid overlapping-label leakage.
- Added cost accounting over the union of previous/current positions.
- Added market and sector neutralization.
- Added realistic deploy/reject gates.

## What Did Not Work

- IC was too small to support deployment.
- Turnover was high relative to signal strength.
- Cost drag exceeded gross edge.
- Synthetic data limits external validity.

## Interview Defense

If asked whether this project "failed", the answer is no. The purpose of quant research is not to force deployment; it is to reject weak signals before capital is allocated. This project demonstrates target hygiene, validation discipline, cost awareness, and research judgment.

## Resume-Safe Bullet

Built cross-sectional alpha validation pipeline with point-in-time volatility-scaled targets, embargoed walk-forward testing, market/sector neutralization, IC/IC-IR diagnostics, and transaction-cost gates; rejected signal after net return fell to -8.24% under 51.68% cost drag, demonstrating deploy/reject research discipline.
