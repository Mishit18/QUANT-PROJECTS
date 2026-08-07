# ATS Screening Pack: Hawkes Process Market Making

## Best-Fit Resume Profiles

- Quant Research
- Quant Trader
- Market Microstructure Research
- HFT Research
- Execution Research

## Recruiter Summary

Built a Hawkes-process limit order book simulator with six event types, stationarity checks, time-rescaling diagnostics, MLE parameter estimation, price-time priority matching, inventory-aware market making, passive fill attribution, and event-driven backtesting.

## ATS Keyword Coverage

Hawkes process, multidimensional Hawkes, limit order book, market microstructure, high-frequency trading, HFT, self-excitation, cross-excitation, order flow, Ogata thinning, exponential kernel, spectral radius, stationarity, time-rescaling theorem, Kolmogorov-Smirnov test, maximum likelihood estimation, MLE, price-time priority, market making, inventory control, reservation price, spread capture, passive fills, backtesting, Python, NumPy, SciPy, Matplotlib, pandas.

## Quant Research Resume Bullets

- Implemented six-type multidimensional Hawkes order-flow simulator for limit, market, and cancel events, validating stationarity via excitation-matrix spectral radius.
- Built residual diagnostics using the time-rescaling theorem and KS tests, identifying mixed goodness-of-fit across simulated limit, market, and cancel event streams.
- Added event-driven limit order book with price-time priority and passive order IDs, enabling fill attribution between market-order flow and resting agent quotes.
- Verified default run with 3,805 simulated events, spectral radius 0.5732, 96 market-maker fills, 1.72 tick spread capture, and explicit model-risk documentation.

## Quant Trader Resume Bullets

- Built market-making simulation framework with Hawkes-generated order flow, LOB matching, inventory-aware quote skew, passive fill detection, and PnL/inventory tracking.
- Fixed backtest fill attribution so resting agent quotes update cash and inventory when hit, converting the project from a no-fill prototype into a functioning simulation backtest.
- Evaluated simulated market-maker behavior with final PnL 8.30, 96 fills, 0.55 max drawdown, 0.096 trades/sec, and final inventory -40 under default configuration.
- Documented limitations around synthetic order flow, simplified latency, no real queue-position calibration, and mixed residual diagnostics.

## Strong Interview Defense

The strongest explanation is:

1. Hawkes processes model clustered order arrivals better than independent Poisson arrivals.
2. Stationarity matters; if the excitation spectral radius exceeds one, intensities can explode.
3. Time-rescaling residual diagnostics test whether the fitted point process is plausible.
4. Market-making simulation needs passive fill attribution; otherwise PnL and inventory are fake.
5. Synthetic validation is useful, but real LOB data is required before production trading claims.

## Claims To Avoid

- Do not claim live HFT readiness.
- Do not claim real-market profitability.
- Do not claim the KS diagnostics all pass; the verified run has mixed p-values.
- Do not claim calibrated market data unless real LOB data is added.

## Upgrade Path For 100/100 Screening

- Add LOBSTER or NASDAQ ITCH data loader.
- Calibrate marked Hawkes process with order sizes.
- Add queue-position and latency simulation.
- Add adverse-selection metric using post-fill mid-price movement.
- Compare exponential, power-law, and sum-of-exponentials kernels.
