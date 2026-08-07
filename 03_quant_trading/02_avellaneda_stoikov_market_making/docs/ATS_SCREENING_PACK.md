# ATS Screening Pack

## Best-Fit Roles

This project is strongest for:

- Quant Trader
- Market Making
- Trading Research
- HFT / Microstructure Research
- Execution Research

It should be the flagship project on a Quant Trader resume because it tests inventory risk, adverse selection, spread capture, queue position, and competitive equilibrium.

## ATS Keyword Coverage

| Area | Keywords |
|---|---|
| Market Making | Avellaneda-Stoikov, optimal quoting, reservation price, bid-ask spread, inventory risk |
| Microstructure | adverse selection, queue position, microprice, fill probability, order flow |
| Quant Methods | HJB, Monte Carlo simulation, stochastic process, risk aversion, arrival intensity |
| Trading | spread capture, PnL decomposition, competition, latency, toxic flow, failure regimes |
| Interview Signals | self-financing PnL, simulated market making, parameter sensitivity, honest limitations |

## Resume Bullets

- Derived and implemented Avellaneda-Stoikov optimal market-making quotes with reservation-price inventory adjustment and asymmetric bid/ask spreads; validated behavior through Monte Carlo simulations across market regimes.
- Built PnL attribution diagnostics decomposing returns into spread capture, inventory PnL, adverse selection, and residual attribution to explain where market-making profits and losses arise.
- Simulated microprice-based quoting, queue-position fill decay, and multi-agent competition; identified failure regimes where high volatility, low fill intensity, toxic flow, or spread compression dominate spread capture.

## 30-Second Interview Pitch

I built an Avellaneda-Stoikov market-making simulator to understand optimal quoting under inventory risk. The model shifts reservation price based on inventory, then quotes asymmetric bid/ask spreads to balance fill probability against risk. I added Monte Carlo validation, PnL decomposition, microprice, queue-position analysis, and multi-agent competition. The key lesson is that market making does not produce free alpha: spread capture is offset by inventory risk, adverse selection, and competition, so the most important part of the project is knowing exactly when the model fails.

## Interview Defense

### Why is this useful if it is simulated?

It proves microstructure intuition and risk-control thinking. The goal is not to claim a live strategy; it is to show how inventory, fill probability, spread, and adverse selection interact.

### What would make it production-grade?

Real LOB calibration, latency modeling, queue priority, hard risk limits, toxicity detection, multi-venue routing, and continuous monitoring of fill intensity.

### Why not optimize parameters for PnL?

Because that would overfit the simulator. The project is stronger when parameters are fixed and the analysis focuses on regime behavior and failure modes.

## Claims To Avoid

- Do not say it is live or production trading.
- Do not claim real exchange calibration.
- Do not overstate simulated Sharpe; the default baseline mean Sharpe is 0.025 across 500 Monte Carlo paths.
- Do not hide adverse selection and competition.
- Do not present this as alpha generation; it is market-making mechanics and risk control.
