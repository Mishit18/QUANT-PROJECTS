# Quant Research + Trading Resume Defense

## Portfolio Story

This resume should be defended as a quant research and trading portfolio, not as a software-engineering resume. The common thread is disciplined signal validation, market microstructure, execution risk, and derivatives/risk modeling.

## Project Buckets

1. **Trading and execution**: Inventory-aware market making, tick-level HFT signals, offline-RL optimal execution, and Hawkes order flow.
2. **Research and alpha validation**: Equity factor modeling, cross-asset alpha validation, and Kalman cointegration pairs.
3. **Risk and derivatives support**: Credit-risk scorecard, production ML monitoring, and Heston/options pricing.

## Strongest Interview Anchors

- Evore Labs: internal research tool prototype for preliminary alpha signal screening, 90 candidate signals, 500 variants, estimated 45% reporting-effort reduction.
- Market making: Avellaneda-Stoikov/HJB framing, inventory penalty, fill-rate diagnostics, toxic-flow sensitivity, and self-financing PnL attribution.
- HFT research: order-flow imbalance, queue imbalance, microprice, spread/depth, expected-value filters, transaction costs, slippage, and adverse selection.
- Options pricing: Heston stochastic volatility, Monte Carlo, PDE limitations, Greeks, implied volatility, volatility surface, and calibration caveats.

## What To Say If Asked About Bloomberg / Refinitiv / KDB+

Do not claim paid terminal access unless you actually had it. The defensible framing is:

"I have not used Bloomberg Terminal or Refinitiv Eikon in a professional seat. For open-source work I used Python-based market-data and research tooling, including OpenBB-style workflows, pandas, NumPy, SciPy, statsmodels, and Jupyter. If the role requires Bloomberg/Eikon, I can learn the interface quickly because the underlying data concepts are already familiar."

## What Not To Overclaim

- Do not claim live trading, deployed alpha, proprietary asset universes, client PnL, or production execution systems.
- Do not mention WorldQuant BRAIN in campus resumes because placement guidance disallows it.
- Do not present synthetic/project data as exchange-colocated or institutional trading data.
- Do not claim Bloomberg Terminal, Refinitiv Eikon, KDB+, or production HFT stack experience unless directly asked and truly trained.

## Likely Cross-Checks

**Why so many projects?**  
The projects cover distinct parts of a quant workflow: signal research, robustness validation, execution, market making, and derivatives/risk. They are not all meant to be production systems.

**What is your best quant project?**  
For trading roles, lead with market making or optimal execution. For quant research roles, lead with Evore Labs, cross-asset validation, or factor modeling. For derivatives/risk roles, lead with Heston/options pricing and credit-risk modeling.

**What is the most honest failure result?**  
Cross-asset alpha validation: a high raw classifier accuracy turned out to be class imbalance, and the signal failed after costs. This is a strong research story because rejecting weak signals is realistic quant work.
