# Recruiter Evidence

## Project

Inventory-Aware Market Making

## Data provenance

Binance trades used for volatility and arrival priors.

## Truth boundary

Real calibration; all quotes, fills and P&L are simulated.

## Primary evidence

- `docs/REAL_MARKET_CALIBRATION.md`
- `docs/QUOTE_STRESS_PACK.md`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
