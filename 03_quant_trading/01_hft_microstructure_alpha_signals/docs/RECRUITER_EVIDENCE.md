# Recruiter Evidence

## Project

Tick-Level HFT Signal Research

## Data provenance

FI-2010 LOB observations, Binance trades, and live top-of-book snapshots.

## Truth boundary

Real data and live snapshots; execution and post-cost fills remain modeled.

## Primary evidence

- `reports/FI2010_REAL_LOB_BENCHMARK.md`
- `reports/REAL_BINANCE_TRADE_BENCHMARK.md`
- `reports/execution_reality_pack.md`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
