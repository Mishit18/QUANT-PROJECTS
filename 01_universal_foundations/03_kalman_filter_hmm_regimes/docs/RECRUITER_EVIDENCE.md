# Recruiter Evidence

## Project

Regime-Aware State-Space Research

## Data provenance

SPY, QQQ and TLT daily prices.

## Truth boundary

Real public data with modeled costs; strategy is rejected against passive benchmark.

## Primary evidence

- `docs/RESULTS_AND_LIMITATIONS.md`
- `docs/LIMITATIONS_AND_FAILURE_MODES.md`
- `docs/SYSTEM_ARCHITECTURE.md`

## Verification

1. Create the environment using the repository dependency specification.
2. Run the test command documented in the README.
3. Run the evidence-generation command documented in the README when compute and data access permit.
4. Compare regenerated outputs with the primary evidence above.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
