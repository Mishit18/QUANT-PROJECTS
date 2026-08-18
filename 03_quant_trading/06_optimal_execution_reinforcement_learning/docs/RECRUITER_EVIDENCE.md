# Recruiter Evidence

## Project

Optimal Execution: Offline RL vs Almgren-Chriss

## Data provenance

Binance-derived volatility/intensity/size priors.

## Truth boundary

Real calibration; execution environment, fills and shortfall are simulated.

## Primary evidence

- `docs/REAL_MARKET_CALIBRATION.md`
- `docs/ATS_SCREENING_PACK.md`

## One-command verification

```bash
python -m pytest -q
```

This command is the clean reproducibility gate for code and invariants. Expensive training or data-refresh commands remain in the README so verification does not silently trigger a multi-hour run.

## Full evidence reproduction

1. Create the environment from the committed lockfile or dependency specification.
2. Run the data or training command documented in the README when compute and data access permit.
3. Compare regenerated outputs with the primary evidence above.
4. Preserve the exact config, seed, dataset version, and hardware notes with the regenerated report.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
