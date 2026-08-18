# Recruiter Evidence

## Project

Options Pricing and Heston Volatility Engine

## Data provenance

Analytical Black-Scholes limits and controlled Heston parameter cases.

## Truth boundary

Numerical validation and local performance benchmark; no live options feed.

## Primary evidence

- `README.md`
- `docs/SDE_SCREENING_PACK.md`
- `tests`

## One-command verification

```bash
cmake -S . -B build && cmake --build build --config Release && ctest --test-dir build -C Release --output-on-failure
```

This command is the clean reproducibility gate for code and invariants. Expensive training or data-refresh commands remain in the README so verification does not silently trigger a multi-hour run.

## Full evidence reproduction

1. Create the environment from the committed lockfile or dependency specification.
2. Run the data or training command documented in the README when compute and data access permit.
3. Compare regenerated outputs with the primary evidence above.
4. Preserve the exact config, seed, dataset version, and hardware notes with the regenerated report.

A committed report is evidence of a recorded run, not proof that every reviewer has reproduced it independently.
