# Architecture

```mermaid
flowchart LR
    N0["Adjusted prices"] --> N1["Standardized returns"]
    N1["Standardized returns"] --> N2["PCA + classical factors"]
    N2["PCA + classical factors"] --> N3["Regime diagnostics"]
    N3["Regime diagnostics"] --> N4["Research/deploy gates"]
    N4["Research/deploy gates"]
```

## Claim boundary

Real public market data; research backtest, not live capital.
