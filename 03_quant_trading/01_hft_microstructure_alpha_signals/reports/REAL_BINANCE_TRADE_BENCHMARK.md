# Real Binance Trade-Flow Benchmark

The benchmark uses 62,691 one-second bars derived from official Binance Vision BTCUSDT aggregate trades. Data is split chronologically into 60% train, 20% validation, and 20% untouched holdout partitions.

- Holdout ROC-AUC: 0.5897
- Holdout balanced accuracy: 0.5680
- Active-signal rate: 13.1%
- Mean net return per active signal after a 1.0 bp cost: -0.973 bps

The negative post-cost result is a research rejection, not a trading-performance claim. Aggregate trades validate trade-flow features but cannot reconstruct queue position or full-depth LOB state.
