# Data Directory

This directory is reserved for:

- Historical order book data (LOBSTER, NASDAQ ITCH format)
- Calibrated Hawkes parameters from real market data
- Simulation outputs and results
- Cached estimation results

## Expected Data Format

If using real data for calibration, place CSV files with columns:
- `timestamp`: Event time (seconds or milliseconds)
- `event_type`: Integer code (0=limit_buy, 1=limit_sell, 2=market_buy, 3=market_sell, 4=cancel_buy, 5=cancel_sell)
- `price`: Order price (optional, for LOB reconstruction)
- `volume`: Order size (optional)

## Synthetic Data

The simulation generates synthetic data automatically. No external data is required to run the project.
