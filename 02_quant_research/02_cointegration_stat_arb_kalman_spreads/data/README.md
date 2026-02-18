# Data Directory

## Structure

- `raw/`: Raw price data downloaded from data sources
- `processed/`: Cleaned, aligned, and preprocessed data ready for analysis

## Data Sources

Primary data source: Yahoo Finance via yfinance library

## Data Quality Checks

All data undergoes the following preprocessing:

1. **Missing Data Handling**
   - Forward fill for gaps < 5 days
   - Remove securities with > 5% missing data
   - Align all series to common trading dates

2. **Corporate Actions**
   - Use adjusted close prices to account for splits and dividends
   - Verify adjustment factors are applied consistently

3. **Outlier Detection**
   - Flag returns > 5 standard deviations
   - Manual review of flagged observations
   - Remove or adjust obvious data errors

4. **Alignment**
   - All series aligned to common date index
   - Handle timezone issues for international securities
   - Ensure consistent frequency (daily EOD)

## Data Format

Processed data saved as:
- CSV format for human readability
- Parquet format for efficient storage and loading
- Index: datetime
- Columns: ticker symbols
- Values: adjusted close prices

## Update Frequency

Data should be refreshed:
- Daily for live trading
- Weekly for research and backtesting
- Full historical refresh quarterly

## Data Retention

- Raw data: Keep all historical data
- Processed data: Regenerate as needed from raw data
- Results: Archive after each major analysis

## Notes

- Always verify data quality before running analysis
- Check for stock splits, mergers, delistings
- Be aware of survivorship bias in historical data
- Consider using point-in-time data for production systems
