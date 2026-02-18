"""
Data cleaning and validation for LOB data.
Ensures data quality and handles edge cases.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LOBDataCleaner:
    """
    Cleans and validates limit order book data.
    
    Checks:
    - Monotonic timestamps
    - Valid bid/ask ordering
    - Positive sizes
    - No extreme outliers
    """
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to LOB data.
        
        Returns cleaned DataFrame with invalid rows removed or corrected.
        """
        df = df.copy()
        
        print(f"Initial rows: {len(df)}")
        
        # Ensure timestamps are sorted
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        print(f"After removing duplicates: {len(df)}")
        
        # Validate bid/ask ordering
        df = self._validate_bid_ask(df)
        print(f"After bid/ask validation: {len(df)}")
        
        # Remove rows with invalid sizes
        df = self._validate_sizes(df)
        print(f"After size validation: {len(df)}")
        
        # Remove extreme price outliers
        df = self._remove_outliers(df)
        print(f"After outlier removal: {len(df)}")
        
        # Fill any remaining NaNs in book levels with forward fill
        book_cols = [col for col in df.columns if 'price' in col or 'size' in col]
        df[book_cols] = df[book_cols].fillna(method='ffill')
        
        return df
    
    def _validate_bid_ask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure bid prices < ask prices at all levels."""
        valid_mask = pd.Series(True, index=df.index)
        
        # Check each level
        for i in range(1, 11):  # assume max 10 levels
            bid_col = f'bid_price_{i}'
            ask_col = f'ask_price_{i}'
            
            if bid_col in df.columns and ask_col in df.columns:
                # Bid must be less than ask
                valid_mask &= (df[bid_col] < df[ask_col]) | df[bid_col].isna()
        
        return df[valid_mask].reset_index(drop=True)
    
    def _validate_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all sizes are positive."""
        valid_mask = pd.Series(True, index=df.index)
        
        size_cols = [col for col in df.columns if 'size' in col and col != 'trade_size']
        
        for col in size_cols:
            if col in df.columns:
                valid_mask &= (df[col] > 0) | df[col].isna()
        
        return df[valid_mask].reset_index(drop=True)
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with extreme price movements (likely errors)."""
        # Compute mid price
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        
        # Compute returns
        df['returns'] = df['mid_price'].pct_change()
        
        # Remove extreme returns (> 5 standard deviations)
        returns_std = df['returns'].std()
        valid_mask = np.abs(df['returns']) < 5 * returns_std
        
        # Keep first row even if NaN return
        valid_mask.iloc[0] = True
        
        df = df[valid_mask].reset_index(drop=True)
        
        # Drop temporary columns
        df = df.drop(columns=['returns'], errors='ignore')
        
        return df


def clean_data(input_path: str, output_path: str, tick_size: float = 0.01):
    """Load, clean, and save LOB data."""
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print("Cleaning data...")
    cleaner = LOBDataCleaner(tick_size=tick_size)
    df_clean = cleaner.clean(df)
    
    print(f"\nSaving cleaned data to {output_path}...")
    df_clean.to_parquet(output_path, index=False)
    
    print(f"Cleaning complete. Final shape: {df_clean.shape}")
    
    return df_clean


if __name__ == "__main__":
    clean_data(
        input_path="data/raw/lob_data.parquet",
        output_path="data/raw/lob_data_clean.parquet"
    )
