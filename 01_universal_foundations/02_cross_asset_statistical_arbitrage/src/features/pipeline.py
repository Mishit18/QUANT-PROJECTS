import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import yaml

from .price import compute_returns_features, compute_price_levels, compute_moving_averages
from .momentum import compute_momentum, compute_acceleration, compute_reversal, compute_trend_strength
from .volatility import compute_realized_vol, compute_parkinson_vol, compute_vol_of_vol, compute_downside_vol
from .volume import compute_volume_features, compute_dollar_volume, compute_amihud_illiquidity
from .cross_sectional import apply_cross_sectional_transforms


class FeaturePipeline:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.features = {}
        
    def compute_all_features(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                            ohlc: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Compute full feature set."""
        returns = prices.pct_change(fill_method=None)
        windows = self.config['features']['lookback_windows']
        vol_windows = self.config['features']['volatility_windows']
        
        self.features.update(compute_returns_features(prices, windows))
        self.features.update(compute_price_levels(prices, windows))
        self.features.update(compute_moving_averages(prices, windows))
        
        self.features.update(compute_momentum(prices, windows))
        self.features.update(compute_acceleration(prices, windows))
        self.features.update(compute_reversal(prices, windows[:4]))
        
        self.features.update(compute_realized_vol(returns, vol_windows))
        self.features.update(compute_downside_vol(returns, vol_windows))
        vol_of_vol = compute_vol_of_vol(returns)
        self.features['vol_of_vol'] = vol_of_vol
        
        if ohlc:
            self.features.update(compute_parkinson_vol(ohlc['high'], ohlc['low'], vol_windows))
        
        self.features.update(compute_volume_features(volumes, vol_windows))
        self.features.update(compute_dollar_volume(prices, volumes, vol_windows))
        self.features.update(compute_amihud_illiquidity(returns, volumes, vol_windows))
        
        if self.config['features']['cross_sectional']:
            cs_features = apply_cross_sectional_transforms(
                {k: v for k, v in list(self.features.items())[:20]}
            )
            self.features.update(cs_features)
        
        return self._stack_features()
    
    def _stack_features(self) -> pd.DataFrame:
        """Stack all features into single wide dataframe."""
        stacked = []
        for name, df in self.features.items():
            if isinstance(df, pd.Series):
                df = df.to_frame(name)
            
            if isinstance(df, pd.DataFrame):
                if df.index.nlevels == 1:
                    df_stacked = df.stack()
                    df_stacked.name = name
                    stacked.append(df_stacked)
                else:
                    df.name = name
                    stacked.append(df)
        
        result = pd.concat(stacked, axis=1)
        return result
    
    def save_features(self, output_path: Path):
        """Save computed features."""
        feature_df = self._stack_features()
        feature_df.to_parquet(output_path)
