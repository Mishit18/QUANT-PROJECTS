import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from .load_data import load_ohlcv, build_panel, filter_universe
from .clean_data import remove_outliers, remove_low_coverage, align_timestamps


class PanelBuilder:
    def __init__(self, data_dir: Path, config: dict):
        self.data_dir = data_dir
        self.config = config
        self.prices = None
        self.volumes = None
        self.returns = None
        
    def build(self, tickers: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build clean OHLCV panels."""
        ohlcv = load_ohlcv(self.data_dir, tickers)
        
        prices = build_panel(ohlcv, 'close')
        volumes = build_panel(ohlcv, 'volume')
        
        prices = filter_universe(
            prices, volumes,
            self.config['data']['min_price'],
            self.config['data']['min_volume']
        )
        
        prices = remove_outliers(prices)
        prices = remove_low_coverage(prices, min_coverage=0.7)
        
        volumes = volumes.loc[prices.index, prices.columns]
        
        returns = prices.pct_change(fill_method=None)
        
        self.prices, self.volumes, self.returns = align_timestamps([prices, volumes, returns])
        
        return self.prices, self.volumes, self.returns
    
    def get_ohlc(self, tickers: list) -> Dict[str, pd.DataFrame]:
        """Get full OHLC for tickers."""
        ohlcv = load_ohlcv(self.data_dir, tickers)
        return {
            'open': build_panel(ohlcv, 'open'),
            'high': build_panel(ohlcv, 'high'),
            'low': build_panel(ohlcv, 'low'),
            'close': build_panel(ohlcv, 'close')
        }
