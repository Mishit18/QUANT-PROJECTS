# src/feature_engineering.py
"""
Comprehensive feature engineering for market data.
Implements 40-80 features including:
- Volatility estimators (Parkinson, Garman-Klass, Yang-Zhang, Rogers-Satchell)
- Rolling statistics (mean, std, skew, kurtosis)
- Microstructure features (OFI, VPIN, volume imbalance)
- Momentum and mean-reversion signals
- Autocorrelation features
- Seasonality features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List

from config import feature_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for market data.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_names = []
        self.target_names = []
    
    def _volume_is_informative(self) -> bool:
        if 'volume' not in self.df.columns:
            return False
        volume = pd.to_numeric(self.df['volume'], errors='coerce').fillna(0.0)
        return bool(volume.abs().sum() > 0 and volume.nunique(dropna=True) > 1)
    
    def add_returns(self) -> 'FeatureEngineer':
        """Add various return measures"""
        if 'close' in self.df.columns:
            close = self.df['close'].where(self.df['close'] > 0)
            self.df['log_return'] = np.log(close).diff()
            self.df['simple_return'] = self.df['close'].pct_change()
            self.df['abs_return'] = self.df['log_return'].abs()
            self.df['squared_return'] = self.df['log_return'] ** 2
            
            self.feature_names.extend(['log_return', 'simple_return', 'abs_return', 'squared_return'])
            logger.info("Added return features")
        
        return self
    
    def add_rolling_statistics(self, windows: List[int] = [5, 15, 60, 240]) -> 'FeatureEngineer':
        """
        Add rolling mean, std, skew, kurtosis for returns.
        """
        if 'log_return' not in self.df.columns:
            self.add_returns()
        
        for window in windows:
            # Rolling mean
            self.df[f'return_mean_{window}'] = self.df['log_return'].rolling(window).mean()
            
            # Rolling std (realized volatility)
            self.df[f'return_std_{window}'] = self.df['log_return'].rolling(window).std()
            
            # Rolling skewness
            self.df[f'return_skew_{window}'] = self.df['log_return'].rolling(window).skew()
            
            # Rolling kurtosis
            self.df[f'return_kurt_{window}'] = self.df['log_return'].rolling(window).kurt()
            
            self.feature_names.extend([
                f'return_mean_{window}', f'return_std_{window}',
                f'return_skew_{window}', f'return_kurt_{window}'
            ])
        
        logger.info(f"Added rolling statistics for windows: {windows}")
        return self
    
    def add_volatility_estimators(self, windows: List[int] = [10, 30, 60]) -> 'FeatureEngineer':
        """
        Add advanced volatility estimators that use OHLC data.
        """
        required = ['open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required):
            logger.warning("OHLC columns not available, skipping volatility estimators")
            return self
        
        safe_ohlc = self.df[required].where(self.df[required] > 0)
        
        for window in windows:
            # Parkinson volatility (uses high-low range)
            hl_ratio = np.log(safe_ohlc['high'] / safe_ohlc['low'])
            self.df[f'parkinson_vol_{window}'] = np.sqrt(
                ((hl_ratio ** 2).rolling(window).mean() / (4 * np.log(2))).clip(lower=0)
            )
            
            # Garman-Klass volatility
            hl = np.log(safe_ohlc['high'] / safe_ohlc['low']) ** 2
            co = np.log(safe_ohlc['close'] / safe_ohlc['open']) ** 2
            self.df[f'garman_klass_vol_{window}'] = np.sqrt(
                (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean().clip(lower=0)
            )
            
            # Rogers-Satchell volatility
            ho = np.log(safe_ohlc['high'] / safe_ohlc['open'])
            hc = np.log(safe_ohlc['high'] / safe_ohlc['close'])
            lo = np.log(safe_ohlc['low'] / safe_ohlc['open'])
            lc = np.log(safe_ohlc['low'] / safe_ohlc['close'])
            
            self.df[f'rogers_satchell_vol_{window}'] = np.sqrt(
                (ho * hc + lo * lc).rolling(window).mean().clip(lower=0)
            )
            
            # Yang-Zhang volatility (combines overnight and intraday)
            oc = np.log(safe_ohlc['open'] / safe_ohlc['close'].shift(1))
            self.df[f'yang_zhang_vol_{window}'] = np.sqrt(
                (oc.rolling(window).var() + 
                self.df[f'rogers_satchell_vol_{window}'] ** 2).clip(lower=0)
            )
            
            self.feature_names.extend([
                f'parkinson_vol_{window}', f'garman_klass_vol_{window}',
                f'rogers_satchell_vol_{window}', f'yang_zhang_vol_{window}'
            ])
        
        logger.info(f"Added volatility estimators for windows: {windows}")
        return self
    
    def add_momentum_features(self, windows: List[int] = [5, 15, 60, 240]) -> 'FeatureEngineer':
        """
        Add momentum and mean-reversion signals.
        """
        if 'close' not in self.df.columns:
            return self
        
        for window in windows:
            # Price momentum (return over window)
            self.df[f'momentum_{window}'] = self.df['close'].pct_change(window)
            
            # Distance from moving average (mean reversion signal)
            ma = self.df['close'].rolling(window).mean()
            self.df[f'dist_from_ma_{window}'] = (self.df['close'] - ma) / ma
            
            # RSI-like momentum
            returns = self.df['close'].diff()
            gains = returns.where(returns > 0, 0).rolling(window).mean()
            losses = -returns.where(returns < 0, 0).rolling(window).mean()
            self.df[f'rsi_{window}'] = 100 - (100 / (1 + gains / (losses + 1e-10)))
            
            self.feature_names.extend([
                f'momentum_{window}', f'dist_from_ma_{window}', f'rsi_{window}'
            ])
        
        logger.info(f"Added momentum features for windows: {windows}")
        return self
    
    def add_autocorrelation_features(self, lags: List[int] = [1, 5, 10, 20]) -> 'FeatureEngineer':
        """
        Add lagged returns and autocorrelation features.
        """
        if 'log_return' not in self.df.columns:
            self.add_returns()
        
        for lag in lags:
            # Lagged returns
            self.df[f'return_lag_{lag}'] = self.df['log_return'].shift(lag)
            
            # Rolling autocorrelation
            self.df[f'return_autocorr_{lag}'] = self.df['log_return'].rolling(60).corr(
                self.df['log_return'].shift(lag)
            )
            
            self.feature_names.extend([f'return_lag_{lag}', f'return_autocorr_{lag}'])
        
        logger.info(f"Added autocorrelation features for lags: {lags}")
        return self
    
    def add_volume_features(self, windows: List[int] = [10, 30, 60]) -> 'FeatureEngineer':
        """
        Add volume-based features.
        """
        if not self._volume_is_informative():
            logger.warning("Volume is missing or constant; skipping volume-derived features")
            return self
        
        for window in windows:
            # Volume moving average
            self.df[f'volume_ma_{window}'] = self.df['volume'].rolling(window).mean()
            
            # Volume ratio (current vs average)
            self.df[f'volume_ratio_{window}'] = (
                self.df['volume'] / self.df[f'volume_ma_{window}'].replace(0, np.nan)
            )
            
            # Volume-weighted price
            self.df[f'vwap_{window}'] = (
                (self.df['close'] * self.df['volume']).rolling(window).sum() /
                self.df['volume'].rolling(window).sum().replace(0, np.nan)
            )
            
            self.feature_names.extend([
                f'volume_ma_{window}', f'volume_ratio_{window}', f'vwap_{window}'
            ])
        
        logger.info(f"Added volume features for windows: {windows}")
        return self
    
    def add_microstructure_features(self, ofi_window: int = 20) -> 'FeatureEngineer':
        """
        Add microstructure features.
        Simplified versions (full implementation requires order book data).
        """
        volume_ok = self._volume_is_informative()
        
        # Order Flow Imbalance proxy (using volume and price direction)
        if volume_ok and 'close' in self.df.columns:
            price_direction = np.sign(self.df['close'].diff())
            self.df['ofi_proxy'] = (price_direction * self.df['volume']).rolling(ofi_window).sum()
            self.feature_names.append('ofi_proxy')
        
        # Volume imbalance (buy vs sell pressure proxy)
        if volume_ok and all(col in self.df.columns for col in ['high', 'low', 'close']):
            # Proxy: if close near high, more buying; near low, more selling
            range_hl = self.df['high'] - self.df['low']
            close_position = (self.df['close'] - self.df['low']) / (range_hl + 1e-10)
            self.df['volume_imbalance'] = (
                (close_position - 0.5) * self.df['volume']
            ).rolling(ofi_window).sum()
            self.feature_names.append('volume_imbalance')
        
        # Effective spread proxy
        if all(col in self.df.columns for col in ['high', 'low']):
            self.df['effective_spread'] = (self.df['high'] - self.df['low']) / self.df['close']
            self.feature_names.append('effective_spread')
        
        logger.info("Added microstructure features")
        return self
    
    def add_seasonality_features(self) -> 'FeatureEngineer':
        """
        Add time-based seasonality features.
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping seasonality features")
            return self
        
        # Hour of day
        self.df['hour'] = self.df.index.hour
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Day of week
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # Minute of hour
        self.df['minute'] = self.df.index.minute
        
        self.feature_names.extend([
            'hour', 'hour_sin', 'hour_cos',
            'day_of_week', 'dow_sin', 'dow_cos', 'minute'
        ])
        
        logger.info("Added seasonality features")
        return self
    
    def add_regime_features(self, vol_window: int = 60, vol_threshold: float = 2.0) -> 'FeatureEngineer':
        """
        Add regime detection features (high vol vs low vol).
        """
        if 'log_return' not in self.df.columns:
            self.add_returns()
        
        # Rolling volatility
        rolling_vol = self.df['log_return'].rolling(vol_window).std()
        vol_ma = rolling_vol.rolling(vol_window).mean()
        vol_std = rolling_vol.rolling(vol_window).std()
        
        # High volatility regime (vol > mean + threshold * std)
        self.df['high_vol_regime'] = (
            rolling_vol > (vol_ma + vol_threshold * vol_std)
        ).astype(int)
        
        # Volatility z-score
        self.df['vol_zscore'] = (rolling_vol - vol_ma) / (vol_std + 1e-10)
        
        self.feature_names.extend(['high_vol_regime', 'vol_zscore'])
        
        logger.info("Added regime features")
        return self
    
    def add_forward_targets(self, horizons: List[int] = None) -> 'FeatureEngineer':
        """
        Add leakage-safe forward returns for downstream alpha research.
        
        These columns are targets, not inputs, and are intentionally excluded
        from ``feature_names``.
        """
        if horizons is None:
            horizons = feature_config.target_horizons
        if 'close' not in self.df.columns:
            return self
        
        close = self.df['close'].where(self.df['close'] > 0)
        log_close = np.log(close)
        for horizon in horizons:
            target = log_close.shift(-horizon) - log_close
            direction = np.sign(target).replace(0, np.nan)
            target_col = f'target_fwd_log_return_{horizon}'
            direction_col = f'target_fwd_direction_{horizon}'
            self.df[target_col] = target
            self.df[direction_col] = direction
            self.target_names.extend([target_col, direction_col])
        
        logger.info(f"Added forward targets for horizons: {horizons}")
        return self
    
    def build_all_features(self) -> pd.DataFrame:
        """
        Build complete feature set.
        """
        logger.info("\n" + "="*60)
        logger.info("Building comprehensive feature set")
        logger.info("="*60)
        
        self.add_returns()
        self.add_rolling_statistics(windows=feature_config.short_windows + feature_config.medium_windows)
        self.add_volatility_estimators(windows=feature_config.vol_windows)
        self.add_momentum_features(windows=feature_config.momentum_windows)
        self.add_autocorrelation_features(lags=[1, 5, 10, 20])
        self.add_volume_features(windows=[10, 30, 60])
        self.add_microstructure_features(ofi_window=20)
        self.add_seasonality_features()
        self.add_regime_features(vol_window=60)
        self.add_forward_targets(horizons=feature_config.target_horizons)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"\nTotal features created: {len(self.feature_names)}")
        logger.info(f"Total targets created: {len(self.target_names)}")
        logger.info(f"DataFrame shape: {self.df.shape}")
        logger.info("="*60 + "\n")
        
        return self.df


if __name__ == "__main__":
    from config import STATIONARY_PARQUET, FEATURES_PARQUET
    
    logger.info("Loading stationary data...")
    df = pd.read_parquet(STATIONARY_PARQUET)
    
    # Build features
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features()
    
    # Save
    df_features.to_parquet(FEATURES_PARQUET)
    logger.info(f"Saved features to {FEATURES_PARQUET}")
    
    # Print feature summary
    print("\nFeature Summary:")
    print(f"Total columns: {len(df_features.columns)}")
    print(f"Feature names: {len(engineer.feature_names)}")
    print("\nSample features:")
    for feat in engineer.feature_names[:20]:
        print(f"  - {feat}")
