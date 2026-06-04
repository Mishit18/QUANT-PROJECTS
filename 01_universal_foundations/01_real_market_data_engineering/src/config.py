# src/config.py
"""
Central configuration for the market data engineering pipeline.

All paths, parameters, and constants are defined here so every script can be
run either from the project root or directly from ``src/``.
"""
from pathlib import Path
from dataclasses import dataclass

# ============================================================================
# PATHS
# ============================================================================
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
DAILY_NOTES_DIR = REPORTS_DIR / "daily_notes"

# Ensure directories exist
for d in [RAW_DIR, CLEANED_DIR, FEATURES_DIR, PLOTS_DIR, DAILY_NOTES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
RAW_CSV = RAW_DIR / "INDIA_VIX_minute.csv"
LOADED_PARQUET = CLEANED_DIR / "INDIA_VIX_loaded.parquet"
CLEANED_PARQUET = CLEANED_DIR / "INDIA_VIX_cleaned.parquet"
STATIONARY_PARQUET = CLEANED_DIR / "INDIA_VIX_stationary.parquet"
FEATURES_PARQUET = FEATURES_DIR / "INDIA_VIX_features.parquet"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MARKET_TIMEZONE = "Asia/Kolkata"

# ============================================================================
# CLEANING PARAMETERS
# ============================================================================
@dataclass
class CleaningConfig:
    """Parameters for data cleaning pipeline"""
    # Hampel filter
    price_window: int = 6
    returns_window: int = 10
    price_nsigma: float = 6.0
    returns_nsigma: float = 8.0
    
    # Imputation
    ffill_limit: int = 3
    bfill_limit: int = 1
    
    # Volume filtering
    min_volume_threshold: float = 0.0
    
    # Price bounds (optional)
    min_price: float = 0.0
    max_price: float = 1e6
    
    # Guard against false positives when local MAD is zero on unchanged ticks.
    min_robust_scale: float = 1e-10

# ============================================================================
# STATIONARITY TEST PARAMETERS
# ============================================================================
@dataclass
class StationarityConfig:
    """Parameters for stationarity testing"""
    adf_maxlag: int = 20
    kpss_nlags: int = 20
    ljung_box_lags: int = 20
    arch_lags: int = 10
    significance_level: float = 0.05
    max_observations: int = 250_000

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
@dataclass
class FeatureConfig:
    """Parameters for feature engineering"""
    # Rolling windows (in minutes for minute data)
    short_windows: list = None
    medium_windows: list = None
    long_windows: list = None
    
    # Volatility estimators
    vol_windows: list = None
    
    # Microstructure
    ofi_window: int = 20
    vpin_window: int = 50
    vpin_buckets: int = 50
    
    # Momentum
    momentum_windows: list = None
    
    # Supervised research targets
    target_horizons: list = None
    
    def __post_init__(self):
        if self.short_windows is None:
            self.short_windows = [5, 10, 15, 30]
        if self.medium_windows is None:
            self.medium_windows = [60, 120, 240]
        if self.long_windows is None:
            self.long_windows = [480, 960, 1440]
        if self.vol_windows is None:
            self.vol_windows = [10, 30, 60, 240]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 15, 60, 240]
        if self.target_horizons is None:
            self.target_horizons = [1, 5, 15, 30, 60]

# ============================================================================
# DRIFT REMOVAL PARAMETERS
# ============================================================================
@dataclass
class DriftConfig:
    """Parameters for drift removal and HP filter"""
    hp_lambda: float = 100000.0
    detrend_method: str = "hp"  # 'hp', 'linear', 'rolling'
    rolling_window: int = 1440  # 1 day for minute data

# ============================================================================
# BACKTESTING PARAMETERS
# ============================================================================
@dataclass
class BacktestConfig:
    """Parameters for backtesting engine"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 10 bps
    slippage_bps: float = 5.0
    execution_lag: int = 1
    
    # Walk-forward validation
    train_period_days: int = 60
    test_period_days: int = 20
    step_days: int = 10
    
    # Risk management
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    
    # Demo signal controls
    momentum_lookback: int = 60
    signal_vol_window: int = 240
    signal_threshold: float = 4.0

# ============================================================================
# INSTANTIATE CONFIGS
# ============================================================================
cleaning_config = CleaningConfig()
stationarity_config = StationarityConfig()
feature_config = FeatureConfig()
drift_config = DriftConfig()
backtest_config = BacktestConfig()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
