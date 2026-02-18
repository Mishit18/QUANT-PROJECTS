"""
Configuration validation tests.
Ensures all modules can be initialized with config parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.config_validator import validate_config
from src.kalman import KalmanHedge
from src.spread_model import SpreadModel
from src.alpha_layer import AlphaLayer
from src.execution import ExecutionModel
from src.regime_filter import RegimeFilter


def test_config_validation():
    """Test config loads and validates correctly."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config = validate_config(config)
    
    assert 'kalman' in config
    assert 'ou_model' in config
    assert 'alpha' in config
    assert 'execution' in config
    
    print("[PASS] Config validation")


def test_module_initialization():
    """Test all modules initialize without errors."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config = validate_config(config)
    
    # Kalman filter
    kf = KalmanHedge(
        transition_cov=config['kalman']['transition_cov'],
        observation_cov=config['kalman']['observation_cov']
    )
    assert kf.Q > 0 and kf.R > 0
    
    # Spread model
    sm = SpreadModel(
        min_r_squared=config['ou_model']['min_r_squared'],
        min_half_life=config['ou_model']['min_half_life'],
        max_half_life=config['ou_model']['max_half_life']
    )
    assert 0 < sm.min_r_squared < 1
    
    # Alpha layer
    al = AlphaLayer(
        entry_z=config['alpha']['entry_z'],
        exit_z=config['alpha']['exit_z'],
        stop_loss_z=config['alpha']['stop_loss_z'],
        max_hold_days=config['alpha']['max_hold_days'],
        velocity_threshold=config['alpha']['velocity_threshold']
    )
    assert al.entry_z > al.exit_z
    
    # Execution model
    em = ExecutionModel(
        transaction_cost_bps=config['execution']['transaction_cost_bps'],
        slippage_bps=config['execution']['slippage_bps']
    )
    assert em.tc_bps >= 0
    
    # Regime filter
    rf = RegimeFilter(
        n_regimes=config['regimes']['n_regimes'],
        random_state=config['regimes']['random_state']
    )
    assert rf.n_regimes > 0
    
    print("[PASS] Module initialization")


if __name__ == "__main__":
    try:
        test_config_validation()
        test_module_initialization()
        print("\n[SUCCESS] All tests passed")
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
