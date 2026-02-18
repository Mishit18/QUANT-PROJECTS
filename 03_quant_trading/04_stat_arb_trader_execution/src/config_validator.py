"""
Configuration validation - production hygiene.

Ensures config.yaml values are properly typed before use.
Prevents silent failures from YAML parsing quirks.
"""

from typing import Any, Dict


def validate_numeric(value: Any, name: str, min_val: float = None, max_val: float = None) -> float:
    """
    Validate and cast numeric config value.
    
    Args:
        value: Config value to validate
        name: Parameter name (for error messages)
        min_val: Optional minimum value
        max_val: Optional maximum value
    
    Returns:
        Validated float value
    
    Raises:
        ValueError: If validation fails
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Config parameter '{name}' must be numeric. "
            f"Got: {value} (type: {type(value).__name__})"
        )
    
    if min_val is not None and numeric_value < min_val:
        raise ValueError(
            f"Config parameter '{name}' must be >= {min_val}. "
            f"Got: {numeric_value}"
        )
    
    if max_val is not None and numeric_value > max_val:
        raise ValueError(
            f"Config parameter '{name}' must be <= {max_val}. "
            f"Got: {numeric_value}"
        )
    
    return numeric_value


def validate_integer(value: Any, name: str, min_val: int = None, max_val: int = None) -> int:
    """
    Validate and cast integer config value.
    
    Args:
        value: Config value to validate
        name: Parameter name (for error messages)
        min_val: Optional minimum value
        max_val: Optional maximum value
    
    Returns:
        Validated int value
    
    Raises:
        ValueError: If validation fails
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Config parameter '{name}' must be an integer. "
            f"Got: {value} (type: {type(value).__name__})"
        )
    
    if min_val is not None and int_value < min_val:
        raise ValueError(
            f"Config parameter '{name}' must be >= {min_val}. "
            f"Got: {int_value}"
        )
    
    if max_val is not None and int_value > max_val:
        raise ValueError(
            f"Config parameter '{name}' must be <= {max_val}. "
            f"Got: {int_value}"
        )
    
    return int_value


def validate_config(config: Dict) -> Dict:
    """
    Validate entire config structure.
    
    Args:
        config: Loaded config dictionary
    
    Returns:
        Validated config with proper types
    
    Raises:
        ValueError: If any validation fails
    """
    # Validate Kalman parameters
    if 'kalman' in config:
        config['kalman']['transition_cov'] = validate_numeric(
            config['kalman']['transition_cov'],
            'kalman.transition_cov',
            min_val=0
        )
        config['kalman']['observation_cov'] = validate_numeric(
            config['kalman']['observation_cov'],
            'kalman.observation_cov',
            min_val=0
        )
    
    # Validate OU model parameters
    if 'ou_model' in config:
        config['ou_model']['min_r_squared'] = validate_numeric(
            config['ou_model']['min_r_squared'],
            'ou_model.min_r_squared',
            min_val=0,
            max_val=1
        )
        config['ou_model']['min_half_life'] = validate_numeric(
            config['ou_model']['min_half_life'],
            'ou_model.min_half_life',
            min_val=0
        )
        config['ou_model']['max_half_life'] = validate_numeric(
            config['ou_model']['max_half_life'],
            'ou_model.max_half_life',
            min_val=0
        )
        config['ou_model']['min_theta'] = validate_numeric(
            config['ou_model']['min_theta'],
            'ou_model.min_theta',
            min_val=0
        )
        
        # Optional new parameters
        if 'strong_r_squared_threshold' in config['ou_model']:
            config['ou_model']['strong_r_squared_threshold'] = validate_numeric(
                config['ou_model']['strong_r_squared_threshold'],
                'ou_model.strong_r_squared_threshold',
                min_val=0,
                max_val=1
            )
    
    # Validate alpha layer parameters
    if 'alpha' in config:
        config['alpha']['entry_z'] = validate_numeric(
            config['alpha']['entry_z'],
            'alpha.entry_z',
            min_val=0
        )
        config['alpha']['exit_z'] = validate_numeric(
            config['alpha']['exit_z'],
            'alpha.exit_z',
            min_val=0
        )
        config['alpha']['stop_loss_z'] = validate_numeric(
            config['alpha']['stop_loss_z'],
            'alpha.stop_loss_z',
            min_val=0
        )
        config['alpha']['max_hold_days'] = validate_integer(
            config['alpha']['max_hold_days'],
            'alpha.max_hold_days',
            min_val=1
        )
        config['alpha']['velocity_threshold'] = validate_numeric(
            config['alpha']['velocity_threshold'],
            'alpha.velocity_threshold',
            min_val=0
        )
        if 'half_life_multiplier' in config['alpha']:
            config['alpha']['half_life_multiplier'] = validate_numeric(
                config['alpha']['half_life_multiplier'],
                'alpha.half_life_multiplier',
                min_val=0
            )
    
    # Validate execution parameters
    if 'execution' in config:
        config['execution']['transaction_cost_bps'] = validate_numeric(
            config['execution']['transaction_cost_bps'],
            'execution.transaction_cost_bps',
            min_val=0
        )
        config['execution']['slippage_bps'] = validate_numeric(
            config['execution']['slippage_bps'],
            'execution.slippage_bps',
            min_val=0
        )
    
    # Validate targets
    if 'targets' in config:
        config['targets']['min_sharpe'] = validate_numeric(
            config['targets']['min_sharpe'],
            'targets.min_sharpe'
        )
        config['targets']['max_drawdown'] = validate_numeric(
            config['targets']['max_drawdown'],
            'targets.max_drawdown',
            min_val=0,
            max_val=1
        )
    
    # Validate portfolio parameters
    if 'portfolio' in config:
        if 'target_volatility' in config['portfolio']:
            config['portfolio']['target_volatility'] = validate_numeric(
                config['portfolio']['target_volatility'],
                'portfolio.target_volatility',
                min_val=0,
                max_val=1
            )
        if 'min_pairs' in config['portfolio']:
            config['portfolio']['min_pairs'] = validate_integer(
                config['portfolio']['min_pairs'],
                'portfolio.min_pairs',
                min_val=1
            )
    
    return config
