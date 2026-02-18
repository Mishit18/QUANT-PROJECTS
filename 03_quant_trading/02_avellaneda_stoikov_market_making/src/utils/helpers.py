"""
Helper Functions

Utility functions for configuration, I/O, and common operations.
"""

import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict[str, Any], output_dir: str, prefix: str = "results"):
    """
    Save simulation results to disk.
    
    Saves in multiple formats:
    - JSON for metadata and metrics
    - Pickle for complete Python objects
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON (for human readability)
    json_path = output_path / f"{prefix}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = _convert_for_json(results)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save as pickle (for complete data)
    pickle_path = output_path / f"{prefix}_{timestamp}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    return json_path, pickle_path


def _convert_for_json(obj: Any) -> Any:
    """
    Convert numpy arrays and other non-JSON-serializable objects.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-serializable object
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    else:
        return obj


def create_output_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create output directory for experiment.
    
    Args:
        base_dir: Base directory path
        experiment_name: Name of experiment
    
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from pickle file.
    
    Args:
        results_path: Path to pickle file
    
    Returns:
        Results dictionary
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results
