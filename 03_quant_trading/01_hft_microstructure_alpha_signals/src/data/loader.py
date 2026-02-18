"""
LOB data loader and synthetic data generator.
Generates realistic limit order book dynamics for research purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import yaml


class LOBDataGenerator:
    """
    Generates synthetic limit order book data with realistic microstructure.
    
    Uses Poisson processes for order arrivals and cancellations.
    Price follows a mean-reverting process with microstructure noise.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.tick_size = config['tick_size']
        self.n_levels = config['n_levels']
        self.initial_price = config['initial_price']
        
    def generate(self, n_events: int) -> pd.DataFrame:
        """
        Generate synthetic LOB event stream.
        
        Returns DataFrame with columns:
        - timestamp: nanosecond timestamp
        - event_type: 'update' or 'trade'
        - bid_price_i, bid_size_i, ask_price_i, ask_size_i for i in 1..n_levels
        - trade_price, trade_size (for trade events)
        """
        np.random.seed(42)
        
        # Initialize price and spread
        mid_price = self.initial_price
        spread_ticks = 1  # minimum spread in ticks
        
        events = []
        current_time = 0
        
        # Initialize order book
        bid_prices, bid_sizes, ask_prices, ask_sizes = self._initialize_book(mid_price)
        
        for i in range(n_events):
            # Time increment (exponential inter-arrival)
            dt = np.random.exponential(1.0 / 100.0)  # 100 events per second average
            current_time += int(dt * 1e9)  # convert to nanoseconds
            
            # Determine event type
            event_type = np.random.choice(['update', 'trade'], p=[0.7, 0.3])
            
            # Update mid price (mean-reverting random walk)
            price_change = np.random.normal(0, self.config['volatility'])
            price_change += -0.01 * (mid_price - self.initial_price)  # mean reversion
            mid_price += price_change
            
            # Update book levels
            bid_prices, bid_sizes, ask_prices, ask_sizes = self._update_book(
                mid_price, bid_prices, bid_sizes, ask_prices, ask_sizes
            )
            
            # Create event record
            event = {
                'timestamp': current_time,
                'event_type': event_type,
            }
            
            # Add book levels
            for level in range(self.n_levels):
                event[f'bid_price_{level+1}'] = bid_prices[level]
                event[f'bid_size_{level+1}'] = bid_sizes[level]
                event[f'ask_price_{level+1}'] = ask_prices[level]
                event[f'ask_size_{level+1}'] = ask_sizes[level]
            
            # Add trade information if trade event
            if event_type == 'trade':
                trade_side = np.random.choice(['buy', 'sell'])
                trade_price = ask_prices[0] if trade_side == 'buy' else bid_prices[0]
                trade_size = np.random.exponential(self.config['mean_order_size'])
                trade_size = np.clip(trade_size, self.config['min_order_size'], 
                                    self.config['max_order_size'])
                
                event['trade_price'] = trade_price
                event['trade_size'] = trade_size
                event['trade_side'] = trade_side
            else:
                event['trade_price'] = np.nan
                event['trade_size'] = np.nan
                event['trade_side'] = None
            
            events.append(event)
        
        df = pd.DataFrame(events)
        return df
    
    def _initialize_book(self, mid_price: float) -> Tuple[np.ndarray, ...]:
        """Initialize order book with reasonable depth profile."""
        bid_prices = np.array([
            mid_price - (i + 0.5) * self.tick_size 
            for i in range(self.n_levels)
        ])
        ask_prices = np.array([
            mid_price + (i + 0.5) * self.tick_size 
            for i in range(self.n_levels)
        ])
        
        # Depth decreases with distance from mid
        bid_sizes = np.array([
            np.random.exponential(self.config['mean_order_size']) * (1.0 / (i + 1))
            for i in range(self.n_levels)
        ])
        ask_sizes = np.array([
            np.random.exponential(self.config['mean_order_size']) * (1.0 / (i + 1))
            for i in range(self.n_levels)
        ])
        
        return bid_prices, bid_sizes, ask_prices, ask_sizes
    
    def _update_book(self, mid_price: float, bid_prices: np.ndarray, 
                     bid_sizes: np.ndarray, ask_prices: np.ndarray, 
                     ask_sizes: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Update book levels with some noise and mean reversion to mid."""
        # Update prices to track mid
        for i in range(self.n_levels):
            bid_prices[i] = mid_price - (i + 0.5) * self.tick_size + \
                           np.random.normal(0, 0.1 * self.tick_size)
            ask_prices[i] = mid_price + (i + 0.5) * self.tick_size + \
                           np.random.normal(0, 0.1 * self.tick_size)
        
        # Round to tick size
        bid_prices = np.round(bid_prices / self.tick_size) * self.tick_size
        ask_prices = np.round(ask_prices / self.tick_size) * self.tick_size
        
        # Update sizes with random walk
        bid_sizes += np.random.normal(0, 10, self.n_levels)
        ask_sizes += np.random.normal(0, 10, self.n_levels)
        
        # Ensure positive and reasonable
        bid_sizes = np.clip(bid_sizes, 10, 1000)
        ask_sizes = np.clip(ask_sizes, 10, 1000)
        
        return bid_prices, bid_sizes, ask_prices, ask_sizes


def load_config(config_path: str = "config/data_config.yaml") -> Dict:
    """Load data configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['data']


def generate_and_save_data(config_path: str = "config/data_config.yaml"):
    """Generate synthetic LOB data and save to disk."""
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(config['raw_data_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print(f"Generating {config['n_events']} LOB events...")
    generator = LOBDataGenerator(config)
    df = generator.generate(config['n_events'])
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    df = generate_and_save_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
