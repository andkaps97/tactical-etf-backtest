"""Helper utilities for the NQ backtest system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import pickle
from pathlib import Path
import hashlib


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe string format.
    
    Args:
        timeframe: Timeframe string (e.g., '1T', '5T', '1H')
        
    Returns:
        True if valid timeframe
    """
    valid_timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D']
    return timeframe in valid_timeframes


def convert_timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Number of minutes
    """
    timeframe_map = {
        '1T': 1,
        '5T': 5,
        '15T': 15,
        '30T': 30,
        '1H': 60,
        '4H': 240,
        '1D': 1440
    }
    return timeframe_map.get(timeframe, 1)


def round_to_tick(price: float, tick_size: float = 0.25) -> float:
    """Round price to the nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Tick size (default NQ = 0.25)
        
    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size


def calculate_position_size(
    capital: float, 
    risk_amount: float, 
    entry_price: float, 
    stop_loss: float,
    tick_value: float = 5.0
) -> int:
    """Calculate position size based on risk management.
    
    Args:
        capital: Available capital
        risk_amount: Amount to risk per trade
        entry_price: Entry price
        stop_loss: Stop loss price
        tick_value: Value per tick (NQ = $5)
        
    Returns:
        Position size in contracts
    """
    if entry_price == stop_loss:
        return 0
    
    risk_per_contract = abs(entry_price - stop_loss) * tick_value / 0.25  # NQ tick size
    
    if risk_per_contract <= 0:
        return 0
    
    position_size = int(risk_amount / risk_per_contract)
    return max(0, position_size)


def is_market_hours(timestamp: datetime, market_open: str = "17:00", market_close: str = "16:00") -> bool:
    """Check if timestamp is within market hours.
    
    Args:
        timestamp: Timestamp to check
        market_open: Market open time (24-hour format)
        market_close: Market close time (24-hour format)
        
    Returns:
        True if within market hours
    """
    # Convert to US/Eastern timezone if needed
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc).astimezone(timezone.utc)
    
    hour = timestamp.hour
    open_hour = int(market_open.split(':')[0])
    close_hour = int(market_close.split(':')[0])
    
    # NQ trades from 6 PM ET Sunday to 5 PM ET Friday
    if open_hour > close_hour:  # Overnight session
        return hour >= open_hour or hour < close_hour
    else:
        return open_hour <= hour < close_hour


def filter_market_days(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include market days (weekdays).
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # Filter out weekends (Saturday=5, Sunday=6)
    weekday_mask = df.index.dayofweek < 5
    return df[weekday_mask]


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLC data to a different timeframe.
    
    Args:
        df: DataFrame with OHLC data
        timeframe: Target timeframe
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Only aggregate columns that exist
    available_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    resampled = df.resample(timeframe).agg(available_agg).dropna()
    return resampled


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string.
    
    Args:
        value: Numeric value (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def save_object(obj: Any, filepath: Union[str, Path], format: str = 'pickle') -> bool:
    """Save object to file.
    
    Args:
        obj: Object to save
        filepath: File path
        format: 'pickle' or 'json'
        
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(obj, f, indent=2, default=str)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
        
        return True
    except Exception:
        return False


def load_object(filepath: Union[str, Path], format: str = 'pickle') -> Any:
    """Load object from file.
    
    Args:
        filepath: File path
        format: 'pickle' or 'json'
        
    Returns:
        Loaded object or None if failed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    try:
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
    except Exception:
        return None


def calculate_file_hash(filepath: Union[str, Path]) -> str:
    """Calculate MD5 hash of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        MD5 hash string
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return ""
    
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def create_data_cache_key(*args, **kwargs) -> str:
    """Create a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Convert arguments to string representation
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (datetime, pd.Timestamp)):
            key_parts.append(arg.isoformat())
        else:
            key_parts.append(str(arg))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (datetime, pd.Timestamp)):
            key_parts.append(f"{key}={value.isoformat()}")
        else:
            key_parts.append(f"{key}={value}")
    
    # Create hash of the combined key
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for NaN values in required columns
    nan_columns = [col for col in required_columns if df[col].isnull().any()]
    if nan_columns:
        return False, f"NaN values found in required columns: {nan_columns}"
    
    return True, "DataFrame structure is valid"


def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    # Recursively merge nested dictionaries
                    merged[key] = merge_configurations(merged[key], value)
                else:
                    merged[key] = value
    
    return merged


def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Estimate memory usage of a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage statistics in MB
    """
    if df.empty:
        return {'total_mb': 0.0, 'per_column_mb': {}}
    
    memory_usage = df.memory_usage(deep=True)
    total_bytes = memory_usage.sum()
    
    # Convert to MB
    total_mb = total_bytes / (1024 * 1024)
    
    per_column_mb = {}
    for column, bytes_used in memory_usage.items():
        per_column_mb[column] = bytes_used / (1024 * 1024)
    
    return {
        'total_mb': total_mb,
        'per_column_mb': per_column_mb,
        'rows': len(df),
        'columns': len(df.columns)
    }


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    if df.empty:
        return df
    
    optimized_df = df.copy()
    
    for column in optimized_df.columns:
        col_type = optimized_df[column].dtype
        
        if col_type == 'object':
            continue
        elif np.issubdtype(col_type, np.integer):
            # Downcast integers
            optimized_df[column] = pd.to_numeric(optimized_df[column], downcast='integer')
        elif np.issubdtype(col_type, np.floating):
            # Downcast floats
            optimized_df[column] = pd.to_numeric(optimized_df[column], downcast='float')
    
    return optimized_df


def create_progress_tracker(total_items: int, description: str = "Processing"):
    """Create a simple progress tracker.
    
    Args:
        total_items: Total number of items to process
        description: Description of the operation
        
    Returns:
        Progress tracker function
    """
    def update_progress(current_item: int, additional_info: str = ""):
        percentage = (current_item / total_items) * 100
        bar_length = 30
        filled_length = int(bar_length * current_item // total_items)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{description}: |{bar}| {percentage:.1f}% ({current_item}/{total_items}) {additional_info}', 
              end='', flush=True)
        
        if current_item == total_items:
            print()  # New line when complete
    
    return update_progress