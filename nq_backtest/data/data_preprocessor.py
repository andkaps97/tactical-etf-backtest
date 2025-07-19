"""Data preprocessing utilities for NQ futures data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging


class DataPreprocessor:
    """Preprocessor for cleaning and validating NQ futures data."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.logger = logging.getLogger(__name__)
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the integrity of market data."""
        
        if df.empty:
            return False, "DataFrame is empty"
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for NaN values
        if df[required_columns].isnull().any().any():
            return False, "Data contains NaN values"
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        
        if invalid_ohlc.any():
            return False, f"Invalid OHLC relationships found in {invalid_ohlc.sum()} bars"
        
        # Check for negative values
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False, "Data contains non-positive prices"
        
        # Check for extreme price movements (>50% in one bar)
        price_changes = df['close'].pct_change().abs()
        extreme_moves = price_changes > 0.5
        if extreme_moves.any():
            self.logger.warning(f"Found {extreme_moves.sum()} bars with >50% price changes")
        
        return True, "Data validation passed"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter market data."""
        
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove bars with zero volume (if volume column exists)
        if 'volume' in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['volume'] > 0]
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} bars with zero volume")
        
        # Remove duplicate timestamps
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        removed_count = initial_count - len(cleaned_df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate timestamps")
        
        # Sort by timestamp
        cleaned_df = cleaned_df.sort_index()
        
        # Forward fill small gaps (up to 5 minutes for minute data)
        if not cleaned_df.empty:
            # Detect frequency
            freq = pd.infer_freq(cleaned_df.index[:10])
            if freq and 'T' in freq:  # Minute data
                max_gap = pd.Timedelta('5T')
            else:  # Hourly or other
                max_gap = pd.Timedelta('2H')
            
            # Forward fill only small gaps
            cleaned_df = self._forward_fill_small_gaps(cleaned_df, max_gap)
        
        return cleaned_df
    
    def _forward_fill_small_gaps(self, df: pd.DataFrame, max_gap: pd.Timedelta) -> pd.DataFrame:
        """Forward fill only small gaps in the data."""
        
        if len(df) < 2:
            return df
        
        # Find gaps
        time_diffs = df.index.to_series().diff()
        large_gaps = time_diffs > max_gap
        
        if large_gaps.any():
            self.logger.info(f"Found {large_gaps.sum()} large gaps (>{max_gap})")
        
        # Create complete time range
        if pd.infer_freq(df.index[:10]):
            freq = pd.infer_freq(df.index[:10])
            complete_range = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=freq
            )
            
            # Reindex and forward fill only small gaps
            df_reindexed = df.reindex(complete_range)
            
            # Only fill gaps smaller than max_gap
            for i in range(1, len(df_reindexed)):
                if pd.isna(df_reindexed.iloc[i].iloc[0]):  # Check if row is NaN
                    gap_size = df_reindexed.index[i] - df_reindexed.index[i-1]
                    if gap_size <= max_gap:
                        # Forward fill
                        df_reindexed.iloc[i] = df_reindexed.iloc[i-1]
            
            # Remove remaining NaN rows
            df_reindexed = df_reindexed.dropna()
            return df_reindexed
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators useful for sweep detection."""
        
        if df.empty:
            return df
        
        # Create a copy
        df_with_indicators = df.copy()
        
        # Add typical price (HLC/3)
        df_with_indicators['typical_price'] = (
            df_with_indicators['high'] + 
            df_with_indicators['low'] + 
            df_with_indicators['close']
        ) / 3
        
        # Add true range for volatility analysis
        df_with_indicators['prev_close'] = df_with_indicators['close'].shift(1)
        df_with_indicators['true_range'] = np.maximum(
            df_with_indicators['high'] - df_with_indicators['low'],
            np.maximum(
                abs(df_with_indicators['high'] - df_with_indicators['prev_close']),
                abs(df_with_indicators['low'] - df_with_indicators['prev_close'])
            )
        )
        
        # Add Average True Range (ATR) - 14 period
        df_with_indicators['atr_14'] = df_with_indicators['true_range'].rolling(14).mean()
        
        # Add price range in ticks (useful for sweep analysis)
        tick_size = 0.25  # NQ tick size
        df_with_indicators['range_ticks'] = (
            (df_with_indicators['high'] - df_with_indicators['low']) / tick_size
        ).round()
        
        # Add time-based features
        df_with_indicators['hour'] = df_with_indicators.index.hour
        df_with_indicators['minute'] = df_with_indicators.index.minute
        df_with_indicators['day_of_week'] = df_with_indicators.index.dayofweek
        
        # Remove intermediate calculations
        df_with_indicators.drop(['prev_close'], axis=1, inplace=True)
        
        return df_with_indicators
    
    def filter_market_hours(
        self, 
        df: pd.DataFrame,
        start_hour: int = 17,  # 5 PM ET
        end_hour: int = 16     # 4 PM ET next day
    ) -> pd.DataFrame:
        """Filter data to only include regular trading hours."""
        
        if df.empty:
            return df
        
        # Convert to ET timezone if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        elif df.index.tz != 'US/Eastern':
            df.index = df.index.tz_convert('US/Eastern')
        
        # Filter for trading hours
        # NQ trades from 6 PM ET Sunday to 5 PM ET Friday
        mask = (
            (df.index.hour >= start_hour) |  # From start_hour onwards
            (df.index.hour < end_hour)       # Until end_hour
        )
        
        # Exclude weekends (Saturday = 5, Sunday = 6 before 6 PM)
        weekend_mask = ~(
            (df.index.dayofweek == 5) |  # Saturday
            ((df.index.dayofweek == 6) & (df.index.hour < start_hour))  # Sunday before start_hour
        )
        
        filtered_df = df[mask & weekend_mask].copy()
        
        self.logger.info(f"Filtered to market hours: {len(df)} -> {len(filtered_df)} bars")
        return filtered_df
    
    def detect_gaps(self, df: pd.DataFrame, expected_freq: str = '1T') -> pd.DataFrame:
        """Detect gaps in the data."""
        
        if len(df) < 2:
            return pd.DataFrame()
        
        # Expected time delta
        if expected_freq == '1T':
            expected_delta = pd.Timedelta('1T')
        elif expected_freq == '1H':
            expected_delta = pd.Timedelta('1H')
        else:
            expected_delta = pd.Timedelta(expected_freq)
        
        # Find actual gaps
        time_diffs = df.index.to_series().diff()[1:]
        gaps = time_diffs[time_diffs > expected_delta * 1.5]  # Allow 50% tolerance
        
        if gaps.empty:
            return pd.DataFrame()
        
        # Create gap summary
        gap_summary = pd.DataFrame({
            'start_time': gaps.index - gaps.values,
            'end_time': gaps.index,
            'gap_duration': gaps.values,
            'gap_minutes': gaps.values.dt.total_seconds() / 60
        })
        
        self.logger.info(f"Found {len(gap_summary)} data gaps")
        return gap_summary
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        
        if df.empty:
            return {"error": "Empty DataFrame"}
        
        report = {
            "basic_stats": {
                "total_bars": len(df),
                "date_range": {
                    "start": df.index[0].isoformat(),
                    "end": df.index[-1].isoformat(),
                    "duration_days": (df.index[-1] - df.index[0]).days
                },
                "columns": list(df.columns)
            },
            "data_quality": {},
            "price_stats": {},
            "gaps": {}
        }
        
        # Data quality checks
        is_valid, message = self.validate_data(df)
        report["data_quality"]["is_valid"] = is_valid
        report["data_quality"]["validation_message"] = message
        
        # Missing data
        if 'volume' in df.columns:
            zero_volume_bars = (df['volume'] == 0).sum()
            report["data_quality"]["zero_volume_bars"] = zero_volume_bars
        
        # Price statistics
        if 'close' in df.columns:
            report["price_stats"] = {
                "min_price": float(df['close'].min()),
                "max_price": float(df['close'].max()),
                "mean_price": float(df['close'].mean()),
                "std_price": float(df['close'].std()),
                "price_range": float(df['close'].max() - df['close'].min())
            }
        
        # Gap analysis
        gaps = self.detect_gaps(df)
        if not gaps.empty:
            report["gaps"] = {
                "total_gaps": len(gaps),
                "largest_gap_minutes": float(gaps['gap_minutes'].max()),
                "total_gap_time_minutes": float(gaps['gap_minutes'].sum())
            }
        else:
            report["gaps"] = {"total_gaps": 0}
        
        return report