"""Sweep detection logic for the NQ futures strategy."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, NamedTuple
import logging


class SweepSignal(NamedTuple):
    """Represents a detected sweep signal."""
    timestamp: datetime
    sweep_type: str  # 'high_sweep' or 'low_sweep'
    h1_open: float
    h1_high: float
    h1_low: float
    sweep_price: float
    sweep_candle_close: float
    minutes_from_h1_open: int
    is_valid: bool
    reason: str = ""


class SweepDetector:
    """Detects H1 high/low sweeps within the first 20 minutes of each hour."""
    
    def __init__(self, sweep_window_minutes: int = 20, min_h1_range_ticks: int = 4):
        """Initialize the sweep detector.
        
        Args:
            sweep_window_minutes: Time window to detect sweeps after H1 open
            min_h1_range_ticks: Minimum H1 range in ticks to consider valid
        """
        self.sweep_window_minutes = sweep_window_minutes
        self.min_h1_range_ticks = min_h1_range_ticks
        self.tick_size = 0.25  # NQ tick size
        self.logger = logging.getLogger(__name__)
        
    def detect_sweeps(
        self, 
        minute_data: pd.DataFrame, 
        hourly_data: pd.DataFrame
    ) -> List[SweepSignal]:
        """Detect all sweep signals in the provided data.
        
        Args:
            minute_data: 1-minute OHLCV data
            hourly_data: 1-hour OHLCV data
            
        Returns:
            List of detected sweep signals
        """
        sweeps = []
        
        if minute_data.empty or hourly_data.empty:
            self.logger.warning("Empty data provided to sweep detector")
            return sweeps
        
        # Process each hourly bar
        for h1_timestamp, h1_bar in hourly_data.iterrows():
            # Get the corresponding minute data for this hour
            hour_start = h1_timestamp
            hour_end = hour_start + timedelta(hours=1)
            
            # Get minute bars within the sweep detection window
            sweep_end = hour_start + timedelta(minutes=self.sweep_window_minutes)
            
            sweep_window_data = minute_data[
                (minute_data.index >= hour_start) & 
                (minute_data.index < sweep_end)
            ]
            
            if sweep_window_data.empty:
                continue
            
            # Check for sweeps
            sweep_signals = self._check_hour_for_sweeps(
                h1_bar, h1_timestamp, sweep_window_data
            )
            
            sweeps.extend(sweep_signals)
        
        self.logger.info(f"Detected {len(sweeps)} sweep signals")
        return sweeps
    
    def _check_hour_for_sweeps(
        self, 
        h1_bar: pd.Series, 
        h1_timestamp: datetime, 
        sweep_window_data: pd.DataFrame
    ) -> List[SweepSignal]:
        """Check a specific hour for sweep patterns.
        
        Args:
            h1_bar: The hourly bar data
            h1_timestamp: The hourly bar timestamp
            sweep_window_data: Minute data within the sweep detection window
            
        Returns:
            List of sweep signals for this hour
        """
        sweeps = []
        
        if sweep_window_data.empty:
            return sweeps
        
        # Extract H1 levels
        h1_open = h1_bar['open']
        h1_high = h1_bar['high']
        h1_low = h1_bar['low']
        
        # Calculate H1 range in ticks
        h1_range_ticks = (h1_high - h1_low) / self.tick_size
        
        # Skip if H1 range is too small
        if h1_range_ticks < self.min_h1_range_ticks:
            return sweeps
        
        # Check each minute candle for sweeps
        for i, (timestamp, minute_bar) in enumerate(sweep_window_data.iterrows()):
            minutes_from_open = int((timestamp - h1_timestamp).total_seconds() / 60)
            
            # Check for high sweep
            if minute_bar['high'] > h1_high:
                sweep_signal = SweepSignal(
                    timestamp=timestamp,
                    sweep_type='high_sweep',
                    h1_open=h1_open,
                    h1_high=h1_high,
                    h1_low=h1_low,
                    sweep_price=minute_bar['high'],
                    sweep_candle_close=minute_bar['close'],
                    minutes_from_h1_open=minutes_from_open,
                    is_valid=True
                )
                
                # Validate the sweep
                validated_sweep = self._validate_sweep(sweep_signal, sweep_window_data, i)
                sweeps.append(validated_sweep)
            
            # Check for low sweep
            if minute_bar['low'] < h1_low:
                sweep_signal = SweepSignal(
                    timestamp=timestamp,
                    sweep_type='low_sweep',
                    h1_open=h1_open,
                    h1_high=h1_high,
                    h1_low=h1_low,
                    sweep_price=minute_bar['low'],
                    sweep_candle_close=minute_bar['close'],
                    minutes_from_h1_open=minutes_from_open,
                    is_valid=True
                )
                
                # Validate the sweep
                validated_sweep = self._validate_sweep(sweep_signal, sweep_window_data, i)
                sweeps.append(validated_sweep)
        
        return sweeps
    
    def _validate_sweep(
        self, 
        sweep: SweepSignal, 
        sweep_window_data: pd.DataFrame, 
        candle_index: int
    ) -> SweepSignal:
        """Validate a sweep signal against quality criteria.
        
        Args:
            sweep: The sweep signal to validate
            sweep_window_data: The minute data for the sweep window
            candle_index: Index of the sweep candle
            
        Returns:
            Validated sweep signal (may be marked invalid)
        """
        reasons = []
        
        # Check if sweep is significant (at least 1 tick beyond H1 level)
        if sweep.sweep_type == 'high_sweep':
            penetration = sweep.sweep_price - sweep.h1_high
        else:
            penetration = sweep.h1_low - sweep.sweep_price
        
        min_penetration = self.tick_size  # At least 1 tick
        if penetration < min_penetration:
            reasons.append(f"Insufficient penetration: {penetration:.2f} < {min_penetration:.2f}")
        
        # Check if this is the first sweep of this type in this window
        prior_candles = sweep_window_data.iloc[:candle_index]
        if not prior_candles.empty:
            if sweep.sweep_type == 'high_sweep':
                prior_sweeps = prior_candles['high'] > sweep.h1_high
            else:
                prior_sweeps = prior_candles['low'] < sweep.h1_low
            
            if prior_sweeps.any():
                reasons.append("Not the first sweep of this type in the window")
        
        # Check for reasonable market conditions (optional additional filters)
        # Could add volume validation, time-of-day filters, etc.
        
        # Determine if sweep is valid
        is_valid = len(reasons) == 0
        reason_str = "; ".join(reasons) if reasons else "Valid sweep"
        
        return sweep._replace(is_valid=is_valid, reason=reason_str)
    
    def get_entry_signals(self, sweeps: List[SweepSignal]) -> List[SweepSignal]:
        """Filter sweeps to get valid entry signals.
        
        Args:
            sweeps: List of all detected sweeps
            
        Returns:
            List of valid entry signals
        """
        entry_signals = [sweep for sweep in sweeps if sweep.is_valid]
        
        self.logger.info(f"Filtered {len(sweeps)} sweeps to {len(entry_signals)} valid entry signals")
        return entry_signals
    
    def analyze_sweep_patterns(self, sweeps: List[SweepSignal]) -> Dict[str, any]:
        """Analyze patterns in detected sweeps for strategy optimization.
        
        Args:
            sweeps: List of detected sweeps
            
        Returns:
            Dictionary containing pattern analysis
        """
        if not sweeps:
            return {"error": "No sweeps to analyze"}
        
        # Convert to DataFrame for analysis
        sweep_data = []
        for sweep in sweeps:
            sweep_data.append({
                'timestamp': sweep.timestamp,
                'sweep_type': sweep.sweep_type,
                'h1_range_ticks': (sweep.h1_high - sweep.h1_low) / self.tick_size,
                'minutes_from_open': sweep.minutes_from_h1_open,
                'penetration_ticks': (
                    (sweep.sweep_price - sweep.h1_high) / self.tick_size 
                    if sweep.sweep_type == 'high_sweep' 
                    else (sweep.h1_low - sweep.sweep_price) / self.tick_size
                ),
                'is_valid': sweep.is_valid,
                'hour': sweep.timestamp.hour,
                'day_of_week': sweep.timestamp.weekday()
            })
        
        df = pd.DataFrame(sweep_data)
        
        analysis = {
            'total_sweeps': len(sweeps),
            'valid_sweeps': df['is_valid'].sum(),
            'sweep_types': df['sweep_type'].value_counts().to_dict(),
            'time_patterns': {
                'by_hour': df.groupby('hour').size().to_dict(),
                'by_day_of_week': df.groupby('day_of_week').size().to_dict(),
                'by_minutes_from_open': df.groupby('minutes_from_open').size().to_dict()
            },
            'statistics': {
                'avg_h1_range_ticks': df['h1_range_ticks'].mean(),
                'avg_penetration_ticks': df['penetration_ticks'].mean(),
                'std_penetration_ticks': df['penetration_ticks'].std(),
                'avg_minutes_to_sweep': df['minutes_from_open'].mean()
            }
        }
        
        return analysis
    
    def optimize_parameters(
        self, 
        minute_data: pd.DataFrame, 
        hourly_data: pd.DataFrame,
        sweep_windows: List[int] = [15, 20, 25, 30],
        min_ranges: List[int] = [3, 4, 5, 6]
    ) -> Dict[str, any]:
        """Optimize sweep detection parameters.
        
        Args:
            minute_data: Minute OHLCV data
            hourly_data: Hourly OHLCV data
            sweep_windows: List of sweep window sizes to test
            min_ranges: List of minimum H1 ranges to test
            
        Returns:
            Dictionary with optimization results
        """
        results = []
        
        original_window = self.sweep_window_minutes
        original_min_range = self.min_h1_range_ticks
        
        try:
            for window in sweep_windows:
                for min_range in min_ranges:
                    self.sweep_window_minutes = window
                    self.min_h1_range_ticks = min_range
                    
                    sweeps = self.detect_sweeps(minute_data, hourly_data)
                    valid_sweeps = [s for s in sweeps if s.is_valid]
                    
                    results.append({
                        'sweep_window': window,
                        'min_h1_range': min_range,
                        'total_sweeps': len(sweeps),
                        'valid_sweeps': len(valid_sweeps),
                        'validity_rate': len(valid_sweeps) / len(sweeps) if sweeps else 0,
                        'sweeps_per_day': len(valid_sweeps) / max(1, len(hourly_data) / 24)
                    })
        
        finally:
            # Restore original parameters
            self.sweep_window_minutes = original_window
            self.min_h1_range_ticks = original_min_range
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Find optimal parameters (balance between validity rate and signal frequency)
            results_df['score'] = results_df['validity_rate'] * results_df['sweeps_per_day']
            best_params = results_df.loc[results_df['score'].idxmax()]
            
            return {
                'optimization_results': results,
                'best_parameters': best_params.to_dict(),
                'summary': {
                    'parameter_combinations_tested': len(results),
                    'best_sweep_window': int(best_params['sweep_window']),
                    'best_min_h1_range': int(best_params['min_h1_range']),
                    'best_score': best_params['score']
                }
            }
        
        return {"error": "No optimization results generated"}