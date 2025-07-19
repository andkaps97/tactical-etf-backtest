"""Strategy configuration parameters for the NQ Sweep-and-Reversion strategy."""

from dataclasses import dataclass
from typing import Optional
import datetime as dt


@dataclass
class StrategyConfig:
    """Configuration parameters for the NQ Sweep-and-Reversion strategy."""
    
    # Core Strategy Parameters
    reference_timeframe: str = "1H"  # H1 candles as primary reference
    sweep_detection_window: int = 20  # Minutes to detect sweep after H1 open
    max_hold_time: int = 20  # Maximum hold time in minutes
    risk_reward_ratio: float = 1.0  # 1:1 risk-reward ratio
    
    # Position Sizing
    position_size: float = 1.0  # Number of contracts per trade
    max_positions: int = 1  # Maximum concurrent positions
    
    # Risk Management
    use_stop_loss: bool = True
    use_take_profit: bool = True
    force_close_on_timeout: bool = True
    
    # Trading Hours (US Central Time)
    market_open: dt.time = dt.time(17, 0)  # 5:00 PM CT (Sunday-Thursday)
    market_close: dt.time = dt.time(16, 0)  # 4:00 PM CT (Monday-Friday)
    
    # Slippage and Commission
    slippage_ticks: float = 0.25  # Slippage in ticks
    commission_per_contract: float = 2.50  # Commission per contract
    tick_size: float = 0.25  # NQ tick size
    tick_value: float = 5.0  # NQ tick value in USD
    
    # Data Parameters
    min_data_lookback_days: int = 180  # Minimum data for backtesting
    max_data_lookback_days: int = 365  # Maximum data to fetch
    
    # Validation Parameters
    min_h1_range_ticks: int = 4  # Minimum H1 range to consider for sweep
    max_h1_range_ticks: int = 200  # Maximum H1 range to consider valid
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.sweep_detection_window <= 0 or self.sweep_detection_window > 60:
            raise ValueError("Sweep detection window must be between 1-60 minutes")
            
        if self.max_hold_time <= 0 or self.max_hold_time > 120:
            raise ValueError("Max hold time must be between 1-120 minutes")
            
        if self.risk_reward_ratio <= 0:
            raise ValueError("Risk reward ratio must be positive")
            
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
            
        if self.tick_size <= 0 or self.tick_value <= 0:
            raise ValueError("Tick size and value must be positive")


@dataclass 
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    initial_capital: float = 100000.0
    
    # Performance Metrics
    benchmark_symbol: str = "NQ"  # Benchmark for comparison
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calculation
    
    # Reporting
    generate_trade_log: bool = True
    generate_performance_report: bool = True
    generate_charts: bool = True
    output_directory: str = "backtest_results"
    
    def validate(self) -> None:
        """Validate backtest configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
            
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("Start date must be before end date")