"""Interactive Brokers API configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IBConfig:
    """Configuration for Interactive Brokers API connection."""
    
    # Connection Parameters
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port (7496 for live)
    client_id: int = 1
    timeout: int = 30  # Connection timeout in seconds
    
    # Contract Specifications
    symbol: str = "NQ"
    sec_type: str = "FUT"  # Futures
    exchange: str = "GLOBEX"
    currency: str = "USD"
    
    # Data Parameters
    data_type: str = "MIDPOINT"  # TRADES, MIDPOINT, BID, ASK
    use_rth: bool = True  # Use regular trading hours only
    keep_up_to_date: bool = False  # For historical data
    
    # Request Limits
    max_requests_per_second: int = 50
    max_historical_requests: int = 60  # Per 10 minutes
    
    def validate(self) -> None:
        """Validate IB configuration parameters."""
        if not (1024 <= self.port <= 65535):
            raise ValueError("Port must be between 1024-65535")
            
        if self.client_id < 0:
            raise ValueError("Client ID must be non-negative")
            
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
            
        if self.symbol not in ["NQ", "MNQ"]:  # NQ and Micro NQ
            raise ValueError("Symbol must be NQ or MNQ")


@dataclass
class ContractConfig:
    """NQ Futures contract specification."""
    
    # Contract Details
    symbol: str = "NQ"
    multiplier: int = 20  # Contract multiplier
    min_tick: float = 0.25  # Minimum price increment
    
    # Trading Specifications  
    initial_margin: float = 19000.0  # Approximate initial margin
    maintenance_margin: float = 17000.0  # Approximate maintenance margin
    
    # Market Hours (UTC)
    sunday_open: str = "23:00"  # Sunday 6 PM ET
    friday_close: str = "22:00"  # Friday 5 PM ET
    daily_close: str = "22:00"  # Daily close 5 PM ET
    daily_open: str = "23:00"   # Daily open 6 PM ET
    
    # Holiday Schedule
    major_holidays: list = None
    
    def __post_init__(self):
        if self.major_holidays is None:
            self.major_holidays = [
                "New Year's Day",
                "Martin Luther King Jr. Day", 
                "Presidents Day",
                "Good Friday",
                "Memorial Day",
                "Independence Day",
                "Labor Day",
                "Thanksgiving",
                "Christmas"
            ]
    
    def validate(self) -> None:
        """Validate contract configuration."""
        if self.multiplier <= 0:
            raise ValueError("Contract multiplier must be positive")
            
        if self.min_tick <= 0:
            raise ValueError("Minimum tick must be positive")
            
        if self.initial_margin <= 0 or self.maintenance_margin <= 0:
            raise ValueError("Margin requirements must be positive")