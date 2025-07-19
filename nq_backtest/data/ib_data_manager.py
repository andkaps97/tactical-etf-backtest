"""Interactive Brokers data manager for NQ futures data."""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from ib_insync import IB, Future, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("ib_insync not available. Using simulated data mode.")

from ..config.ib_config import IBConfig, ContractConfig
from ..config.strategy_config import StrategyConfig


class IBDataManager:
    """Manages data fetching and caching from Interactive Brokers."""
    
    def __init__(self, ib_config: IBConfig, contract_config: ContractConfig):
        """Initialize the IB data manager."""
        self.ib_config = ib_config
        self.contract_config = contract_config
        self.ib = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        # Data cache
        self._minute_data_cache = {}
        self._hourly_data_cache = {}
        
    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        if not IB_AVAILABLE:
            self.logger.warning("IB not available, using mock connection")
            self.connected = True
            return True
            
        try:
            self.ib = IB()
            await self.ib.connectAsync(
                host=self.ib_config.host,
                port=self.ib_config.port,
                clientId=self.ib_config.client_id,
                timeout=self.ib_config.timeout
            )
            self.connected = True
            self.logger.info(f"Connected to IB at {self.ib_config.host}:{self.ib_config.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IB")
    
    def _create_nq_contract(self, expiry: Optional[str] = None) -> 'Future':
        """Create NQ futures contract."""
        if not IB_AVAILABLE:
            return None
            
        # Use front month contract if no expiry specified
        if expiry is None:
            expiry = self._get_front_month_expiry()
            
        return Future(
            symbol=self.ib_config.symbol,
            lastTradeDateOrContractMonth=expiry,
            exchange=self.ib_config.exchange,
            currency=self.ib_config.currency
        )
    
    def _get_front_month_expiry(self) -> str:
        """Get the front month expiry for NQ futures."""
        # NQ expires on the third Friday of March, June, September, December
        now = datetime.now()
        quarterly_months = [3, 6, 9, 12]
        
        for month in quarterly_months:
            if month >= now.month:
                # Find third Friday of the month
                year = now.year
                # Simple approximation - use 15th of month for now
                expiry_date = datetime(year, month, 15)
                return expiry_date.strftime("%Y%m")
                
        # If no expiry this year, use March of next year
        return datetime(now.year + 1, 3, 15).strftime("%Y%m")
    
    async def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = "1 min",
        data_type: str = "MIDPOINT"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical minute data from IB."""
        
        if not IB_AVAILABLE:
            return self._generate_mock_data(start_date, end_date, bar_size)
        
        if not self.connected:
            await self.connect()
            
        if not self.connected:
            self.logger.error("Cannot fetch data - not connected to IB")
            return None
            
        try:
            contract = self._create_nq_contract()
            if not contract:
                return None
                
            # Qualify the contract
            await self.ib.qualifyContractsAsync(contract)
            
            # Calculate duration
            duration = end_date - start_date
            duration_str = f"{duration.days} D"
            
            # Fetch historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=data_type,
                useRTH=self.ib_config.use_rth,
                keepUpToDate=self.ib_config.keep_up_to_date
            )
            
            if not bars:
                self.logger.warning("No historical data received")
                return None
                
            # Convert to DataFrame
            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return None
            
            self.logger.info(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None
    
    def _generate_mock_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = "1 min"
    ) -> pd.DataFrame:
        """Generate mock NQ data for testing when IB is not available."""
        
        self.logger.info("Generating mock NQ data for testing")
        
        # Create minute-by-minute data
        freq = "1T" if "min" in bar_size else "1H"
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Filter for market hours (simplified)
        market_hours = date_range[
            (date_range.hour >= 17) | (date_range.hour < 16)
        ]
        
        n_bars = len(market_hours)
        if n_bars == 0:
            return pd.DataFrame()
        
        # Generate realistic NQ price data
        np.random.seed(42)  # For reproducible data
        base_price = 15000.0
        
        # Generate random walk with some trending
        returns = np.random.normal(0.0001, 0.002, n_bars)  # Small returns with volatility
        returns[0] = 0  # Start at base price
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = []
        for i, (timestamp, price) in enumerate(zip(market_hours, prices)):
            if i == 0:
                open_price = price
            else:
                open_price = data[-1]['close']
            
            # Generate high/low around the price
            volatility = 0.001
            high = price * (1 + np.random.exponential(volatility))
            low = price * (1 - np.random.exponential(volatility))
            
            # Ensure open/close are within high/low
            close_price = np.random.uniform(low, high)
            open_price = max(min(open_price, high), low)
            
            # Round to tick size (0.25)
            tick_size = 0.25
            open_price = round(open_price / tick_size) * tick_size
            high = round(high / tick_size) * tick_size
            low = round(low / tick_size) * tick_size
            close_price = round(close_price / tick_size) * tick_size
            
            data.append({
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': np.random.randint(100, 1000),
                'average': (high + low + close_price) / 3,
                'barCount': 1
            })
        
        df = pd.DataFrame(data, index=market_hours)
        df.index.name = 'date'
        
        self.logger.info(f"Generated {len(df)} mock bars from {df.index[0]} to {df.index[-1]}")
        return df
    
    async def get_minute_data(
        self,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get minute data with caching."""
        
        cache_key = f"{start_date}_{end_date}_1min"
        
        if use_cache and cache_key in self._minute_data_cache:
            self.logger.debug("Returning cached minute data")
            return self._minute_data_cache[cache_key]
        
        data = await self.fetch_historical_data(start_date, end_date, "1 min")
        
        if data is not None and use_cache:
            self._minute_data_cache[cache_key] = data
            
        return data
    
    def resample_to_hourly(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """Resample minute data to hourly bars."""
        
        if minute_data.empty:
            return pd.DataFrame()
        
        # Resample to hourly
        hourly = minute_data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        self.logger.debug(f"Resampled {len(minute_data)} minute bars to {len(hourly)} hourly bars")
        return hourly
    
    async def get_hourly_data(
        self,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get hourly data with caching."""
        
        cache_key = f"{start_date}_{end_date}_1hour"
        
        if use_cache and cache_key in self._hourly_data_cache:
            self.logger.debug("Returning cached hourly data")
            return self._hourly_data_cache[cache_key]
        
        # Get minute data and resample
        minute_data = await self.get_minute_data(start_date, end_date, use_cache)
        
        if minute_data is None:
            return None
            
        hourly_data = self.resample_to_hourly(minute_data)
        
        if use_cache:
            self._hourly_data_cache[cache_key] = hourly_data
            
        return hourly_data
    
    def clear_cache(self):
        """Clear all cached data."""
        self._minute_data_cache.clear()
        self._hourly_data_cache.clear()
        self.logger.info("Cleared data cache")