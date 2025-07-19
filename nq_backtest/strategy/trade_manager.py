"""Trade management for the NQ futures strategy."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, NamedTuple
from enum import Enum
import logging


class TradeDirection(Enum):
    """Trade direction enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "OPEN"
    CLOSED_PROFIT = "CLOSED_PROFIT"
    CLOSED_LOSS = "CLOSED_LOSS"
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT"


class Trade(NamedTuple):
    """Represents a trade."""
    trade_id: int
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    max_hold_time: timedelta
    sweep_type: str
    h1_open: float
    h1_high: float
    h1_low: float
    
    # Optional fields (set when trade closes)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl_ticks: Optional[float] = None
    pnl_dollars: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    hold_time_minutes: Optional[int] = None
    exit_reason: Optional[str] = None


class TradeManager:
    """Manages individual trades and position sizing."""
    
    def __init__(
        self,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 0.25,
        max_positions: int = 1
    ):
        """Initialize the trade manager.
        
        Args:
            tick_size: Size of one tick (NQ = 0.25)
            tick_value: Dollar value per tick (NQ = $5)
            commission_per_contract: Commission cost per contract
            slippage_ticks: Estimated slippage in ticks
            max_positions: Maximum concurrent positions
        """
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.max_positions = max_positions
        
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.next_trade_id = 1
        
        self.logger = logging.getLogger(__name__)
    
    def can_open_trade(self) -> bool:
        """Check if a new trade can be opened."""
        return len(self.open_trades) < self.max_positions
    
    def create_trade_from_sweep(
        self,
        sweep_signal,  # SweepSignal from sweep_detector
        entry_time: datetime,
        entry_price: float,
        position_size: float = 1.0,
        max_hold_time_minutes: int = 20,
        risk_reward_ratio: float = 1.0
    ) -> Optional[Trade]:
        """Create a trade from a sweep signal.
        
        Args:
            sweep_signal: The sweep signal that triggered the trade
            entry_time: Time of trade entry
            entry_price: Entry price (after sweep candle closes)
            position_size: Number of contracts
            max_hold_time_minutes: Maximum hold time in minutes
            risk_reward_ratio: Risk-reward ratio (1.0 = 1:1)
            
        Returns:
            Created trade or None if trade cannot be opened
        """
        if not self.can_open_trade():
            self.logger.warning("Cannot open trade - maximum positions reached")
            return None
        
        # Determine trade direction
        if sweep_signal.sweep_type == 'high_sweep':
            direction = TradeDirection.SHORT
        else:
            direction = TradeDirection.LONG
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_take_levels(
            entry_price=entry_price,
            direction=direction,
            h1_open=sweep_signal.h1_open,
            risk_reward_ratio=risk_reward_ratio
        )
        
        # Create trade
        trade = Trade(
            trade_id=self.next_trade_id,
            entry_time=entry_time,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_time=timedelta(minutes=max_hold_time_minutes),
            sweep_type=sweep_signal.sweep_type,
            h1_open=sweep_signal.h1_open,
            h1_high=sweep_signal.h1_high,
            h1_low=sweep_signal.h1_low
        )
        
        self.open_trades.append(trade)
        self.next_trade_id += 1
        
        self.logger.info(f"Opened {direction.value} trade #{trade.trade_id} at {entry_price}")
        return trade
    
    def _calculate_stop_take_levels(
        self,
        entry_price: float,
        direction: TradeDirection,
        h1_open: float,
        risk_reward_ratio: float = 1.0
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels.
        
        The strategy uses distance from entry to H1 open for both SL and TP.
        
        Args:
            entry_price: Trade entry price
            direction: Trade direction
            h1_open: H1 open price
            risk_reward_ratio: Risk-reward ratio
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        distance_to_h1_open = abs(entry_price - h1_open)
        
        if direction == TradeDirection.LONG:
            # Long trade: SL below entry, TP above entry
            stop_loss = entry_price - distance_to_h1_open
            take_profit = entry_price + (distance_to_h1_open * risk_reward_ratio)
        else:
            # Short trade: SL above entry, TP below entry
            stop_loss = entry_price + distance_to_h1_open
            take_profit = entry_price - (distance_to_h1_open * risk_reward_ratio)
        
        # Round to tick size
        stop_loss = round(stop_loss / self.tick_size) * self.tick_size
        take_profit = round(take_profit / self.tick_size) * self.tick_size
        
        return stop_loss, take_profit
    
    def update_trades(self, current_time: datetime, current_price: float) -> List[Trade]:
        """Update all open trades and close any that hit exit conditions.
        
        Args:
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            List of trades that were closed
        """
        closed_trades = []
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            closed_trade = self._check_trade_exit(trade, current_time, current_price)
            if closed_trade:
                closed_trades.append(closed_trade)
                self.open_trades.remove(trade)
                self.closed_trades.append(closed_trade)
        
        return closed_trades
    
    def _check_trade_exit(
        self, 
        trade: Trade, 
        current_time: datetime, 
        current_price: float
    ) -> Optional[Trade]:
        """Check if a trade should be closed.
        
        Args:
            trade: The trade to check
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            Closed trade if exit condition met, None otherwise
        """
        # Check timeout first
        time_in_trade = current_time - trade.entry_time
        if time_in_trade >= trade.max_hold_time:
            return self._close_trade(trade, current_time, current_price, "TIMEOUT")
        
        # Check stop loss and take profit
        if trade.direction == TradeDirection.LONG:
            if current_price <= trade.stop_loss:
                return self._close_trade(trade, current_time, current_price, "STOP_LOSS")
            elif current_price >= trade.take_profit:
                return self._close_trade(trade, current_time, current_price, "TAKE_PROFIT")
        else:  # SHORT
            if current_price >= trade.stop_loss:
                return self._close_trade(trade, current_time, current_price, "STOP_LOSS")
            elif current_price <= trade.take_profit:
                return self._close_trade(trade, current_time, current_price, "TAKE_PROFIT")
        
        return None
    
    def _close_trade(
        self, 
        trade: Trade, 
        exit_time: datetime, 
        exit_price: float, 
        exit_reason: str
    ) -> Trade:
        """Close a trade and calculate final P&L.
        
        Args:
            trade: The trade to close
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Closed trade with P&L calculated
        """
        # Apply slippage
        if trade.direction == TradeDirection.LONG:
            # For long trades, slippage hurts on exit (lower price)
            slipped_exit_price = exit_price - (self.slippage_ticks * self.tick_size)
        else:
            # For short trades, slippage hurts on exit (higher price)
            slipped_exit_price = exit_price + (self.slippage_ticks * self.tick_size)
        
        # Calculate P&L in ticks
        if trade.direction == TradeDirection.LONG:
            pnl_ticks = (slipped_exit_price - trade.entry_price) / self.tick_size
        else:
            pnl_ticks = (trade.entry_price - slipped_exit_price) / self.tick_size
        
        # Calculate P&L in dollars
        pnl_dollars = pnl_ticks * self.tick_value * trade.position_size
        
        # Subtract commission (both entry and exit)
        total_commission = self.commission_per_contract * trade.position_size * 2
        pnl_dollars -= total_commission
        
        # Calculate total slippage cost
        total_slippage = (self.slippage_ticks * 2) * self.tick_value * trade.position_size
        
        # Determine status
        if exit_reason == "TIMEOUT":
            status = TradeStatus.CLOSED_TIMEOUT
        elif pnl_dollars > 0:
            status = TradeStatus.CLOSED_PROFIT
        else:
            status = TradeStatus.CLOSED_LOSS
        
        # Calculate hold time
        hold_time_minutes = int((exit_time - trade.entry_time).total_seconds() / 60)
        
        # Create closed trade
        closed_trade = trade._replace(
            exit_time=exit_time,
            exit_price=slipped_exit_price,
            status=status,
            pnl_ticks=pnl_ticks,
            pnl_dollars=pnl_dollars,
            commission=total_commission,
            slippage=total_slippage,
            hold_time_minutes=hold_time_minutes,
            exit_reason=exit_reason
        )
        
        self.logger.info(
            f"Closed {trade.direction.value} trade #{trade.trade_id}: "
            f"{exit_reason}, P&L: {pnl_dollars:.2f} ({pnl_ticks:.1f} ticks)"
        )
        
        return closed_trade
    
    def force_close_all_trades(self, current_time: datetime, current_price: float) -> List[Trade]:
        """Force close all open trades (e.g., at end of session).
        
        Args:
            current_time: Current market time
            current_price: Current market price
            
        Returns:
            List of all closed trades
        """
        closed_trades = []
        
        for trade in self.open_trades[:]:
            closed_trade = self._close_trade(trade, current_time, current_price, "FORCED_CLOSE")
            closed_trades.append(closed_trade)
            self.closed_trades.append(closed_trade)
        
        self.open_trades.clear()
        
        if closed_trades:
            self.logger.info(f"Force closed {len(closed_trades)} open trades")
        
        return closed_trades
    
    def get_open_positions_count(self) -> int:
        """Get number of open positions."""
        return len(self.open_trades)
    
    def get_trade_history(self) -> List[Trade]:
        """Get all closed trades."""
        return self.closed_trades.copy()
    
    def get_current_positions(self) -> List[Trade]:
        """Get all open trades."""
        return self.open_trades.copy()
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for all open trades.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total unrealized P&L in dollars
        """
        total_unrealized = 0.0
        
        for trade in self.open_trades:
            if trade.direction == TradeDirection.LONG:
                unrealized_ticks = (current_price - trade.entry_price) / self.tick_size
            else:
                unrealized_ticks = (trade.entry_price - current_price) / self.tick_size
            
            unrealized_dollars = unrealized_ticks * self.tick_value * trade.position_size
            total_unrealized += unrealized_dollars
        
        return total_unrealized
    
    def reset(self):
        """Reset the trade manager for a new backtest."""
        self.open_trades.clear()
        self.closed_trades.clear()
        self.next_trade_id = 1
        self.logger.info("Trade manager reset")