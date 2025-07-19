"""Main strategy engine for the NQ Sweep-and-Reversion strategy."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from .sweep_detector import SweepDetector, SweepSignal
from .trade_manager import TradeManager, Trade, TradeDirection, TradeStatus
from ..config.strategy_config import StrategyConfig


class StrategyEngine:
    """Main engine that orchestrates the NQ Sweep-and-Reversion strategy."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize the strategy engine.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sweep_detector = SweepDetector(
            sweep_window_minutes=config.sweep_detection_window,
            min_h1_range_ticks=config.min_h1_range_ticks
        )
        
        self.trade_manager = TradeManager(
            tick_size=config.tick_size,
            tick_value=config.tick_value,
            commission_per_contract=config.commission_per_contract,
            slippage_ticks=config.slippage_ticks,
            max_positions=config.max_positions
        )
        
        # Strategy state
        self.is_initialized = False
        self.current_equity = 0.0
        self.strategy_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'signals_skipped': 0
        }
    
    def initialize(self, initial_capital: float = 100000.0):
        """Initialize the strategy with starting capital.
        
        Args:
            initial_capital: Starting capital for the strategy
        """
        self.current_equity = initial_capital
        self.is_initialized = True
        self.logger.info(f"Strategy initialized with ${initial_capital:,.2f}")
    
    def process_data(
        self, 
        minute_data: pd.DataFrame, 
        hourly_data: pd.DataFrame
    ) -> Dict[str, any]:
        """Process market data and execute strategy logic.
        
        Args:
            minute_data: 1-minute OHLCV data
            hourly_data: 1-hour OHLCV data
            
        Returns:
            Dictionary containing execution results
        """
        if not self.is_initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        if minute_data.empty or hourly_data.empty:
            self.logger.warning("Empty data provided to strategy")
            return self._create_empty_result()
        
        self.logger.info(f"Processing data: {len(minute_data)} minute bars, {len(hourly_data)} hourly bars")
        
        # Detect sweep signals
        sweep_signals = self.sweep_detector.detect_sweeps(minute_data, hourly_data)
        entry_signals = self.sweep_detector.get_entry_signals(sweep_signals)
        
        self.strategy_stats['signals_generated'] += len(entry_signals)
        
        # Execute trades based on signals
        execution_results = self._execute_strategy(entry_signals, minute_data)
        
        # Update all open trades throughout the data period
        self._update_trades_with_market_data(minute_data)
        
        # Compile results
        results = {
            'sweep_signals': sweep_signals,
            'entry_signals': entry_signals,
            'execution_results': execution_results,
            'final_equity': self.current_equity,
            'open_trades': self.trade_manager.get_current_positions(),
            'closed_trades': self.trade_manager.get_trade_history(),
            'strategy_stats': self.strategy_stats.copy()
        }
        
        return results
    
    def _execute_strategy(
        self, 
        entry_signals: List[SweepSignal], 
        minute_data: pd.DataFrame
    ) -> List[Dict[str, any]]:
        """Execute trades based on entry signals.
        
        Args:
            entry_signals: List of valid entry signals
            minute_data: Minute market data for execution
            
        Returns:
            List of execution results
        """
        execution_results = []
        
        for signal in entry_signals:
            result = self._execute_signal(signal, minute_data)
            execution_results.append(result)
            
            if result['executed']:
                self.strategy_stats['trades_executed'] += 1
            else:
                self.strategy_stats['signals_skipped'] += 1
        
        return execution_results
    
    def _execute_signal(
        self, 
        signal: SweepSignal, 
        minute_data: pd.DataFrame
    ) -> Dict[str, any]:
        """Execute a single entry signal.
        
        Args:
            signal: The sweep signal to execute
            minute_data: Minute market data
            
        Returns:
            Execution result dictionary
        """
        # Find the entry bar (after the sweep candle closes)
        entry_time = signal.timestamp + timedelta(minutes=1)
        
        # Find the corresponding market data
        entry_data = minute_data[minute_data.index >= entry_time]
        
        if entry_data.empty:
            return {
                'signal': signal,
                'executed': False,
                'reason': 'No market data available for entry',
                'trade': None
            }
        
        # Entry is at the open of the next candle after sweep
        entry_bar = entry_data.iloc[0]
        entry_price = entry_bar['open']
        
        # Check if we can open a trade
        if not self.trade_manager.can_open_trade():
            return {
                'signal': signal,
                'executed': False,
                'reason': 'Maximum positions already open',
                'trade': None
            }
        
        # Check if we have sufficient capital (margin requirements)
        if not self._check_margin_requirements():
            return {
                'signal': signal,
                'executed': False,
                'reason': 'Insufficient margin',
                'trade': None
            }
        
        # Create and execute trade
        trade = self.trade_manager.create_trade_from_sweep(
            sweep_signal=signal,
            entry_time=entry_time,
            entry_price=entry_price,
            position_size=self.config.position_size,
            max_hold_time_minutes=self.config.max_hold_time,
            risk_reward_ratio=self.config.risk_reward_ratio
        )
        
        if trade:
            return {
                'signal': signal,
                'executed': True,
                'reason': 'Trade opened successfully',
                'trade': trade
            }
        else:
            return {
                'signal': signal,
                'executed': False,
                'reason': 'Failed to create trade',
                'trade': None
            }
    
    def _update_trades_with_market_data(self, minute_data: pd.DataFrame):
        """Update all trades with market data to check for exits.
        
        Args:
            minute_data: Minute market data
        """
        if self.trade_manager.get_open_positions_count() == 0:
            return
        
        for timestamp, bar in minute_data.iterrows():
            # Use typical price for exit calculations
            current_price = (bar['high'] + bar['low'] + bar['close']) / 3
            
            # Update trades
            closed_trades = self.trade_manager.update_trades(timestamp, current_price)
            
            # Update equity for closed trades
            for trade in closed_trades:
                self.current_equity += trade.pnl_dollars
    
    def _check_margin_requirements(self) -> bool:
        """Check if there's sufficient margin for a new trade.
        
        Returns:
            True if margin requirements are met
        """
        # Simplified margin check - in reality this would be more complex
        required_margin = 20000.0 * self.config.position_size  # Approximate NQ margin
        available_equity = self.current_equity
        
        # Account for unrealized P&L if there are open positions
        if self.trade_manager.get_open_positions_count() > 0:
            # We'd need current price to calculate this properly
            # For now, assume adequate margin if we have positive equity
            pass
        
        return available_equity >= required_margin
    
    def _create_empty_result(self) -> Dict[str, any]:
        """Create empty result structure."""
        return {
            'sweep_signals': [],
            'entry_signals': [],
            'execution_results': [],
            'final_equity': self.current_equity,
            'open_trades': [],
            'closed_trades': [],
            'strategy_stats': self.strategy_stats.copy()
        }
    
    def force_close_all_positions(self, current_time: datetime, current_price: float):
        """Force close all open positions (e.g., at end of trading session).
        
        Args:
            current_time: Current time
            current_price: Current market price
        """
        closed_trades = self.trade_manager.force_close_all_trades(current_time, current_price)
        
        # Update equity
        for trade in closed_trades:
            self.current_equity += trade.pnl_dollars
        
        self.logger.info(f"Force closed {len(closed_trades)} positions")
    
    def get_strategy_statistics(self) -> Dict[str, any]:
        """Get comprehensive strategy statistics.
        
        Returns:
            Dictionary of strategy statistics
        """
        closed_trades = self.trade_manager.get_trade_history()
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'avg_hold_time': 0.0
            }
        
        # Basic statistics
        total_trades = len(closed_trades)
        winners = [t for t in closed_trades if t.pnl_dollars > 0]
        losers = [t for t in closed_trades if t.pnl_dollars <= 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(t.pnl_dollars for t in closed_trades)
        
        avg_winner = np.mean([t.pnl_dollars for t in winners]) if winners else 0.0
        avg_loser = np.mean([t.pnl_dollars for t in losers]) if losers else 0.0
        
        gross_profit = sum(t.pnl_dollars for t in winners)
        gross_loss = abs(sum(t.pnl_dollars for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_hold_time = np.mean([t.hold_time_minutes for t in closed_trades]) if closed_trades else 0.0
        
        # Trade distribution by exit reason
        exit_reasons = {}
        for trade in closed_trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # P&L distribution
        pnl_values = [t.pnl_dollars for t in closed_trades]
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_hold_time_minutes': avg_hold_time,
            'max_winner': max(pnl_values) if pnl_values else 0.0,
            'max_loser': min(pnl_values) if pnl_values else 0.0,
            'exit_reason_distribution': exit_reasons,
            'strategy_stats': self.strategy_stats.copy()
        }
    
    def reset(self):
        """Reset the strategy for a new backtest."""
        self.trade_manager.reset()
        self.current_equity = 0.0
        self.is_initialized = False
        self.strategy_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'signals_skipped': 0
        }
        self.logger.info("Strategy engine reset")
    
    def validate_configuration(self) -> Tuple[bool, str]:
        """Validate the strategy configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.config.validate()
            return True, "Configuration is valid"
        except ValueError as e:
            return False, str(e)