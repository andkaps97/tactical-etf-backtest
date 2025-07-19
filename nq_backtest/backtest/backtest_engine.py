"""Backtesting engine for the NQ Sweep-and-Reversion strategy."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from pathlib import Path

from ..config.strategy_config import StrategyConfig, BacktestConfig
from ..config.ib_config import IBConfig, ContractConfig
from ..data.ib_data_manager import IBDataManager
from ..data.data_preprocessor import DataPreprocessor
from ..strategy.strategy_engine import StrategyEngine
from ..strategy.trade_manager import Trade


class BacktestEngine:
    """Main backtesting engine for the NQ strategy."""
    
    def __init__(
        self,
        strategy_config: StrategyConfig,
        backtest_config: BacktestConfig,
        ib_config: IBConfig,
        contract_config: ContractConfig
    ):
        """Initialize the backtest engine.
        
        Args:
            strategy_config: Strategy parameters
            backtest_config: Backtest configuration
            ib_config: Interactive Brokers configuration
            contract_config: Futures contract configuration
        """
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config
        self.ib_config = ib_config
        self.contract_config = contract_config
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = IBDataManager(ib_config, contract_config)
        self.data_preprocessor = DataPreprocessor()
        self.strategy_engine = StrategyEngine(strategy_config)
        
        # Backtest state
        self.is_initialized = False
        self.backtest_results = None
        
    async def initialize(self) -> bool:
        """Initialize the backtest engine.
        
        Returns:
            True if initialization successful
        """
        try:
            # Validate configurations
            self.strategy_config.validate()
            self.backtest_config.validate()
            self.ib_config.validate()
            self.contract_config.validate()
            
            # Initialize strategy
            self.strategy_engine.initialize(self.backtest_config.initial_capital)
            
            self.is_initialized = True
            self.logger.info("Backtest engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backtest engine: {e}")
            return False
    
    async def run_backtest(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run the complete backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing backtest results
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.is_initialized:
            raise RuntimeError("Backtest engine not initialized")
        
        # Set date range
        if start_date is None:
            start_date = self.backtest_config.start_date or (date.today() - timedelta(days=180))
        if end_date is None:
            end_date = self.backtest_config.end_date or date.today()
        
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        
        try:
            # Step 1: Fetch and prepare data
            self.logger.info("Fetching market data...")
            minute_data, hourly_data = await self._fetch_and_prepare_data(start_date, end_date)
            
            if minute_data.empty or hourly_data.empty:
                raise RuntimeError("No market data available for backtest period")
            
            # Step 2: Run strategy
            self.logger.info("Running strategy...")
            strategy_results = await self._run_strategy(minute_data, hourly_data)
            
            # Step 3: Calculate performance metrics
            self.logger.info("Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics(
                strategy_results, minute_data, hourly_data
            )
            
            # Step 4: Compile results
            self.backtest_results = {
                'config': {
                    'strategy': self.strategy_config,
                    'backtest': self.backtest_config,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'data_summary': {
                    'minute_bars': len(minute_data),
                    'hourly_bars': len(hourly_data),
                    'date_range': {
                        'start': minute_data.index[0].isoformat(),
                        'end': minute_data.index[-1].isoformat()
                    }
                },
                'strategy_results': strategy_results,
                'performance_metrics': performance_metrics,
                'trades': self._format_trades_for_output(strategy_results['closed_trades']),
                'equity_curve': self._calculate_equity_curve(strategy_results['closed_trades'])
            }
            
            # Step 5: Save results if requested
            if save_results:
                await self._save_results()
            
            self.logger.info("Backtest completed successfully")
            return self.backtest_results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    async def _fetch_and_prepare_data(
        self, 
        start_date: date, 
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and prepare market data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (minute_data, hourly_data)
        """
        # Convert dates to datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        
        # Fetch minute data
        minute_data = await self.data_manager.get_minute_data(start_datetime, end_datetime)
        
        if minute_data is None or minute_data.empty:
            raise RuntimeError("Failed to fetch minute data")
        
        # Clean and validate data
        is_valid, message = self.data_preprocessor.validate_data(minute_data)
        if not is_valid:
            self.logger.warning(f"Data validation failed: {message}")
        
        minute_data = self.data_preprocessor.clean_data(minute_data)
        minute_data = self.data_preprocessor.add_technical_indicators(minute_data)
        
        # Filter for market hours
        minute_data = self.data_preprocessor.filter_market_hours(minute_data)
        
        # Create hourly data
        hourly_data = self.data_manager.resample_to_hourly(minute_data)
        
        self.logger.info(f"Prepared {len(minute_data)} minute bars and {len(hourly_data)} hourly bars")
        
        return minute_data, hourly_data
    
    async def _run_strategy(
        self, 
        minute_data: pd.DataFrame, 
        hourly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run the strategy on the prepared data.
        
        Args:
            minute_data: Prepared minute data
            hourly_data: Prepared hourly data
            
        Returns:
            Strategy execution results
        """
        # Reset strategy state
        self.strategy_engine.reset()
        self.strategy_engine.initialize(self.backtest_config.initial_capital)
        
        # Process data in chunks to simulate real-time execution
        # For now, process all data at once
        results = self.strategy_engine.process_data(minute_data, hourly_data)
        
        # Force close any remaining positions
        if results['open_trades']:
            last_timestamp = minute_data.index[-1]
            last_price = minute_data.iloc[-1]['close']
            self.strategy_engine.force_close_all_positions(last_timestamp, last_price)
            
            # Update results with forced closures
            results['closed_trades'] = self.strategy_engine.trade_manager.get_trade_history()
            results['open_trades'] = self.strategy_engine.trade_manager.get_current_positions()
            results['final_equity'] = self.strategy_engine.current_equity
        
        return results
    
    def _calculate_performance_metrics(
        self, 
        strategy_results: Dict[str, Any], 
        minute_data: pd.DataFrame, 
        hourly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Args:
            strategy_results: Results from strategy execution
            minute_data: Minute market data
            hourly_data: Hourly market data
            
        Returns:
            Dictionary of performance metrics
        """
        closed_trades = strategy_results['closed_trades']
        initial_capital = self.backtest_config.initial_capital
        final_equity = strategy_results['final_equity']
        
        if not closed_trades:
            return self._create_empty_performance_metrics()
        
        # Basic metrics
        total_return = (final_equity - initial_capital) / initial_capital
        total_trades = len(closed_trades)
        
        # Trade statistics
        pnl_values = [trade.pnl_dollars for trade in closed_trades]
        winners = [pnl for pnl in pnl_values if pnl > 0]
        losers = [pnl for pnl in pnl_values if pnl <= 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0
        avg_winner = np.mean(winners) if winners else 0.0
        avg_loser = np.mean(losers) if losers else 0.0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        daily_returns = self._calculate_daily_returns(closed_trades, minute_data)
        
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(closed_trades, initial_capital)
        
        # Time-based metrics
        trade_durations = [trade.hold_time_minutes for trade in closed_trades if trade.hold_time_minutes]
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
        
        # Calculate CAGR
        start_date = minute_data.index[0].date()
        end_date = minute_data.index[-1].date()
        days_in_backtest = (end_date - start_date).days
        years_in_backtest = days_in_backtest / 365.25
        
        cagr = (final_equity / initial_capital) ** (1 / years_in_backtest) - 1 if years_in_backtest > 0 else 0.0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_winner': max(pnl_values) if pnl_values else 0.0,
            'max_loser': min(pnl_values) if pnl_values else 0.0,
            'avg_trade_duration_minutes': avg_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'backtest_period_days': days_in_backtest,
            'backtest_period_years': years_in_backtest
        }
    
    def _calculate_daily_returns(self, trades: List[Trade], minute_data: pd.DataFrame) -> pd.Series:
        """Calculate daily returns for risk metric calculations."""
        if not trades:
            return pd.Series(dtype=float)
        
        # Group trades by date and sum P&L
        daily_pnl = {}
        for trade in trades:
            if trade.exit_time:
                trade_date = trade.exit_time.date()
                daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + trade.pnl_dollars
        
        # Create complete date range
        start_date = minute_data.index[0].date()
        end_date = minute_data.index[-1].date()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Convert to returns series
        daily_returns = pd.Series(index=date_range, dtype=float)
        daily_returns = daily_returns.fillna(0.0)
        
        for trade_date, pnl in daily_pnl.items():
            if trade_date in daily_returns.index:
                daily_returns[trade_date] = pnl / self.backtest_config.initial_capital
        
        return daily_returns
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
        
        excess_returns = daily_returns - (self.backtest_config.risk_free_rate / 365)
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if daily_returns.empty:
            return 0.0
        
        downside_returns = daily_returns[daily_returns < 0]
        if downside_returns.empty or downside_returns.std() == 0:
            return float('inf') if daily_returns.mean() > 0 else 0.0
        
        excess_returns = daily_returns - (self.backtest_config.risk_free_rate / 365)
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, trades: List[Trade], initial_capital: float) -> float:
        """Calculate maximum drawdown."""
        if not trades:
            return 0.0
        
        # Calculate running equity
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl_dollars)
        
        # Calculate drawdowns
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_equity_curve(self, trades: List[Trade]) -> pd.DataFrame:
        """Calculate equity curve over time."""
        if not trades:
            return pd.DataFrame()
        
        equity_data = []
        running_equity = self.backtest_config.initial_capital
        
        equity_data.append({
            'timestamp': trades[0].entry_time if trades else datetime.now(),
            'equity': running_equity,
            'trade_pnl': 0.0,
            'cumulative_pnl': 0.0
        })
        
        cumulative_pnl = 0.0
        for trade in trades:
            if trade.exit_time:
                running_equity += trade.pnl_dollars
                cumulative_pnl += trade.pnl_dollars
                
                equity_data.append({
                    'timestamp': trade.exit_time,
                    'equity': running_equity,
                    'trade_pnl': trade.pnl_dollars,
                    'cumulative_pnl': cumulative_pnl
                })
        
        df = pd.DataFrame(equity_data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _format_trades_for_output(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Format trades for output/export."""
        formatted_trades = []
        
        for trade in trades:
            formatted_trades.append({
                'trade_id': trade.trade_id,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'direction': trade.direction.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl_dollars': trade.pnl_dollars,
                'pnl_ticks': trade.pnl_ticks,
                'commission': trade.commission,
                'hold_time_minutes': trade.hold_time_minutes,
                'exit_reason': trade.exit_reason,
                'sweep_type': trade.sweep_type,
                'h1_open': trade.h1_open,
                'h1_high': trade.h1_high,
                'h1_low': trade.h1_low,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit
            })
        
        return formatted_trades
    
    def _create_empty_performance_metrics(self) -> Dict[str, Any]:
        """Create empty performance metrics structure."""
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'max_winner': 0.0,
            'max_loser': 0.0,
            'avg_trade_duration_minutes': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'initial_capital': self.backtest_config.initial_capital,
            'final_equity': self.backtest_config.initial_capital,
            'backtest_period_days': 0,
            'backtest_period_years': 0.0
        }
    
    async def _save_results(self):
        """Save backtest results to files."""
        if not self.backtest_results:
            return
        
        output_dir = Path(self.backtest_config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade log
        if self.backtest_config.generate_trade_log and self.backtest_results['trades']:
            trades_df = pd.DataFrame(self.backtest_results['trades'])
            trade_file = output_dir / f"trade_log_{timestamp}.csv"
            trades_df.to_csv(trade_file, index=False)
            self.logger.info(f"Saved trade log to {trade_file}")
        
        # Save equity curve
        equity_curve = self.backtest_results['equity_curve']
        if not equity_curve.empty:
            equity_file = output_dir / f"equity_curve_{timestamp}.csv"
            equity_curve.to_csv(equity_file)
            self.logger.info(f"Saved equity curve to {equity_file}")
        
        # Save performance metrics
        metrics_df = pd.DataFrame([self.backtest_results['performance_metrics']])
        metrics_file = output_dir / f"performance_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        self.logger.info(f"Saved performance metrics to {metrics_file}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.data_manager:
            await self.data_manager.disconnect()
        self.logger.info("Backtest engine cleanup completed")