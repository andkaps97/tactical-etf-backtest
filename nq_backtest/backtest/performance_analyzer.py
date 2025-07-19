"""Performance analysis for backtest results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PerformanceAnalyzer:
    """Comprehensive performance analysis for backtest results."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the performance analyzer.
        
        Args:
            output_dir: Directory to save reports and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of backtest results.
        
        Args:
            backtest_results: Complete backtest results
            
        Returns:
            Dictionary containing detailed analysis
        """
        self.logger.info("Starting comprehensive performance analysis")
        
        trades = backtest_results.get('trades', [])
        performance_metrics = backtest_results.get('performance_metrics', {})
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        
        analysis = {
            'summary': self._create_performance_summary(performance_metrics),
            'trade_analysis': self._analyze_trades(trades),
            'risk_analysis': self._analyze_risk_metrics(trades, equity_curve),
            'time_analysis': self._analyze_time_patterns(trades),
            'drawdown_analysis': self._analyze_drawdowns(equity_curve),
            'distribution_analysis': self._analyze_pnl_distribution(trades)
        }
        
        # Add strategy-specific analysis
        strategy_results = backtest_results.get('strategy_results', {})
        if strategy_results:
            analysis['sweep_analysis'] = self._analyze_sweep_patterns(strategy_results)
        
        self.logger.info("Performance analysis completed")
        return analysis
    
    def _create_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a high-level performance summary."""
        return {
            'total_return_pct': metrics.get('total_return', 0.0) * 100,
            'cagr_pct': metrics.get('cagr', 0.0) * 100,
            'max_drawdown_pct': metrics.get('max_drawdown', 0.0) * 100,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'sortino_ratio': metrics.get('sortino_ratio', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'win_rate_pct': metrics.get('win_rate', 0.0) * 100,
            'total_trades': metrics.get('total_trades', 0),
            'avg_trade_duration_hours': metrics.get('avg_trade_duration_minutes', 0) / 60,
            'calmar_ratio': (
                metrics.get('cagr', 0.0) / metrics.get('max_drawdown', 0.001)
                if metrics.get('max_drawdown', 0) > 0 else 0.0
            )
        }
    
    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual trade performance."""
        if not trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(df)
        winners = df[df['pnl_dollars'] > 0]
        losers = df[df['pnl_dollars'] <= 0]
        
        # Trade direction analysis
        direction_stats = df['direction'].value_counts()
        direction_pnl = df.groupby('direction')['pnl_dollars'].agg(['count', 'sum', 'mean', 'std'])
        
        # Exit reason analysis
        exit_reason_stats = df['exit_reason'].value_counts()
        exit_reason_pnl = df.groupby('exit_reason')['pnl_dollars'].agg(['count', 'sum', 'mean'])
        
        # Consecutive wins/losses
        df['is_winner'] = df['pnl_dollars'] > 0
        consecutive_stats = self._calculate_consecutive_runs(df['is_winner'])
        
        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'direction_statistics': direction_stats.to_dict(),
            'direction_pnl_analysis': direction_pnl.to_dict(),
            'exit_reason_statistics': exit_reason_stats.to_dict(),
            'exit_reason_pnl_analysis': exit_reason_pnl.to_dict(),
            'consecutive_runs': consecutive_stats,
            'best_trade': float(df['pnl_dollars'].max()),
            'worst_trade': float(df['pnl_dollars'].min()),
            'avg_winner': float(winners['pnl_dollars'].mean()) if not winners.empty else 0.0,
            'avg_loser': float(losers['pnl_dollars'].mean()) if not losers.empty else 0.0,
            'largest_winner_ticks': float(df['pnl_ticks'].max()),
            'largest_loser_ticks': float(df['pnl_ticks'].min()),
        }
    
    def _analyze_risk_metrics(
        self, 
        trades: List[Dict[str, Any]], 
        equity_curve: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze risk-related metrics."""
        if not trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(trades)
        
        # VaR calculation (95% and 99%)
        pnl_values = df['pnl_dollars'].values
        var_95 = np.percentile(pnl_values, 5) if len(pnl_values) > 0 else 0.0
        var_99 = np.percentile(pnl_values, 1) if len(pnl_values) > 0 else 0.0
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(pnl_values[pnl_values <= var_95]) if len(pnl_values) > 0 else 0.0
        
        # Risk-adjusted returns
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            returns = equity_curve['equity'].pct_change().dropna()
            
            # Ulcer Index
            running_max = equity_curve['equity'].expanding().max()
            drawdowns = (equity_curve['equity'] - running_max) / running_max
            ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
            
            # Gain-to-Pain ratio
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            gain_to_pain = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        else:
            ulcer_index = 0.0
            gain_to_pain = 0.0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'expected_shortfall_95': es_95,
            'ulcer_index': ulcer_index,
            'gain_to_pain_ratio': gain_to_pain,
            'largest_loss_streak': self._calculate_largest_loss_streak(df),
            'recovery_factor': self._calculate_recovery_factor(trades, equity_curve)
        }
    
    def _analyze_time_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time-based patterns in trading performance."""
        if not trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(trades)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Extract time components
        df['hour'] = df['entry_time'].dt.hour
        df['day_of_week'] = df['entry_time'].dt.dayofweek
        df['month'] = df['entry_time'].dt.month
        
        # Performance by time periods
        hourly_stats = df.groupby('hour')['pnl_dollars'].agg(['count', 'sum', 'mean']).round(2)
        daily_stats = df.groupby('day_of_week')['pnl_dollars'].agg(['count', 'sum', 'mean']).round(2)
        monthly_stats = df.groupby('month')['pnl_dollars'].agg(['count', 'sum', 'mean']).round(2)
        
        # Hold time analysis
        hold_times = df['hold_time_minutes'].dropna()
        hold_time_stats = {
            'mean_minutes': float(hold_times.mean()) if not hold_times.empty else 0.0,
            'median_minutes': float(hold_times.median()) if not hold_times.empty else 0.0,
            'std_minutes': float(hold_times.std()) if not hold_times.empty else 0.0,
            'min_minutes': float(hold_times.min()) if not hold_times.empty else 0.0,
            'max_minutes': float(hold_times.max()) if not hold_times.empty else 0.0
        }
        
        return {
            'hourly_performance': hourly_stats.to_dict(),
            'daily_performance': daily_stats.to_dict(),
            'monthly_performance': monthly_stats.to_dict(),
            'hold_time_statistics': hold_time_stats,
            'trades_per_day': len(df) / len(df['entry_time'].dt.date.unique()) if not df.empty else 0.0
        }
    
    def _analyze_drawdowns(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {'error': 'No equity curve data available'}
        
        equity = equity_curve['equity']
        
        # Calculate running maximum and drawdowns
        running_max = equity.expanding().max()
        drawdowns = (equity - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_changes = in_drawdown.diff()
        
        # Start and end points of drawdown periods
        dd_starts = drawdown_changes[drawdown_changes == True].index
        dd_ends = drawdown_changes[drawdown_changes == False].index
        
        if len(dd_starts) > len(dd_ends):
            # Add the final timestamp if we end in a drawdown
            dd_ends = dd_ends.append(pd.Index([equity.index[-1]]))
        
        drawdown_periods = []
        for start, end in zip(dd_starts, dd_ends):
            period_dd = drawdowns[start:end]
            if not period_dd.empty:
                drawdown_periods.append({
                    'start': start,
                    'end': end,
                    'duration_days': (end - start).days,
                    'max_drawdown': float(period_dd.min()),
                    'recovery_time_days': 0  # Would need additional calculation
                })
        
        # Statistics
        if drawdown_periods:
            avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            avg_duration = np.mean([dd['duration_days'] for dd in drawdown_periods])
            max_drawdown_period = min(drawdown_periods, key=lambda x: x['max_drawdown'])
        else:
            avg_drawdown = 0.0
            avg_duration = 0.0
            max_drawdown_period = None
        
        return {
            'max_drawdown': float(drawdowns.min()),
            'average_drawdown': avg_drawdown,
            'number_of_drawdown_periods': len(drawdown_periods),
            'average_drawdown_duration_days': avg_duration,
            'longest_drawdown_period': max_drawdown_period,
            'current_drawdown': float(drawdowns.iloc[-1]) if not drawdowns.empty else 0.0
        }
    
    def _analyze_pnl_distribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze P&L distribution characteristics."""
        if not trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(trades)
        pnl_values = df['pnl_dollars'].values
        
        # Distribution statistics
        distribution_stats = {
            'mean': float(np.mean(pnl_values)),
            'median': float(np.median(pnl_values)),
            'std': float(np.std(pnl_values)),
            'skewness': float(self._calculate_skewness(pnl_values)),
            'kurtosis': float(self._calculate_kurtosis(pnl_values)),
            'percentile_5': float(np.percentile(pnl_values, 5)),
            'percentile_25': float(np.percentile(pnl_values, 25)),
            'percentile_75': float(np.percentile(pnl_values, 75)),
            'percentile_95': float(np.percentile(pnl_values, 95))
        }
        
        # Create bins for histogram analysis
        n_bins = min(20, len(pnl_values) // 5) if len(pnl_values) > 10 else 10
        hist_counts, hist_bins = np.histogram(pnl_values, bins=n_bins)
        
        return {
            'distribution_statistics': distribution_stats,
            'histogram': {
                'counts': hist_counts.tolist(),
                'bins': hist_bins.tolist()
            }
        }
    
    def _analyze_sweep_patterns(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sweep-specific patterns."""
        sweep_signals = strategy_results.get('sweep_signals', [])
        entry_signals = strategy_results.get('entry_signals', [])
        
        if not sweep_signals:
            return {'error': 'No sweep signals to analyze'}
        
        # Convert to DataFrame for analysis
        sweep_data = []
        for signal in sweep_signals:
            sweep_data.append({
                'sweep_type': signal.sweep_type,
                'minutes_from_h1_open': signal.minutes_from_h1_open,
                'is_valid': signal.is_valid,
                'h1_range_ticks': (signal.h1_high - signal.h1_low) / 0.25,  # Assuming NQ tick size
                'hour': signal.timestamp.hour
            })
        
        df = pd.DataFrame(sweep_data)
        
        # Sweep statistics
        sweep_stats = {
            'total_sweeps_detected': len(sweep_signals),
            'valid_sweeps': len(entry_signals),
            'sweep_validity_rate': len(entry_signals) / len(sweep_signals) if sweep_signals else 0.0,
            'sweep_type_distribution': df['sweep_type'].value_counts().to_dict(),
            'avg_minutes_to_sweep': float(df['minutes_from_h1_open'].mean()),
            'avg_h1_range_ticks': float(df['h1_range_ticks'].mean()),
            'hourly_sweep_distribution': df['hour'].value_counts().to_dict()
        }
        
        return sweep_stats
    
    def _calculate_consecutive_runs(self, is_winner: pd.Series) -> Dict[str, Any]:
        """Calculate consecutive win/loss runs."""
        runs = []
        current_run = 0
        current_type = None
        
        for winner in is_winner:
            if winner == current_type:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append({'type': current_type, 'length': current_run})
                current_run = 1
                current_type = winner
        
        # Add final run
        if current_run > 0:
            runs.append({'type': current_type, 'length': current_run})
        
        if not runs:
            return {'max_winning_streak': 0, 'max_losing_streak': 0}
        
        winning_runs = [r['length'] for r in runs if r['type'] == True]
        losing_runs = [r['length'] for r in runs if r['type'] == False]
        
        return {
            'max_winning_streak': max(winning_runs) if winning_runs else 0,
            'max_losing_streak': max(losing_runs) if losing_runs else 0,
            'avg_winning_streak': np.mean(winning_runs) if winning_runs else 0.0,
            'avg_losing_streak': np.mean(losing_runs) if losing_runs else 0.0
        }
    
    def _calculate_largest_loss_streak(self, df: pd.DataFrame) -> float:
        """Calculate the largest consecutive loss in dollars."""
        if df.empty:
            return 0.0
        
        # Calculate cumulative sum of losses during losing streaks
        df['is_loss'] = df['pnl_dollars'] < 0
        df['loss_group'] = (df['is_loss'] != df['is_loss'].shift()).cumsum()
        
        loss_streaks = df[df['is_loss']].groupby('loss_group')['pnl_dollars'].sum()
        
        return abs(float(loss_streaks.min())) if not loss_streaks.empty else 0.0
    
    def _calculate_recovery_factor(
        self, 
        trades: List[Dict[str, Any]], 
        equity_curve: pd.DataFrame
    ) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        if not trades or equity_curve.empty:
            return 0.0
        
        net_profit = sum(trade['pnl_dollars'] for trade in trades)
        
        if 'equity' in equity_curve.columns:
            equity = equity_curve['equity']
            running_max = equity.expanding().max()
            drawdowns = (equity - running_max) / running_max
            max_drawdown = abs(float(drawdowns.min()))
        else:
            max_drawdown = 0.001  # Avoid division by zero
        
        return net_profit / max_drawdown if max_drawdown > 0 else 0.0
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        skew = np.sum(((values - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return skew
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of a distribution."""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        kurt = (np.sum(((values - mean) / std) ** 4) * n * (n + 1) / 
                ((n - 1) * (n - 2) * (n - 3))) - (3 * (n - 1) ** 2 / 
                ((n - 2) * (n - 3)))
        return kurt
    
    def generate_charts(
        self, 
        backtest_results: Dict[str, Any], 
        save_charts: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive charts for the backtest results.
        
        Args:
            backtest_results: Complete backtest results
            save_charts: Whether to save charts to files
            
        Returns:
            Dictionary containing chart information
        """
        self.logger.info("Generating performance charts")
        
        charts_created = []
        
        # 1. Equity Curve
        if not backtest_results.get('equity_curve', pd.DataFrame()).empty:
            equity_chart = self._create_equity_curve_chart(
                backtest_results['equity_curve'], save_charts
            )
            charts_created.append(equity_chart)
        
        # 2. Trade Distribution
        if backtest_results.get('trades'):
            trade_dist_chart = self._create_trade_distribution_chart(
                backtest_results['trades'], save_charts
            )
            charts_created.append(trade_dist_chart)
        
        # 3. Monthly Returns Heatmap
        if backtest_results.get('trades'):
            monthly_chart = self._create_monthly_returns_chart(
                backtest_results['trades'], save_charts
            )
            charts_created.append(monthly_chart)
        
        # 4. Drawdown Chart
        if not backtest_results.get('equity_curve', pd.DataFrame()).empty:
            drawdown_chart = self._create_drawdown_chart(
                backtest_results['equity_curve'], save_charts
            )
            charts_created.append(drawdown_chart)
        
        self.logger.info(f"Generated {len(charts_created)} charts")
        return {'charts_created': charts_created}
    
    def _create_equity_curve_chart(self, equity_curve: pd.DataFrame, save: bool) -> str:
        """Create equity curve chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(equity_curve.index, equity_curve['equity'], linewidth=2, color='blue')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / 'equity_curve.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filename)
        else:
            plt.show()
            return "equity_curve_displayed"
    
    def _create_trade_distribution_chart(self, trades: List[Dict], save: bool) -> str:
        """Create trade P&L distribution chart."""
        df = pd.DataFrame(trades)
        pnl_values = df['pnl_dollars']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(pnl_values, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(pnl_values, vert=True)
        ax2.set_title('Trade P&L Box Plot', fontweight='bold')
        ax2.set_ylabel('P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / 'trade_distribution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filename)
        else:
            plt.show()
            return "trade_distribution_displayed"
    
    def _create_monthly_returns_chart(self, trades: List[Dict], save: bool) -> str:
        """Create monthly returns heatmap."""
        df = pd.DataFrame(trades)
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['year'] = df['exit_time'].dt.year
        df['month'] = df['exit_time'].dt.month
        
        # Group by year and month
        monthly_pnl = df.groupby(['year', 'month'])['pnl_dollars'].sum().reset_index()
        
        # Create pivot table
        pivot_table = monthly_pnl.pivot(index='year', columns='month', values='pnl_dollars')
        pivot_table = pivot_table.fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'P&L ($)'})
        
        ax.set_title('Monthly Returns Heatmap', fontweight='bold', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / 'monthly_returns.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filename)
        else:
            plt.show()
            return "monthly_returns_displayed"
    
    def _create_drawdown_chart(self, equity_curve: pd.DataFrame, save: bool) -> str:
        """Create drawdown chart."""
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Equity curve
        ax1.plot(equity.index, equity, linewidth=2, color='blue', label='Equity')
        ax1.plot(running_max.index, running_max, linewidth=1, color='red', 
                alpha=0.7, label='Running Maximum')
        ax1.set_title('Equity Curve with Running Maximum', fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        ax2.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown, linewidth=1, color='red')
        ax2.set_title('Drawdown', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / 'drawdown_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filename)
        else:
            plt.show()
            return "drawdown_analysis_displayed"