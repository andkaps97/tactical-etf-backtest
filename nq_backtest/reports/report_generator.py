"""Report generator for comprehensive backtest results."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

from ..utils.helpers import format_currency, format_percentage


class ReportGenerator:
    """Generate comprehensive reports for backtest results."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_full_report(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        format: str = "html"
    ) -> str:
        """Generate a comprehensive backtest report.
        
        Args:
            backtest_results: Complete backtest results
            analysis_results: Performance analysis results
            format: Report format ('html', 'pdf', or 'text')
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "html":
            return self._generate_html_report(backtest_results, analysis_results, timestamp)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(backtest_results, analysis_results, timestamp)
        elif format.lower() == "text":
            return self._generate_text_report(backtest_results, analysis_results, timestamp)
        else:
            raise ValueError("Format must be 'html', 'pdf', or 'text'")
    
    def _generate_html_report(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        timestamp: str
    ) -> str:
        """Generate HTML report."""
        
        if not JINJA_AVAILABLE:
            return self._generate_simple_html_report(backtest_results, analysis_results, timestamp)
        
        # HTML template
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NQ Sweep-and-Reversion Strategy - Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }
        .metric-label { font-weight: bold; color: #666; font-size: 0.9em; }
        .metric-value { font-size: 1.2em; font-weight: bold; color: #333; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .neutral { color: #ffc107; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .trade-summary { background-color: #e8f4fd; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NQ Sweep-and-Reversion Strategy</h1>
        <h2>Backtest Report</h2>
        <p><strong>Generated:</strong> {{ report_date }}</p>
        <p><strong>Period:</strong> {{ start_date }} to {{ end_date }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {{ 'positive' if summary.total_return_pct > 0 else 'negative' }}">
                    {{ "%.2f"|format(summary.total_return_pct) }}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value {{ 'positive' if summary.cagr_pct > 0 else 'negative' }}">
                    {{ "%.2f"|format(summary.cagr_pct) }}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{{ "%.2f"|format(summary.max_drawdown_pct) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {{ 'positive' if summary.sharpe_ratio > 1 else 'neutral' if summary.sharpe_ratio > 0 else 'negative' }}">
                    {{ "%.2f"|format(summary.sharpe_ratio) }}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {{ 'positive' if summary.win_rate_pct > 50 else 'negative' }}">
                    {{ "%.1f"|format(summary.win_rate_pct) }}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {{ 'positive' if summary.profit_factor > 1 else 'negative' }}">
                    {{ "%.2f"|format(summary.profit_factor) }}
                </div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Trade Analysis</h2>
        <div class="trade-summary">
            <p><strong>Total Trades:</strong> {{ trade_analysis.total_trades }}</p>
            <p><strong>Winning Trades:</strong> {{ trade_analysis.winners }}</p>
            <p><strong>Losing Trades:</strong> {{ trade_analysis.losers }}</p>
            <p><strong>Average Winner:</strong> {{ "%.2f"|format(trade_analysis.avg_winner) }}</p>
            <p><strong>Average Loser:</strong> {{ "%.2f"|format(trade_analysis.avg_loser) }}</p>
            <p><strong>Best Trade:</strong> {{ "%.2f"|format(trade_analysis.best_trade) }}</p>
            <p><strong>Worst Trade:</strong> {{ "%.2f"|format(trade_analysis.worst_trade) }}</p>
        </div>
    </div>

    {% if sweep_analysis %}
    <div class="section">
        <h2>Sweep Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Sweeps Detected</div>
                <div class="metric-value">{{ sweep_analysis.total_sweeps_detected }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Valid Sweeps</div>
                <div class="metric-value">{{ sweep_analysis.valid_sweeps }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Validity Rate</div>
                <div class="metric-value">{{ "%.1f"|format(sweep_analysis.sweep_validity_rate * 100) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Time to Sweep</div>
                <div class="metric-value">{{ "%.1f"|format(sweep_analysis.avg_minutes_to_sweep) }} min</div>
            </div>
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>Risk Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value">{{ "%.2f"|format(summary.sortino_ratio) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-value">{{ "%.2f"|format(summary.calmar_ratio) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Value at Risk (95%)</div>
                <div class="metric-value negative">{{ "%.2f"|format(risk_analysis.value_at_risk_95) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recovery Factor</div>
                <div class="metric-value">{{ "%.2f"|format(risk_analysis.recovery_factor) }}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Sweep Detection Window</td><td>{{ config.strategy.sweep_detection_window }} minutes</td></tr>
            <tr><td>Max Hold Time</td><td>{{ config.strategy.max_hold_time }} minutes</td></tr>
            <tr><td>Risk-Reward Ratio</td><td>{{ config.strategy.risk_reward_ratio }}:1</td></tr>
            <tr><td>Position Size</td><td>{{ config.strategy.position_size }} contracts</td></tr>
            <tr><td>Commission per Contract</td><td>${{ config.strategy.commission_per_contract }}</td></tr>
            <tr><td>Initial Capital</td><td>${{ "{:,.2f}"|format(config.backtest.initial_capital) }}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Disclaimer</h2>
        <p><em>This backtest report is for educational and research purposes only. Past performance does not guarantee future results. 
        Trading futures involves substantial risk and is not suitable for all investors. Always consult with a qualified financial advisor 
        before making trading decisions.</em></p>
    </div>
</body>
</html>
        """
        
        # Prepare template data
        template_data = {
            'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'start_date': backtest_results['config']['start_date'],
            'end_date': backtest_results['config']['end_date'],
            'summary': analysis_results['summary'],
            'trade_analysis': analysis_results['trade_analysis'],
            'risk_analysis': analysis_results.get('risk_analysis', {}),
            'sweep_analysis': analysis_results.get('sweep_analysis', {}),
            'config': backtest_results['config']
        }
        
        # Render template
        template = jinja2.Template(template_str)
        html_content = template.render(**template_data)
        
        # Save to file
        filename = self.output_dir / f"nq_backtest_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated HTML report: {filename}")
        return str(filename)
    
    def _generate_simple_html_report(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        timestamp: str
    ) -> str:
        """Generate simple HTML report without Jinja2."""
        
        performance = backtest_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NQ Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ margin: 10px 0; padding: 10px; background-color: #f5f5f5; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>NQ Sweep-and-Reversion Strategy Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Summary</h2>
    <div class="metric">Total Return: <span class="{'positive' if performance.get('total_return', 0) > 0 else 'negative'}">{format_percentage(performance.get('total_return', 0))}</span></div>
    <div class="metric">CAGR: <span class="{'positive' if performance.get('cagr', 0) > 0 else 'negative'}">{format_percentage(performance.get('cagr', 0))}</span></div>
    <div class="metric">Max Drawdown: <span class="negative">{format_percentage(performance.get('max_drawdown', 0))}</span></div>
    <div class="metric">Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}</div>
    <div class="metric">Win Rate: {format_percentage(performance.get('win_rate', 0))}</div>
    <div class="metric">Total Trades: {performance.get('total_trades', 0)}</div>
    
    <h2>Trade Statistics</h2>
    <div class="metric">Profit Factor: {performance.get('profit_factor', 0):.2f}</div>
    <div class="metric">Average Winner: {format_currency(performance.get('avg_winner', 0))}</div>
    <div class="metric">Average Loser: {format_currency(performance.get('avg_loser', 0))}</div>
    <div class="metric">Max Winner: {format_currency(performance.get('max_winner', 0))}</div>
    <div class="metric">Max Loser: {format_currency(performance.get('max_loser', 0))}</div>
    
</body>
</html>
        """
        
        filename = self.output_dir / f"nq_backtest_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated simple HTML report: {filename}")
        return str(filename)
    
    def _generate_pdf_report(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        timestamp: str
    ) -> str:
        """Generate PDF report."""
        
        if not PDF_AVAILABLE:
            self.logger.warning("PDF generation not available. Install fpdf2: pip install fpdf2")
            return self._generate_text_report(backtest_results, analysis_results, timestamp)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Title
        pdf.cell(0, 10, 'NQ Sweep-and-Reversion Strategy Report', ln=True, align='C')
        pdf.ln(10)
        
        # Performance metrics
        performance = backtest_results.get('performance_metrics', {})
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Performance Summary', ln=True)
        pdf.set_font('Arial', '', 12)
        
        metrics = [
            ('Total Return', format_percentage(performance.get('total_return', 0))),
            ('CAGR', format_percentage(performance.get('cagr', 0))),
            ('Max Drawdown', format_percentage(performance.get('max_drawdown', 0))),
            ('Sharpe Ratio', f"{performance.get('sharpe_ratio', 0):.2f}"),
            ('Win Rate', format_percentage(performance.get('win_rate', 0))),
            ('Total Trades', str(performance.get('total_trades', 0))),
            ('Profit Factor', f"{performance.get('profit_factor', 0):.2f}"),
        ]
        
        for label, value in metrics:
            pdf.cell(60, 8, label + ':', border=0)
            pdf.cell(0, 8, value, border=0, ln=True)
        
        filename = self.output_dir / f"nq_backtest_report_{timestamp}.pdf"
        pdf.output(str(filename))
        
        self.logger.info(f"Generated PDF report: {filename}")
        return str(filename)
    
    def _generate_text_report(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        timestamp: str
    ) -> str:
        """Generate text report."""
        
        performance = backtest_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        trade_analysis = analysis_results.get('trade_analysis', {})
        
        report_lines = [
            "=" * 60,
            "NQ SWEEP-AND-REVERSION STRATEGY BACKTEST REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Period: {backtest_results['config']['start_date']} to {backtest_results['config']['end_date']}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 20,
            f"Total Return:        {format_percentage(performance.get('total_return', 0))}",
            f"CAGR:               {format_percentage(performance.get('cagr', 0))}",
            f"Max Drawdown:       {format_percentage(performance.get('max_drawdown', 0))}",
            f"Sharpe Ratio:       {performance.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio:      {performance.get('sortino_ratio', 0):.2f}",
            f"Win Rate:           {format_percentage(performance.get('win_rate', 0))}",
            f"Profit Factor:      {performance.get('profit_factor', 0):.2f}",
            "",
            "TRADE STATISTICS",
            "-" * 16,
            f"Total Trades:       {performance.get('total_trades', 0)}",
            f"Winning Trades:     {performance.get('winning_trades', 0)}",
            f"Losing Trades:      {performance.get('losing_trades', 0)}",
            f"Average Winner:     {format_currency(performance.get('avg_winner', 0))}",
            f"Average Loser:      {format_currency(performance.get('avg_loser', 0))}",
            f"Best Trade:         {format_currency(performance.get('max_winner', 0))}",
            f"Worst Trade:        {format_currency(performance.get('max_loser', 0))}",
            f"Avg Hold Time:      {performance.get('avg_trade_duration_minutes', 0):.1f} minutes",
            "",
            "STRATEGY CONFIGURATION",
            "-" * 22,
            f"Sweep Window:       {backtest_results['config']['strategy'].sweep_detection_window} minutes",
            f"Max Hold Time:      {backtest_results['config']['strategy'].max_hold_time} minutes",
            f"Risk-Reward:        {backtest_results['config']['strategy'].risk_reward_ratio}:1",
            f"Position Size:      {backtest_results['config']['strategy'].position_size} contracts",
            f"Initial Capital:    {format_currency(backtest_results['config']['backtest'].initial_capital)}",
            "",
            "DISCLAIMER",
            "-" * 10,
            "This backtest report is for educational purposes only.",
            "Past performance does not guarantee future results.",
            "Trading futures involves substantial risk.",
            "",
            "=" * 60
        ]
        
        # Add sweep analysis if available
        if 'sweep_analysis' in analysis_results:
            sweep = analysis_results['sweep_analysis']
            sweep_lines = [
                "",
                "SWEEP ANALYSIS",
                "-" * 14,
                f"Total Sweeps:       {sweep.get('total_sweeps_detected', 0)}",
                f"Valid Sweeps:       {sweep.get('valid_sweeps', 0)}",
                f"Validity Rate:      {format_percentage(sweep.get('sweep_validity_rate', 0))}",
                f"Avg Time to Sweep:  {sweep.get('avg_minutes_to_sweep', 0):.1f} minutes",
            ]
            # Insert after trade statistics
            insert_index = report_lines.index("STRATEGY CONFIGURATION")
            report_lines = report_lines[:insert_index] + sweep_lines + [""] + report_lines[insert_index:]
        
        report_content = "\n".join(report_lines)
        
        filename = self.output_dir / f"nq_backtest_report_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Generated text report: {filename}")
        return str(filename)
    
    def generate_trade_journal(self, trades: List[Dict[str, Any]]) -> str:
        """Generate a detailed trade journal CSV.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Path to generated trade journal file
        """
        if not trades:
            self.logger.warning("No trades to generate journal for")
            return ""
        
        df = pd.DataFrame(trades)
        
        # Add additional calculated columns
        df['r_multiple'] = df['pnl_dollars'] / df.apply(
            lambda row: abs(row['entry_price'] - row['stop_loss']) * 5 / 0.25, axis=1
        )
        df['hold_time_hours'] = df['hold_time_minutes'] / 60
        df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.day_name()
        df['hour_of_day'] = pd.to_datetime(df['entry_time']).dt.hour
        
        # Reorder columns for better readability
        column_order = [
            'trade_id', 'entry_time', 'exit_time', 'direction', 'sweep_type',
            'entry_price', 'exit_price', 'stop_loss', 'take_profit',
            'pnl_dollars', 'pnl_ticks', 'r_multiple', 'hold_time_minutes',
            'exit_reason', 'commission', 'day_of_week', 'hour_of_day'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df_ordered = df[available_columns]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"trade_journal_{timestamp}.csv"
        
        df_ordered.to_csv(filename, index=False)
        self.logger.info(f"Generated trade journal: {filename}")
        
        return str(filename)
    
    def generate_summary_dashboard(
        self, 
        backtest_results: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate a summary dashboard CSV with key metrics.
        
        Args:
            backtest_results: Complete backtest results
            analysis_results: Performance analysis results
            
        Returns:
            Path to generated dashboard file
        """
        performance = backtest_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        
        # Create summary data
        dashboard_data = {
            'Metric': [
                'Total Return (%)', 'CAGR (%)', 'Max Drawdown (%)', 'Sharpe Ratio',
                'Sortino Ratio', 'Calmar Ratio', 'Win Rate (%)', 'Profit Factor',
                'Total Trades', 'Winning Trades', 'Losing Trades',
                'Average Winner ($)', 'Average Loser ($)', 'Best Trade ($)',
                'Worst Trade ($)', 'Average Hold Time (min)', 'Initial Capital ($)',
                'Final Equity ($)'
            ],
            'Value': [
                f"{performance.get('total_return', 0) * 100:.2f}",
                f"{performance.get('cagr', 0) * 100:.2f}",
                f"{performance.get('max_drawdown', 0) * 100:.2f}",
                f"{performance.get('sharpe_ratio', 0):.2f}",
                f"{performance.get('sortino_ratio', 0):.2f}",
                f"{summary.get('calmar_ratio', 0):.2f}",
                f"{performance.get('win_rate', 0) * 100:.1f}",
                f"{performance.get('profit_factor', 0):.2f}",
                str(performance.get('total_trades', 0)),
                str(performance.get('winning_trades', 0)),
                str(performance.get('losing_trades', 0)),
                f"{performance.get('avg_winner', 0):.2f}",
                f"{performance.get('avg_loser', 0):.2f}",
                f"{performance.get('max_winner', 0):.2f}",
                f"{performance.get('max_loser', 0):.2f}",
                f"{performance.get('avg_trade_duration_minutes', 0):.1f}",
                f"{performance.get('initial_capital', 0):,.2f}",
                f"{performance.get('final_equity', 0):,.2f}"
            ]
        }
        
        df = pd.DataFrame(dashboard_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"summary_dashboard_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Generated summary dashboard: {filename}")
        
        return str(filename)