"""Main execution script for the NQ Sweep-and-Reversion Strategy backtest."""

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
import argparse

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nq_backtest.config.strategy_config import StrategyConfig, BacktestConfig
from nq_backtest.config.ib_config import IBConfig, ContractConfig
from nq_backtest.backtest.backtest_engine import BacktestEngine
from nq_backtest.backtest.performance_analyzer import PerformanceAnalyzer
from nq_backtest.reports.report_generator import ReportGenerator
from nq_backtest.utils.logging_config import setup_logging


def create_default_configs():
    """Create default configuration objects."""
    
    # Strategy configuration
    strategy_config = StrategyConfig(
        sweep_detection_window=20,  # 20 minutes
        max_hold_time=20,          # 20 minutes max hold
        risk_reward_ratio=1.0,     # 1:1 risk-reward
        position_size=1.0,         # 1 contract
        commission_per_contract=2.50,
        slippage_ticks=0.25
    )
    
    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=date.today() - timedelta(days=90),  # Last 3 months
        end_date=date.today() - timedelta(days=1),     # Yesterday
        initial_capital=100000.0,
        generate_trade_log=True,
        generate_performance_report=True,
        generate_charts=True
    )
    
    # Interactive Brokers configuration
    ib_config = IBConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=1
    )
    
    # Contract configuration
    contract_config = ContractConfig()
    
    return strategy_config, backtest_config, ib_config, contract_config


async def run_backtest(args):
    """Run the complete backtest process."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting NQ Sweep-and-Reversion Strategy Backtest")
    
    try:
        # Create configurations
        strategy_config, backtest_config, ib_config, contract_config = create_default_configs()
        
        # Override with command line arguments
        if args.start_date:
            backtest_config.start_date = date.fromisoformat(args.start_date)
        if args.end_date:
            backtest_config.end_date = date.fromisoformat(args.end_date)
        if args.capital:
            backtest_config.initial_capital = args.capital
        if args.sweep_window:
            strategy_config.sweep_detection_window = args.sweep_window
        if args.max_hold:
            strategy_config.max_hold_time = args.max_hold
        if args.ib_port:
            ib_config.port = args.ib_port
        
        # Initialize backtest engine
        logger.info("Initializing backtest engine...")
        backtest_engine = BacktestEngine(
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            ib_config=ib_config,
            contract_config=contract_config
        )
        
        # Run backtest
        logger.info("Running backtest...")
        backtest_results = await backtest_engine.run_backtest(
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
            save_results=True
        )
        
        # Performance analysis
        logger.info("Performing analysis...")
        analyzer = PerformanceAnalyzer(output_dir=backtest_config.output_directory)
        analysis_results = analyzer.analyze_results(backtest_results)
        
        # Generate charts
        if backtest_config.generate_charts:
            logger.info("Generating charts...")
            analyzer.generate_charts(backtest_results, save_charts=True)
        
        # Generate reports
        if backtest_config.generate_performance_report:
            logger.info("Generating reports...")
            report_generator = ReportGenerator(output_dir=backtest_config.output_directory)
            
            # Generate HTML report
            html_report = report_generator.generate_full_report(
                backtest_results, analysis_results, format="html"
            )
            logger.info(f"HTML report generated: {html_report}")
            
            # Generate trade journal
            if backtest_results.get('trades'):
                trade_journal = report_generator.generate_trade_journal(backtest_results['trades'])
                logger.info(f"Trade journal generated: {trade_journal}")
            
            # Generate summary dashboard
            dashboard = report_generator.generate_summary_dashboard(
                backtest_results, analysis_results
            )
            logger.info(f"Summary dashboard generated: {dashboard}")
        
        # Print summary to console
        print_summary(backtest_results, analysis_results)
        
        # Cleanup
        await backtest_engine.cleanup()
        
        logger.info("Backtest completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return False


def print_summary(backtest_results, analysis_results):
    """Print a summary of results to console."""
    
    performance = backtest_results.get('performance_metrics', {})
    summary = analysis_results.get('summary', {})
    
    print("\n" + "="*60)
    print("NQ SWEEP-AND-REVERSION STRATEGY - BACKTEST SUMMARY")
    print("="*60)
    
    print(f"Period: {backtest_results['config']['start_date']} to {backtest_results['config']['end_date']}")
    print(f"Initial Capital: ${performance.get('initial_capital', 0):,.2f}")
    print(f"Final Equity: ${performance.get('final_equity', 0):,.2f}")
    print()
    
    print("PERFORMANCE METRICS:")
    print(f"  Total Return:     {performance.get('total_return', 0)*100:+.2f}%")
    print(f"  CAGR:            {performance.get('cagr', 0)*100:+.2f}%")
    print(f"  Max Drawdown:    {performance.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Sharpe Ratio:    {performance.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:   {performance.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio:    {summary.get('calmar_ratio', 0):.2f}")
    print()
    
    print("TRADE STATISTICS:")
    print(f"  Total Trades:    {performance.get('total_trades', 0)}")
    print(f"  Win Rate:        {performance.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor:   {performance.get('profit_factor', 0):.2f}")
    print(f"  Average Winner:  ${performance.get('avg_winner', 0):,.2f}")
    print(f"  Average Loser:   ${performance.get('avg_loser', 0):,.2f}")
    print(f"  Best Trade:      ${performance.get('max_winner', 0):,.2f}")
    print(f"  Worst Trade:     ${performance.get('max_loser', 0):,.2f}")
    print()
    
    # Sweep analysis if available
    if 'sweep_analysis' in analysis_results:
        sweep = analysis_results['sweep_analysis']
        print("SWEEP ANALYSIS:")
        print(f"  Total Sweeps:    {sweep.get('total_sweeps_detected', 0)}")
        print(f"  Valid Sweeps:    {sweep.get('valid_sweeps', 0)}")
        print(f"  Validity Rate:   {sweep.get('sweep_validity_rate', 0)*100:.1f}%")
        print(f"  Avg Time to Sweep: {sweep.get('avg_minutes_to_sweep', 0):.1f} minutes")
        print()
    
    print("="*60)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="NQ Sweep-and-Reversion Strategy Backtest")
    
    # Date parameters
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Strategy parameters
    parser.add_argument('--sweep-window', type=int, help='Sweep detection window in minutes (default: 20)')
    parser.add_argument('--max-hold', type=int, help='Maximum hold time in minutes (default: 20)')
    parser.add_argument('--capital', type=float, help='Initial capital (default: 100000)')
    
    # IB parameters
    parser.add_argument('--ib-port', type=int, help='Interactive Brokers port (default: 7497)')
    
    # Logging parameters
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, help='Log file name (auto-generated if not specified)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                       help='Output directory for results (default: backtest_results)')
    
    # Execution parameters
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-reports', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_console=True,
        enable_file=True
    )
    
    # Run backtest
    success = asyncio.run(run_backtest(args))
    
    if success:
        print("\nBacktest completed successfully! Check the output directory for detailed results.")
        sys.exit(0)
    else:
        print("\nBacktest failed. Check the log files for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()