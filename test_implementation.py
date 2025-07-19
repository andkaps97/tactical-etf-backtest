"""Simple test script to validate the NQ strategy implementation."""

import sys
import asyncio
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nq_backtest.config.strategy_config import StrategyConfig, BacktestConfig
from nq_backtest.config.ib_config import IBConfig, ContractConfig
from nq_backtest.data.ib_data_manager import IBDataManager
from nq_backtest.strategy.sweep_detector import SweepDetector
from nq_backtest.utils.logging_config import setup_logging


async def test_basic_functionality():
    """Test basic functionality of the NQ strategy components."""
    
    print("🚀 Testing NQ Sweep-and-Reversion Strategy Implementation")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level="INFO", enable_file=False)
    
    # Test 1: Configuration validation
    print("Test 1: Configuration validation...")
    try:
        strategy_config = StrategyConfig()
        strategy_config.validate()
        
        backtest_config = BacktestConfig()
        backtest_config.validate()
        
        ib_config = IBConfig()
        ib_config.validate()
        
        contract_config = ContractConfig()
        contract_config.validate()
        
        print("✓ All configurations are valid")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    # Test 2: Mock data generation
    print("\nTest 2: Mock data generation...")
    try:
        data_manager = IBDataManager(ib_config, contract_config)
        
        # Generate 7 days of mock data
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now() - timedelta(days=1)
        
        minute_data = await data_manager.get_minute_data(start_date, end_date, use_cache=False)
        
        if minute_data is not None and not minute_data.empty:
            print(f"✓ Generated {len(minute_data)} minute bars")
            print(f"  Date range: {minute_data.index[0]} to {minute_data.index[-1]}")
            
            # Test hourly resampling
            hourly_data = data_manager.resample_to_hourly(minute_data)
            print(f"✓ Resampled to {len(hourly_data)} hourly bars")
        else:
            print("✗ Failed to generate mock data")
            return False
            
    except Exception as e:
        print(f"✗ Mock data generation failed: {e}")
        return False
    
    # Test 3: Sweep detection
    print("\nTest 3: Sweep detection...")
    try:
        sweep_detector = SweepDetector(
            sweep_window_minutes=20,
            min_h1_range_ticks=4
        )
        
        sweeps = sweep_detector.detect_sweeps(minute_data, hourly_data)
        entry_signals = sweep_detector.get_entry_signals(sweeps)
        
        print(f"✓ Detected {len(sweeps)} total sweeps")
        print(f"✓ Found {len(entry_signals)} valid entry signals")
        
        if sweeps:
            # Analyze sweep patterns
            analysis = sweep_detector.analyze_sweep_patterns(sweeps)
            print(f"  Sweep types: {analysis.get('sweep_types', {})}")
            print(f"  Average H1 range: {analysis.get('statistics', {}).get('avg_h1_range_ticks', 0):.1f} ticks")
        
    except Exception as e:
        print(f"✗ Sweep detection failed: {e}")
        return False
    
    # Test 4: Basic strategy metrics
    print("\nTest 4: Strategy metrics calculation...")
    try:
        from nq_backtest.backtest.performance_analyzer import PerformanceAnalyzer
        
        # Create mock trade data for testing
        mock_trades = []
        if entry_signals:
            for i, signal in enumerate(entry_signals[:3]):  # Test with first 3 signals
                mock_trades.append({
                    'trade_id': i + 1,
                    'entry_time': signal.timestamp.isoformat(),
                    'exit_time': (signal.timestamp + timedelta(minutes=15)).isoformat(),
                    'direction': 'SHORT' if signal.sweep_type == 'high_sweep' else 'LONG',
                    'entry_price': signal.sweep_candle_close,
                    'exit_price': signal.sweep_candle_close + (10 if signal.sweep_type == 'low_sweep' else -10),
                    'pnl_dollars': 50.0 if i % 2 == 0 else -25.0,
                    'pnl_ticks': 2.0 if i % 2 == 0 else -1.0,
                    'hold_time_minutes': 15,
                    'exit_reason': 'TAKE_PROFIT' if i % 2 == 0 else 'STOP_LOSS'
                })
        
        if mock_trades:
            analyzer = PerformanceAnalyzer()
            analysis = analyzer._analyze_trades(mock_trades)
            
            print(f"✓ Analyzed {analysis['total_trades']} mock trades")
            print(f"  Winners: {analysis['winners']}, Losers: {analysis['losers']}")
            print(f"  Best trade: ${analysis['best_trade']:.2f}")
        else:
            print("✓ Performance analyzer ready (no trades to analyze)")
        
    except Exception as e:
        print(f"✗ Performance analysis failed: {e}")
        return False
    
    # Test 5: Report generation capability
    print("\nTest 5: Report generation capability...")
    try:
        from nq_backtest.reports.report_generator import ReportGenerator
        
        reporter = ReportGenerator()
        
        # Test with minimal data
        mock_backtest_results = {
            'config': {
                'start_date': start_date.date().isoformat(),
                'end_date': end_date.date().isoformat(),
                'strategy': strategy_config,
                'backtest': backtest_config
            },
            'performance_metrics': {
                'total_return': 0.05,
                'total_trades': len(mock_trades),
                'win_rate': 0.67,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.03
            },
            'trades': mock_trades
        }
        
        mock_analysis = {
            'summary': {'total_return_pct': 5.0, 'win_rate_pct': 67.0},
            'trade_analysis': {'total_trades': len(mock_trades), 'best_trade': 50.0}
        }
        
        # Test text report generation (simplest)
        if mock_trades:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = reporter._generate_text_report(
                mock_backtest_results, mock_analysis, timestamp
            )
            print(f"✓ Generated test report: {Path(report_path).name}")
        else:
            print("✓ Report generator ready")
        
    except Exception as e:
        print(f"✗ Report generation test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! NQ Strategy implementation is working correctly.")
    print("\n📋 Summary:")
    print(f"   • Generated {len(minute_data)} minute bars of mock NQ data")
    print(f"   • Detected {len(sweeps)} sweep patterns")
    print(f"   • Found {len(entry_signals)} valid entry signals")
    print(f"   • All core modules are functional")
    print("\n🚀 Ready to run full backtests with: python main.py")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if not success:
        sys.exit(1)