# NQ E-mini Futures Sweep-and-Reversion Strategy

A comprehensive Python backtesting framework for Nasdaq 100 E-mini Futures implementing a sweep-and-reversion strategy with Interactive Brokers data integration.

## Strategy Overview

The NQ Sweep-and-Reversion strategy is a short-term trading approach that:

1. **Reference Timeframe**: Uses H1 (1-hour) candles as the primary reference
2. **Sweep Detection**: Monitors if the H1 high or low is breached within the first 20 minutes of each hour
3. **Entry Logic**: 
   - Enter SHORT if H1 high is swept (breached above)
   - Enter LONG if H1 low is swept (breached below)
   - Entry occurs after the sweep candle closes
4. **Risk Management**:
   - Take Profit: Distance from entry price to H1 open (1:1 risk-reward)
   - Stop Loss: Same distance as take profit
   - Maximum hold time: 20 minutes
   - Force close at market if neither TP nor SL is hit within max hold time

## Features

- **Interactive Brokers Integration**: Real-time and historical data fetching using `ib_insync`
- **Comprehensive Backtesting**: Full trade simulation with proper slippage and commission modeling
- **Performance Analysis**: Detailed metrics including Sharpe ratio, maximum drawdown, profit factor
- **Risk Management**: Built-in position sizing and risk controls
- **Reporting**: HTML, PDF, and text reports with charts and trade journals
- **Data Validation**: Robust data cleaning and validation processes
- **Configurable Parameters**: Easy-to-modify strategy and backtest settings

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd tactical-etf-backtest
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Interactive Brokers Setup (Optional):**
   - Install and configure TWS (Trader Workstation) or IB Gateway
   - Enable API connections in TWS/Gateway settings
   - The system will use mock data if IB is not available

## Quick Start

### Basic Usage

Run a backtest with default settings:

```bash
python main.py
```

### Custom Parameters

Run a backtest with custom parameters:

```bash
python main.py --start-date 2024-01-01 --end-date 2024-06-30 --capital 50000 --sweep-window 15 --max-hold 25
```

### Command Line Options

```bash
python main.py --help
```

Available options:
- `--start-date`: Start date for backtest (YYYY-MM-DD)
- `--end-date`: End date for backtest (YYYY-MM-DD)
- `--sweep-window`: Sweep detection window in minutes (default: 20)
- `--max-hold`: Maximum hold time in minutes (default: 20)
- `--capital`: Initial capital (default: 100000)
- `--ib-port`: Interactive Brokers port (default: 7497)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--output-dir`: Output directory for results

## Configuration

### Strategy Configuration

Modify strategy parameters in `nq_backtest/config/strategy_config.py`:

```python
@dataclass
class StrategyConfig:
    sweep_detection_window: int = 20  # Minutes to detect sweep
    max_hold_time: int = 20          # Maximum hold time
    risk_reward_ratio: float = 1.0   # Risk-reward ratio
    position_size: float = 1.0       # Contracts per trade
    commission_per_contract: float = 2.50
    slippage_ticks: float = 0.25
```

### Interactive Brokers Configuration

Configure IB connection in `nq_backtest/config/ib_config.py`:

```python
@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for paper, 7496 for live
    client_id: int = 1
```

## Project Structure

```
nq_backtest/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── strategy_config.py      # Strategy parameters
│   └── ib_config.py           # IB connection settings
├── data/
│   ├── __init__.py
│   ├── ib_data_manager.py     # IB data fetching
│   └── data_preprocessor.py   # Data cleaning & validation
├── strategy/
│   ├── __init__.py
│   ├── sweep_detector.py      # Sweep detection logic
│   ├── strategy_engine.py     # Main strategy implementation
│   └── trade_manager.py       # Trade execution & management
├── backtest/
│   ├── __init__.py
│   ├── backtest_engine.py     # Backtesting framework
│   └── performance_analyzer.py # Performance metrics
├── utils/
│   ├── __init__.py
│   ├── logging_config.py      # Logging setup
│   └── helpers.py             # Utility functions
├── reports/
│   ├── __init__.py
│   └── report_generator.py    # Report generation
main.py                        # Main execution script
requirements.txt               # Dependencies
README.md                      # This file
```

## Output Files

The system generates several output files in the `backtest_results/` directory:

- **HTML Report**: Comprehensive performance report with charts
- **Trade Journal**: Detailed CSV with all trade information
- **Equity Curve**: CSV file showing portfolio value over time
- **Performance Metrics**: Summary of key statistics
- **Charts**: PNG files with equity curve, drawdown, and trade distribution plots

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Return Metrics**: Total return, CAGR, monthly/quarterly returns
- **Risk Metrics**: Maximum drawdown, Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Metrics**: Win rate, profit factor, average winner/loser
- **Time Metrics**: Average hold time, trade frequency
- **Sweep Analysis**: Sweep detection efficiency and patterns

## Data Sources

### Interactive Brokers (Recommended)
- Real-time and historical minute data
- Accurate market hours and holidays
- Proper contract specifications

### Mock Data (Fallback)
- Realistic synthetic NQ data
- Used when IB is not available
- Good for testing and development

## Risk Disclaimers

⚠️ **Important Risk Disclaimers**

- This software is for educational and research purposes only
- Past performance does not guarantee future results
- Trading futures involves substantial risk of loss
- You can lose more than your initial investment
- Only trade with capital you can afford to lose
- Always consult with a qualified financial advisor
- The authors are not responsible for any trading losses

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black nq_backtest/
isort nq_backtest/
```

### Type Checking

```bash
mypy nq_backtest/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Provide relevant log files and configuration

## Changelog

### Version 1.0.0
- Initial implementation of NQ Sweep-and-Reversion strategy
- Interactive Brokers data integration
- Comprehensive backtesting framework
- Performance analysis and reporting
- HTML, PDF, and text report generation
- Configurable strategy parameters
- Risk management and position sizing
- Data validation and error handling

## Acknowledgments

- Interactive Brokers for providing market data API
- The `ib_insync` library for IB integration
- The Python trading community for inspiration and best practices
