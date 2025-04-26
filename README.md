# tactical-etf-backtest
A self-contained Google Colab notebook demonstrating a rules-based tactical ETF trading strategy on weekly Vanguard Total Market (VTI) data

## 📖 Overview

This project implements:

1. **Indicator Calculations**  
   - EMA13, EMA21, SMA50  
   - MACD histogram (12-26-9)  
   - Elder Impulse System filter (Green/Neutral/Red bars)

2. **Signal Logic**  
   - Primary trend filter: EMA21 ↔️ SMA50 crossover  
   - Entry/exit rules with optional Elder bar validation  
   - Position states: Long, Short, Cash

3. **Portfolio Simulation**  
   - Weekly rebalancing across leveraged ETFs (SPXL/SPXS), benchmark (SPX), gold/mining (GLDM/SPTS), and T-bills (BIL)  
   - 40/60 allocations for Long/Short, 60/40 for Cash  
   - Equity curve calculation

4. **Performance Analytics**  
   - Total Return, CAGR, Max Drawdown  
   - Sharpe, Sortino, Calmar ratios  
   - Export to CSV: equity_curve.csv, trade_log.csv, performance_metrics.csv

5. **Visualizations**  
   - Static plots via Matplotlib  
   - Interactive charts via Plotly
