#!/usr/bin/env python3
"""
stg_6_performance.py - Performance Analytics on approved trades.
Simulates trade outcomes and calculates comprehensive metrics.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.analytics.metrics_engine import MetricsEngine
from agentic_trading_system.analytics.dashboards.plot_generator import PlotGenerator
from agentic_trading_system.analytics.dashboards.html_reporter import HTMLReporter

async def simulate_trade(trade: dict) -> dict:
    """
    Simulate a trade. If trade timestamp is today, use yesterday as entry.
    """
    symbol = trade["ticker"]
    entry_price = trade["entry_price"]
    stop_price = trade["stop_price"]
    max_hold_days = 5

    import pytz
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now().astimezone(eastern)

    entry_time_str = trade.get("timestamp")
    if entry_time_str:
        entry_time = datetime.fromisoformat(entry_time_str)
        if entry_time.tzinfo is None:
            entry_time = eastern.localize(entry_time)
    else:
        entry_time = now - timedelta(days=1)

    # If entry_time is today (or in the future), move it back to yesterday
    if entry_time.date() >= now.date():
        entry_time = now - timedelta(days=1)
        print(f"   {symbol}: trade timestamp is today, using entry time {entry_time.date()}")

    end_date = entry_time + timedelta(days=max_hold_days + 5)
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=entry_time - timedelta(days=1), end=end_date)

    if hist.empty:
        print(f"   {symbol}: no historical data for period")
        return None

    # Filter bars after entry_time
    future_bars = hist[hist.index > entry_time]
    if future_bars.empty:
        # Try using the last bar (same day) – but that's not realistic
        print(f"   {symbol}: no bars after {entry_time}, using last available bar")
        future_bars = hist.tail(1)
        if future_bars.empty:
            return None

    entry_bar = future_bars.iloc[0]
    actual_entry_price = entry_bar['Open']
    position_value = trade["position_size"]
    shares = position_value / actual_entry_price

    exit_time = None
    exit_price = None
    stop_hit = False

    for idx, row in hist.iterrows():
        if idx <= entry_time:
            continue
        if row['Low'] <= stop_price:
            exit_time = idx
            exit_price = stop_price
            stop_hit = True
            break
        days_held = (idx - entry_time).days
        if days_held >= max_hold_days:
            exit_time = idx
            exit_price = row['Close']
            break

    if exit_time is None:
        exit_time = hist.index[-1]
        exit_price = hist['Close'].iloc[-1]

    pnl = shares * (exit_price - actual_entry_price)
    return_pct = (exit_price - actual_entry_price) / actual_entry_price

    return {
        "symbol": symbol,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": actual_entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "position_value": position_value,
        "pnl": pnl,
        "return_pct": return_pct,
        "stop_hit": stop_hit,
        "days_held": (exit_time - entry_time).days,
        "original_risk_score": trade.get("risk_score", 0),
    }

async def main():
    # 1. Find the latest discovery run and risk output
    discovery_base = Path("discovery_outputs")
    run_dirs = sorted(discovery_base.glob("2026*_*"), reverse=True)
    if not run_dirs:
        print("No discovery runs found")
        return

    latest_run = run_dirs[0]
    risk_dir = latest_run / "risk"
    queue_file = risk_dir / "approved_queue.json"

    if not queue_file.exists():
        print(f"Approved queue not found: {queue_file}")
        return

    with open(queue_file) as f:
        approved_trades = json.load(f)

    print(f"📋 Simulating {len(approved_trades)} approved trades...")

    simulated_trades = []
    for trade in approved_trades:
        sim = await simulate_trade(trade)
        if sim:
            simulated_trades.append(sim)
            print(f"   {sim['symbol']}: entry ${sim['entry_price']:.2f}, exit ${sim['exit_price']:.2f}, "
                  f"P&L ${sim['pnl']:.2f} ({sim['return_pct']*100:.1f}%), stop hit: {sim['stop_hit']}")

    if not simulated_trades:
        print("No trades could be simulated.")
        return

    # 2. Build equity curve from trade returns (sequential)
    equity_curve = [100000]  # starting capital
    for trade in simulated_trades:
        equity_curve.append(equity_curve[-1] * (1 + trade["return_pct"]))

    # 3. Initialize Metrics Engine
    metrics_engine = MetricsEngine("MetricsEngine", {})

    # 4. Calculate all metrics
    metrics = await metrics_engine.calculate_all_metrics(
        trades=simulated_trades,
        equity_curve=equity_curve,
        positions=[],
        current_prices={}
    )

    print("\n📊 Performance Metrics Summary:")
    print(f"   Total P&L: ${metrics['pnl']['total_pnl']:,.2f}")
    print(f"   Sharpe Ratio: {metrics['risk_adjusted']['sharpe_ratio'].get('annualized_sharpe', 0):.2f}")
    print(f"   Max Drawdown: {metrics['drawdown']['max_drawdown'].get('max_drawdown_pct', 0):.1f}%")
    print(f"   Win Rate: {metrics['trading_stats']['win_rate'].get('win_rate', 0)*100:.1f}%")
    print(f"   Profit Factor: {metrics['trading_stats']['profit_factor'].get('profit_factor', 0):.2f}")

    # 5. Generate plots
    plotter = PlotGenerator({"output_dir": str(risk_dir / "performance_charts")})
    equity_curve_plot = plotter.plot_equity_curve(equity_curve, title="Simulated Equity Curve")
    drawdown_plot = plotter.plot_drawdown(equity_curve, title="Drawdown Chart")
    returns = [t["return_pct"] for t in simulated_trades]
    dist_plot = plotter.plot_return_distribution(returns, title="Return Distribution")

    # 6. Generate HTML report
    reporter = HTMLReporter({"output_dir": str(risk_dir / "reports")})
    report_path = reporter.generate_report(
        metrics=metrics,
        plots=[equity_curve_plot, drawdown_plot, dist_plot],
        title="Trading Performance Report"
    )

    print(f"\n💾 Performance report saved to: {report_path}")
    print(f"💾 Charts saved to: {risk_dir / 'performance_charts'}")

if __name__ == "__main__":
    asyncio.run(main())