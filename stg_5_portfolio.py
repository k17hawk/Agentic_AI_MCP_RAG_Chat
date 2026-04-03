#!/usr/bin/env python3
"""
stg_5_portfolio.py - Run portfolio optimizer on approved trades.
"""

import asyncio
import json
from pathlib import Path
import pandas as pd
import yfinance as yf

from agentic_trading_system.portfolio.portfolio_optimizer import PortfolioOptimizer

async def main():
    # 1. Find the latest discovery run and its risk output
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
    
    # 2. Load approved trades
    with open(queue_file) as f:
        approved_trades = json.load(f)
    
    print(f"📋 Loaded {len(approved_trades)} approved trades:")
    for trade in approved_trades:
        print(f"   {trade['ticker']}: risk score {trade['risk_score']:.2f}, "
              f"pre‑risk size ${trade['position_size']:,.2f}")
    
    # 3. Fetch historical prices for all tickers
    tickers = [trade["ticker"] for trade in approved_trades]
    if not tickers:
        print("No tickers to optimize")
        return
    
    print(f"\n📊 Fetching historical prices for {tickers}...")
    prices = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=False)
    
    # If yfinance returns MultiIndex, flatten to simple DataFrame
    if len(tickers) == 1:
        prices = prices["Close"].to_frame(tickers[0])
    else:
        prices = {ticker: prices[ticker]["Close"] for ticker in tickers}
        prices = pd.DataFrame(prices)
    
    print(f"   Price data shape: {prices.shape}")
    
    # 4. Build current portfolio (start with cash only, no existing positions)
    current_portfolio = {
        "total_value": 100000,
        "cash": 100000,
        "positions": []
    }
    
    # 5. Initialize portfolio optimizer
    optimizer = PortfolioOptimizer("PortfolioOptimizer", config={
        "initial_value": 100000,
        "initial_cash": 100000,
        "constraints": {
            "max_position": 0.25,
            "min_position": 0.01,
            "max_sector": 0.30,
            "max_turnover": 0.20
        }
    })
    
    # 6. Set market regime (from your earlier analysis)
    optimizer.current_regime = "bear_trending_high_vol"
    
    # 7. Run optimization (e.g., max Sharpe ratio)
    print("\n🧮 Running portfolio optimization (max Sharpe)...")
    result = await optimizer.optimize(prices, method="max_sharpe")
    
    print(f"\n📈 Optimization result:")
    print(f"   Expected return: {result['expected_return']:.2%}")
    print(f"   Volatility: {result['volatility']:.2%}")
    print(f"   Sharpe ratio: {result['sharpe_ratio']:.2f}")
    print(f"\n🎯 Target allocation:")
    for asset, weight in result["weights"].items():
        print(f"   {asset}: {weight:.2%}")
    
    # 8. Generate rebalancing recommendations
    print("\n🔄 Checking rebalancing needs...")
    recommendations = await optimizer.generate_recommendations()
    
    print(f"\n📋 Recommendations:")
    print(f"   Summary: {recommendations['summary']}")
    print(f"   Priority: {recommendations['priority']}")
    
    if recommendations["trades"]:
        print("\n   Trades:")
        for trade in recommendations["trades"]:
            print(f"      {trade['action']} {trade['symbol']}: "
                  f"${trade['value']:,.2f} ({trade['priority']} priority)")
    
    if recommendations["risk_alerts"]:
        print("\n   Risk alerts:")
        for alert in recommendations["risk_alerts"]:
            print(f"      {alert['type']}: {alert['message']}")
    
    # 9. Save results
    output_dir = risk_dir / "portfolio"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "optimization_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    with open(output_dir / "recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())