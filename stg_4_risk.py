#!/usr/bin/env python3
"""
stg_4_risk.py - Run risk management on analysis results and save outputs.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.risk.risk_manager import RiskManager

async def fetch_market_data(ticker: str):
    """Fetch current price, volatility, ATR, volume data."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    info = stock.info
    
    if hist.empty:
        return None
    
    price = hist['Close'].iloc[-1]
    returns = hist['Close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * np.sqrt(252)
    
    # ATR (14-day)
    high = hist['High']
    low = hist['Low']
    close = hist['Close']
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift()),
        'lc': abs(low - close.shift())
    }).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    volume = hist['Volume'].iloc[-1]
    avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
    
    bid = info.get('bid', price * 0.999)
    ask = info.get('ask', price * 1.001)
    spread = (ask - bid) / price if price > 0 else 0.001
    
    return {
        "price": float(price),
        "volatility": float(volatility),
        "atr": float(atr),
        "volume": int(volume),
        "avg_volume": int(avg_volume),
        "bid_ask_spread": float(spread),
        "win_rate": 0.55,
        "avg_win": 0.10,
        "avg_loss": 0.05
    }

async def main():
    # Find the latest discovery run and analysis folder
    discovery_base = Path("discovery_outputs")
    run_dirs = sorted(discovery_base.glob("2026*_*"), reverse=True)
    if not run_dirs:
        print("No discovery runs found")
        return
    
    latest_run = run_dirs[0]
    analysis_dir = latest_run / "analysis"
    if not analysis_dir.exists():
        print(f"No analysis directory found at {analysis_dir}")
        return
    
    # Create a risk output directory
    risk_dir = latest_run / "risk"
    risk_dir.mkdir(exist_ok=True)
    
    # Initialize risk manager
    risk_config = {
        "initial_capital": 100000,
        "max_concurrent_analyses": 5,
        "kelly_config": {"max_fraction": 0.25},
        "half_kelly_config": {"max_fraction": 0.15},
        "fixed_fraction_config": {"default_fraction": 0.02},
        "vol_adjusted_config": {"base_fraction": 0.02},
        "atr_stop_config": {"default_multiplier": 2.0},
        "vol_stop_config": {"default_multiplier": 2.0},
        "queue_config": {"max_size": 100, "item_ttl_seconds": 300},
        "scorer_config": {}
    }
    
    risk_manager = RiskManager("RiskManager", risk_config)
    risk_manager.current_regime = "bear_trending_high_vol"  # from earlier analysis
    
    # Process each analysis file
    analysis_files = list(analysis_dir.glob("*_analysis.json"))
    approved_trades = []
    
    for ana_file in analysis_files:
        with open(ana_file) as f:
            analysis = json.load(f)
        
        ticker = analysis.get("ticker")
        print(f"\n🛡️ Running risk for {ticker}")
        
        market = await fetch_market_data(ticker)
        if not market:
            print(f"   Could not fetch market data for {ticker}")
            continue
        
        risk_input = {
            "symbol": ticker,
            "final_score": analysis.get("final_score", 0.5),
            "action": analysis.get("recommendation", "WATCH"),
            "confidence": analysis.get("confidence", 0.5),
            "price": market["price"],
            "volatility": market["volatility"],
            "atr": market["atr"],
            "volume": market["volume"],
            "avg_volume": market["avg_volume"],
            "bid_ask_spread": market["bid_ask_spread"],
            "win_rate": market["win_rate"],
            "avg_win": market["avg_win"],
            "avg_loss": market["avg_loss"],
        }
        
        response = await risk_manager.calculate_risk(risk_input, "test")
        trade = response.content
        
        # Save individual trade risk assessment
        trade_file = risk_dir / f"{ticker}_risk.json"
        with open(trade_file, "w") as f:
            json.dump(trade, f, indent=2, default=str)
        print(f"   💾 Saved risk assessment to {trade_file}")
        
        if trade["should_trade"]:
            approved_trades.append(trade)
            print(f"   ✅ Approved: position ${trade['position_size']:,.2f}, stop ${trade['stop_price']:.2f}")
        else:
            print(f"   ❌ Rejected: {trade.get('risk_warnings', ['No reason given'])[0]}")
    
    # Save the entire queue (all approved trades)
    queue_items = await risk_manager.approved_queue.get_all()
    queue_file = risk_dir / "approved_queue.json"
    with open(queue_file, "w") as f:
        json.dump(queue_items, f, indent=2, default=str)
    print(f"\n💾 Saved approved queue ({len(queue_items)} items) to {queue_file}")
    
    # Also save queue statistics
    stats = risk_manager.approved_queue.get_stats()
    stats_file = risk_dir / "queue_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"💾 Saved queue stats to {stats_file}")

if __name__ == "__main__":
    asyncio.run(main())