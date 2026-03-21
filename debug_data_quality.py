# debug_data_quality.py
import asyncio
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.prefilter.data_quality_checker import DataQualityChecker

async def debug_data_quality():
    """Debug why data quality checker is failing"""
    
    print("=" * 60)
    print("🔍 DEBUGGING DATA QUALITY CHECKER")
    print("=" * 60)
    
    # Create checker with 60-day requirement
    checker = DataQualityChecker({
        "min_history_days": 60,
        "min_data_points": 50,
        "max_data_age_hours": 24
    })
    
    # Test with AAPL
    symbol = "AAPL"
    print(f"\n📊 Testing {symbol}...")
    
    # Get real data
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="6mo")
    
    print(f"   Data points: {len(hist)}")
    print(f"   Exchange: {info.get('exchange')}")
    print(f"   Current price: ${info.get('currentPrice', info.get('regularMarketPrice', 0))}")
    
    # Create info dict
    enriched_info = {
        "symbol": symbol,
        "exchange": info.get("exchange", "NMS"),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "volume": info.get("volume", 0),
        "market_cap": info.get("marketCap", 0)
    }
    
    # Run validation
    result = await checker.validate(symbol, enriched_info)
    
    print(f"\n📊 Validation Result:")
    print(f"   Passed: {result['passed']}")
    print(f"   Quality Score: {result.get('quality_score', 'N/A')}")
    print(f"   Issues: {result.get('issues', [])}")
    print(f"   Warnings: {result.get('warnings', [])}")
    print(f"   Historical Data: {result.get('historical_data', {})}")
    
    # Check each component
    print(f"\n🔍 Component Checks:")
    print(f"   • Has sufficient history: {result.get('historical_data', {}).get('days_available', 0) >= 60}")
    print(f"   • Data fresh: {not result.get('freshness', {}).get('stale', True)}")
    
    return result

async def debug_data_quality_internal():
    """Debug internal methods of data quality checker"""
    
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING INTERNAL METHODS")
    print("=" * 60)
    
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="6mo")
    
    # Create checker instance
    checker = DataQualityChecker({"min_history_days": 60})
    
    # Test each internal method
    print(f"\n📊 Testing historical data check...")
    hist_data = await checker._check_historical_data(symbol)
    print(f"   Days available: {hist_data.get('days_available', 0)}")
    print(f"   Data points: {hist_data.get('data_points', 0)}")
    print(f"   Issues: {hist_data.get('issues', [])}")
    print(f"   Has issues: {hist_data.get('has_issues', False)}")
    
    print(f"\n📊 Testing freshness check...")
    freshness = await checker._check_data_freshness(symbol)
    print(f"   Stale: {freshness.get('stale', True)}")
    print(f"   Age hours: {freshness.get('age_hours', 0)}")
    print(f"   Last trade: {freshness.get('last_trade', 'N/A')}")
    
    print(f"\n📊 Testing corporate actions...")
    corp_actions = await checker._check_corporate_actions(symbol)
    print(f"   Has splits: {corp_actions.get('has_splits', False)}")
    print(f"   Has dividends: {corp_actions.get('has_dividends', False)}")
    

    warnings = []
    if freshness.get('stale', True):
        warnings.append("Data is stale")
    
    # Get issues list from hist_data
    issues = hist_data.get('issues', [])
    
    quality_score = checker._calculate_quality_score(
        issues,  # issues list
        warnings,  # warnings list (not boolean)
        hist_data,
        freshness
    )
    print(f"\n📊 Quality Score: {quality_score}")

if __name__ == "__main__":
    asyncio.run(debug_data_quality())
    asyncio.run(debug_data_quality_internal())


