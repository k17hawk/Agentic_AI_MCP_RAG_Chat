#!/usr/bin/env python3
"""
Quick test to verify exchange validator fix
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.prefilter.exchange_validator import ExchangeValidator

async def test_exchange_validator():
    """Test exchange validator with real data"""
    
    print("=" * 60)
    print("🔧 TESTING EXCHANGE VALIDATOR FIX")
    print("=" * 60)
    
    # Initialize with config
    config = {
        "allowed_exchanges": ["NYSE", "NASDAQ", "AMEX", "BATS", "IEX"],
        "blocked_exchanges": ["OTC", "PINK", "GREY", "YHD"]
    }
    
    validator = ExchangeValidator(config)
    
    # Test cases
    test_cases = [
        {"ticker": "AAPL", "exchange": "NMS", "expected": True},
        {"ticker": "MSFT", "exchange": "NMS", "expected": True},
        {"ticker": "GOOGL", "exchange": "NMS", "expected": True},
        {"ticker": "TSLA", "exchange": "NMS", "expected": True},
        {"ticker": "NVDA", "exchange": "NMS", "expected": True},
        {"ticker": "META", "exchange": "NMS", "expected": True},
        {"ticker": "BAD", "exchange": "OTC", "expected": False},
        {"ticker": "FAKE", "exchange": "PINK", "expected": False},
    ]
    
    print("\n📊 Test Results:")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        info = {"exchange": test["exchange"]}
        result = await validator.validate(test["ticker"], info)
        
        status = "✅ PASS" if result['passed'] == test["expected"] else "❌ FAIL"
        if result['passed'] == test["expected"]:
            passed += 1
        else:
            failed += 1
        
        print(f"   {test['ticker']}: Exchange='{test['exchange']}' → {status}")
        if not result['passed']:
            print(f"      Reason: {result.get('reason', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ Exchange validator is working correctly!")
    else:
        print("❌ Some tests failed. Check the validator implementation.")
    
    return passed, failed

async def test_with_yfinance():
    """Test with actual yfinance data"""
    
    print("\n" + "=" * 60)
    print("📊 TESTING WITH REAL YFINANCE DATA")
    print("=" * 60)
    
    import yfinance as yf
    
    config = {
        "allowed_exchanges": ["NYSE", "NASDAQ", "AMEX", "BATS", "IEX"],
        "blocked_exchanges": ["OTC", "PINK", "GREY", "YHD"]
    }
    
    validator = ExchangeValidator(config)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
    
    print("\n📊 Results from yFinance:")
    print("-" * 40)
    
    all_passed = True
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        exchange = info.get("exchange", "Unknown")
        
        result = await validator.validate(symbol, {"exchange": exchange})
        
        if result['passed']:
            print(f"   ✅ {symbol}: Exchange='{exchange}' → PASSED")
        else:
            print(f"   ❌ {symbol}: Exchange='{exchange}' → FAILED")
            print(f"      Reason: {result.get('reason', 'Unknown')}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All symbols passed exchange validation!")
    else:
        print("❌ Some symbols failed. Check exchange mapping.")
    
    return all_passed

if __name__ == "__main__":
    print("\n🚀 Testing Exchange Validator Fix")
    print("=" * 60)
    
    # Run tests
    passed, failed = asyncio.run(test_exchange_validator())
    
    if passed > 0:
        asyncio.run(test_with_yfinance())