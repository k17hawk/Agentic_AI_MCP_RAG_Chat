#!/usr/bin/env python3
"""
Fixed test for prefilter with corrected gap detection
"""
import asyncio
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.prefilter.exchange_validator import ExchangeValidator
from agentic_trading_system.prefilter.price_range_checker import PriceRangeChecker
from agentic_trading_system.prefilter.volume_checker import VolumeChecker
from agentic_trading_system.prefilter.market_cap_checker import MarketCapChecker
from agentic_trading_system.prefilter.data_quality_checker import DataQualityChecker

class FixedPrefilterTest:
    """Test prefilter with fixed gap detection"""
    
    def __init__(self):
        self.results = []
    
    async def test_symbol(self, symbol):
        """Test a single symbol"""
        print(f"\n📊 Testing {symbol}...")
        print("-" * 40)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")
            
            days = len(hist)
            print(f"   Days available: {days}")
            print(f"   Exchange: {info.get('exchange', 'N/A')}")
            print(f"   Price: ${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
            print(f"   Volume: {info.get('volume', 0):,}")
            print(f"   Market Cap: ${info.get('marketCap', 0):,.0f}")
            
            # Create info dict with all needed fields
            enriched_info = {
                "symbol": symbol,
                "exchange": info.get("exchange", "NMS"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "volume": info.get("volume", 0),
                "average_volume": info.get("averageVolume", 0),
                "market_cap": info.get("marketCap", 0),
                "long_name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                # Additional fields to prevent "Insufficient basic information"
                "longName": info.get("longName", symbol),
                "shortName": info.get("shortName", symbol),
                "regularMarketPrice": info.get("regularMarketPrice", 0),
                "regularMarketVolume": info.get("regularMarketVolume", 0)
            }
            
            # Run validators
            exchange = ExchangeValidator({"allowed_exchanges": ["NMS", "NYQ", "NGM", "ASE"]})
            price = PriceRangeChecker({"min_price": 1.0})
            volume = VolumeChecker({"min_volume": 100000})
            market_cap = MarketCapChecker({"min_market_cap": 50000000})
            quality = DataQualityChecker({"min_history_days": 60})
            
            exchange_result = await exchange.validate(symbol, enriched_info)
            price_result = await price.validate(symbol, enriched_info)
            volume_result = await volume.validate(symbol, enriched_info)
            market_cap_result = await market_cap.validate(symbol, enriched_info)
            quality_result = await quality.validate(symbol, enriched_info)
            
            print(f"\n   Results:")
            print(f"   Exchange: {'✅ PASS' if exchange_result['passed'] else '❌ FAIL'} - {info.get('exchange', 'N/A')}")
            print(f"   Price: {'✅ PASS' if price_result['passed'] else '❌ FAIL'} - ${enriched_info['current_price']:.2f}")
            print(f"   Volume: {'✅ PASS' if volume_result['passed'] else '❌ FAIL'} - {enriched_info['volume']:,}")
            print(f"   Market Cap: {'✅ PASS' if market_cap_result['passed'] else '❌ FAIL'} - ${enriched_info['market_cap']:,.0f}")
            print(f"   Data Quality: {'✅ PASS' if quality_result['passed'] else '❌ FAIL'} - {days} days (need 60)")
            
            if quality_result.get('issues'):
                print(f"      Issues: {quality_result['issues']}")
            
            all_passed = all([
                exchange_result['passed'],
                price_result['passed'],
                volume_result['passed'],
                market_cap_result['passed'],
                quality_result['passed']
            ])
            
            if all_passed:
                print(f"\n   ✅ {symbol} PASSED all checks!")
            else:
                print(f"\n   ❌ {symbol} FAILED")
            
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    async def run(self):
        """Run tests for all symbols"""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
        
        print("=" * 60)
        print("🚀 FIXED PREFILTER TEST")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for symbol in symbols:
            if await self.test_symbol(symbol):
                passed += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   Pass Rate: {passed / (passed + failed) * 100:.1f}%")
        
        if passed == len(symbols):
            print("\n🎉 ALL SYMBOLS PASSED! Ready for Analysis phase!")
        else:
            print("\n⚠️ Some symbols failed. Check the reasons above.")

async def main():
    tester = FixedPrefilterTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())