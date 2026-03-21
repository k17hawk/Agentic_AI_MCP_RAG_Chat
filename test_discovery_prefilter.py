#!/usr/bin/env python3
"""
Fixed Complete Sequential Test: Discovery → Prefilter
Tests the full pipeline from data discovery to quality gating
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.discovery.search_aggregator import SearchAggregator
from agentic_trading_system.prefilter.exchange_validator import ExchangeValidator
from agentic_trading_system.prefilter.price_range_checker import PriceRangeChecker
from agentic_trading_system.prefilter.volume_checker import VolumeChecker
from agentic_trading_system.prefilter.market_cap_checker import MarketCapChecker
from agentic_trading_system.prefilter.data_quality_checker import DataQualityChecker
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class DiscoveryToPrefilterPipeline:
    """
    Fixed Sequential test: Discovery → Prefilter
    """
    
    def __init__(self):
        self.discovery_results = []
        self.prefilter_results = []
        self.stats = {
            "discovery": {},
            "prefilter": {},
            "passed_tickers": [],
            "rejected_tickers": []
        }
    
    async def run_discovery(self, symbols: list, sources: list = None):
        """
        Step 1: Run discovery to get data for symbols
        """
        print("\n" + "="*70)
        print("🔍 STEP 1: DISCOVERY PHASE")
        print("="*70)
        
        discovery_config = {
            "tavily_config": {},
            "news_config": {},
            "social_config": {},
            "source_weights": {
                "tavily": 0.25,
                "news": 0.20,
                "social": 0.15,
                "sec": 0.15,
                "options": 0.15,
                "macro": 0.10
            }
        }
        
        discovery = SearchAggregator("Discovery", discovery_config)
        
        for symbol in symbols:
            print(f"\n📡 Discovering data for {symbol}...")
            
            result = await discovery.discover(
                query=f"{symbol} stock",
                options={
                    "max_results": 5,
                    "sources": sources or ["news", "tavily", "social"]
                }
            )
            
            # Store results with extracted info
            self.discovery_results.append({
                "symbol": symbol,
                "discovery_result": result,
                "tickers": result.get('entities', {}).get('tickers', [symbol]),
                "items_found": result.get('total_items', 0),
                "source_stats": result.get('source_stats', {})
            })
            
            print(f"   ✅ Found {result.get('total_items', 0)} items")
            print(f"   📊 Sources: {list(result.get('source_stats', {}).keys())}")
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Update stats
        self.stats["discovery"] = {
            "symbols_processed": len(symbols),
            "total_items": sum(r["items_found"] for r in self.discovery_results)
        }
        
        return self.discovery_results
    
    async def run_prefilter(self, symbols: list):
        """
        Step 2: Run prefilter on discovered symbols using individual validators directly
        (FIX: Don't use QualityGates wrapper, use validators directly)
        """
        print("\n" + "="*70)
        print("🔍 STEP 2: PREFILTER PHASE")
        print("="*70)
        
        # Initialize validators directly (FIXED)
        exchange_validator = ExchangeValidator({
            "allowed_exchanges": ["NYSE", "NASDAQ", "AMEX", "BATS", "IEX"],
            "blocked_exchanges": ["OTC", "PINK", "GREY", "YHD"]
        })
        
        price_checker = PriceRangeChecker({
            "min_price": 1.0,
            "max_price": 10000.0
        })
        
        volume_checker = VolumeChecker({
            "min_volume": 100000,
            "min_avg_volume": 50000
        })
        
        market_cap_checker = MarketCapChecker({
            "min_market_cap": 50000000  # $50M
        })
        
        data_quality_checker = DataQualityChecker({
            "min_history_days": 60  # YOUR 60-DAY REQUIREMENT!
        })
        
        # Process each symbol
        for discovery_result in self.discovery_results:
            symbol = discovery_result["symbol"]
            print(f"\n📊 Running quality gates for {symbol}...")
            
            # Get info from yfinance (real data)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="6mo")
                
                # Create info dict with all needed fields
                enriched_info = {
                    "symbol": symbol,
                    "exchange": info.get("exchange", "NMS"),
                    "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "previous_close": info.get("regularMarketPreviousClose", 0),
                    "volume": info.get("volume", 0),
                    "average_volume": info.get("averageVolume", 0),
                    "market_cap": info.get("marketCap", 0),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "long_name": info.get("longName", symbol),
                    "data_days": len(hist),
                    # Additional fields
                    "longName": info.get("longName", symbol),
                    "shortName": info.get("shortName", symbol),
                    "regularMarketPrice": info.get("regularMarketPrice", 0),
                    "regularMarketVolume": info.get("regularMarketVolume", 0)
                }
                
                # Run all quality checks
                print(f"\n   🔍 Running individual checks for {symbol}:")
                
                # 1. Exchange Check
                exchange_result = await exchange_validator.validate(symbol, enriched_info)
                print(f"      Exchange: {enriched_info['exchange']} → {'✅ PASS' if exchange_result['passed'] else '❌ FAIL'}")
                if not exchange_result['passed']:
                    print(f"         Reason: {exchange_result.get('reason', 'Unknown')}")
                
                # 2. Price Check
                price_result = await price_checker.validate(symbol, enriched_info)
                print(f"      Price: ${enriched_info['current_price']:.2f} → {'✅ PASS' if price_result['passed'] else '❌ FAIL'}")
                if not price_result['passed']:
                    print(f"         Reason: {price_result.get('reason', 'Unknown')}")
                
                # 3. Volume Check
                volume_result = await volume_checker.validate(symbol, enriched_info)
                print(f"      Volume: {enriched_info['volume']:,} → {'✅ PASS' if volume_result['passed'] else '❌ FAIL'}")
                if not volume_result['passed']:
                    print(f"         Reason: {volume_result.get('reason', 'Unknown')}")
                
                # 4. Market Cap Check
                mcap_result = await market_cap_checker.validate(symbol, enriched_info)
                print(f"      Market Cap: ${enriched_info['market_cap']:,.0f} → {'✅ PASS' if mcap_result['passed'] else '❌ FAIL'}")
                if not mcap_result['passed']:
                    print(f"         Reason: {mcap_result.get('reason', 'Unknown')}")
                
                # 5. Data Quality Check (YOUR 60-DAY!)
                quality_result = await data_quality_checker.validate(symbol, enriched_info)
                days = enriched_info['data_days']
                print(f"      Data Quality: {days} days → {'✅ PASS' if quality_result['passed'] else '❌ FAIL'} (60-DAY REQUIREMENT: {'✅' if days >= 60 else '❌'})")
                if not quality_result['passed']:
                    print(f"         Issues: {quality_result.get('issues', [])}")
                
                # Overall pass/fail
                all_passed = all([
                    exchange_result['passed'],
                    price_result['passed'],
                    volume_result['passed'],
                    mcap_result['passed'],
                    quality_result['passed']
                ])
                
                result = {
                    "symbol": symbol,
                    "passed": all_passed,
                    "details": {
                        "exchange": exchange_result,
                        "price": price_result,
                        "volume": volume_result,
                        "market_cap": mcap_result,
                        "data_quality": quality_result
                    },
                    "info": enriched_info,
                    "discovery_items": discovery_result["items_found"]
                }
                
                if all_passed:
                    self.stats["passed_tickers"].append(result)
                    print(f"\n   ✅ {symbol} PASSED all quality gates!")
                else:
                    self.stats["rejected_tickers"].append(result)
                    print(f"\n   ❌ {symbol} REJECTED - see reasons above")
                
                self.prefilter_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                self.stats["rejected_tickers"].append({
                    "symbol": symbol,
                    "passed": False,
                    "error": str(e)
                })
        
        # Update stats
        self.stats["prefilter"] = {
            "processed": len(symbols),
            "passed": len(self.stats["passed_tickers"]),
            "rejected": len(self.stats["rejected_tickers"]),
            "pass_rate": len(self.stats["passed_tickers"]) / len(symbols) * 100 if symbols else 0
        }
        
        return self.prefilter_results
    
    def print_summary(self):
        """Print detailed summary"""
        print("\n" + "="*70)
        print("📊 SEQUENTIAL TEST SUMMARY: DISCOVERY → PREFILTER")
        print("="*70)
        
        # Discovery Summary
        print("\n📡 DISCOVERY PHASE SUMMARY:")
        print(f"   • Symbols Processed: {self.stats['discovery']['symbols_processed']}")
        print(f"   • Total Items Found: {self.stats['discovery']['total_items']}")
        
        print("\n📊 PREFILTER PHASE SUMMARY:")
        print(f"   • Symbols Processed: {self.stats['prefilter']['processed']}")
        print(f"   • ✅ Passed: {self.stats['prefilter']['passed']}")
        print(f"   • ❌ Rejected: {self.stats['prefilter']['rejected']}")
        print(f"   • Pass Rate: {self.stats['prefilter']['pass_rate']:.1f}%")
        
        # Passed Tickers
        if self.stats["passed_tickers"]:
            print(f"\n✅ PASSED TICKERS (Ready for Analysis):")
            for ticker in self.stats["passed_tickers"]:
                info = ticker["info"]
                print(f"\n   📈 {ticker['symbol']} - {info.get('long_name', 'Unknown')}")
                print(f"      • Exchange: {info['exchange']}")
                print(f"      • Price: ${info['current_price']:.2f}")
                print(f"      • Volume: {info['volume']:,}")
                print(f"      • Market Cap: ${info['market_cap']:,.0f}")
                print(f"      • Data Days: {info['data_days']} (60-day requirement: {'✅' if info['data_days'] >= 60 else '❌'})")
                print(f"      • Discovery Items: {ticker['discovery_items']}")
        
        # Rejected Tickers
        if self.stats["rejected_tickers"]:
            print(f"\n❌ REJECTED TICKERS:")
            for ticker in self.stats["rejected_tickers"]:
                print(f"\n   📉 {ticker['symbol']}")
                if 'error' in ticker:
                    print(f"      • Error: {ticker['error']}")
                else:
                    for check_name, check_result in ticker.get('details', {}).items():
                        if not check_result.get('passed', True):
                            print(f"      • Failed: {check_name} - {check_result.get('reason', 'Unknown')}")
        
        # 60-Day Requirement Report
        print(f"\n🎯 60-DAY REQUIREMENT REPORT:")
        for ticker in self.stats["passed_tickers"] + self.stats["rejected_tickers"]:
            if isinstance(ticker, dict) and 'info' in ticker:
                symbol = ticker['symbol']
                days = ticker['info']['data_days']
                status = "✅ MET" if days >= 60 else f"❌ NEED {60-days} MORE DAYS"
                print(f"   • {symbol}: {days} days - {status}")

async def run_sequential_test():
    """Run the complete sequential test"""
    
    print("\n" + "="*70)
    print("🚀 SEQUENTIAL TEST: DISCOVERY → PREFILTER")
    print("="*70)
    
    # Initialize pipeline
    pipeline = DiscoveryToPrefilterPipeline()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
    
    # STEP 1: Run Discovery
    discovery_results = await pipeline.run_discovery(test_symbols, sources=["news", "tavily"])
    
    # STEP 2: Run Prefilter on discovered symbols
    prefilter_results = await pipeline.run_prefilter(test_symbols)
    
    # STEP 3: Print Summary
    pipeline.print_summary()
    
    # Save results to file
    output = {
        "timestamp": datetime.now().isoformat(),
        "discovery": pipeline.stats["discovery"],
        "prefilter": pipeline.stats["prefilter"],
        "passed_tickers": [
            {
                "symbol": t["symbol"],
                "price": t["info"]["current_price"],
                "exchange": t["info"]["exchange"],
                "data_days": t["info"]["data_days"],
                "discovery_items": t["discovery_items"]
            }
            for t in pipeline.stats["passed_tickers"]
        ],
        "rejected_tickers": [
            {
                "symbol": t["symbol"],
                "reasons": [r.get('reason', r) for r in t.get('details', {}).values() if not r.get('passed', True)]
            }
            for t in pipeline.stats["rejected_tickers"]
        ]
    }
    
    # Save to file
    with open("data/discovery_to_prefilter_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: data/discovery_to_prefilter_results.json")
    
    return pipeline

async def main():
    """Main entry point"""
    try:
        result = await run_sequential_test()
        
        # Final verdict
        if result.stats["prefilter"]["passed"] > 0:
            print("\n🎉 SUCCESS: Discovery → Prefilter pipeline working!")
            print(f"   ✅ {result.stats['prefilter']['passed']} symbols passed all quality gates")
            print("   📊 These symbols are ready for ANALYSIS phase")
        else:
            print("\n⚠️ No symbols passed prefilter. Check the rejection reasons above.")
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())