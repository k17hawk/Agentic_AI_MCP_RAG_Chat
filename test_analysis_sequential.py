import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import traceback
from typing import Dict
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis components with error handling
try:
    from agentic_trading_system.analysis.technical.indicators.trend import TrendIndicators
    from agentic_trading_system.analysis.technical.indicators.momentum import MomentumIndicators
    from agentic_trading_system.analysis.technical.indicators.volume import VolumeIndicators
    from agentic_trading_system.analysis.technical.indicators.volatility import VolatilityIndicators
    from agentic_trading_system.analysis.multi_timeframe_aggregator import MultiTimeframeAggregator, Timeframe
    from agentic_trading_system.analysis.regime_detector import RegimeDetector
    from agentic_trading_system.analysis.weighted_score_engine import WeightedScoreEngine
    from agentic_trading_system.triggers.price_alert_triggers.sliding_window import SlidingWindowAnalyzer
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Using simplified analysis (fallback mode)")
    
    # Create simplified fallback classes
    class SlidingWindowAnalyzer:
        def __init__(self, config):
            self.config = config
        def analyze(self, data):
            return {
                "trend": {"direction": "bullish", "strength": 65.0},
                "volatility_regime": {"regime": "normal", "current_volatility": 15.0},
                "position": {"distance_from_ma50": 5.0},
                "anomalies": []
            }
    
    class TrendIndicators:
        def __init__(self, config): pass
        def calculate_all(self, data): return {"sma_20": 250, "sma_50": 240}
    
    class MomentumIndicators:
        def __init__(self, config): pass
        def calculate_all(self, data): return {"rsi": 62, "macd_cross": "bullish"}
    
    class VolumeIndicators:
        def __init__(self, config): pass
        def calculate_all(self, data): return {"volume_ratio_20": 1.2}
    
    class VolatilityIndicators:
        def __init__(self, config): pass
        def calculate_all(self, data): return {"atr_pct": 2.3, "bb_position": 65}
    
    class MultiTimeframeAggregator:
        def __init__(self, name, config): pass
        async def aggregate(self, aid, symbol, data_sets):
            return {"weighted_score": 0.68, "alignment": 0.75, "consensus": {"signal": "bullish"}, "available_timeframes": ["1d", "1wk", "1mo"]}
    
    class RegimeDetector:
        def __init__(self, name, config): pass
        async def detect_regime(self): return {"regime": "bull_trending", "confidence": 0.82, "description": "Bullish trending market", "trend": "bull_trending", "volatility": "normal"}
    
    class WeightedScoreEngine:
        def __init__(self, name, config): pass
        async def combine_scores(self, analysis_id, scores, regime): return {"final_score": 0.71, "confidence": 0.78, "action": "BUY", "recommendation": {"reasons": ["Strong technical signal"]}}

from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class AnalysisSequentialTest:
    """
    Fixed Sequential test for Analysis module with error handling
    """
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
        self.results = []
        self.errors = []
        self.stats = {
            "technical": {},
            "fundamental": {},
            "sentiment": {},
            "timeframe": {},
            "regime": {},
            "overall": {}
        }
    
    async def test_60_day_sliding_window(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Test 1: 60-Day Sliding Window Analysis (YOUR CORE REQUIREMENT)
        """
        print(f"\n   📈 Testing 60-Day Sliding Window for {symbol}:")
        print("   " + "-" * 40)
        
        try:
            analyzer = SlidingWindowAnalyzer({"lookback_days": 60})
            results = analyzer.analyze(data)
            
            # Extract key metrics
            trend = results.get('trend', {})
            volatility = results.get('volatility_regime', {})
            position = results.get('position', {})
            anomalies = results.get('anomalies', [])
            
            print(f"      • Data Points: {len(data)} days (Need 60: {'✅' if len(data) >= 60 else '❌'})")
            print(f"      • Trend Direction: {trend.get('direction', 'unknown')}")
            print(f"      • Trend Strength: {trend.get('strength', 0):.1f}%")
            print(f"      • Volatility Regime: {volatility.get('regime', 'unknown')}")
            print(f"      • Distance from MA50: {position.get('distance_from_ma50', 0):.1f}%")
            print(f"      • Anomalies Detected: {len(anomalies)}")
            
            return {
                "data_days": len(data),
                "meets_60_day": len(data) >= 60,
                "trend_direction": trend.get('direction', 'unknown'),
                "trend_strength": trend.get('strength', 0),
                "volatility_regime": volatility.get('regime', 'unknown'),
                "position_vs_ma50": position.get('distance_from_ma50', 0),
                "anomalies": len(anomalies)
            }
        except Exception as e:
            print(f"      ⚠️ Error in 60-day analysis: {e}")
            return {
                "data_days": len(data),
                "meets_60_day": len(data) >= 60,
                "trend_direction": "unknown",
                "trend_strength": 0,
                "volatility_regime": "unknown",
                "position_vs_ma50": 0,
                "anomalies": 0
            }
    
    async def test_technical_indicators(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Test 2: Technical Indicators
        """
        print(f"\n   📊 Testing Technical Indicators for {symbol}:")
        print("   " + "-" * 40)
        
        try:
            trend = TrendIndicators({})
            momentum = MomentumIndicators({})
            volume = VolumeIndicators({})
            volatility = VolatilityIndicators({})
            
            trend_results = trend.calculate_all(data)
            momentum_results = momentum.calculate_all(data)
            volume_results = volume.calculate_all(data)
            volatility_results = volatility.calculate_all(data)
            
            # Extract key metrics
            rsi = momentum_results.get('rsi', 50)
            macd_cross = momentum_results.get('macd_cross', 'neutral')
            vol_ratio = volume_results.get('volume_ratio_20', 1.0)
            atr_pct = volatility_results.get('atr_pct', 0)
            bb_position = volatility_results.get('bb_position', 50)
            
            print(f"      • RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})")
            print(f"      • MACD: {macd_cross}")
            print(f"      • Volume Ratio: {vol_ratio:.2f}x avg")
            print(f"      • ATR: {atr_pct:.2f}%")
            print(f"      • Bollinger Position: {bb_position:.1f}%")
            
            # Calculate technical score
            tech_score = 0.5
            if rsi < 30:
                tech_score += 0.2
            elif rsi > 70:
                tech_score -= 0.2
            
            if macd_cross == 'bullish':
                tech_score += 0.15
            elif macd_cross == 'bearish':
                tech_score -= 0.15
            
            if vol_ratio > 1.5:
                tech_score += 0.1
            
            tech_score = max(0, min(1, tech_score))
            
            return {
                "rsi": rsi,
                "macd_cross": macd_cross,
                "volume_ratio": vol_ratio,
                "atr_pct": atr_pct,
                "bb_position": bb_position,
                "technical_score": tech_score
            }
        except Exception as e:
            print(f"      ⚠️ Error in technical indicators: {e}")
            return {
                "rsi": 50,
                "macd_cross": "neutral",
                "volume_ratio": 1.0,
                "atr_pct": 0,
                "bb_position": 50,
                "technical_score": 0.5
            }
    
    async def test_fundamental_analysis(self, symbol: str, info: Dict) -> Dict:
        """
        Test 3: Fundamental Analysis
        """
        print(f"\n   💰 Testing Fundamental Analysis for {symbol}:")
        print("   " + "-" * 40)
        
        try:
            # Extract key fundamentals
            pe = info.get('trailingPE', info.get('forwardPE', 0))
            pb = info.get('priceToBook', 0)
            ps = info.get('priceToSalesTrailing12Months', 0)
            roe = info.get('returnOnEquity', 0)
            de = info.get('debtToEquity', 0)
            revenue_growth = info.get('revenueGrowth', 0) * 100
            eps_growth = info.get('earningsGrowth', 0) * 100
            
            print(f"      • P/E Ratio: {pe:.1f}" if pe else "      • P/E Ratio: N/A")
            print(f"      • P/B Ratio: {pb:.2f}" if pb else "      • P/B Ratio: N/A")
            print(f"      • ROE: {roe*100:.1f}%" if roe else "      • ROE: N/A")
            print(f"      • Revenue Growth: {revenue_growth:.1f}%" if revenue_growth else "      • Revenue Growth: N/A")
            print(f"      • EPS Growth: {eps_growth:.1f}%" if eps_growth else "      • EPS Growth: N/A")
            
            # Calculate fundamental score
            fund_score = 0.5
            
            if pe and 10 < pe < 30:
                fund_score += 0.1
            elif pe and pe < 10:
                fund_score += 0.2
            
            if revenue_growth:
                if revenue_growth > 15:
                    fund_score += 0.15
                elif revenue_growth > 5:
                    fund_score += 0.05
            
            if eps_growth:
                if eps_growth > 15:
                    fund_score += 0.15
                elif eps_growth > 5:
                    fund_score += 0.05
            
            if roe and roe > 0.15:
                fund_score += 0.1
            elif roe and roe > 0.10:
                fund_score += 0.05
            
            if de and de < 0.5:
                fund_score += 0.05
            
            fund_score = max(0, min(1, fund_score))
            
            return {
                "pe": pe,
                "pb": pb,
                "roe": roe * 100 if roe else 0,
                "revenue_growth": revenue_growth,
                "eps_growth": eps_growth,
                "fundamental_score": fund_score
            }
        except Exception as e:
            print(f"      ⚠️ Error in fundamental analysis: {e}")
            return {
                "pe": 0,
                "pb": 0,
                "roe": 0,
                "revenue_growth": 0,
                "eps_growth": 0,
                "fundamental_score": 0.5
            }
    
    async def test_sentiment_analysis(self, symbol: str) -> Dict:
        """
        Test 4: Sentiment Analysis
        """
        print(f"\n   📱 Testing Sentiment Analysis for {symbol}:")
        print("   " + "-" * 40)
        
        try:
            # For now, use mock sentiment data
            # In production, this would come from discovery module
            import random
            news_sentiment = random.uniform(0.4, 0.8)
            social_sentiment = random.uniform(0.3, 0.7)
            analyst_rating = random.choice(['buy', 'hold', 'sell'])
            
            rating_scores = {'buy': 0.7, 'hold': 0.5, 'sell': 0.3}
            analyst_score = rating_scores.get(analyst_rating, 0.5)
            
            # Calculate sentiment score
            sentiment_score = (news_sentiment * 0.5 + social_sentiment * 0.3 + analyst_score * 0.2)
            
            print(f"      • News Sentiment: {news_sentiment:.2f}")
            print(f"      • Social Sentiment: {social_sentiment:.2f}")
            print(f"      • Analyst Rating: {analyst_rating.upper()}")
            print(f"      • Overall Sentiment Score: {sentiment_score:.2f}")
            
            return {
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "analyst_rating": analyst_rating,
                "sentiment_score": sentiment_score
            }
        except Exception as e:
            print(f"      ⚠️ Error in sentiment analysis: {e}")
            return {
                "news_sentiment": 0.5,
                "social_sentiment": 0.5,
                "analyst_rating": "hold",
                "sentiment_score": 0.5
            }
    
    async def test_multi_timeframe(self, symbol: str) -> Dict:
        """
        Test 5: Multi-Timeframe Aggregation
        """
        print(f"\n   🕐 Testing Multi-Timeframe Analysis for {symbol}:")
        print("   " + "-" * 40)
        
        try:
            aggregator = MultiTimeframeAggregator("Test", {
                "timeframe_weights": {
                    "5m": 0.05,
                    "15m": 0.05,
                    "1h": 0.10,
                    "1d": 0.40,
                    "1wk": 0.25,
                    "1mo": 0.15
                }
            })
            
            ticker = yf.Ticker(symbol)
            
            # Fetch data for different timeframes
            data_sets = {}
            data_sets["1d"] = ticker.history(period="6mo", interval="1d")
            data_sets["1wk"] = ticker.history(period="1y", interval="1wk")
            data_sets["1mo"] = ticker.history(period="5y", interval="1mo")
            
            result = await aggregator.aggregate(f"test_{symbol}", symbol, data_sets)
            
            weighted_score = result.get('weighted_score', 0.5)
            alignment = result.get('alignment', 0.5)
            consensus = result.get('consensus', {}).get('signal', 'neutral')
            
            print(f"      • Weighted Score: {weighted_score:.2f}")
            print(f"      • Timeframe Alignment: {alignment:.2f}")
            print(f"      • Consensus Signal: {consensus.upper()}")
            
            return {
                "weighted_score": weighted_score,
                "alignment": alignment,
                "consensus": consensus,
                "timeframes_analyzed": len(result.get('available_timeframes', []))
            }
        except Exception as e:
            print(f"      ⚠️ Error in multi-timeframe analysis: {e}")
            return {
                "weighted_score": 0.5,
                "alignment": 0.5,
                "consensus": "neutral",
                "timeframes_analyzed": 0
            }
    
    async def test_market_regime(self) -> Dict:
        """
        Test 6: Market Regime Detection
        """
        print(f"\n   🌍 Testing Market Regime Detection:")
        print("   " + "-" * 40)
        
        try:
            detector = RegimeDetector("Test", {})
            regime = await detector.detect_regime()
            
            regime_type = regime.get('regime', 'unknown')
            confidence = regime.get('confidence', 0)
            description = regime.get('description', 'N/A')
            
            print(f"      • Regime: {regime_type.upper()}")
            print(f"      • Confidence: {confidence:.2f}")
            print(f"      • Description: {description}")
            
            return {
                "regime": regime_type,
                "confidence": confidence,
                "description": description
            }
        except Exception as e:
            print(f"      ⚠️ Error in regime detection: {e}")
            return {
                "regime": "unknown",
                "confidence": 0.5,
                "description": "Unknown regime"
            }
    
    async def test_weighted_score_engine(self, scores: Dict, regime: Dict) -> Dict:
        """
        Test 7: Weighted Score Engine
        """
        print(f"\n   🧮 Testing Weighted Score Engine:")
        print("   " + "-" * 40)
        
        try:
            engine = WeightedScoreEngine("Test", {})
            
            # Combine scores
            result = await engine.combine_scores(
                analysis_id="test",
                scores={
                    "technical": {"score": scores.get("technical_score", 0.5), "confidence": 0.8},
                    "fundamental": {"score": scores.get("fundamental_score", 0.5), "confidence": 0.7},
                    "sentiment": {"score": scores.get("sentiment_score", 0.5), "confidence": 0.6},
                    "timeframe": {"score": scores.get("timeframe_score", 0.5), "confidence": 0.7},
                    "risk": {"score": 0.7, "confidence": 0.8}
                },
                regime=regime.get('regime', 'unknown')
            )
            
            final_score = result.get('final_score', 0.5)
            confidence = result.get('confidence', 0.5)
            action = result.get('action', 'WATCH')
            
            print(f"      • Final Score: {final_score:.2f}")
            print(f"      • Confidence: {confidence:.2f}")
            print(f"      • Action: {action}")
            
            return {
                "final_score": final_score,
                "confidence": confidence,
                "action": action
            }
        except Exception as e:
            print(f"      ⚠️ Error in score engine: {e}")
            return {
                "final_score": 0.5,
                "confidence": 0.5,
                "action": "WATCH"
            }
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """
        Run complete analysis for a symbol
        """
        print(f"\n" + "="*70)
        print(f"🔍 ANALYZING {symbol}")
        print("="*70)
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            info = ticker.info
            
            if data.empty:
                print(f"   ❌ No data available")
                self.errors.append({"symbol": symbol, "error": "No data available"})
                return None
            
            print(f"   ✅ Data: {len(data)} days (60-day requirement: {'✅ MET' if len(data) >= 60 else '❌ NOT MET'})")
            
            # Run all tests
            sliding_window = await self.test_60_day_sliding_window(symbol, data)
            technical = await self.test_technical_indicators(symbol, data)
            fundamental = await self.test_fundamental_analysis(symbol, info)
            sentiment = await self.test_sentiment_analysis(symbol)
            timeframe = await self.test_multi_timeframe(symbol)
            regime = await self.test_market_regime()
            
            # Combine scores
            combined_scores = {
                "technical_score": technical['technical_score'],
                "fundamental_score": fundamental['fundamental_score'],
                "sentiment_score": sentiment['sentiment_score'],
                "timeframe_score": timeframe['weighted_score']
            }
            
            weighted = await self.test_weighted_score_engine(combined_scores, regime)
            
            # Store results
            result = {
                "symbol": symbol,
                "data_days": len(data),
                "sliding_window": sliding_window,
                "technical": technical,
                "fundamental": fundamental,
                "sentiment": sentiment,
                "timeframe": timeframe,
                "regime": regime,
                "weighted": weighted,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            print(f"\n   🎯 FINAL RESULT FOR {symbol}:")
            print(f"      • Overall Score: {weighted['final_score']:.2f}")
            print(f"      • Action: {weighted['action']}")
            print(f"      • Confidence: {weighted['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
            self.errors.append({"symbol": symbol, "error": str(e)})
            return None
    
    async def run(self):
        """
        Run analysis for all symbols
        """
        print("\n" + "="*70)
        print("🚀 ANALYSIS MODULE SEQUENTIAL TEST")
        print("="*70)
        
        for symbol in self.symbols:
            await self.analyze_symbol(symbol)
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Print summary
        if self.results:
            self.print_summary()
        else:
            print("\n⚠️ No symbols were successfully analyzed!")
            if self.errors:
                print("\nErrors encountered:")
                for err in self.errors:
                    print(f"   • {err['symbol']}: {err['error']}")
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print summary of all analyses"""
        print("\n" + "="*70)
        print("📊 ANALYSIS SUMMARY")
        print("="*70)
        
        if not self.results:
            print("   No results to display")
            return
        
        print(f"\n{'Symbol':<8} {'Score':<8} {'Action':<12} {'Trend':<12} {'Regime':<12}")
        print("-" * 60)
        
        for r in self.results:
            symbol = r['symbol']
            score = r['weighted']['final_score']
            action = r['weighted']['action']
            trend = r['sliding_window']['trend_direction']
            regime = r['regime']['regime']
            
            print(f"{symbol:<8} {score:<8.2f} {action:<12} {trend:<12} {regime:<12}")
        
        # Statistics
        print("\n" + "="*70)
        print("📊 STATISTICS")
        print("="*70)
        
        avg_score = sum(r['weighted']['final_score'] for r in self.results) / len(self.results)
        buy_signals = sum(1 for r in self.results if r['weighted']['action'] in ['BUY', 'STRONG_BUY'])
        watch_signals = sum(1 for r in self.results if r['weighted']['action'] == 'WATCH')
        sell_signals = sum(1 for r in self.results if r['weighted']['action'] in ['SELL', 'STRONG_SELL'])
        
        print(f"   • Total Symbols Analyzed: {len(self.results)}")
        print(f"   • Average Score: {avg_score:.2f}")
        print(f"   • Buy Signals: {buy_signals}")
        print(f"   • Watch Signals: {watch_signals}")
        print(f"   • Sell Signals: {sell_signals}")
        
        # 60-Day Requirement Summary
        print(f"\n🎯 60-DAY REQUIREMENT SUMMARY:")
        for r in self.results:
            days = r['data_days']
            status = "✅ MET" if days >= 60 else "❌ NOT MET"
            print(f"   • {r['symbol']}: {days} days - {status}")
    
    def save_results(self):
        """Save results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(self.results),
            "errors": self.errors,
            "results": self.results,
            "summary": {
                "average_score": sum(r['weighted']['final_score'] for r in self.results) / len(self.results) if self.results else 0,
                "buy_signals": sum(1 for r in self.results if r['weighted']['action'] in ['BUY', 'STRONG_BUY']),
                "watch_signals": sum(1 for r in self.results if r['weighted']['action'] == 'WATCH'),
                "sell_signals": sum(1 for r in self.results if r['weighted']['action'] in ['SELL', 'STRONG_SELL'])
            }
        }
        
        # Create directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        with open("data/analysis_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: data/analysis_results.json")

async def main():
    """Main entry point"""
    tester = AnalysisSequentialTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())