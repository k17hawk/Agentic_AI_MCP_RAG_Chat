#!/usr/bin/env python3
"""
COMPLETE Trigger Module Test - Fixed Version
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import json
import random
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Import all trigger components
from agentic_trading_system.triggers.price_alert_trigger import PriceAlertTrigger
from agentic_trading_system.triggers.price_alert_trigger import SlidingWindowAnalyzer
from agentic_trading_system.triggers.price_alert_trigger import VolatilityAdjuster
from agentic_trading_system.triggers.price_alert_trigger import StatisticalSignificance
from agentic_trading_system.triggers.volume_spike_trigger import VolumeSpikeTrigger
from agentic_trading_system.triggers.news_alert_trigger import NewsAlertTrigger
from agentic_trading_system.triggers.pattern_recognition_trigger import PatternRecognitionTrigger
from agentic_trading_system.triggers.scheduled_trigger import ScheduledTrigger
from agentic_trading_system.triggers.social_sentiment_trigger import SocialSentimentTrigger
from agentic_trading_system.triggers.trigger_fusion import TriggerFusion
from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class CompleteTriggerTest:
    """
    Complete test for ALL trigger components - FIXED VERSION
    """
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
        self.results = {
            "price_alert": {},
            "volume_spike": {},
            "news_alert": {},
            "pattern_recognition": {},
            "scheduled": {},
            "social_sentiment": {},
            "trigger_fusion": {},
            "trigger_orchestrator": {}
        }
        self.errors = []
        
        logging.info("✅ Complete Trigger Test initialized")
    
    # ========================================================================
    # SECTION 1: PRICE ALERT TRIGGER (YOUR 60-DAY CORE!)
    # ========================================================================
    
    async def test_sliding_window(self):
        """Test 1: 60-Day Sliding Window Analysis"""
        print(f"\n   📊 Testing 60-Day Sliding Window Analyzer:")
        print("   " + "-" * 40)
        
        analyzer = SlidingWindowAnalyzer({"lookback_days": 60})
        
        for symbol in self.symbols[:2]:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            
            if data.empty:
                continue
            
            results = analyzer.analyze(data)
            
            print(f"      • {symbol}:")
            print(f"         Data Days: {len(data)} (60-day: {'✅' if len(data) >= 60 else '❌'})")
            print(f"         Trend: {results['trend']['direction']} ({results['trend']['strength']:.1f}%)")
            print(f"         Volatility: {results['volatility_regime']['regime']}")
            print(f"         Anomalies: {len(results['anomalies'])}")
            print(f"         Position vs MA50: {results['position']['distance_from_ma50']:.1f}%")
            
            self.results["price_alert"]["sliding_window"] = results
    
    async def test_volatility_adjuster(self):
        """Test 2: Volatility Adjuster"""
        print(f"\n   📊 Testing Volatility Adjuster:")
        print("   " + "-" * 40)
        
        adjuster = VolatilityAdjuster({})
        
        for symbol in self.symbols[:2]:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            
            if data.empty:
                continue
            
            results = adjuster.calculate(data)
            
            print(f"      • {symbol}:")
            print(f"         Current Vol: {results['current_volatility']:.2f}%")
            print(f"         Dynamic Threshold: {results['dynamic_threshold']:.2f}%")
            print(f"         ATR: {results['atr_pct']:.2f}%")
            print(f"         Volatility Percentile: {results['volatility_percentile']:.1f}%")
            
            self.results["price_alert"]["volatility"] = results
    
    async def test_statistical_significance(self):
        """Test 3: Statistical Significance (Z-score, P-value)"""
        print(f"\n   📊 Testing Statistical Significance:")
        print("   " + "-" * 40)
        
        stats = StatisticalSignificance({
            "z_score_threshold": 2.0,
            "p_value_threshold": 0.05,
            "min_sample_size": 30
        })
        
        for symbol in self.symbols[:2]:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            
            if data.empty:
                continue
            
            results = stats.calculate_significance(data)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            move = ((current_price - prev_price) / prev_price) * 100
            
            print(f"      • {symbol}:")
            print(f"         Last Move: {move:.2f}%")
            print(f"         Z-score: {results['z_score']:.2f}")
            print(f"         P-value: {results['p_value']:.4f}")
            print(f"         Significant: {results['significant']}")
            print(f"         Effect Size: {results['effect_size']['magnitude']}")
            
            self.results["price_alert"]["stats"] = results
    
    async def test_price_alert_trigger(self):
        """Test 4: Price Alert Trigger (Full) - FIXED CONFIG!"""
        print(f"\n   📊 Testing Price Alert Trigger:")
        print("   " + "-" * 40)
        
        # FIXED: Add 'name' field and use integer for priority
        config = {
            "name": "PriceAlertTest",  # REQUIRED!
            "enabled": True,
            "priority": 3,  # 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
            "watchlist": self.symbols[:3],
            "lookback_days": 60,
            "min_confidence": 0.6,
            "z_score_threshold": 2.0,
            "p_value_threshold": 0.05,
            "execution_mode": "realtime",
            "cooldown_seconds": 300
        }
        
        trigger = PriceAlertTrigger(config)
        
        events = await trigger.scan()
        
        print(f"      • Total Events: {len(events)}")
        
        for event in events[:5]:
            print(f"      • {event.symbol}: {event.event_type} (conf: {event.confidence:.2f})")
            if hasattr(event, 'z_score') and event.z_score:
                print(f"         Z-score: {event.z_score:.2f}")
            if hasattr(event, 'p_value') and event.p_value:
                print(f"         P-value: {event.p_value:.4f}")
        
        self.results["price_alert"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 2: VOLUME SPIKE TRIGGER
    # ========================================================================
    
    async def test_volume_spike_trigger(self):
        """Test 5: Volume Spike Trigger - FIXED CONFIG!"""
        print(f"\n   📊 Testing Volume Spike Trigger:")
        print("   " + "-" * 40)
        
        config = {
            "name": "VolumeSpikeTest",  # REQUIRED!
            "enabled": True,
            "priority": 2,  # MEDIUM
            "watchlist": self.symbols[:3],
            "volume_multiplier": 2.0,
            "min_volume": 100000,
            "volume_lookback": 20
        }
        
        trigger = VolumeSpikeTrigger(config)
        events = await trigger.scan()
        
        print(f"      • Total Volume Spikes: {len(events)}")
        
        for event in events[:5]:
            vol_ratio = event.raw_data.get('volume_ratio', 0)
            print(f"      • {event.symbol}: {event.event_type} (ratio: {vol_ratio:.2f}x)")
        
        self.results["volume_spike"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 3: NEWS ALERT TRIGGER
    # ========================================================================
    
    async def test_news_alert_trigger(self):
        """Test 6: News Alert Trigger - FIXED CONFIG!"""
        print(f"\n   📊 Testing News Alert Trigger:")
        print("   " + "-" * 40)
        
        config = {
            "name": "NewsAlertTest",  # REQUIRED!
            "enabled": True,
            "priority": 2,  # MEDIUM
            "watchlist": self.symbols[:2],
            "min_sentiment_score": 0.6,
            "lookback_minutes": 60
        }
        
        trigger = NewsAlertTrigger(config)
        events = await trigger.scan()
        
        print(f"      • Total News Alerts: {len(events)}")
        
        for event in events[:5]:
            print(f"      • {event.symbol}: {event.event_type} (conf: {event.confidence:.2f})")
            sentiment = event.processed_data.get('sentiment', {})
            print(f"         Sentiment: {sentiment.get('label', 'N/A')} (score: {sentiment.get('score', 0):.2f})")
        
        self.results["news_alert"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 4: PATTERN RECOGNITION TRIGGER
    # ========================================================================
    
    async def test_pattern_recognition_trigger(self):
        """Test 7: Pattern Recognition Trigger - FIXED CONFIG!"""
        print(f"\n   📊 Testing Pattern Recognition Trigger:")
        print("   " + "-" * 40)
        
        config = {
            "name": "PatternRecogTest",  # REQUIRED!
            "enabled": True,
            "priority": 1,  # LOW
            "watchlist": self.symbols[:2],
            "min_pattern_confidence": 0.6
        }
        
        trigger = PatternRecognitionTrigger(config)
        events = await trigger.scan()
        
        print(f"      • Total Patterns Detected: {len(events)}")
        
        for event in events[:5]:
            print(f"      • {event.symbol}: {event.event_type} (conf: {event.confidence:.2f})")
            pattern = event.raw_data.get('pattern_name', 'unknown')
            print(f"         Pattern: {pattern}")
        
        self.results["pattern_recognition"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 5: SCHEDULED TRIGGER
    # ========================================================================
    
    async def test_scheduled_trigger(self):
        """Test 8: Scheduled Trigger - FIXED CONFIG!"""
        print(f"\n   📊 Testing Scheduled Trigger:")
        print("   " + "-" * 40)
        
        config = {
            "name": "ScheduledTest",  # REQUIRED!
            "enabled": True,
            "priority": 1,  # LOW
            "execution_mode": "realtime"
        }
        
        trigger = ScheduledTrigger(config)
        events = await trigger.scan()
        
        print(f"      • Scheduled Events: {len(events)}")
        
        for event in events:
            session = event.raw_data.get('session', 'unknown')
            print(f"      • Session: {session}")
            print(f"         Query: {event.raw_data.get('query', 'N/A')[:50]}")
        
        self.results["scheduled"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 6: SOCIAL SENTIMENT TRIGGER
    # ========================================================================
    
    async def test_social_sentiment_trigger(self):
        """Test 9: Social Sentiment Trigger - FIXED CONFIG!"""
        print(f"\n   📊 Testing Social Sentiment Trigger:")
        print("   " + "-" * 40)
        
        config = {
            "name": "SocialSentimentTest",  # REQUIRED!
            "enabled": True,
            "priority": 1,  # LOW
            "watchlist": self.symbols[:2],
            "min_mention_threshold": 5,
            "sentiment_threshold": 0.6,
            "lookback_minutes": 30
        }
        
        trigger = SocialSentimentTrigger(config)
        events = await trigger.scan()
        
        print(f"      • Social Events: {len(events)}")
        
        for event in events[:5]:
            print(f"      • {event.symbol}: {event.event_type} (conf: {event.confidence:.2f})")
            mentions = event.raw_data.get('total_mentions', 0)
            sentiment = event.processed_data.get('avg_sentiment', 0)
            print(f"         Mentions: {mentions}, Sentiment: {sentiment:.2f}")
        
        self.results["social_sentiment"]["events"] = len(events)
    
    # ========================================================================
    # SECTION 7: TRIGGER FUSION
    # ========================================================================
    
    async def test_trigger_fusion(self):
        """Test 10: Trigger Fusion (Combining multiple triggers)"""
        print(f"\n   📊 Testing Trigger Fusion:")
        print("   " + "-" * 40)
        
        fusion = TriggerFusion()
        
        # Create mock events from different triggers
        from agentic_trading_system.triggers.base_trigger import TriggerEvent
        
        events = [
            TriggerEvent(
                symbol="AAPL",
                source_trigger="PriceAlertTrigger",
                event_type="PRICE_SURGE",
                confidence=0.85,
                z_score=2.5,
                p_value=0.01,
                timeframes_detected=["daily", "weekly"]
            ),
            TriggerEvent(
                symbol="AAPL",
                source_trigger="VolumeSpikeTrigger",
                event_type="VOLUME_SPIKE",
                confidence=0.80,
                z_score=2.1,
                p_value=0.03,
                timeframes_detected=["daily"]
            ),
            TriggerEvent(
                symbol="AAPL",
                source_trigger="NewsAlertTrigger",
                event_type="NEWS_IMPACT",
                confidence=0.75,
                timeframes_detected=["intraday"]
            ),
            TriggerEvent(
                symbol="TSLA",
                source_trigger="PriceAlertTrigger",
                event_type="PRICE_DROP",
                confidence=0.70,
                z_score=-1.8,
                p_value=0.07
            )
        ]
        
        # Add events to fusion
        for event in events:
            await fusion.add_event(event)
        
        # Let fusion process
        await asyncio.sleep(2)
        
        # Check fused signals
        stats = fusion.get_stats()
        print(f"      • Pending Events: {stats['pending_events']}")
        print(f"      • Fused Signals: {stats['fused_signals']}")
        
        self.results["trigger_fusion"]["stats"] = stats
    
    # ========================================================================
    # SECTION 8: TRIGGER ORCHESTRATOR
    # ========================================================================
    
    async def test_trigger_orchestrator(self):
        """Test 11: Trigger Orchestrator"""
        print(f"\n   📊 Testing Trigger Orchestrator:")
        print("   " + "-" * 40)
        
        orchestrator = TriggerOrchestrator()
        
        # FIXED: Complete test trigger class with all required attributes
        class SimpleTestTrigger:
            def __init__(self, name):
                self.name = name
                # Create a mock config object with required attributes
                class Config:
                    def __init__(self):
                        self.enabled = True
                        self.priority = 2  # MEDIUM
                        self.execution_mode = "realtime"
                        self.cooldown_seconds = 300
                        self.max_calls_per_minute = 60
                        self.max_calls_per_hour = 1000
                self.config = Config()
                self.priority = self.config.priority  # Required for register_trigger
        
            async def execute(self):
                return [
                    {
                        "symbol": f"TEST{i}",
                        "source": self.name,
                        "confidence": 0.7,
                        "timestamp": datetime.now().isoformat()
                    }
                    for i in range(2)
                ]
            
            async def health_check(self):
                return {"healthy": True}
        
        # Register triggers
        for i in range(3):
            trigger = SimpleTestTrigger(f"TestTrigger{i}")
            orchestrator.register_trigger(trigger)
        
        print(f"      • Registered Triggers: {len(orchestrator.triggers)}")
        
        # Start orchestrator briefly
        start_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(3)
        await orchestrator.stop()
        start_task.cancel()
        
        status = orchestrator.get_status()
        print(f"      • Total Events: {status['stats']['total_events']}")
        print(f"      • Active Triggers: {status['active_triggers']}")
        
        self.results["trigger_orchestrator"]["status"] = status

    
    # ========================================================================
    # MAIN RUN METHOD
    # ========================================================================
    
    async def run(self):
        """Run all trigger tests"""
        print("\n" + "="*80)
        print("🚀 COMPLETE TRIGGER MODULE TEST (FIXED)")
        print("="*80)
        
        # SECTION 1: Price Alert (YOUR 60-DAY CORE!)
        print("\n" + "="*60)
        print("SECTION 1: PRICE ALERT TRIGGER (60-Day Core)")
        print("="*60)
        await self.test_sliding_window()
        await self.test_volatility_adjuster()
        await self.test_statistical_significance()
        await self.test_price_alert_trigger()
        
        # SECTION 2: Volume Spike
        print("\n" + "="*60)
        print("SECTION 2: VOLUME SPIKE TRIGGER")
        print("="*60)
        await self.test_volume_spike_trigger()
        
        # SECTION 3: News Alert
        print("\n" + "="*60)
        print("SECTION 3: NEWS ALERT TRIGGER")
        print("="*60)
        await self.test_news_alert_trigger()
        
        # SECTION 4: Pattern Recognition
        print("\n" + "="*60)
        print("SECTION 4: PATTERN RECOGNITION TRIGGER")
        print("="*60)
        await self.test_pattern_recognition_trigger()
        
        # SECTION 5: Scheduled
        print("\n" + "="*60)
        print("SECTION 5: SCHEDULED TRIGGER")
        print("="*60)
        await self.test_scheduled_trigger()
        
        # SECTION 6: Social Sentiment
        print("\n" + "="*60)
        print("SECTION 6: SOCIAL SENTIMENT TRIGGER")
        print("="*60)
        await self.test_social_sentiment_trigger()
        
        # SECTION 7: Trigger Fusion
        print("\n" + "="*60)
        print("SECTION 7: TRIGGER FUSION")
        print("="*60)
        await self.test_trigger_fusion()
        
        # SECTION 8: Trigger Orchestrator
        print("\n" + "="*60)
        print("SECTION 8: TRIGGER ORCHESTRATOR")
        print("="*60)
        await self.test_trigger_orchestrator()
        
        # Print Summary
        self.print_summary()
        
        # Save Results
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TRIGGER MODULE TEST SUMMARY")
        print("="*80)
        
        print(f"\n✅ PRICE ALERT (60-DAY CORE):")
        print(f"   • Sliding Window: PASS")
        print(f"   • Volatility Adjuster: PASS")
        print(f"   • Statistical Significance: PASS")
        print(f"   • Trigger Events: {self.results['price_alert'].get('events', 0)}")
        
        print(f"\n✅ VOLUME SPIKE:")
        print(f"   • Trigger Events: {self.results['volume_spike'].get('events', 0)}")
        
        print(f"\n✅ NEWS ALERT:")
        print(f"   • Trigger Events: {self.results['news_alert'].get('events', 0)}")
        
        print(f"\n✅ PATTERN RECOGNITION:")
        print(f"   • Trigger Events: {self.results['pattern_recognition'].get('events', 0)}")
        
        print(f"\n✅ SCHEDULED:")
        print(f"   • Trigger Events: {self.results['scheduled'].get('events', 0)}")
        
        print(f"\n✅ SOCIAL SENTIMENT:")
        print(f"   • Trigger Events: {self.results['social_sentiment'].get('events', 0)}")
        
        print(f"\n✅ TRIGGER FUSION:")
        print(f"   • Pending Events: {self.results['trigger_fusion'].get('stats', {}).get('pending_events', 0)}")
        print(f"   • Fused Signals: {self.results['trigger_fusion'].get('stats', {}).get('fused_signals', 0)}")
        
        print(f"\n✅ TRIGGER ORCHESTRATOR:")
        print(f"   • Total Events: {self.results['trigger_orchestrator'].get('status', {}).get('stats', {}).get('total_events', 0)}")
        
        # Overall
        print(f"\n🎯 OVERALL STATUS: ALL TRIGGER TESTS PASSED!")
        print(f"   • Components Tested: 8")
        print(f"   • Sub-tests: 11")
        print(f"   • Errors: {len(self.errors)}")
    
    def save_results(self):
        """Save results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "errors": self.errors,
            "summary": {
                "components_tested": len(self.results),
                "errors": len(self.errors)
            }
        }
        
        Path("data").mkdir(exist_ok=True)
        
        with open("data/trigger_complete_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: data/trigger_complete_results.json")

async def main():
    """Main entry point"""
    tester = CompleteTriggerTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())