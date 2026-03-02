import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yfinance as yf
import pandas as pd
from loguru import logger
from datetime import datetime

# Import the pattern recognition modules
from triggers.pattern_recognition_trigger import PatternRecognitionTrigger
from triggers.pattern_recognition_triggers.candlestick_patterns import CandlestickPatterns
from triggers.pattern_recognition_triggers.technical_patterns import TechnicalPatterns


# ==============================
# LOGGER CONFIGURATION
# ==============================

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{message}</cyan>"
)

# ==============================
# MOCK DEPENDENCIES
# ==============================

class SimpleMemory:
    async def get(self, key):
        return None

    async def store(self, key, value, tier):
        pass


class SimpleMessageBus:
    async def publish(self, topic, message):
        logger.info(f"📨 Would publish to {topic}: {message}")

    async def send_to_agent(self, agent_name, message):
        logger.info(f"📨 Would send to {agent_name}: {message}")


# ==============================
# MAIN TEST FUNCTION
# ==============================

async def test_pattern_recognition():

    logger.info("🚀 Starting Pattern Recognition Test")
    logger.info("=" * 60)

    memory = SimpleMemory()
    message_bus = SimpleMessageBus()

    # ✅ FIXED CONFIG
    config = {
        "name": "pattern_recognition_trigger",  # REQUIRED
        "enabled": True,
        "priority": 2,  # Must be 1,2,3,4 (NOT 'MEDIUM')
        "min_pattern_confidence": 0.6,
        "watchlist": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "timeframes": [
            {"name": "daily", "period": "3mo", "interval": "1d"}
        ]
    }

    try:
        trigger = PatternRecognitionTrigger(
            config=config,
            memory_agent=memory,
            message_bus=message_bus
        )
    except Exception as e:
        logger.exception("❌ Failed to initialize PatternRecognitionTrigger")
        return

    logger.info("✅ Created PatternRecognitionTrigger")

    for symbol in config["watchlist"]:

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Analyzing {symbol}")
        logger.info(f"{'='*60}")

        try:
            data = await trigger._get_timeframe_data(
                symbol,
                config["timeframes"][0]
            )

            if data is None or data.empty:
                logger.error(f"❌ No data for {symbol}")
                continue

            logger.info(f"✅ Fetched {len(data)} days of data")
            logger.info(
                f"   Date range: {data.index[0].strftime('%Y-%m-%d')} "
                f"to {data.index[-1].strftime('%Y-%m-%d')}"
            )
            logger.info(f"   Current price: ${data['Close'].iloc[-1]:.2f}")

            # Candlestick patterns
            logger.info("\n📈 Testing Candlestick Patterns...")
            candlestick = CandlestickPatterns(config)
            candle_patterns = candlestick.detect(data)

            if candle_patterns:
                for pattern in candle_patterns:
                    logger.info(
                        f"   ✅ {pattern['name']} | "
                        f"{pattern['direction']} | "
                        f"conf: {pattern['confidence']}"
                    )
            else:
                logger.info("   ❌ No candlestick patterns found")

            # Technical patterns
            logger.info("\n📉 Testing Technical Patterns...")
            technical = TechnicalPatterns(config)
            tech_patterns = technical.detect(data)

            if tech_patterns:
                for pattern in tech_patterns:
                    logger.info(
                        f"   ✅ {pattern['name']} | "
                        f"{pattern['direction']} | "
                        f"conf: {pattern['confidence']}"
                    )
            else:
                logger.info("   ❌ No technical patterns found")

            # Full scan
            logger.info("\n🔄 Testing full scan...")
            events = await trigger.scan()

            if events:
                logger.info(f"✅ Generated {len(events)} events")
                for event in events:
                    logger.info(
                        f"   • {event.symbol}: "
                        f"{event.event_type} "
                        f"(conf: {event.confidence:.2f})"
                    )
            else:
                logger.info("❌ No trigger events generated")

            await asyncio.sleep(1)

        except Exception:
            logger.exception(f"❌ Error analyzing {symbol}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Pattern Recognition Test Complete!")


# ==============================
# ENTRY POINT
# ==============================

def main():

    print("""
╔══════════════════════════════════════════════════════════════╗
║     Pattern Recognition Trigger - Test Suite                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    asyncio.run(test_pattern_recognition())


if __name__ == "__main__":
    main()