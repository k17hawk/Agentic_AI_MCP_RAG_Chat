"""
Pattern Recognition Trigger - Detects chart patterns
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from logger import logging as logger

from triggers.base_trigger import BaseTrigger, TriggerEvent
from triggers.pattern_recognition_triggers.candlestick_patterns import CandlestickPatterns
from triggers.pattern_recognition_triggers.technical_patterns import TechnicalPatterns

class PatternRecognitionTrigger(BaseTrigger):
    """
    Detects technical chart patterns
    - Candlestick patterns (doji, engulfing, hammer, etc.)
    - Chart patterns (head & shoulders, double top/bottom, etc.)
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="PatternRecognitionTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Initialize pattern detectors
        self.candlestick = CandlestickPatterns(config)
        self.technical = TechnicalPatterns(config)
        
        # Configuration
        self.min_pattern_confidence = config.get("min_pattern_confidence", 0.6)
        self.require_volume_confirmation = config.get("require_volume_confirmation", True)
        
        # Watchlist
        self.watchlist = config.get("watchlist", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
            "META", "TSLA", "JPM", "V", "WMT"
        ])
        
        # Timeframes to check
        self.timeframes = [
            {"name": "daily", "period": "3mo", "interval": "1d"},
            {"name": "weekly", "period": "1y", "interval": "1wk"},
            {"name": "4h", "period": "1mo", "interval": "1h"}  # For intraday patterns
        ]
        
        logger.info("🔍 PatternRecognitionTrigger initialized")
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan for chart patterns
        """
        events = []
        
        for symbol in self.watchlist:
            try:
                # Check cooldown
                if not self._check_cooldown(symbol):
                    continue
                
                # Check each timeframe
                for tf in self.timeframes:
                    # Get data
                    data = await self._get_timeframe_data(symbol, tf)
                    
                    if data is None or len(data) < 50:
                        continue
                    
                    # Detect patterns
                    patterns = await self._detect_patterns(symbol, data, tf["name"])
                    
                    for pattern in patterns:
                        if pattern["confidence"] >= self.min_pattern_confidence:
                            event = await self._create_event(symbol, pattern, tf["name"])
                            events.append(event)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate pattern recognition events"""
        # Check confidence
        if event.confidence < self.config.min_confidence:
            return False
        
        return True
    
    async def _get_timeframe_data(self, symbol: str, tf: Dict) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe"""
        cache_key = f"pattern_{symbol}_{tf['name']}"
        cached = self.get_cache(cache_key)
        
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=tf["period"], interval=tf["interval"])
            
            if not data.empty:
                self.set_cache(cache_key, data, ttl_seconds=1800)  # 30 min cache
                return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    async def _detect_patterns(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[Dict]:
        """
        Detect all patterns in the data
        """
        patterns = []
        
        # Detect candlestick patterns
        candle_patterns = self.candlestick.detect(data)
        for pattern in candle_patterns:
            pattern["type"] = "candlestick"
            pattern["timeframe"] = timeframe
            patterns.append(pattern)
        
        # Detect technical patterns
        tech_patterns = self.technical.detect(data)
        for pattern in tech_patterns:
            pattern["type"] = "technical"
            pattern["timeframe"] = timeframe
            patterns.append(pattern)
        
        # Filter by confidence
        patterns = [p for p in patterns if p["confidence"] > 0.3]
        
        return patterns
    
    async def _create_event(self, symbol: str, pattern: Dict, timeframe: str) -> TriggerEvent:
        """Create a trigger event from pattern detection"""
        return TriggerEvent(
            symbol=symbol,
            source_trigger=self.name,
            event_type=f"PATTERN_{pattern['name'].upper()}",
            confidence=pattern["confidence"],
            raw_data={
                "pattern_name": pattern["name"],
                "pattern_type": pattern["type"],
                "direction": pattern.get("direction", "neutral"),
                "timeframe": timeframe
            },
            processed_data={
                "description": pattern.get("description", ""),
                "signals": pattern.get("signals", []),
                "confirmation_needed": pattern.get("needs_confirmation", True)
            },
            timeframes_detected=[timeframe],
            primary_timeframe=timeframe,
            market_regime=await self._get_market_regime(),
            correlation_id=f"pattern_{symbol}_{pattern['name']}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        )
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None