"""
Volume Spike Trigger - Detects unusual trading volume
"""
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from agentic_trading_system.triggers.base_trigger import BaseTrigger, TriggerEvent

class VolumeSpikeTrigger(BaseTrigger):
    """
    Detects unusual volume spikes that may indicate institutional activity
    """
    
    def __init__(self, name: str, config: dict, memory_agent=None, message_bus=None, priority=None):
        if priority is not None:
            config['priority'] = priority
        super().__init__(
            name=name,
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Configuration
        self.volume_multiplier = config.get("volume_multiplier", 1.5)
        self.volume_lookback = config.get("lookback_days", 20)
        self.min_volume = config.get("min_absolute_volume", 50000)
        self.require_price_move = config.get("require_price_move", True)
        self.min_price_move = config.get("min_price_move", 1.0)
        
        # Watchlist
        self.watchlist = config.get("symbols", config.get("watchlist", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
            "META", "TSLA", "JPM", "V", "WMT"
        ]))
        
        logger.info(f"📊 {self.name} initialized with {len(self.watchlist)} stocks, multiplier: {self.volume_multiplier}")
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan for volume spikes
        """
        events = []
        
        for symbol in self.watchlist:
            try:
                # Check cooldown
                if not self._check_cooldown(symbol):
                    continue
                
                # Get volume data
                data = await self._get_volume_data(symbol)
                
                if data is None or data.empty:
                    logger.debug(f"No volume data for {symbol}, skipping")
                    continue
                
                # Analyze volume
                analysis = self._analyze_volume(data, symbol)
                
                if analysis and analysis.get("spike_detected", False):
                    event = await self._create_event(symbol, analysis)
                    events.append(event)
                    logger.info(f"📊 Volume spike detected for {symbol}: {analysis.get('volume_ratio', 0):.1f}x average, confidence: {event.confidence:.2f}")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        volume = event.raw_data.get("current_volume", 0)
        if volume < self.min_volume:
            logger.debug(f"Validation failed for {event.symbol}: volume {volume} < {self.min_volume}")
            return False
        if event.confidence < 0.35:
            logger.debug(f"Validation failed for {event.symbol}: confidence {event.confidence} < 0.35")
            return False
        return True
    
    async def _get_volume_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get volume data with caching and error handling"""
        cache_key = f"vol_{symbol}"
        cached = self.get_cache(cache_key)
        
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(symbol)
            
            try:
                # Use appropriate period based on lookback
                period = f"{max(self.volume_lookback + 10, 30)}d"
                data = stock.history(period=period, interval="1d")
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {e}")
                return None
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            if 'Volume' not in data.columns:
                logger.warning(f"No Volume column in data for {symbol}")
                return None
            
            data = data.dropna(subset=['Volume'])
            
            if len(data) < self.volume_lookback:
                logger.debug(f"Insufficient data for {symbol}: only {len(data)} days, need {self.volume_lookback}")
                return None
            
            self.set_cache(cache_key, data, ttl_seconds=300)
            logger.debug(f"Cached volume data for {symbol}: {len(data)} days")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {type(e).__name__}: {e}")
            return None
    
    def _analyze_volume(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Analyze volume patterns
        """
        try:
            if data.empty or len(data) < self.volume_lookback:
                return {"spike_detected": False, "error": "insufficient_data"}
            
            volumes = data['Volume'].values
            closes = data['Close'].values
            
            # Current volume (most recent trading day)
            current_volume = float(volumes[-1]) if len(volumes) > 0 else 0
            
            # Average volume (excluding current day)
            avg_volume = float(np.mean(volumes[-self.volume_lookback-1:-1])) if len(volumes) > self.volume_lookback else 0
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Check for spike - use configured multiplier
            spike_detected = volume_ratio >= self.volume_multiplier
            
            # Price movement
            current_price = float(closes[-1]) if len(closes) > 0 else 0
            prev_price = float(closes[-2]) if len(closes) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
            
            # Volume trend
            recent_volumes = volumes[-5:] if len(volumes) >= 5 else volumes
            older_volumes = volumes[-20:-5] if len(volumes) >= 20 else volumes[:len(volumes)-5]
            
            recent_avg = np.mean(recent_volumes) if len(recent_volumes) > 0 else 0
            older_avg = np.mean(older_volumes) if len(older_volumes) > 0 else recent_avg
            
            volume_trend = "increasing" if recent_avg > older_avg * 1.2 else "decreasing" if recent_avg < older_avg * 0.8 else "stable"
            
            # Simple VWAP
            typical_prices = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_prices * data['Volume']).sum() / data['Volume'].sum() if data['Volume'].sum() > 0 else current_price
            price_vs_vwap = ((current_price - vwap) / vwap) * 100
            
            # Calculate confidence - MORE GENEROUS for volume spikes
            # Base confidence from volume ratio
            if volume_ratio >= self.volume_multiplier:
                # Scale confidence: 1.5x = 0.5, 2x = 0.65, 3x = 0.8, 4x+ = 0.9
                volume_confidence = min(0.85, 0.4 + (volume_ratio - self.volume_multiplier) * 0.25)
            else:
                volume_confidence = 0.3
            
            # Price movement boost
            price_boost = 0
            if abs(price_change) > 2:
                price_boost = 0.2
            elif abs(price_change) > 1:
                price_boost = 0.1
            
            # Trend boost
            trend_boost = 0.1 if volume_trend == "increasing" else 0
            
            # Final confidence (capped at 0.9 for volume spikes)
            confidence = min(0.9, volume_confidence + price_boost + trend_boost)
            
            # Institutional score
            institutional_score = volume_confidence + price_boost
            
            result = {
                "spike_detected": spike_detected,
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "price_change": price_change,
                "current_price": current_price,
                "volume_trend": volume_trend,
                "vwap": float(vwap),
                "price_vs_vwap": float(price_vs_vwap),
                "institutional_score": min(1.0, institutional_score),
                "confidence": confidence,
                "data_points": len(data)
            }
            
            if spike_detected:
                logger.debug(f"Volume spike for {symbol}: {volume_ratio:.1f}x, price change: {price_change:.2f}%, confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing volume for {symbol}: {e}")
            return {"spike_detected": False, "error": str(e)}
    
    async def _create_event(self, symbol: str, analysis: Dict) -> TriggerEvent:
        try:
            # Determine event type based on price direction
            price_change = analysis.get("price_change", 0)
            if price_change > 1:
                event_type = "VOLUME_SPIKE_BULLISH"
            elif price_change < -1:
                event_type = "VOLUME_SPIKE_BEARISH"
            else:
                event_type = "VOLUME_SPIKE_NEUTRAL"

            return TriggerEvent(
                symbol=symbol,
                source_trigger=self.name,
                event_type=event_type,
                confidence=analysis.get("confidence", 0.5),
                raw_data={
                    "current_volume": analysis["current_volume"],
                    "avg_volume": analysis["avg_volume"],
                    "volume_ratio": analysis["volume_ratio"],
                    "price_change": analysis["price_change"],
                    "current_price": analysis["current_price"]
                },
                processed_data={
                    "volume_trend": analysis["volume_trend"],
                    "vwap": analysis["vwap"],
                    "price_vs_vwap": analysis["price_vs_vwap"],
                    "institutional_score": analysis["institutional_score"]
                },
                z_score=analysis["volume_ratio"],
                sample_size=analysis["data_points"],
                volatility=abs(analysis["price_change"]) / 100,
                timeframes_detected=["daily"],
                primary_timeframe="daily",
                market_regime=await self._get_market_regime(),
                correlation_id=f"vol_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
            )
        except Exception as e:
            logger.error(f"Failed to create event for {symbol}: {e}")
            raise
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            try:
                return await self.memory.get("current_market_regime")
            except:
                pass
        return None
    
    