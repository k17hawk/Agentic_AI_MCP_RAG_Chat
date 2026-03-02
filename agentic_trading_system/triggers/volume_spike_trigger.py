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
from triggers.base_trigger import BaseTrigger, TriggerEvent

class VolumeSpikeTrigger(BaseTrigger):
    """
    Detects unusual volume spikes that may indicate institutional activity
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="VolumeSpikeTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Configuration
        self.volume_multiplier = config.get("volume_multiplier", 2.5)  # 2.5x average
        self.volume_lookback = config.get("volume_lookback", 20)  # 20-day average
        self.min_volume = config.get("min_volume", 100000)  # Min 100k shares
        self.require_price_move = config.get("require_price_move", True)
        self.min_price_move = config.get("min_price_move", 1.0)  # 1% move
        
        # Watchlist
        self.watchlist = config.get("watchlist", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
            "META", "TSLA", "JPM", "V", "WMT"
        ])
        
        logger.info("📊 VolumeSpikeTrigger initialized")
    
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
                
                if data is None:
                    continue
                
                # Analyze volume
                analysis = self._analyze_volume(data)
                
                if analysis["spike_detected"]:
                    event = await self._create_event(symbol, analysis)
                    events.append(event)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate volume spike events"""
        # Check minimum volume
        volume = event.raw_data.get("current_volume", 0)
        if volume < self.min_volume:
            return False
        
        # Check confidence
        if event.confidence < self.config.min_confidence:
            return False
        
        return True
    
    async def _get_volume_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get volume data with caching"""
        cache_key = f"vol_{symbol}"
        cached = self.get_cache(cache_key)
        
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="2mo")  # 2 months for volume analysis
            
            if not data.empty:
                self.set_cache(cache_key, data, ttl_seconds=300)  # 5 min cache
                return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns
        """
        if data.empty or len(data) < 30:
            return {"spike_detected": False}
        
        # Current volume
        current_volume = data['Volume'].iloc[-1]
        
        # Average volume
        avg_volume = data['Volume'].iloc[-self.volume_lookback:-1].mean()
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Check for spike
        spike_detected = volume_ratio >= self.volume_multiplier
        
        # Price movement
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Volume trend
        volume_trend = self._calculate_volume_trend(data)
        
        # Volume profile
        volume_profile = self._volume_profile(data)
        
        # Institutional vs retail (simplified)
        institutional_score = self._institutional_score(data, volume_ratio)
        
        return {
            "spike_detected": spike_detected,
            "current_volume": float(current_volume),
            "avg_volume": float(avg_volume),
            "volume_ratio": float(volume_ratio),
            "price_change": float(price_change),
            "volume_trend": volume_trend,
            "volume_profile": volume_profile,
            "institutional_score": institutional_score,
            "data_points": len(data)
        }
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> Dict:
        """Calculate volume trend"""
        volumes = data['Volume'].iloc[-10:]  # Last 10 days
        
        # Linear regression on volume
        x = np.arange(len(volumes))
        y = volumes.values
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Direction
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "flat"
        
        return {
            "direction": direction,
            "slope": float(slope),
            "strength": float(abs(slope) / volumes.mean()) if volumes.mean() > 0 else 0
        }
    
    def _volume_profile(self, data: pd.DataFrame) -> Dict:
        """Analyze volume profile"""
        volumes = data['Volume']
        prices = data['Close']
        
        # Find price levels with highest volume
        volume_price = pd.DataFrame({
            'volume': volumes,
            'price': prices
        }).dropna()
        
        # Simple volume-weighted average price
        vwap = (volume_price['volume'] * volume_price['price']).sum() / volume_price['volume'].sum()
        
        # Current price relative to VWAP
        current_price = prices.iloc[-1]
        price_vs_vwap = (current_price - vwap) / vwap * 100
        
        return {
            "vwap": float(vwap),
            "price_vs_vwap": float(price_vs_vwap),
            "above_vwap": current_price > vwap
        }
    
    def _institutional_score(self, data: pd.DataFrame, volume_ratio: float) -> float:
        """
        Estimate if volume is institutional (high ratio + price move)
        """
        score = 0.0
        
        # Volume ratio contribution
        if volume_ratio > 3:
            score += 0.5
        elif volume_ratio > 2:
            score += 0.3
        
        # Check for price movement with volume
        price_change = data['Close'].pct_change().iloc[-1] * 100
        if abs(price_change) > self.min_price_move:
            score += 0.3
        
        # Check if volume is unusual compared to recent
        recent_volumes = data['Volume'].iloc[-5:-1]
        if volume_ratio > recent_volumes.max() / data['Volume'].mean():
            score += 0.2
        
        return min(1.0, score)
    
    async def _create_event(self, symbol: str, analysis: Dict) -> TriggerEvent:
        """Create a trigger event from volume spike"""
        confidence = min(1.0, analysis["volume_ratio"] / 5)  # 5x = 100% confidence
        
        return TriggerEvent(
            symbol=symbol,
            source_trigger=self.name,
            event_type="VOLUME_SPIKE",
            confidence=confidence,
            raw_data={
                "current_volume": analysis["current_volume"],
                "avg_volume": analysis["avg_volume"],
                "volume_ratio": analysis["volume_ratio"],
                "price_change": analysis["price_change"]
            },
            processed_data={
                "volume_trend": analysis["volume_trend"],
                "volume_profile": analysis["volume_profile"],
                "institutional_score": analysis["institutional_score"]
            },
            z_score=analysis["volume_ratio"],  # Using ratio as proxy for z-score
            sample_size=analysis["data_points"],
            volatility=analysis["volume_trend"]["strength"],
            timeframes_detected=["intraday", "daily"],
            primary_timeframe="daily",
            market_regime=await self._get_market_regime(),
            correlation_id=f"vol_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        )
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None