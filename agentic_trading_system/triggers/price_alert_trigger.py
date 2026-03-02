"""
Price Alert Trigger - Detects significant price movements with statistical rigor
This is your ENHANCED version with 60-day sliding window analysis
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from triggers.base_trigger import BaseTrigger, TriggerEvent
from triggers.price_alert_triggers.sliding_window import SlidingWindowAnalyzer
from triggers.price_alert_triggers.volatility_adjusted import VolatilityAdjuster
from triggers.price_alert_triggers.statistical_significance import StatisticalSignificance

class PriceAlertTrigger(BaseTrigger):
    """
    Advanced price movement detection with:
    - 60-day sliding window analysis (YOUR REQUIREMENT)
    - Volatility-adjusted thresholds
    - Statistical significance testing
    - Multi-timeframe confirmation
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="PriceAlertTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Components
        self.sliding_window = SlidingWindowAnalyzer(config)
        self.volatility_adjuster = VolatilityAdjuster(config)
        self.statistics = StatisticalSignificance(config)
        
        # Watchlist (from config or default)
        self.watchlist = config.get("watchlist", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
            "META", "TSLA", "JPM", "V", "WMT"
        ])
        
        # Timeframes to analyze
        self.timeframes = [
            {"name": "intraday", "period": "1d", "interval": "5m", "weight": 0.2},
            {"name": "daily", "period": "3mo", "interval": "1d", "weight": 0.4},
            {"name": "weekly", "period": "1y", "interval": "1wk", "weight": 0.3},
            {"name": "monthly", "period": "5y", "interval": "1mo", "weight": 0.1}
        ]
        
        # Cache for historical data
        self.historical_cache = {}
        
        logger.info(f"📈 PriceAlertTrigger initialized with {len(self.watchlist)} stocks")
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan watchlist for significant price movements
        This is the main entry point called by orchestrator
        """
        events = []
        
        for symbol in self.watchlist:
            try:
                # Check cooldown
                if not self._check_cooldown(symbol):
                    continue
                
                # Get historical data
                data = await self._get_historical_data(symbol)
                
                if data is None or len(data) < 60:  # Need at least 60 days
                    continue
                
                # Analyze price movement
                analysis = await self._analyze_symbol(symbol, data)
                
                if analysis and analysis["trigger"]:
                    # Create trigger event
                    event = await self._create_event(symbol, analysis)
                    events.append(event)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate price alert events"""
        # Check statistical significance (YOUR 60-DAY REQUIREMENT)
        if not event.is_statistically_significant():
            logger.debug(f"Event {event.event_id} not statistically significant")
            return False
        
        # Check minimum confidence
        if event.confidence < self.config.min_confidence:
            return False
        
        return True
    
    async def _analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Comprehensive analysis of a symbol across timeframes
        """
        # Get 60-day sliding window analysis
        window_analysis = self.sliding_window.analyze(data, window_days=60)
        
        # Get volatility-adjusted metrics
        volatility_metrics = self.volatility_adjuster.calculate(data)
        
        # Calculate statistical significance
        stats = self.statistics.calculate_significance(data)
        
        # Multi-timeframe analysis
        timeframe_results = []
        for tf in self.timeframes:
            tf_data = await self._get_timeframe_data(symbol, tf)
            if tf_data is not None:
                tf_analysis = self._analyze_timeframe(tf_data, tf["name"])
                timeframe_results.append(tf_analysis)
        
        # Calculate current movement (last 2 days)
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        movement_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Determine if this triggers
        trigger = (
            stats["significant"] and
            abs(movement_pct) >= volatility_metrics["dynamic_threshold"]
        )
        
        if trigger:
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "movement_pct": float(movement_pct),
                "z_score": stats["z_score"],
                "p_value": stats["p_value"],
                "volatility": volatility_metrics["current_volatility"],
                "threshold": volatility_metrics["dynamic_threshold"],
                "sample_size": stats["sample_size"],
                "timeframe_alignment": self._calculate_alignment(timeframe_results),
                "timeframes": [tf["name"] for tf in timeframe_results if tf["aligned"]],
                "trigger": True
            }
        
        return None
    
    def _analyze_timeframe(self, data: pd.DataFrame, tf_name: str) -> Dict:
        """Analyze a specific timeframe"""
        if data.empty or len(data) < 20:
            return {"name": tf_name, "aligned": False, "confidence": 0}
        
        # Simple trend detection
        sma20 = data['Close'].rolling(20).mean().iloc[-1]
        sma50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
        current = data['Close'].iloc[-1]
        
        aligned = current > sma20
        if sma50:
            aligned = aligned and current > sma50
        
        return {
            "name": tf_name,
            "aligned": aligned,
            "confidence": 0.7 if aligned else 0.3,
            "price": float(current),
            "sma20": float(sma20),
            "sma50": float(sma50) if sma50 else None
        }
    
    def _calculate_alignment(self, timeframe_results: List[Dict]) -> float:
        """Calculate how well timeframes align"""
        if not timeframe_results:
            return 0.0
        
        aligned_count = sum(1 for tf in timeframe_results if tf["aligned"])
        return aligned_count / len(timeframe_results) * 100
    
    async def _create_event(self, symbol: str, analysis: Dict) -> TriggerEvent:
        """Create a trigger event from analysis"""
        return TriggerEvent(
            symbol=symbol,
            source_trigger=self.name,
            event_type="PRICE_SURGE" if analysis["movement_pct"] > 0 else "PRICE_DROP",
            confidence=min(1.0, (abs(analysis["z_score"]) / 3) * 0.8 + 0.2),
            z_score=analysis["z_score"],
            p_value=analysis["p_value"],
            sample_size=analysis["sample_size"],
            volatility=analysis["volatility"],
            raw_data={
                "movement_pct": analysis["movement_pct"],
                "current_price": analysis["current_price"],
                "threshold": analysis["threshold"]
            },
            processed_data={
                "timeframe_alignment": analysis["timeframe_alignment"],
                "timeframes_detected": analysis["timeframes"]
            },
            timeframes_detected=analysis["timeframes"],
            primary_timeframe="daily",
            market_regime=await self._get_market_regime(),
            correlation_id=f"price_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        )
    
    async def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
        cache_key = f"hist_{symbol}"
        cached = self.get_cache(cache_key)
        
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")  # 6 months for 60-day window + buffer
            
            if not data.empty:
                self.set_cache(cache_key, data, ttl_seconds=3600)  # Cache for 1 hour
                return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    async def _get_timeframe_data(self, symbol: str, tf: Dict) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe"""
        cache_key = f"tf_{symbol}_{tf['period']}"
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
            logger.error(f"Error fetching {symbol} {tf['period']}: {e}")
        
        return None
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None