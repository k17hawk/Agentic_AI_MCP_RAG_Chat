"""
Technical Analyzer - Performs technical analysis on stocks
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

class TechnicalAnalyzer(BaseAgent):
    """
    Technical Analysis Agent
    Calculates various technical indicators and generates scores
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Performs technical analysis on stocks",
            config=config
        )
        
        # Data cache
        self.cache = {}
        
        # Indicator weights
        self.weights = {
            "trend": 0.3,
            "momentum": 0.25,
            "volume": 0.2,
            "volatility": 0.15,
            "pattern": 0.1
        }
        
        logging.info(f"✅ TechnicalAnalyzer initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process technical analysis requests
        """
        if message.message_type == "analysis_request":
            analysis_id = message.content.get("analysis_id")
            symbol = message.content.get("symbol")
            
            # Perform analysis
            score, details = await self.analyze(symbol)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="analysis_result",
                content={
                    "analysis_id": analysis_id,
                    "agent": self.name,
                    "score": score,
                    "details": details
                }
            )
        
        return None
    
    async def analyze(self, symbol: str) -> tuple[float, Dict]:
        """
        Perform comprehensive technical analysis
        Returns score (0-1) and detailed breakdown
        """
        try:
            # Get data
            data = await self._get_stock_data(symbol)
            
            if data is None or len(data) < 50:
                return 0.5, {"error": "Insufficient data"}
            
            # Calculate indicators
            trend_score = self._analyze_trend(data)
            momentum_score = self._analyze_momentum(data)
            volume_score = self._analyze_volume(data)
            volatility_score = self._analyze_volatility(data)
            pattern_score = self._analyze_patterns(data)
            
            # Calculate weighted score
            total_score = (
                trend_score * self.weights["trend"] +
                momentum_score * self.weights["momentum"] +
                volume_score * self.weights["volume"] +
                volatility_score * self.weights["volatility"] +
                pattern_score * self.weights["pattern"]
            )
            
            details = {
                "trend": {"score": trend_score},
                "momentum": {"score": momentum_score},
                "volume": {"score": volume_score},
                "volatility": {"score": volatility_score},
                "pattern": {"score": pattern_score},
                "current_price": float(data['Close'].iloc[-1])
            }
            
            return float(total_score), details
            
        except Exception as e:
            logging.error(f"Error in technical analysis for {symbol}: {e}")
            return 0.5, {"error": str(e)}
    
    def _analyze_trend(self, data: pd.DataFrame) -> float:
        """Analyze trend using moving averages"""
        close = data['Close']
        current = close.iloc[-1]
        
        # Calculate MAs
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(data) >= 50 else ma20
        ma200 = close.rolling(200).mean().iloc[-1] if len(data) >= 200 else ma50
        
        # Score based on price relative to MAs
        score = 0.5
        
        if current > ma20:
            score += 0.1
        if current > ma50:
            score += 0.15
        if current > ma200:
            score += 0.2
        
        # ADX for trend strength (simplified)
        high = data['High']
        low = data['Low']
        
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift()),
                abs(low - close.shift())
            )
        )
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Strong trend if ATR is high relative to price
        atr_pct = (atr / current) * 100
        if atr_pct > 3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _analyze_momentum(self, data: pd.DataFrame) -> float:
        """Analyze momentum using RSI, MACD, etc."""
        close = data['Close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Score based on RSI
        if current_rsi < 30:
            rsi_score = 0.8  # Oversold - potential buy
        elif current_rsi > 70:
            rsi_score = 0.2  # Overbought - potential sell
        else:
            rsi_score = 0.5 + (current_rsi - 50) / 100
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        macd_score = 0.6 if macd.iloc[-1] > signal.iloc[-1] else 0.4
        
        # Combine scores
        score = (rsi_score * 0.6) + (macd_score * 0.4)
        
        return float(min(1.0, max(0.0, score)))
    
    def _analyze_volume(self, data: pd.DataFrame) -> float:
        """Analyze volume patterns"""
        volume = data['Volume']
        close = data['Close']
        
        # Volume ratio
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # OBV (simplified)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_trend = 0.6 if obv.iloc[-1] > obv.iloc[-20] else 0.4
        
        # Score based on volume ratio
        if vol_ratio > 2:
            vol_score = 0.8  # High volume - strong signal
        elif vol_ratio > 1.5:
            vol_score = 0.7
        elif vol_ratio > 1:
            vol_score = 0.6
        else:
            vol_score = 0.5
        
        # Combine
        score = (vol_score * 0.6) + (obv_trend * 0.4)
        
        return float(min(1.0, max(0.0, score)))
    
    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """Analyze volatility"""
        close = data['Close']
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        
        upper = ma20 + (std20 * 2)
        lower = ma20 - (std20 * 2)
        
        current = close.iloc[-1]
        
        # Position within bands
        if current > upper.iloc[-1]:
            bb_score = 0.3  # Overextended
        elif current < lower.iloc[-1]:
            bb_score = 0.7  # Oversold
        else:
            # Normal range
            range_pct = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            bb_score = 0.5 + (range_pct - 0.5) * 0.2
        
        # ATR
        high = data['High']
        low = data['Low']
        
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift()),
                abs(low - close.shift())
            )
        )
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / current) * 100
        
        # Lower ATR is better for entry
        atr_score = max(0, 1 - (atr_pct / 10))
        
        score = (bb_score * 0.5) + (atr_score * 0.5)
        
        return float(min(1.0, max(0.0, score)))
    
    def _analyze_patterns(self, data: pd.DataFrame) -> float:
        """Analyze chart patterns (simplified)"""
        # This would integrate with your pattern recognition triggers
        # For now, return neutral score
        return 0.5
    
    async def _get_stock_data(self, symbol: str, period: str = "6mo"):
        """Get stock data with caching"""
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = (datetime.now(), data)
                return data
        except Exception as e:
            logging.error(f"Error fetching {symbol}: {e}")
        
        return None