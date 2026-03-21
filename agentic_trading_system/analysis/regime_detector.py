"""
Regime Detector - Identifies current market regime
"""
import numpy as np
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

class RegimeDetector(BaseAgent):
    """
    Detects market regime:
    - Bull/Bear Trending
    - Ranging
    - High/Low Volatility
    - Transition
    
    Uses multiple market indicators to classify current regime
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Detects current market regime",
            config=config
        )
        
        # Cache for market data
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Market proxies
        self.market_symbols = config.get("market_symbols", {
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
            "dow": "^DJI",
            "vix": "^VIX",
            "russell": "^RUT"
        })
        
        # Regime detection thresholds
        self.adx_threshold = config.get("adx_threshold", 25)  # ADX > 25 = trending
        self.vix_threshold = config.get("vix_threshold", 20)  # VIX > 20 = high volatility
        self.vix_panic_threshold = config.get("vix_panic_threshold", 30)  # VIX > 30 = panic
        self.correlation_threshold = config.get("correlation_threshold", 0.7)  # High correlation
        
        # Moving average periods
        self.ma_periods = config.get("ma_periods", [20, 50, 200])
        
        logging.info(f"✅ RegimeDetector initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process regime detection requests
        """
        if message.message_type == "regime_request":
            analysis_id = message.content.get("analysis_id")
            symbol = message.content.get("symbol")
            
            regime = await self.detect_regime(symbol)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="regime_result",
                content={
                    "analysis_id": analysis_id,
                    "regime": regime["regime"],
                    "details": regime
                }
            )
        
        elif message.message_type == "broadcast_regime":
            # Broadcast current regime to all agents
            regime = await self.detect_regime()
            return AgentMessage(
                sender=self.name,
                receiver="broadcast",
                message_type="regime_update",
                content=regime,
                priority=4
            )
        
        return None
    
    async def detect_regime(self, symbol: str = "SPY") -> Dict[str, Any]:
        """
        Detect current market regime
        Uses SPY as market proxy if no symbol provided
        """
        try:
            # Get market data for multiple indices
            market_data = await self._get_market_data()
            
            if not market_data:
                return {"regime": "unknown", "confidence": 0, "error": "No data"}
            
            # Get primary market data (SPY)
            spy_data = market_data.get("sp500")
            if spy_data is None or spy_data.empty:
                return {"regime": "unknown", "confidence": 0, "error": "No SPY data"}
            
            # Calculate indicators
            returns = spy_data['Close'].pct_change().dropna() * 100
            
            # Volatility (VIX)
            vix_level = await self._get_vix_level()
            vix_data = market_data.get("vix")
            
            # Trend (using moving averages and ADX)
            ma20 = spy_data['Close'].tail(20).mean()
            ma50 = spy_data['Close'].tail(50).mean() if len(spy_data) >= 50 else ma20
            ma200 = spy_data['Close'].tail(200).mean() if len(spy_data) >= 200 else ma50
            
            current_price = spy_data['Close'].iloc[-1]
            price_1m_ago = spy_data['Close'].iloc[-20] if len(spy_data) >= 20 else current_price
            price_3m_ago = spy_data['Close'].iloc[-60] if len(spy_data) >= 60 else current_price
            
            # Trend direction
            above_ma20 = current_price > ma20
            above_ma50 = current_price > ma50
            above_ma200 = current_price > ma200
            
            # Short-term momentum
            momentum_1m = (current_price - price_1m_ago) / price_1m_ago * 100
            momentum_3m = (current_price - price_3m_ago) / price_3m_ago * 100
            
            # Calculate ADX for trend strength
            adx = self._calculate_adx(spy_data)
            is_trending = adx > self.adx_threshold
            
            # Calculate market breadth (simplified)
            breadth_score = await self._calculate_market_breadth()
            
            # Volatility analysis
            current_vol = returns.tail(20).std()
            historical_vol = returns.tail(60).std()
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            
            # VIX analysis
            vix_analysis = self._analyze_vix(vix_level, vix_data)
            is_high_volatility = vix_analysis["is_high_volatility"] or vol_ratio > 1.5
            is_panic = vix_analysis["is_panic"]
            
            # Correlation analysis
            correlation = await self._calculate_market_correlation(market_data)
            
            # Determine regime
            regime = self._classify_regime(
                above_ma20, above_ma50, above_ma200,
                momentum_1m, momentum_3m,
                vol_ratio, is_trending, is_high_volatility, is_panic,
                adx, vix_level, correlation
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(spy_data, adx, vix_level, correlation)
            
            # Generate regime description
            description = self._generate_regime_description(regime, vix_level, adx, momentum_1m)
            
            return {
                "regime": regime["type"],
                "confidence": confidence,
                "trend": regime["trend"],
                "volatility": regime["volatility"],
                "adx": float(adx),
                "vix": float(vix_level) if vix_level else None,
                "vix_signal": vix_analysis["signal"],
                "vol_ratio": float(vol_ratio),
                "momentum_1m": float(momentum_1m),
                "momentum_3m": float(momentum_3m),
                "correlation": float(correlation),
                "breadth": float(breadth_score) if breadth_score else None,
                "description": description,
                "details": regime,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error detecting regime: {e}")
            return {"regime": "unknown", "confidence": 0, "error": str(e)}
    
    def _classify_regime(self, above_ma20: bool, above_ma50: bool, above_ma200: bool,
                        momentum_1m: float, momentum_3m: float,
                        vol_ratio: float, is_trending: bool, is_high_volatility: bool, is_panic: bool,
                        adx: float, vix: Optional[float], correlation: float) -> Dict[str, Any]:
        """
        Classify market regime based on indicators
        """
        # Count bullish signals
        bullish_count = sum([above_ma20, above_ma50, above_ma200])
        
        # Check for panic first
        if is_panic:
            return {
                "type": "panic",
                "trend": "panic",
                "volatility": "extreme",
                "description": "Panic selling - extreme volatility",
                "bullish_signals": bullish_count,
                "adx": float(adx)
            }
        
        # Determine trend based on moving averages and momentum
        if bullish_count >= 2 and momentum_1m > 0:
            if is_trending:
                trend = "strong_bull_trending"
                trend_desc = "Strong bullish trending market"
            else:
                trend = "bull_trending"
                trend_desc = "Bullish trending market"
        elif bullish_count <= 1 and momentum_1m < 0:
            if is_trending:
                trend = "strong_bear_trending"
                trend_desc = "Strong bearish trending market"
            else:
                trend = "bear_trending"
                trend_desc = "Bearish trending market"
        elif bullish_count >= 2:
            trend = "bull_ranging"
            trend_desc = "Bullish ranging market"
        elif bullish_count <= 1:
            trend = "bear_ranging"
            trend_desc = "Bearish ranging market"
        else:
            trend = "neutral_ranging"
            trend_desc = "Neutral ranging market"
        
        # Determine volatility regime
        if is_high_volatility:
            if vol_ratio > 2.0:
                volatility = "extreme_volatility"
                vol_desc = "Extreme volatility"
            else:
                volatility = "high_volatility"
                vol_desc = "High volatility environment"
        elif vol_ratio < 0.8:
            volatility = "low_volatility"
            vol_desc = "Low volatility environment"
        else:
            volatility = "normal_volatility"
            vol_desc = "Normal volatility environment"
        
        # Check for transition (ADX rising but not yet trending)
        is_transition = 20 < adx < 25 and abs(momentum_1m) > 3
        
        # Combine into final regime
        if is_transition:
            regime_type = "transition"
            description = "Market in transition - trend emerging"
        elif "trending" in trend:
            if "extreme" in volatility:
                regime_type = f"{trend.split('_')[0]}_trending_extreme_vol"
                description = f"{trend_desc} with {vol_desc.lower()}"
            elif "high" in volatility:
                regime_type = f"{trend.split('_')[0]}_trending_high_vol"
                description = f"{trend_desc} with {vol_desc.lower()}"
            else:
                regime_type = trend
                description = trend_desc
        else:
            if "extreme" in volatility:
                regime_type = "extreme_volatility_ranging"
                description = f"Ranging market with {vol_desc.lower()}"
            elif "high" in volatility:
                regime_type = "high_volatility_ranging"
                description = f"Ranging market with {vol_desc.lower()}"
            elif "low" in volatility:
                regime_type = "low_volatility_ranging"
                description = f"Ranging market with {vol_desc.lower()}"
            else:
                regime_type = "ranging"
                description = "Ranging market"
        
        return {
            "type": regime_type,
            "trend": trend,
            "volatility": volatility,
            "description": description,
            "bullish_signals": bullish_count,
            "adx": float(adx)
        }
    
    def _analyze_vix(self, vix_level: Optional[float], vix_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze VIX for volatility signals
        """
        result = {
            "is_high_volatility": False,
            "is_panic": False,
            "signal": "normal"
        }
        
        if vix_level:
            if vix_level > self.vix_panic_threshold:
                result["is_panic"] = True
                result["is_high_volatility"] = True
                result["signal"] = "panic"
            elif vix_level > self.vix_threshold:
                result["is_high_volatility"] = True
                result["signal"] = "elevated"
            elif vix_level < 15:
                result["signal"] = "complacent"
        
        # Check VIX trend if data available
        if vix_data is not None and len(vix_data) > 5:
            vix_current = vix_data['Close'].iloc[-1]
            vix_5d_ago = vix_data['Close'].iloc[-5]
            vix_trend = (vix_current - vix_5d_ago) / vix_5d_ago * 100
            result["vix_trend"] = float(vix_trend)
        
        return result
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ADX for trend strength
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        if len(data) < period * 2:
            return 20.0  # Default neutral
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0
    
    async def _calculate_market_breadth(self) -> Optional[float]:
        """
        Calculate market breadth (advancing/declining ratio)
        Simplified - would need more data in production
        """
        # Placeholder - in production, would use advance/decline data
        return 0.5
    
    async def _calculate_market_correlation(self, market_data: Dict) -> float:
        """
        Calculate average correlation between major indices
        """
        correlations = []
        symbols = list(market_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sym1 = symbols[i]
                sym2 = symbols[j]
                
                data1 = market_data.get(sym1)
                data2 = market_data.get(sym2)
                
                if data1 is not None and data2 is not None and len(data1) > 20 and len(data2) > 20:
                    # Calculate correlation of returns
                    returns1 = data1['Close'].pct_change().dropna().tail(20)
                    returns2 = data2['Close'].pct_change().dropna().tail(20)
                    
                    if len(returns1) == len(returns2) and len(returns1) > 5:
                        corr = returns1.corr(returns2)
                        if not pd.isna(corr):
                            correlations.append(corr)
        
        if correlations:
            return float(np.mean(correlations))
        
        return 0.7  # Default high correlation
    
    def _calculate_confidence(self, data: pd.DataFrame, adx: float, 
                            vix: Optional[float], correlation: float) -> float:
        """
        Calculate confidence in regime detection
        """
        confidence = 0.7  # Base confidence
        
        # ADX confidence (higher ADX = more confident in trend)
        if adx > 30:
            confidence += 0.15
        elif adx < 20:
            confidence -= 0.1
        
        # VIX confidence (clear signal = more confident)
        if vix:
            if vix > 30 or vix < 15:
                confidence += 0.1
        else:
            confidence -= 0.1  # No VIX data
        
        # Correlation confidence (high correlation = clearer regime)
        if correlation > 0.8:
            confidence += 0.1
        elif correlation < 0.5:
            confidence -= 0.1
        
        # Data quality (more data = more confident)
        if len(data) > 200:
            confidence += 0.1
        elif len(data) < 50:
            confidence -= 0.1
        
        return float(min(1.0, max(0.0, confidence)))
    
    def _generate_regime_description(self, regime: Dict, vix: Optional[float], 
                                    adx: float, momentum: float) -> str:
        """
        Generate human-readable regime description
        """
        regime_type = regime["type"]
        
        descriptions = {
            "strong_bull_trending": "Strong Bull Market - Clear uptrend with momentum",
            "bull_trending": "Bull Market - Uptrending with good momentum",
            "strong_bear_trending": "Strong Bear Market - Clear downtrend with selling pressure",
            "bear_trending": "Bear Market - Downtrending with negative momentum",
            "bull_ranging": "Bullish Consolidation - Sideways but biased higher",
            "bear_ranging": "Bearish Consolidation - Sideways but biased lower",
            "neutral_ranging": "Neutral Market - No clear direction",
            "high_volatility_ranging": "High Volatility - Wide ranges, uncertain direction",
            "low_volatility_ranging": "Low Volatility - Tight ranges, potential breakout",
            "transition": "Market in Transition - Trend emerging",
            "panic": "PANIC MODE - Extreme fear, high selling pressure",
            "extreme_volatility_ranging": "Extreme Volatility - Very wide ranges, high uncertainty"
        }
        
        base_desc = descriptions.get(regime_type, "Mixed market conditions")
        
        # Add details
        details = []
        if vix:
            if vix > 30:
                details.append(f"VIX at {vix:.1f} (panic level)")
            elif vix > 20:
                details.append(f"VIX at {vix:.1f} (elevated fear)")
            elif vix < 15:
                details.append(f"VIX at {vix:.1f} (complacent)")
        
        if adx > 30:
            details.append(f"Strong trend (ADX: {adx:.1f})")
        elif adx < 20:
            details.append(f"Weak trend (ADX: {adx:.1f})")
        
        if abs(momentum) > 5:
            details.append(f"Strong momentum: {momentum:+.1f}%")
        
        if details:
            return f"{base_desc} - {' • '.join(details)}"
        
        return base_desc
    
    async def _get_vix_level(self) -> Optional[float]:
        """
        Get current VIX level (fear index)
        """
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logging.debug(f"Could not fetch VIX: {e}")
        
        return None
    
    async def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get market data for all indices with caching
        """
        market_data = {}
        
        for name, symbol in self.market_symbols.items():
            cache_key = f"market_{name}"
            
            # Check cache
            if cache_key in self.cache:
                cached_time, data = self.cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    market_data[name] = data
                    continue
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")
                
                if not data.empty:
                    self.cache[cache_key] = (datetime.now(), data)
                    market_data[name] = data
                    
            except Exception as e:
                logging.debug(f"Could not fetch {name} data: {e}")
        
        return market_data