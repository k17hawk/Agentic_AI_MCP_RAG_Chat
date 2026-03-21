"""
Market Regime Risk - Adjusts risk parameters based on market regime
"""
from typing import Dict, Any, Optional
from agentic_trading_system.utils.logger import logger as logging

class MarketRegimeRisk:
    """
    Market Regime Risk - Adjusts risk parameters based on detected market regime
    
    Different regimes require different risk approaches:
    - Bull Trending: Can take more risk
    - Bear Trending: Reduce risk, use tighter stops
    - High Volatility: Reduce position sizes
    - Ranging: Mean reversion strategies, moderate risk
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk multipliers by regime
        self.risk_multipliers = config.get("risk_multipliers", {
            "strong_bull_trending": 1.2,
            "bull_trending": 1.1,
            "bull_ranging": 1.0,
            "neutral_ranging": 1.0,
            "bear_ranging": 0.9,
            "bear_trending": 0.8,
            "strong_bear_trending": 0.7,
            "high_volatility": 0.6,
            "extreme_volatility_ranging": 0.5,
            "transition": 0.7,
            "panic": 0.4
        })
        
        # Stop loss multipliers by regime
        self.stop_multipliers = config.get("stop_multipliers", {
            "strong_bull_trending": 2.5,
            "bull_trending": 2.0,
            "bull_ranging": 1.8,
            "neutral_ranging": 1.5,
            "bear_ranging": 1.3,
            "bear_trending": 1.2,
            "strong_bear_trending": 1.0,
            "high_volatility": 1.5,
            "extreme_volatility_ranging": 1.8,
            "transition": 1.2,
            "panic": 1.0
        })
        
        # Max position size by regime (as fraction of portfolio)
        self.max_position_sizes = config.get("max_position_sizes", {
            "strong_bull_trending": 0.25,
            "bull_trending": 0.20,
            "bull_ranging": 0.15,
            "neutral_ranging": 0.12,
            "bear_ranging": 0.10,
            "bear_trending": 0.08,
            "strong_bear_trending": 0.05,
            "high_volatility": 0.06,
            "extreme_volatility_ranging": 0.04,
            "transition": 0.07,
            "panic": 0.03
        })
        
        logging.info(f"✅ MarketRegimeRisk initialized")
    
    def adjust_risk(self, base_risk_params: Dict[str, Any], 
                   regime: str) -> Dict[str, Any]:
        """
        Adjust risk parameters based on market regime
        """
        # Get multipliers for this regime
        risk_mult = self.risk_multipliers.get(regime, 1.0)
        stop_mult = self.stop_multipliers.get(regime, 1.5)
        max_position = self.max_position_sizes.get(regime, 0.10)
        
        adjusted = base_risk_params.copy()
        
        # Adjust position size
        if "position_size" in adjusted:
            adjusted["position_size"] = adjusted["position_size"] * risk_mult
        
        if "position_fraction" in adjusted:
            adjusted["position_fraction"] = min(
                adjusted["position_fraction"] * risk_mult,
                max_position
            )
        
        # Adjust stop loss distance
        if "stop_distance" in adjusted:
            adjusted["stop_distance"] = adjusted["stop_distance"] * stop_mult
        
        if "stop_percent" in adjusted:
            adjusted["stop_percent"] = adjusted["stop_percent"] * stop_mult
        
        # Adjust risk per trade
        if "risk_per_trade" in adjusted:
            adjusted["risk_per_trade"] = adjusted["risk_per_trade"] * risk_mult
        
        # Add regime info
        adjusted["regime"] = regime
        adjusted["risk_multiplier"] = risk_mult
        adjusted["stop_multiplier"] = stop_mult
        adjusted["max_position_allowed"] = max_position
        adjusted["regime_description"] = self._get_regime_description(regime)
        
        return adjusted
    
    def get_max_position(self, regime: str) -> float:
        """Get maximum allowed position size for regime"""
        return self.max_position_sizes.get(regime, 0.10)
    
    def get_risk_appetite(self, regime: str) -> str:
        """Get risk appetite description for regime"""
        risk_levels = {
            "strong_bull_trending": "Aggressive",
            "bull_trending": "Above Normal",
            "bull_ranging": "Normal",
            "neutral_ranging": "Normal",
            "bear_ranging": "Cautious",
            "bear_trending": "Conservative",
            "strong_bear_trending": "Very Conservative",
            "high_volatility": "Defensive",
            "extreme_volatility_ranging": "Highly Defensive",
            "transition": "Cautious",
            "panic": "Risk Off"
        }
        return risk_levels.get(regime, "Normal")
    
    def should_trade(self, regime: str) -> bool:
        """Determine if we should trade in this regime"""
        # Avoid trading in extreme regimes
        no_trade_regimes = ["panic", "extreme_volatility_ranging"]
        return regime not in no_trade_regimes
    
    def _get_regime_description(self, regime: str) -> str:
        """Get human-readable regime description"""
        descriptions = {
            "strong_bull_trending": "Strong bull market - can take more risk",
            "bull_trending": "Bull market - normal risk levels",
            "bull_ranging": "Bullish consolidation - moderate risk",
            "neutral_ranging": "Sideways market - balanced risk",
            "bear_ranging": "Bearish consolidation - reduce risk",
            "bear_trending": "Bear market - be conservative",
            "strong_bear_trending": "Strong bear market - very conservative",
            "high_volatility": "High volatility - reduce position sizes",
            "extreme_volatility_ranging": "Extreme volatility - stay in cash",
            "transition": "Market in transition - be cautious",
            "panic": "Panic mode - capital preservation"
        }
        return descriptions.get(regime, "Unknown regime")