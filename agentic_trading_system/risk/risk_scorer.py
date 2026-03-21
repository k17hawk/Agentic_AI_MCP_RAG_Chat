"""
Risk Scorer - Scores overall risk of a trade or portfolio
"""
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
from agentic_trading_system.utils.logger import logger as logging

class RiskScorer:
    """
    Risk Scorer - Calculates overall risk score for trades and portfolios
    
    Factors for trade risk:
    - Position size risk
    - Stop loss distance
    - Volatility risk
    - Market regime risk
    - Liquidity risk
    - Correlation risk
    - Leverage risk
    - Time horizon risk
    
    Factors for portfolio risk:
    - Concentration risk
    - Correlation risk
    - VaR/ES levels
    - Drawdown risk
    - Sector exposure
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Weights for different risk factors (trade level)
        self.trade_weights = config.get("trade_weights", {
            "position_size": 0.20,
            "stop_distance": 0.15,
            "volatility": 0.15,
            "market_regime": 0.15,
            "liquidity": 0.10,
            "correlation": 0.10,
            "leverage": 0.10,
            "time_horizon": 0.05
        })
        
        # Weights for portfolio risk factors
        self.portfolio_weights = config.get("portfolio_weights", {
            "concentration": 0.25,
            "correlation": 0.20,
            "var": 0.20,
            "drawdown": 0.15,
            "sector_exposure": 0.10,
            "leverage": 0.10
        })
        
        # Risk thresholds
        self.low_risk_threshold = config.get("low_risk_threshold", 0.3)
        self.medium_risk_threshold = config.get("medium_risk_threshold", 0.6)
        self.high_risk_threshold = config.get("high_risk_threshold", 0.8)
        
        # Risk tolerance levels
        self.max_position_size = config.get("max_position_size", 0.25)  # 25% of portfolio
        self.max_leverage = config.get("max_leverage", 2.0)  # 2x leverage
        self.max_sector_exposure = config.get("max_sector_exposure", 0.30)  # 30% per sector
        
        logging.info(f"✅ RiskScorer initialized")
    
    def score_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk score for a trade (0 = lowest risk, 1 = highest risk)
        
        Args:
            trade_params: Dictionary containing:
                - position_fraction: Fraction of portfolio
                - stop_percent: Stop loss percentage
                - volatility: Expected volatility
                - regime: Market regime
                - volume: Current volume
                - avg_volume: Average volume
                - correlation: Correlation with portfolio
                - leverage: Leverage used
                - time_horizon: Expected holding period in days
        """
        # Calculate individual risk components
        position_risk = self._score_position_size(trade_params)
        stop_risk = self._score_stop_distance(trade_params)
        volatility_risk = self._score_volatility(trade_params)
        regime_risk = self._score_market_regime(trade_params)
        liquidity_risk = self._score_liquidity(trade_params)
        correlation_risk = self._score_correlation(trade_params)
        leverage_risk = self._score_leverage(trade_params)
        time_risk = self._score_time_horizon(trade_params)
        
        # Store component scores
        components = {
            "position_size_risk": float(position_risk),
            "stop_distance_risk": float(stop_risk),
            "volatility_risk": float(volatility_risk),
            "market_regime_risk": float(regime_risk),
            "liquidity_risk": float(liquidity_risk),
            "correlation_risk": float(correlation_risk),
            "leverage_risk": float(leverage_risk),
            "time_horizon_risk": float(time_risk)
        }
        
        # Weighted average
        total_risk = (
            position_risk * self.trade_weights["position_size"] +
            stop_risk * self.trade_weights["stop_distance"] +
            volatility_risk * self.trade_weights["volatility"] +
            regime_risk * self.trade_weights["market_regime"] +
            liquidity_risk * self.trade_weights["liquidity"] +
            correlation_risk * self.trade_weights["correlation"] +
            leverage_risk * self.trade_weights["leverage"] +
            time_risk * self.trade_weights["time_horizon"]
        )
        
        # Normalize to 0-1
        total_risk = max(0.0, min(1.0, total_risk))
        
        # Determine risk level
        risk_level = self._get_risk_level(total_risk)
        
        # Determine if trade should be executed
        should_trade = total_risk <= self.medium_risk_threshold
        
        # Generate warnings
        warnings = self._generate_trade_warnings(components, trade_params)
        
        return {
            "risk_score": float(total_risk),
            "risk_level": risk_level,
            "components": components,
            "weights_used": self.trade_weights,
            "should_trade": should_trade,
            "warnings": warnings,
            "max_position_allowed": self.max_position_size,
            "max_leverage_allowed": self.max_leverage,
            "timestamp": datetime.now().isoformat()
        }
    
    def score_portfolio(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk score for an entire portfolio
        
        Args:
            portfolio: Dictionary containing:
                - positions: List of position dictionaries
                - returns: List of historical returns
                - value: Total portfolio value
                - sector_exposures: Dict of sector exposures
                - correlations: Correlation matrix
        """
        positions = portfolio.get("positions", [])
        
        if not positions:
            return {
                "risk_score": 0.0,
                "risk_level": "NO_POSITIONS",
                "message": "No positions in portfolio"
            }
        
        # Calculate component scores
        concentration_risk = self._score_concentration(portfolio)
        correlation_risk = self._score_portfolio_correlation(portfolio)
        var_risk = self._score_var(portfolio)
        drawdown_risk = self._score_drawdown(portfolio)
        sector_risk = self._score_sector_exposure(portfolio)
        leverage_risk = self._score_portfolio_leverage(portfolio)
        
        components = {
            "concentration_risk": float(concentration_risk),
            "correlation_risk": float(correlation_risk),
            "var_risk": float(var_risk),
            "drawdown_risk": float(drawdown_risk),
            "sector_exposure_risk": float(sector_risk),
            "leverage_risk": float(leverage_risk)
        }
        
        # Weighted average
        total_risk = (
            concentration_risk * self.portfolio_weights["concentration"] +
            correlation_risk * self.portfolio_weights["correlation"] +
            var_risk * self.portfolio_weights["var"] +
            drawdown_risk * self.portfolio_weights["drawdown"] +
            sector_risk * self.portfolio_weights["sector_exposure"] +
            leverage_risk * self.portfolio_weights["leverage"]
        )
        
        total_risk = max(0.0, min(1.0, total_risk))
        risk_level = self._get_risk_level(total_risk)
        
        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(components, portfolio)
        
        return {
            "risk_score": float(total_risk),
            "risk_level": risk_level,
            "components": components,
            "weights_used": self.portfolio_weights,
            "recommendations": recommendations,
            "num_positions": len(positions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _score_position_size(self, params: Dict) -> float:
        """Score position size risk (0-1)"""
        position_fraction = params.get("position_fraction", 0)
        
        if position_fraction <= 0:
            return 0.0
        elif position_fraction <= 0.05:
            return 0.1
        elif position_fraction <= 0.10:
            return 0.3
        elif position_fraction <= 0.15:
            return 0.5
        elif position_fraction <= 0.20:
            return 0.7
        elif position_fraction <= 0.25:
            return 0.9
        else:
            return 1.0
    
    def _score_stop_distance(self, params: Dict) -> float:
        """Score stop loss distance risk (0-1)"""
        stop_percent = abs(params.get("stop_percent", 0))
        
        if stop_percent <= 1:
            return 0.1
        elif stop_percent <= 2:
            return 0.3
        elif stop_percent <= 3:
            return 0.5
        elif stop_percent <= 5:
            return 0.7
        elif stop_percent <= 8:
            return 0.8
        elif stop_percent <= 10:
            return 0.9
        else:
            return 1.0
    
    def _score_volatility(self, params: Dict) -> float:
        """Score volatility risk (0-1)"""
        volatility = params.get("volatility", 0.20)  # Annualized
        
        if volatility <= 0.10:
            return 0.1
        elif volatility <= 0.15:
            return 0.2
        elif volatility <= 0.20:
            return 0.3
        elif volatility <= 0.25:
            return 0.4
        elif volatility <= 0.30:
            return 0.5
        elif volatility <= 0.40:
            return 0.6
        elif volatility <= 0.50:
            return 0.7
        elif volatility <= 0.60:
            return 0.8
        elif volatility <= 0.75:
            return 0.9
        else:
            return 1.0
    
    def _score_market_regime(self, params: Dict) -> float:
        """Score market regime risk (0-1)"""
        regime = params.get("regime", "neutral_ranging")
        
        # Risk levels by regime
        regime_risk = {
            "strong_bull_trending": 0.1,
            "bull_trending": 0.2,
            "bull_ranging": 0.3,
            "neutral_ranging": 0.4,
            "low_volatility_ranging": 0.3,
            "bear_ranging": 0.6,
            "bear_trending": 0.7,
            "strong_bear_trending": 0.8,
            "high_volatility": 0.7,
            "high_volatility_ranging": 0.7,
            "extreme_volatility_ranging": 0.9,
            "transition": 0.7,
            "panic": 1.0
        }
        
        return regime_risk.get(regime, 0.5)
    
    def _score_liquidity(self, params: Dict) -> float:
        """Score liquidity risk (0-1)"""
        volume = params.get("volume", 0)
        avg_volume = params.get("avg_volume", 1)
        bid_ask_spread = params.get("bid_ask_spread", 0.01)  # Default 1%
        
        if avg_volume <= 0:
            return 0.5
        
        volume_ratio = volume / avg_volume
        
        # Volume score
        if volume_ratio > 2.0:
            volume_score = 0.1
        elif volume_ratio > 1.5:
            volume_score = 0.2
        elif volume_ratio > 1.0:
            volume_score = 0.3
        elif volume_ratio > 0.5:
            volume_score = 0.5
        elif volume_ratio > 0.2:
            volume_score = 0.7
        else:
            volume_score = 1.0
        
        # Spread score
        if bid_ask_spread <= 0.001:  # 0.1%
            spread_score = 0.1
        elif bid_ask_spread <= 0.002:  # 0.2%
            spread_score = 0.2
        elif bid_ask_spread <= 0.005:  # 0.5%
            spread_score = 0.3
        elif bid_ask_spread <= 0.01:   # 1.0%
            spread_score = 0.5
        elif bid_ask_spread <= 0.02:   # 2.0%
            spread_score = 0.7
        else:
            spread_score = 1.0
        
        # Combine (70% volume, 30% spread)
        return volume_score * 0.7 + spread_score * 0.3
    
    def _score_correlation(self, params: Dict) -> float:
        """Score correlation risk with existing portfolio (0-1)"""
        correlation = params.get("avg_portfolio_correlation", 0.5)
        
        if correlation < 0.2:
            return 0.1
        elif correlation < 0.4:
            return 0.3
        elif correlation < 0.6:
            return 0.5
        elif correlation < 0.8:
            return 0.7
        else:
            return 0.9
    
    def _score_leverage(self, params: Dict) -> float:
        """Score leverage risk (0-1)"""
        leverage = params.get("leverage", 1.0)
        
        if leverage <= 1.0:
            return 0.0
        elif leverage <= 1.2:
            return 0.2
        elif leverage <= 1.5:
            return 0.4
        elif leverage <= 2.0:
            return 0.6
        elif leverage <= 2.5:
            return 0.8
        else:
            return 1.0
    
    def _score_time_horizon(self, params: Dict) -> float:
        """Score time horizon risk (0-1)"""
        days = params.get("time_horizon_days", 5)
        
        if days <= 1:
            return 0.8  # Very short term is risky
        elif days <= 3:
            return 0.6
        elif days <= 7:
            return 0.4
        elif days <= 14:
            return 0.3
        elif days <= 30:
            return 0.2
        else:
            return 0.1  # Longer term less risky
    
    def _score_concentration(self, portfolio: Dict) -> float:
        """Score portfolio concentration risk (0-1)"""
        positions = portfolio.get("positions", [])
        
        if not positions:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        total_value = sum(p.get("value", 0) for p in positions)
        if total_value == 0:
            return 1.0
        
        hhi = sum((p.get("value", 0) / total_value) ** 2 for p in positions)
        
        # Convert HHI to risk score (0.0-1.0)
        if hhi <= 0.1:
            return 0.1
        elif hhi <= 0.2:
            return 0.3
        elif hhi <= 0.3:
            return 0.5
        elif hhi <= 0.4:
            return 0.7
        elif hhi <= 0.5:
            return 0.8
        else:
            return 1.0
    
    def _score_portfolio_correlation(self, portfolio: Dict) -> float:
        """Score portfolio correlation risk (0-1)"""
        correlations = portfolio.get("correlations", [])
        
        if not correlations or len(correlations) < 2:
            return 0.5
        
        # Calculate average absolute correlation
        avg_corr = np.mean([abs(c) for c in correlations])
        
        if avg_corr < 0.3:
            return 0.1
        elif avg_corr < 0.5:
            return 0.3
        elif avg_corr < 0.7:
            return 0.6
        else:
            return 0.9
    
    def _score_var(self, portfolio: Dict) -> float:
        """Score VaR risk (0-1)"""
        var_95 = portfolio.get("var_95", 0.02)  # 2% daily VaR as baseline
        
        if var_95 <= 0.01:
            return 0.1
        elif var_95 <= 0.02:
            return 0.3
        elif var_95 <= 0.03:
            return 0.5
        elif var_95 <= 0.04:
            return 0.6
        elif var_95 <= 0.05:
            return 0.7
        elif var_95 <= 0.07:
            return 0.8
        elif var_95 <= 0.10:
            return 0.9
        else:
            return 1.0
    
    def _score_drawdown(self, portfolio: Dict) -> float:
        """Score drawdown risk (0-1)"""
        max_drawdown = portfolio.get("max_drawdown", 0.10)  # 10% as baseline
        
        if max_drawdown <= 0.05:
            return 0.1
        elif max_drawdown <= 0.10:
            return 0.3
        elif max_drawdown <= 0.15:
            return 0.5
        elif max_drawdown <= 0.20:
            return 0.6
        elif max_drawdown <= 0.25:
            return 0.7
        elif max_drawdown <= 0.30:
            return 0.8
        elif max_drawdown <= 0.40:
            return 0.9
        else:
            return 1.0
    
    def _score_sector_exposure(self, portfolio: Dict) -> float:
        """Score sector exposure risk (0-1)"""
        sector_exposures = portfolio.get("sector_exposures", {})
        
        if not sector_exposures:
            return 0.5
        
        max_exposure = max(sector_exposures.values()) if sector_exposures else 0
        
        if max_exposure <= 0.20:
            return 0.1
        elif max_exposure <= 0.30:
            return 0.3
        elif max_exposure <= 0.40:
            return 0.5
        elif max_exposure <= 0.50:
            return 0.7
        elif max_exposure <= 0.60:
            return 0.8
        else:
            return 1.0
    
    def _score_portfolio_leverage(self, portfolio: Dict) -> float:
        """Score portfolio leverage risk (0-1)"""
        leverage = portfolio.get("leverage", 1.0)
        
        if leverage <= 1.0:
            return 0.0
        elif leverage <= 1.2:
            return 0.3
        elif leverage <= 1.5:
            return 0.5
        elif leverage <= 2.0:
            return 0.7
        elif leverage <= 2.5:
            return 0.8
        else:
            return 1.0
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level description"""
        if score <= self.low_risk_threshold:
            return "VERY_LOW"
        elif score <= self.medium_risk_threshold:
            return "LOW"
        elif score <= self.high_risk_threshold:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_trade_warnings(self, components: Dict, params: Dict) -> List[str]:
        """Generate warnings for high-risk components"""
        warnings = []
        
        if components["position_size_risk"] > 0.7:
            warnings.append(f"Position size is large ({params.get('position_fraction', 0)*100:.1f}% of portfolio)")
        
        if components["stop_distance_risk"] > 0.7:
            warnings.append(f"Stop loss distance is wide ({params.get('stop_percent', 0):.1f}%)")
        
        if components["volatility_risk"] > 0.7:
            warnings.append(f"High volatility stock ({params.get('volatility', 0)*100:.1f}% annualized)")
        
        if components["liquidity_risk"] > 0.7:
            warnings.append("Low liquidity - may have slippage")
        
        if components["leverage_risk"] > 0.5:
            warnings.append(f"Using leverage ({params.get('leverage', 1.0):.1f}x)")
        
        return warnings
    
    def _generate_portfolio_recommendations(self, components: Dict, portfolio: Dict) -> List[str]:
        """Generate recommendations for portfolio optimization"""
        recommendations = []
        
        if components["concentration_risk"] > 0.7:
            recommendations.append("Consider diversifying - portfolio too concentrated")
        
        if components["correlation_risk"] > 0.7:
            recommendations.append("Assets highly correlated - add uncorrelated assets")
        
        if components["var_risk"] > 0.7:
            recommendations.append("VaR too high - reduce position sizes")
        
        if components["drawdown_risk"] > 0.7:
            recommendations.append("Historical drawdown too large - add stops")
        
        if components["sector_exposure_risk"] > 0.7:
            recommendations.append("Sector concentration too high - diversify sectors")
        
        return recommendations
    
    def get_risk_appetite(self, score: float) -> str:
        """Get risk appetite description based on score"""
        if score <= 0.2:
            return "VERY_CONSERVATIVE"
        elif score <= 0.4:
            return "CONSERVATIVE"
        elif score <= 0.6:
            return "MODERATE"
        elif score <= 0.8:
            return "AGGRESSIVE"
        else:
            return "VERY_AGGRESSIVE"
    
    def compare_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        Compare multiple trades and rank by risk
        """
        scored_trades = []
        
        for trade in trades:
            score_result = self.score_trade(trade)
            scored_trades.append({
                **trade,
                "risk_score": score_result["risk_score"],
                "risk_level": score_result["risk_level"],
                "should_trade": score_result["should_trade"]
            })
        
        # Sort by risk score (lowest risk first)
        scored_trades.sort(key=lambda x: x["risk_score"])
        
        return scored_trades