"""
Recommendation Generator - Creates actionable portfolio recommendations
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logger import logger as  logging

class RecommendationGenerator:
    """
    Recommendation Generator - Creates actionable portfolio recommendations
    
    Generates:
    - Buy/Sell recommendations
    - Rebalancing orders
    - Risk warnings
    - Performance insights
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Recommendation thresholds
        self.buy_threshold = config.get("buy_threshold", 0.02)  # 2% underweight
        self.sell_threshold = config.get("sell_threshold", 0.02)  # 2% overweight
        
        # Priority levels
        self.priority_levels = {
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1
        }
        
        logging.info(f"✅ RecommendationGenerator initialized")
    
    def generate_recommendations(self, allocation_plan: Dict[str, Any],
                                 rebalance_signal: Dict[str, Any],
                                 market_outlook: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio recommendations
        """
        recommendations = {
            "trades": [],
            "rebalancing": None,
            "risk_alerts": [],
            "performance_insights": [],
            "priority": "LOW",
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate trade recommendations
        if "trades" in allocation_plan:
            recommendations["trades"] = self._generate_trade_recommendations(
                allocation_plan["trades"]
            )
        
        # Generate rebalancing recommendation
        if rebalance_signal["needs_rebalance"]:
            recommendations["rebalancing"] = self._generate_rebalancing_recommendation(
                rebalance_signal
            )
        
        # Generate risk alerts
        recommendations["risk_alerts"] = self._generate_risk_alerts(
            allocation_plan, market_outlook
        )
        
        # Generate performance insights
        recommendations["performance_insights"] = self._generate_performance_insights(
            allocation_plan
        )
        
        # Determine overall priority
        recommendations["priority"] = self._determine_priority(recommendations)
        
        # Generate summary
        recommendations["summary"] = self._generate_summary(recommendations)
        
        return recommendations
    
    def _generate_trade_recommendations(self, trades: List[Dict]) -> List[Dict]:
        """Generate detailed trade recommendations"""
        recommendations = []
        
        for trade in trades:
            action = trade["action"]
            symbol = trade["symbol"]
            value = trade["value"]
            
            # Determine priority based on trade size
            if trade.get("difference", 0) > 0.05:
                priority = "HIGH"
            elif trade.get("difference", 0) > 0.02:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            recommendation = {
                "type": "trade",
                "symbol": symbol,
                "action": action,
                "value": value,
                "shares": self._calculate_shares(value, symbol),
                "current_weight": trade.get("current_weight", 0),
                "target_weight": trade.get("target_weight", 0),
                "difference": trade.get("difference", 0),
                "priority": priority,
                "reason": self._get_trade_reason(action, trade),
                "urgency": "IMMEDIATE" if priority == "HIGH" else "SCHEDULED"
            }
            
            # Add execution instructions
            recommendation["execution"] = self._get_execution_instructions(recommendation)
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_rebalancing_recommendation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rebalancing recommendation"""
        urgency = signal["urgency"]
        
        return {
            "type": "rebalancing",
            "needed": True,
            "urgency": urgency,
            "reason": signal["reasons"][0] if signal["reasons"] else "Portfolio drift detected",
            "max_drift": signal["max_drift"],
            "avg_drift": signal["avg_drift"],
            "action": "REBALANCE_NOW" if urgency == "HIGH" else "SCHEDULE_REBALANCE",
            "priority": urgency
        }
    
    def _generate_risk_alerts(self, allocation_plan: Dict[str, Any],
                             market_outlook: Dict[str, Any] = None) -> List[Dict]:
        """Generate risk alerts"""
        alerts = []
        
        # Check concentration risk
        if "weights" in allocation_plan:
            max_weight = max(allocation_plan["weights"].values())
            if max_weight > 0.25:
                alerts.append({
                    "type": "concentration_risk",
                    "severity": "HIGH",
                    "message": f"High concentration in single asset ({max_weight:.1%})",
                    "action": "Consider diversifying"
                })
        
        # Check sector concentration
        # This would need sector mapping
        
        # Check market outlook if available
        if market_outlook and market_outlook.get("risk_level") == "HIGH":
            alerts.append({
                "type": "market_risk",
                "severity": market_outlook["risk_level"],
                "message": market_outlook.get("warning", "Elevated market risk"),
                "action": "Consider reducing exposure"
            })
        
        return alerts
    
    def _generate_performance_insights(self, allocation_plan: Dict[str, Any]) -> List[Dict]:
        """Generate performance insights"""
        insights = []
        
        # Expected return insight
        if "expected_return" in allocation_plan:
            exp_return = allocation_plan["expected_return"]
            if exp_return > 0.15:
                insights.append({
                    "type": "return_outlook",
                    "sentiment": "positive",
                    "message": f"Strong expected return of {exp_return:.1%}"
                })
            elif exp_return < 0.05:
                insights.append({
                    "type": "return_outlook",
                    "sentiment": "neutral",
                    "message": f"Modest expected return of {exp_return:.1%}"
                })
        
        # Volatility insight
        if "volatility" in allocation_plan:
            vol = allocation_plan["volatility"]
            if vol > 0.25:
                insights.append({
                    "type": "risk_outlook",
                    "sentiment": "warning",
                    "message": f"High portfolio volatility ({vol:.1%})"
                })
        
        # Sharpe ratio insight
        if "sharpe_ratio" in allocation_plan:
            sharpe = allocation_plan["sharpe_ratio"]
            if sharpe > 1.0:
                insights.append({
                    "type": "efficiency",
                    "sentiment": "positive",
                    "message": f"Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})"
                })
        
        return insights
    
    def _determine_priority(self, recommendations: Dict) -> str:
        """Determine overall priority of recommendations"""
        priorities = []
        
        # Check trade priorities
        for trade in recommendations.get("trades", []):
            priorities.append(self.priority_levels.get(trade["priority"], 1))
        
        # Check rebalancing priority
        if recommendations.get("rebalancing"):
            priorities.append(self.priority_levels.get(
                recommendations["rebalancing"]["priority"], 2
            ))
        
        # Check risk alerts
        for alert in recommendations.get("risk_alerts", []):
            if alert.get("severity") == "HIGH":
                priorities.append(3)
        
        if not priorities:
            return "LOW"
        
        avg_priority = sum(priorities) / len(priorities)
        
        if avg_priority >= 2.5:
            return "HIGH"
        elif avg_priority >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_summary(self, recommendations: Dict) -> str:
        """Generate human-readable summary"""
        parts = []
        
        trade_count = len(recommendations.get("trades", []))
        if trade_count > 0:
            buys = sum(1 for t in recommendations["trades"] if t["action"] == "BUY")
            sells = sum(1 for t in recommendations["trades"] if t["action"] == "SELL")
            parts.append(f"{buys} buy orders, {sells} sell orders")
        
        if recommendations.get("rebalancing"):
            parts.append(f"Rebalancing needed ({recommendations['rebalancing']['urgency']} urgency)")
        
        alert_count = len(recommendations.get("risk_alerts", []))
        if alert_count > 0:
            parts.append(f"{alert_count} risk alerts")
        
        insight_count = len(recommendations.get("performance_insights", []))
        if insight_count > 0:
            parts.append(f"{insight_count} insights")
        
        if parts:
            return " | ".join(parts)
        else:
            return "No actionable recommendations at this time"
    
    def _calculate_shares(self, value: float, symbol: str) -> int:
        """Calculate number of shares to trade"""
        # This would need current price
        # Placeholder
        return int(value / 100)  # Assume $100/share
    
    def _get_trade_reason(self, action: str, trade: Dict) -> str:
        """Get reason for trade recommendation"""
        diff = abs(trade.get("difference", 0))
        
        if action == "BUY":
            return f"Underweight by {diff:.1%} relative to target"
        else:
            return f"Overweight by {diff:.1%} relative to target"
    
    def _get_execution_instructions(self, recommendation: Dict) -> Dict[str, Any]:
        """Get execution instructions for trade"""
        instructions = {
            "order_type": "MARKET",
            "time_in_force": "DAY",
            "notes": []
        }
        
        if recommendation["priority"] == "HIGH":
            instructions["order_type"] = "MARKET"
            instructions["time_in_force"] = "DAY"
            instructions["notes"].append("Execute immediately")
        elif recommendation["priority"] == "MEDIUM":
            instructions["order_type"] = "LIMIT"
            instructions["time_in_force"] = "GOOD_TILL_CANCEL"
            instructions["notes"].append("Use limit orders for better price")
        else:
            instructions["order_type"] = "LIMIT"
            instructions["time_in_force"] = "GOOD_TILL_CANCEL"
            instructions["notes"].append("Low priority - can wait for favorable prices")
        
        return instructions