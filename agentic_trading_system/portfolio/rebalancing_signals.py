"""
Rebalancing Signals - Detects when portfolio needs rebalancing
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta
from utils.logger import logger as logging

class RebalancingSignals:
    """
    Rebalancing Signals - Determines when to rebalance portfolio
    
    Signals based on:
    - Drift from target allocation
    - Time-based rebalancing
    - Volatility-based thresholds
    - Cash flow events
    - Tax optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Rebalancing thresholds
        self.absolute_threshold = config.get("absolute_threshold", 0.05)  # 5% absolute drift
        self.relative_threshold = config.get("relative_threshold", 0.20)  # 20% relative drift
        
        # Time-based rebalancing
        self.calendar_days = config.get("calendar_days", 90)  # Rebalance quarterly
        self.last_rebalance = None
        
        # Volatility-based bands
        self.use_volatility_bands = config.get("use_volatility_bands", True)
        self.volatility_multiplier = config.get("volatility_multiplier", 2.0)
        
        # Tax optimization
        self.tax_harvest_threshold = config.get("tax_harvest_threshold", -0.10)  # -10% loss
        
        logging.info(f"✅ RebalancingSignals initialized")
    
    def check_rebalance(self, current_allocation: Dict[str, float],
                       target_allocation: Dict[str, float],
                       portfolio_value: float,
                       volatility: float = None,
                       cash_flow: float = 0) -> Dict[str, Any]:
        """
        Check if portfolio needs rebalancing
        """
        signals = []
        reasons = []
        
        # Check absolute drift
        abs_drift_result = self._check_absolute_drift(current_allocation, target_allocation)
        if abs_drift_result["needs_rebalance"]:
            signals.append("absolute_drift")
            reasons.append(abs_drift_result["reason"])
        
        # Check relative drift
        rel_drift_result = self._check_relative_drift(current_allocation, target_allocation)
        if rel_drift_result["needs_rebalance"]:
            signals.append("relative_drift")
            reasons.append(rel_drift_result["reason"])
        
        # Check time-based rebalancing
        time_result = self._check_time_based()
        if time_result["needs_rebalance"]:
            signals.append("time_based")
            reasons.append(time_result["reason"])
        
        # Check volatility-based bands
        if volatility and self.use_volatility_bands:
            vol_result = self._check_volatility_bands(volatility)
            if vol_result["needs_rebalance"]:
                signals.append("volatility_bands")
                reasons.append(vol_result["reason"])
        
        # Check cash flow
        if abs(cash_flow) > 0:
            cash_result = self._check_cash_flow(cash_flow, portfolio_value)
            if cash_result["needs_rebalance"]:
                signals.append("cash_flow")
                reasons.append(cash_result["reason"])
        
        # Check tax harvesting opportunities
        tax_result = self._check_tax_harvesting(current_allocation)
        if tax_result["has_opportunity"]:
            signals.append("tax_harvest")
            reasons.append(tax_result["reason"])
        
        # Determine urgency
        urgency = self._determine_urgency(signals)
        
        # Calculate drift metrics
        max_drift, avg_drift = self._calculate_drift_metrics(current_allocation, target_allocation)
        
        return {
            "needs_rebalance": len(signals) > 0,
            "urgency": urgency,
            "signals": signals,
            "reasons": reasons,
            "max_drift": max_drift,
            "avg_drift": avg_drift,
            "thresholds": {
                "absolute": self.absolute_threshold,
                "relative": self.relative_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_absolute_drift(self, current: Dict[str, float],
                             target: Dict[str, float]) -> Dict[str, Any]:
        """Check absolute drift from target"""
        max_drift = 0
        drifting_assets = []
        
        all_symbols = set(current.keys()) | set(target.keys())
        
        for symbol in all_symbols:
            current_weight = current.get(symbol, 0)
            target_weight = target.get(symbol, 0)
            
            drift = abs(current_weight - target_weight)
            
            if drift > self.absolute_threshold:
                drifting_assets.append({
                    "symbol": symbol,
                    "current": current_weight,
                    "target": target_weight,
                    "drift": drift
                })
                
                if drift > max_drift:
                    max_drift = drift
        
        return {
            "needs_rebalance": len(drifting_assets) > 0,
            "max_drift": max_drift,
            "drifting_assets": drifting_assets,
            "reason": f"{len(drifting_assets)} assets exceeded absolute drift threshold"
        }
    
    def _check_relative_drift(self, current: Dict[str, float],
                             target: Dict[str, float]) -> Dict[str, Any]:
        """Check relative drift from target"""
        max_rel_drift = 0
        drifting_assets = []
        
        for symbol, target_weight in target.items():
            if target_weight == 0:
                continue
            
            current_weight = current.get(symbol, 0)
            rel_drift = abs(current_weight - target_weight) / target_weight
            
            if rel_drift > self.relative_threshold:
                drifting_assets.append({
                    "symbol": symbol,
                    "current": current_weight,
                    "target": target_weight,
                    "rel_drift": rel_drift
                })
                
                if rel_drift > max_rel_drift:
                    max_rel_drift = rel_drift
        
        return {
            "needs_rebalance": len(drifting_assets) > 0,
            "max_rel_drift": max_rel_drift,
            "drifting_assets": drifting_assets,
            "reason": f"{len(drifting_assets)} assets exceeded relative drift threshold"
        }
    
    def _check_time_based(self) -> Dict[str, Any]:
        """Check if it's time to rebalance based on calendar"""
        now = datetime.now()
        
        if self.last_rebalance is None:
            self.last_rebalance = now
            return {
                "needs_rebalance": False,
                "reason": "First run - no rebalance needed"
            }
        
        days_since = (now - self.last_rebalance).days
        
        if days_since >= self.calendar_days:
            self.last_rebalance = now
            return {
                "needs_rebalance": True,
                "days_since": days_since,
                "reason": f"{days_since} days since last rebalance"
            }
        
        return {
            "needs_rebalance": False,
            "days_since": days_since,
            "reason": f"Only {days_since} days since last rebalance"
        }
    
    def _check_volatility_bands(self, volatility: float) -> Dict[str, Any]:
        """Check if volatility triggers rebalancing"""
        # Wider bands for higher volatility
        dynamic_threshold = self.absolute_threshold * (1 + self.volatility_multiplier * volatility)
        
        return {
            "needs_rebalance": False,  # This is just informational
            "dynamic_threshold": dynamic_threshold,
            "reason": f"Volatility-based threshold: {dynamic_threshold:.2%}"
        }
    
    def _check_cash_flow(self, cash_flow: float, portfolio_value: float) -> Dict[str, Any]:
        """Check if cash flow triggers rebalancing"""
        cash_flow_pct = abs(cash_flow) / portfolio_value if portfolio_value > 0 else 0
        
        # Significant cash flow (>5% of portfolio)
        if cash_flow_pct > 0.05:
            return {
                "needs_rebalance": True,
                "cash_flow_pct": cash_flow_pct,
                "reason": f"Cash flow of {cash_flow_pct:.1%} of portfolio"
            }
        
        return {
            "needs_rebalance": False,
            "cash_flow_pct": cash_flow_pct
        }
    
    def _check_tax_harvesting(self, current_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Check for tax loss harvesting opportunities"""
        # This would need position cost basis data
        # Simplified version
        return {
            "has_opportunity": False,
            "reason": "No tax harvesting opportunities detected"
        }
    
    def _determine_urgency(self, signals: List[str]) -> str:
        """Determine rebalancing urgency"""
        if len(signals) >= 3:
            return "HIGH"
        elif len(signals) >= 2:
            return "MEDIUM"
        elif len(signals) >= 1:
            return "LOW"
        else:
            return "NONE"
    
    def _calculate_drift_metrics(self, current: Dict[str, float],
                                target: Dict[str, float]) -> tuple:
        """Calculate max and average drift"""
        drifts = []
        
        all_symbols = set(current.keys()) | set(target.keys())
        
        for symbol in all_symbols:
            current_weight = current.get(symbol, 0)
            target_weight = target.get(symbol, 0)
            drifts.append(abs(current_weight - target_weight))
        
        if drifts:
            return max(drifts), sum(drifts) / len(drifts)
        
        return 0, 0