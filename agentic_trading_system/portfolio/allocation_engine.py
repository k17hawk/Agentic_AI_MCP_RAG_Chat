"""
Allocation Engine - Core engine for portfolio allocation
"""
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
from agentic_trading_system.utils.logger import logger as logging

class AllocationEngine:
    """
    Allocation Engine - Core engine for portfolio allocation
    
    Combines multiple optimization methods and handles:
    - Multi-asset allocation
    - Tax-efficient placement
    - Cash flow management
    - Rebalancing
    - Constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Import optimization models
        from portfolio.efficient_frontier import EfficientFrontier
        from portfolio.black_litterman import BlackLitterman
        from portfolio.risk_parity import RiskParity
        from portfolio.hierarchical_risk_parity import HierarchicalRiskParity
        
        self.ef = EfficientFrontier(config.get("ef_config", {}))
        self.bl = BlackLitterman(config.get("bl_config", {}))
        self.rp = RiskParity(config.get("rp_config", {}))
        self.hrp = HierarchicalRiskParity(config.get("hrp_config", {}))
        
        # Allocation constraints
        self.asset_class_limits = config.get("asset_class_limits", {})
        self.sector_limits = config.get("sector_limits", {})
        self.max_turnover = config.get("max_turnover", 0.20)  # 20% max turnover
        
        # Tax efficiency
        self.tax_aware = config.get("tax_aware", True)
        self.tax_rates = config.get("tax_rates", {
            "short_term": 0.35,
            "long_term": 0.15
        })
        
        logging.info(f"✅ AllocationEngine initialized")
    
    def allocate(self, current_portfolio: Dict[str, Any],
                target_allocation: Dict[str, float],
                prices: pd.DataFrame,
                cash: float = 0,
                options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate allocation plan to move from current to target
        """
        options = options or {}
        
        # Get current positions
        current_positions = current_portfolio.get("positions", [])
        current_weights = self._calculate_current_weights(current_positions, prices)
        
        # Calculate trades needed
        trades = self._calculate_trades(current_weights, target_allocation, current_portfolio["total_value"])
        
        # Check turnover
        turnover = self._calculate_turnover(trades, current_portfolio["total_value"])
        if turnover > self.max_turnover and not options.get("force_rebalance", False):
            # Scale back trades to meet turnover limit
            trades = self._scale_trades(trades, self.max_turnover / turnover)
        
        # Apply tax optimization if enabled
        if self.tax_aware:
            trades = self._optimize_taxes(trades, current_positions)
        
        # Calculate cash needed
        net_cash = self._calculate_net_cash(trades)
        
        # Adjust for available cash
        if net_cash > cash:
            # Need to raise cash - scale down buys
            trades = self._adjust_for_cash(trades, cash, net_cash)
        
        return {
            "trades": trades,
            "target_allocation": target_allocation,
            "current_allocation": current_weights,
            "turnover": turnover,
            "net_cash_required": net_cash,
            "timestamp": datetime.now().isoformat()
        }
    
    def optimize(self, prices: pd.DataFrame, method: str = "max_sharpe",
                constraints: Dict = None, views: List = None) -> Dict[str, Any]:
        """
        Find optimal portfolio allocation using specified method
        """
        returns = prices.pct_change().dropna()
        
        if method == "max_sharpe":
            result = self.ef.optimize_max_sharpe(returns)
        elif method == "min_volatility":
            result = self.ef.optimize_min_volatility(returns)
        elif method == "risk_parity":
            result = self.rp.optimize(returns)
        elif method == "hierarchical_risk_parity":
            result = self.hrp.optimize(returns)
        elif method == "black_litterman":
            # Need market caps and views for Black-Litterman
            market_caps = constraints.get("market_caps", {})
            views = views or []
            result = self.bl.optimize(prices, market_caps, views)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Apply constraints
        if constraints:
            result["weights"] = self._apply_constraints(result["weights"], constraints)
        
        return result
    
    def _calculate_current_weights(self, positions: List[Dict], 
                                  prices: pd.DataFrame) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        weights = {}
        total_value = 0
        
        for position in positions:
            symbol = position["symbol"]
            shares = position["shares"]
            
            if symbol in prices.columns:
                current_price = prices[symbol].iloc[-1]
                value = shares * current_price
                weights[symbol] = value
                total_value += value
        
        # Normalize
        if total_value > 0:
            for symbol in weights:
                weights[symbol] /= total_value
        
        return weights
    
    def _calculate_trades(self, current: Dict[str, float], 
                         target: Dict[str, float],
                         portfolio_value: float) -> List[Dict]:
        """Calculate trades needed to achieve target allocation"""
        trades = []
        
        all_symbols = set(current.keys()) | set(target.keys())
        
        for symbol in all_symbols:
            current_weight = current.get(symbol, 0)
            target_weight = target.get(symbol, 0)
            
            if abs(target_weight - current_weight) < 0.001:
                continue
            
            # Calculate trade value
            current_value = current_weight * portfolio_value
            target_value = target_weight * portfolio_value
            trade_value = target_value - current_value
            
            trades.append({
                "symbol": symbol,
                "action": "BUY" if trade_value > 0 else "SELL",
                "value": abs(trade_value),
                "current_weight": current_weight,
                "target_weight": target_weight,
                "difference": target_weight - current_weight
            })
        
        # Sort by absolute difference
        trades.sort(key=lambda x: abs(x["difference"]), reverse=True)
        
        return trades
    
    def _calculate_turnover(self, trades: List[Dict], portfolio_value: float) -> float:
        """Calculate portfolio turnover"""
        total_trade_value = sum(t["value"] for t in trades)
        return total_trade_value / (2 * portfolio_value) if portfolio_value > 0 else 0
    
    def _scale_trades(self, trades: List[Dict], scale_factor: float) -> List[Dict]:
        """Scale trades to meet turnover limit"""
        scaled_trades = []
        for trade in trades:
            scaled_trade = trade.copy()
            scaled_trade["value"] *= scale_factor
            scaled_trade["scaled"] = True
            scaled_trades.append(scaled_trade)
        return scaled_trades
    
    def _optimize_taxes(self, trades: List[Dict], positions: List[Dict]) -> List[Dict]:
        """
        Optimize trades for tax efficiency
        - Harvest losses
        - Prefer long-term gains
        """
        # Group positions by holding period
        long_term_positions = {}
        short_term_positions = {}
        
        for position in positions:
            symbol = position["symbol"]
            entry_time = datetime.fromisoformat(position["entry_time"])
            holding_days = (datetime.now() - entry_time).days
            
            if holding_days > 365:
                long_term_positions[symbol] = position
            else:
                short_term_positions[symbol] = position
        
        # Prioritize selling short-term losers first
        optimized_trades = []
        for trade in trades:
            if trade["action"] == "SELL":
                symbol = trade["symbol"]
                
                # Check if we have this position
                if symbol in short_term_positions:
                    # Check if it's a loser
                    position = short_term_positions[symbol]
                    if position["current_price"] < position["entry_price"]:
                        trade["tax_efficient"] = True
                        trade["tax_rate"] = self.tax_rates["short_term"]
                
                elif symbol in long_term_positions:
                    position = long_term_positions[symbol]
                    if position["current_price"] < position["entry_price"]:
                        trade["tax_efficient"] = True
                        trade["tax_rate"] = self.tax_rates["long_term"]
            
            optimized_trades.append(trade)
        
        return optimized_trades
    
    def _calculate_net_cash(self, trades: List[Dict]) -> float:
        """Calculate net cash required"""
        net_cash = 0
        for trade in trades:
            if trade["action"] == "BUY":
                net_cash += trade["value"]
            else:
                net_cash -= trade["value"]
        return net_cash
    
    def _adjust_for_cash(self, trades: List[Dict], available_cash: float, 
                         required_cash: float) -> List[Dict]:
        """Adjust trades based on available cash"""
        if required_cash <= available_cash:
            return trades
        
        # Need to reduce buys
        cash_shortfall = required_cash - available_cash
        reduction_factor = 1 - (cash_shortfall / required_cash)
        
        adjusted_trades = []
        for trade in trades:
            if trade["action"] == "BUY":
                trade["value"] *= reduction_factor
                trade["cash_adjusted"] = True
            adjusted_trades.append(trade)
        
        return adjusted_trades
    
    def _apply_constraints(self, weights: Dict[str, float], 
                          constraints: Dict) -> Dict[str, float]:
        """Apply allocation constraints"""
        # Apply asset class limits
        if "asset_class_limits" in constraints:
            for asset_class, limit in constraints["asset_class_limits"].items():
                # Sum weights in this asset class
                class_weight = sum(
                    w for a, w in weights.items() 
                    if constraints.get("asset_class", {}).get(a) == asset_class
                )
                
                if class_weight > limit:
                    # Scale down this asset class
                    scale = limit / class_weight
                    for a in weights:
                        if constraints.get("asset_class", {}).get(a) == asset_class:
                            weights[a] *= scale
        
        # Apply sector limits
        if "sector_limits" in constraints:
            for sector, limit in constraints["sector_limits"].items():
                sector_weight = sum(
                    w for a, w in weights.items() 
                    if constraints.get("sectors", {}).get(a) == sector
                )
                
                if sector_weight > limit:
                    scale = limit / sector_weight
                    for a in weights:
                        if constraints.get("sectors", {}).get(a) == sector:
                            weights[a] *= scale
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights