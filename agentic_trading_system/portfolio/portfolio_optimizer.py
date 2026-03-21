"""
Portfolio Optimizer - Main orchestrator for portfolio optimization
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import all portfolio components
from agentic_trading_system.portfolio.efficient_frontier import EfficientFrontier
from agentic_trading_system.portfolio.black_litterman import BlackLitterman
from agentic_trading_system.portfolio.risk_parity import RiskParity
from agentic_trading_system.portfolio.hierarchical_risk_parity import HierarchicalRiskParity
from agentic_trading_system.portfolio.allocation_engine import AllocationEngine
from agentic_trading_system.portfolio.rebalancing_signals import RebalancingSignals
from agentic_trading_system.portfolio.recommendation_generator import RecommendationGenerator

class PortfolioOptimizer(BaseAgent):
    """
    Portfolio Optimizer - Main orchestrator for portfolio management
    
    Responsibilities:
    - Optimize portfolio allocation
    - Monitor for rebalancing needs
    - Generate recommendations
    - Execute rebalancing plans
    - Track portfolio performance
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Portfolio optimization and management",
            config=config
        )
        
        # Initialize components
        self.ef = EfficientFrontier(config.get("ef_config", {}))
        self.bl = BlackLitterman(config.get("bl_config", {}))
        self.rp = RiskParity(config.get("rp_config", {}))
        self.hrp = HierarchicalRiskParity(config.get("hrp_config", {}))
        self.allocation_engine = AllocationEngine(config.get("allocation_config", {}))
        self.rebalancing = RebalancingSignals(config.get("rebalancing_config", {}))
        self.recommendation_generator = RecommendationGenerator(config.get("recommendation_config", {}))
        
        # Portfolio state
        self.portfolio = {
            "total_value": config.get("initial_value", 100000),
            "cash": config.get("initial_cash", 100000),
            "positions": [],
            "target_allocation": {},
            "last_rebalance": None,
            "performance_history": []
        }
        
        # Current market regime (from risk manager)
        self.current_regime = "neutral_ranging"
        
        # Constraints
        self.constraints = config.get("constraints", {
            "max_position": 0.25,
            "min_position": 0.01,
            "max_sector": 0.30,
            "max_turnover": 0.20
        })
        
        logging.info(f"✅ PortfolioOptimizer initialized with value: ${self.portfolio['total_value']:,.2f}")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process portfolio optimization requests
        """
        if message.message_type == "optimize_portfolio":
            # Optimize portfolio allocation
            prices = message.content.get("prices")
            method = message.content.get("method", "max_sharpe")
            views = message.content.get("views", [])
            
            result = await self.optimize(prices, method, views)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="optimization_result",
                content=result
            )
        
        elif message.message_type == "check_rebalance":
            # Check if rebalancing is needed
            result = await self.check_rebalance()
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="rebalance_check",
                content=result
            )
        
        elif message.message_type == "generate_recommendations":
            # Generate portfolio recommendations
            market_outlook = message.content.get("market_outlook")
            result = await self.generate_recommendations(market_outlook)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="recommendations",
                content=result
            )
        
        elif message.message_type == "update_portfolio":
            # Update portfolio with new positions/prices
            updates = message.content
            await self._update_portfolio(updates)
            return None
        
        elif message.message_type == "update_regime":
            # Update market regime
            self.current_regime = message.content.get("regime", self.current_regime)
            return None
        
        elif message.message_type == "get_portfolio_status":
            # Get current portfolio status
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="portfolio_status",
                content=self.get_status()
            )
        
        return None
    
    async def optimize(self, prices: pd.DataFrame, method: str = "max_sharpe",
                      views: List[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio allocation
        """
        logging.info(f"📊 Optimizing portfolio using {method}")
        
        # Prepare constraints
        constraints = {
            "max_weight": self.constraints["max_position"],
            "min_weight": self.constraints["min_position"],
            "market_caps": self._get_market_caps(prices.columns)
        }
        
        # Run optimization
        if method == "max_sharpe":
            result = self.ef.optimize_max_sharpe(prices.pct_change().dropna())
        elif method == "min_volatility":
            result = self.ef.optimize_min_volatility(prices.pct_change().dropna())
        elif method == "risk_parity":
            result = self.rp.optimize(prices.pct_change().dropna())
        elif method == "hierarchical_risk_parity":
            result = self.hrp.optimize(prices.pct_change().dropna())
        elif method == "black_litterman":
            if not views:
                views = []
            result = self.bl.optimize(prices, constraints["market_caps"], views)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store target allocation
        self.portfolio["target_allocation"] = result["weights"]
        self.portfolio["last_optimization"] = datetime.now().isoformat()
        
        # Add metadata
        result["method"] = method
        result["timestamp"] = datetime.now().isoformat()
        result["constraints"] = constraints
        
        logging.info(f"✅ Optimization complete: expected return={result['expected_return']:.2%}, "
                    f"volatility={result['volatility']:.2%}")
        
        return result
    
    async def check_rebalance(self) -> Dict[str, Any]:
        """
        Check if portfolio needs rebalancing
        """
        if not self.portfolio["target_allocation"]:
            return {
                "needs_rebalance": False,
                "reason": "No target allocation set"
            }
        
        # Calculate current weights
        current_weights = self._calculate_current_weights()
        
        # Get market volatility
        volatility = self._get_portfolio_volatility()
        
        # Check rebalancing signals
        signal = self.rebalancing.check_rebalance(
            current_weights,
            self.portfolio["target_allocation"],
            self.portfolio["total_value"],
            volatility,
            cash_flow=0  # Would come from cash flow events
        )
        
        return signal
    
    async def generate_recommendations(self, market_outlook: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate portfolio recommendations
        """
        # Check if rebalancing needed
        rebalance_signal = await self.check_rebalance()

        # Generate allocation plan if needed
        allocation_plan = None
        if rebalance_signal["needs_rebalance"]:
            # Use cached price data if available; AllocationEngine now handles
            # an empty DataFrame gracefully by falling back to stored values.
            price_data = self._get_price_data()
            allocation_plan = self.allocation_engine.allocate(
                self.portfolio,
                self.portfolio["target_allocation"],
                price_data,
                self.portfolio["cash"]
            )

        # Generate recommendations
        recommendations = self.recommendation_generator.generate_recommendations(
            allocation_plan or {},
            rebalance_signal,
            market_outlook
        )

        return recommendations
    
    def _calculate_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        weights = {}
        total_value = self.portfolio["total_value"]
        
        for position in self.portfolio["positions"]:
            symbol = position["symbol"]
            value = position["value"]
            weights[symbol] = value / total_value if total_value > 0 else 0
        
        return weights
    
    def _get_portfolio_volatility(self) -> float:
        """Estimate portfolio volatility"""
        # This would use historical returns
        # Placeholder
        return 0.15
    
    def _get_market_caps(self, symbols: List[str]) -> Dict[str, float]:
        """Get market caps for symbols"""
        # This would fetch from data source
        # Placeholder
        return {symbol: 1e9 for symbol in symbols}
    
    def _get_price_data(self) -> pd.DataFrame:
        """
        Return the most recent cached price DataFrame.
        Populated via `update_portfolio` with action='update_prices'.
        Returns an empty DataFrame when no cache exists — callers must handle this.
        """
        return getattr(self, "_price_cache", pd.DataFrame())
    
    async def _update_portfolio(self, updates: Dict[str, Any]):
        """Update portfolio state"""
        action = updates.get("action")
        
        if action == "add_position":
            position = updates.get("position")
            self.portfolio["positions"].append(position)
            self.portfolio["cash"] -= position["value"]
            
        elif action == "remove_position":
            symbol = updates.get("symbol")
            self.portfolio["positions"] = [
                p for p in self.portfolio["positions"] 
                if p["symbol"] != symbol
            ]
            
        elif action == "update_prices":
            prices = updates.get("prices", {})
            for position in self.portfolio["positions"]:
                if position["symbol"] in prices:
                    position["current_price"] = prices[position["symbol"]]
                    position["value"] = position["shares"] * prices[position["symbol"]]
            # Keep a single-row price DataFrame so generate_recommendations can use it
            if prices:
                self._price_cache = pd.DataFrame([prices])
        
        # Recalculate total value
        total_invested = sum(p.get("value", 0) for p in self.portfolio["positions"])
        self.portfolio["total_value"] = self.portfolio["cash"] + total_invested
        
        # Add to history
        self.portfolio["performance_history"].append({
            "timestamp": datetime.now().isoformat(),
            "total_value": self.portfolio["total_value"],
            "cash": self.portfolio["cash"]
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "portfolio": {
                "total_value": self.portfolio["total_value"],
                "cash": self.portfolio["cash"],
                "invested": self.portfolio["total_value"] - self.portfolio["cash"],
                "num_positions": len(self.portfolio["positions"]),
                "has_target": bool(self.portfolio["target_allocation"]),
                "last_optimization": self.portfolio.get("last_optimization"),
                "last_rebalance": self.portfolio.get("last_rebalance")
            },
            "current_regime": self.current_regime,
            "constraints": self.constraints
        }