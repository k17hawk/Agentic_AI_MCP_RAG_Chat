"""
Risk Manager - Main orchestrator for all risk management
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from utils.logger import logger as logging
from agents.base_agent import BaseAgent, AgentMessage  # IMPORT AgentMessage!

# Import all risk components
from risk.market_regime_risk import MarketRegimeRisk
from risk.position_sizing.kelly_criterion import KellyCriterion
from risk.position_sizing.half_kelly import HalfKelly
from risk.position_sizing.fixed_fraction import FixedFraction
from risk.position_sizing.volatility_adjusted import VolatilityAdjusted
from risk.stop_loss_optimizer.atr_stop import ATRStop
from risk.stop_loss_optimizer.volatility_stop import VolatilityStop
from risk.stop_loss_optimizer.trailing_stop import TrailingStop
from risk.stop_loss_optimizer.time_stop import TimeStop
from risk.portfolio_risk.var_calculator import VaRCalculator
from risk.portfolio_risk.expected_shortfall import ExpectedShortfall
from risk.portfolio_risk.correlation_matrix import CorrelationMatrix
from risk.portfolio_risk.diversification_score import DiversificationScore
from risk.portfolio_risk.stress_tester import StressTester
from risk.risk_scorer import RiskScorer
from risk.risk_approved_queue import RiskApprovedQueue

class RiskManager(BaseAgent):
    """
    Risk Manager - Main orchestrator for all risk management
    
    Responsibilities:
    - Calculate position sizes
    - Set stop losses
    - Monitor portfolio risk
    - Score and approve/reject trades
    - Maintain approved queue
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Comprehensive risk management",
            config=config
        )
        
        # Initialize position sizing strategies
        self.kelly = KellyCriterion(config.get("kelly_config", {}))
        self.half_kelly = HalfKelly(config.get("half_kelly_config", {}))
        self.fixed_fraction = FixedFraction(config.get("fixed_fraction_config", {}))
        self.vol_adjusted = VolatilityAdjusted(config.get("vol_adjusted_config", {}))
        
        # Initialize stop loss strategies
        self.atr_stop = ATRStop(config.get("atr_stop_config", {}))
        self.vol_stop = VolatilityStop(config.get("vol_stop_config", {}))
        self.trailing_stop = TrailingStop(config.get("trailing_stop_config", {}))
        self.time_stop = TimeStop(config.get("time_stop_config", {}))
        
        # Initialize portfolio risk tools
        self.var_calc = VaRCalculator(config.get("var_config", {}))
        self.es_calc = ExpectedShortfall(config.get("es_config", {}))
        self.corr_matrix = CorrelationMatrix(config.get("correlation_config", {}))
        self.diversification = DiversificationScore(config.get("diversification_config", {}))
        self.stress_tester = StressTester(config.get("stress_config", {}))
        
        # Initialize regime risk adapter
        self.regime_risk = MarketRegimeRisk(config.get("regime_risk_config", {}))
        
        # Initialize risk scorer and queue
        self.risk_scorer = RiskScorer(config.get("scorer_config", {}))
        self.approved_queue = RiskApprovedQueue(config.get("queue_config", {}))
        
        # Portfolio state
        self.portfolio = {
            "capital": config.get("initial_capital", 100000),
            "positions": [],
            "cash": config.get("initial_capital", 100000),
            "last_updated": datetime.now().isoformat()
        }
        
        # Current market regime
        self.current_regime = "neutral_ranging"
        
        logging.info(f"✅ RiskManager initialized with capital: ${self.portfolio['capital']:,.2f}")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process risk management requests
        """
        if message.message_type == "analysis_complete":
            # New analysis complete, calculate risk
            analysis = message.content
            return await self.calculate_risk(analysis, message.sender)
        
        elif message.message_type == "get_risk_approval":
            # Get next approved trade
            trade = await self.approved_queue.get_next()
            if trade:
                return AgentMessage(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="risk_approved_trade",
                    content=trade
                )
            else:
                return AgentMessage(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="risk_queue_empty",
                    content={"message": "No approved trades in queue"}
                )
        
        elif message.message_type == "update_portfolio":
            # Update portfolio with new positions
            updates = message.content
            await self._update_portfolio(updates)
            return None
        
        elif message.message_type == "update_regime":
            # Update current market regime
            self.current_regime = message.content.get("regime", self.current_regime)
            logging.info(f"📊 RiskManager updated regime: {self.current_regime}")
            return None
        
        elif message.message_type == "get_portfolio_risk":
            # Get portfolio risk metrics
            risk_metrics = await self.calculate_portfolio_risk()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="portfolio_risk",
                content=risk_metrics
            )
        
        elif message.message_type == "health_check":
            # Health check request
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="health_response",
                content=self.get_status()
            )
        
        return None
    
    async def calculate_risk(self, analysis: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Calculate risk for a potential trade
        """
        symbol = analysis.get("symbol")
        score = analysis.get("final_score", 0.5)
        
        logging.info(f"🛡️ Calculating risk for {symbol}")
        
        # Get basic info
        signal = analysis.get("action", "WATCH")
        confidence = analysis.get("confidence", 0.5)
        
        # Get price (would come from analysis details)
        price = 100.0  # Placeholder - would come from analysis
        
        # Get volatility data (would come from analysis)
        volatility = 0.20  # Placeholder
        
        # Calculate position size using multiple methods
        kelly_size = self.kelly.calculate(
            win_rate=0.55,  # Would come from historical data
            avg_win=0.10,
            avg_loss=0.05
        )
        
        fixed_size = self.fixed_fraction.calculate(
            capital=self.portfolio["capital"],
            confidence=confidence
        )
        
        vol_adjusted_size = self.vol_adjusted.calculate(
            capital=self.portfolio["capital"],
            volatility=volatility,
            confidence=confidence
        )
        
        # Choose position size (using most conservative for now)
        position_size = min(
            kelly_size.get("recommended_fraction", 0) * self.portfolio["capital"],
            fixed_size.get("position_value", 0),
            vol_adjusted_size.get("position_value", 0)
        )
        
        position_fraction = position_size / self.portfolio["capital"] if self.portfolio["capital"] > 0 else 0
        
        # Calculate stop loss
        atr = price * 0.02  # 2% ATR placeholder
        stop_result = self.atr_stop.calculate(
            entry_price=price,
            atr=atr,
            multiplier=self.regime_risk.stop_multipliers.get(self.current_regime, 2.0)
        )
        
        # Apply regime adjustments
        risk_params = {
            "position_size": position_size,
            "position_fraction": position_fraction,
            "stop_distance": stop_result["stop_distance"],
            "stop_percent": stop_result["stop_percent"],
            "stop_price": stop_result["stop_price"],
            "risk_per_trade": stop_result["risk_per_share"] * (position_size / price) if price > 0 else 0
        }
        
        adjusted_params = self.regime_risk.adjust_risk(risk_params, self.current_regime)
        
        # Calculate risk score
        risk_score_result = self.risk_scorer.score_trade({
            **adjusted_params,
            "volatility": volatility,
            "regime": self.current_regime,
            "volume": 1000000,  # Placeholder
            "avg_volume": 800000,  # Placeholder
            "bid_ask_spread": 0.001,  # Placeholder
            "avg_portfolio_correlation": 0.5,  # Placeholder
            "leverage": 1.0,  # Placeholder
            "time_horizon_days": 5  # Placeholder
        })
        
        # Create trade object
        trade = {
            "ticker": symbol,
            "analysis": analysis,
            "position_size": adjusted_params["position_size"],
            "position_fraction": adjusted_params["position_fraction"],
            "entry_price": price,
            "stop_price": adjusted_params["stop_price"],
            "stop_percent": adjusted_params["stop_percent"],
            "risk_per_trade": adjusted_params["risk_per_trade"],
            "risk_score": risk_score_result["risk_score"],
            "risk_level": risk_score_result["risk_level"],
            "risk_components": risk_score_result["components"],
            "risk_warnings": risk_score_result["warnings"],
            "regime": self.current_regime,
            "regime_description": self.regime_risk._get_regime_description(self.current_regime),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to queue if approved
        if risk_score_result["should_trade"]:
            await self.approved_queue.add(trade)
            logging.info(f"✅ Trade approved for {symbol} (risk: {risk_score_result['risk_score']:.2f})")
            message_type = "risk_approved"
        else:
            logging.info(f"❌ Trade rejected for {symbol} (risk: {risk_score_result['risk_score']:.2f})")
            message_type = "risk_rejected"
        
        return AgentMessage(
            sender=self.name,
            receiver=requester,
            message_type=message_type,
            content=trade
        )
    
    async def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """
        Calculate overall portfolio risk metrics
        """
        if not self.portfolio["positions"]:
            return {
                "message": "No positions",
                "total_value": self.portfolio["capital"],
                "cash": self.portfolio["cash"],
                "invested": self.portfolio["capital"] - self.portfolio["cash"],
                "num_positions": 0
            }
        
        # Calculate portfolio returns (would use historical data)
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]  # Placeholder
        
        # Calculate VaR
        var = self.var_calc.calculate(returns, self.portfolio["capital"])
        
        # Calculate Expected Shortfall
        es = self.es_calc.calculate(returns, self.portfolio["capital"])
        
        # Calculate diversification
        diversification = self.diversification.calculate(self.portfolio["positions"])
        
        # Run stress tests
        stress = self.stress_tester.test_portfolio(self.portfolio["positions"])
        
        # Calculate portfolio risk score
        portfolio_for_scoring = {
            "positions": self.portfolio["positions"],
            "returns": returns,
            "value": self.portfolio["capital"],
            "var_95": var.get("var_percent", 2) / 100,
            "max_drawdown": 0.10,  # Placeholder
            "sector_exposures": self._calculate_sector_exposures(),
            "leverage": 1.0,  # Placeholder
            "correlations": [0.5, 0.6, 0.4]  # Placeholder
        }
        
        portfolio_risk = self.risk_scorer.score_portfolio(portfolio_for_scoring)
        
        return {
            "total_value": self.portfolio["capital"],
            "cash": self.portfolio["cash"],
            "invested": self.portfolio["capital"] - self.portfolio["cash"],
            "num_positions": len(self.portfolio["positions"]),
            "var": var,
            "expected_shortfall": es,
            "diversification": diversification,
            "stress_test": stress,
            "portfolio_risk_score": portfolio_risk,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """Calculate sector exposures for current portfolio"""
        sector_exposures = {}
        
        for position in self.portfolio["positions"]:
            sector = position.get("sector", "Unknown")
            value = position.get("value", 0)
            sector_exposures[sector] = sector_exposures.get(sector, 0) + value
        
        # Normalize
        total = sum(sector_exposures.values())
        if total > 0:
            for sector in sector_exposures:
                sector_exposures[sector] /= total
        
        return sector_exposures
    
    async def _update_portfolio(self, updates: Dict[str, Any]):
        """
        Update portfolio with new positions or changes
        """
        action = updates.get("action")
        symbol = updates.get("symbol")
        
        if action == "buy":
            # Add new position
            position = {
                "symbol": symbol,
                "shares": updates.get("shares"),
                "entry_price": updates.get("price"),
                "current_price": updates.get("price"),
                "value": updates.get("value"),
                "stop_loss": updates.get("stop_loss"),
                "take_profit": updates.get("take_profit"),
                "sector": updates.get("sector", "Unknown"),
                "entry_time": datetime.now().isoformat()
            }
            self.portfolio["positions"].append(position)
            self.portfolio["cash"] -= updates.get("value", 0)
            logging.info(f"➕ Added position: {symbol} for ${updates.get('value', 0):,.2f}")
            
        elif action == "sell":
            # Remove position
            initial_count = len(self.portfolio["positions"])
            self.portfolio["positions"] = [
                p for p in self.portfolio["positions"] 
                if p["symbol"] != symbol
            ]
            if len(self.portfolio["positions"]) < initial_count:
                self.portfolio["cash"] += updates.get("value", 0)
                logging.info(f"➖ Sold position: {symbol} for ${updates.get('value', 0):,.2f}")
        
        elif action == "update_price":
            # Update current prices
            for position in self.portfolio["positions"]:
                if position["symbol"] == symbol:
                    position["current_price"] = updates.get("price")
                    position["value"] = position["shares"] * updates.get("price", 0)
                    break
        
        self.portfolio["last_updated"] = datetime.now().isoformat()
        
        # Recalculate total capital
        total_invested = sum(p.get("value", 0) for p in self.portfolio["positions"])
        self.portfolio["capital"] = self.portfolio["cash"] + total_invested
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "queue_size": self.message_queue.qsize(),
            "portfolio": {
                "capital": self.portfolio["capital"],
                "cash": self.portfolio["cash"],
                "positions": len(self.portfolio["positions"]),
                "invested_pct": (self.portfolio["capital"] - self.portfolio["cash"]) / self.portfolio["capital"] * 100 if self.portfolio["capital"] > 0 else 0
            },
            "queue": self.approved_queue.get_stats(),
            "current_regime": self.current_regime,
            "risk_appetite": self.regime_risk.get_risk_appetite(self.current_regime)
        }