"""
Risk Manager - Main orchestrator for all risk management
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import numpy as np

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import all risk components
from agentic_trading_system.risk.market_regime_risk import MarketRegimeRisk
from agentic_trading_system.risk.position_sizing.kelly_criterion import KellyCriterion
from agentic_trading_system.risk.position_sizing.half_kelly import HalfKelly
from agentic_trading_system.risk.position_sizing.fixed_fraction import FixedFraction
from agentic_trading_system.risk.position_sizing.volatility_adjusted import VolatilityAdjusted
from agentic_trading_system.risk.stop_loss_optimizer.atr_stop import ATRStop
from agentic_trading_system.risk.stop_loss_optimizer.volatility_stop import VolatilityStop
from agentic_trading_system.risk.stop_loss_optimizer.trailing_stop import TrailingStop
from agentic_trading_system.risk.stop_loss_optimizer.time_stop import TimeStop
from agentic_trading_system.risk.portfolio_risk.var_calculator import VaRCalculator
from agentic_trading_system.risk.portfolio_risk.expected_shortfall import ExpectedShortfall
from agentic_trading_system.risk.portfolio_risk.correlation_matrix import CorrelationMatrix
from agentic_trading_system.risk.portfolio_risk.diversification_score import DiversificationScore
from agentic_trading_system.risk.portfolio_risk.stress_tester import StressTester
from agentic_trading_system.risk.risk_scorer import RiskScorer
from agentic_trading_system.risk.risk_approved_queue import RiskApprovedQueue

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
            "last_updated": datetime.now().isoformat(),
            "historical_returns": []  # Store historical returns for correlation
        }
        
        # Current market regime
        self.current_regime = "neutral_ranging"
        
        # Historical data for correlations
        self.price_history = {}  # Store price history for each symbol
        
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
        
        elif message.message_type == "update_price_history":
            # Update price history for correlation calculations
            symbol = message.content.get("symbol")
            price = message.content.get("price")
            if symbol and price:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append({
                    "timestamp": datetime.now(),
                    "price": price
                })
                # Keep last 100 prices
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
            return None
        
        return None
    
    async def calculate_risk(self, analysis: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Calculate risk for a potential trade using actual market data
        """
        # Extract all available data from analysis
        symbol = analysis.get("symbol")
        score = analysis.get("final_score", 0.5)
        signal = analysis.get("action", "WATCH")
        confidence = analysis.get("confidence", 0.5)
        
        # Get actual market data (FIX: Use real values from analysis)
        price = analysis.get("price", 100.0)
        volatility = analysis.get("volatility", 0.20)
        atr = analysis.get("atr", price * volatility / 16)  # Estimate if not provided
        volume = analysis.get("volume", 1000000)
        avg_volume = analysis.get("avg_volume", 800000)
        bid_ask_spread = analysis.get("bid_ask_spread", 0.001)
        
        # Get historical win rate if available
        win_rate = analysis.get("win_rate", 0.55)
        avg_win = analysis.get("avg_win", 0.10)
        avg_loss = analysis.get("avg_loss", 0.05)
        
        logging.info(f"🛡️ Calculating risk for {symbol} at ${price:.2f} (vol: {volatility*100:.1f}%, ATR: ${atr:.2f})")
        
        # ===== POSITION SIZING CALCULATIONS =====
        
        # Kelly Criterion
        kelly_result = self.kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        kelly_fraction = min(kelly_result.get("recommended_fraction", 0), 
                            self.config.get("kelly_config", {}).get("max_fraction", 0.25))
        kelly_size = kelly_fraction * self.portfolio["capital"]
        
        # Half Kelly (more conservative)
        half_kelly_result = self.half_kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        half_kelly_fraction = half_kelly_result.get("recommended_fraction", 0)
        half_kelly_size = half_kelly_fraction * self.portfolio["capital"]
        
        # Fixed Fraction
        fixed_result = self.fixed_fraction.calculate(
            capital=self.portfolio["capital"],
            confidence=confidence
        )
        fixed_size = fixed_result.get("position_value", 0)
        
        # Volatility Adjusted (PRIMARY METHOD)
        vol_adjusted_result = self.vol_adjusted.calculate(
            capital=self.portfolio["capital"],
            volatility=volatility,
            confidence=confidence
        )
        vol_adjusted_size = vol_adjusted_result.get("position_value", 0)
        
        # Choose position size (prioritize volatility-adjusted)
        position_size = vol_adjusted_size
        
        # Consider half-kelly if more conservative
        if half_kelly_size > 0 and half_kelly_size < position_size:
            position_size = half_kelly_size
        
        # Apply max position size based on regime
        max_regime_fraction = self.regime_risk.max_position_sizes.get(self.current_regime, 0.25)
        max_position = self.portfolio["capital"] * max_regime_fraction
        position_size = min(position_size, max_position)
        
        # Minimum position size
        min_position = self.portfolio["capital"] * 0.005  # 0.5% minimum
        position_size = max(position_size, min_position)
        
        position_fraction = position_size / self.portfolio["capital"] if self.portfolio["capital"] > 0 else 0
        
        # ===== STOP LOSS CALCULATIONS =====
        
        # Get regime-adjusted stop multiplier
        regime_stop_multiplier = self.regime_risk.stop_multipliers.get(self.current_regime, 2.0)
        
        # Calculate ATR-based stop loss
        stop_result = self.atr_stop.calculate(
            entry_price=price,
            atr=atr,
            multiplier=regime_stop_multiplier
        )
        
        # Also calculate volatility stop for comparison
        # Generate mock returns for volatility stop (in production, use historical)
        mock_returns = np.random.normal(0, volatility / np.sqrt(252), 20)
        vol_stop_result = self.vol_stop.calculate(
            entry_price=price,
            returns=mock_returns.tolist(),
            multiplier=regime_stop_multiplier
        )
        
        # Use the more conservative stop
        if vol_stop_result.get("stop_price", 0) > stop_result.get("stop_price", 0):
            stop_price = vol_stop_result["stop_price"]
            stop_distance = vol_stop_result["stop_distance"]
            stop_percent = vol_stop_result["stop_percent"]
        else:
            stop_price = stop_result["stop_price"]
            stop_distance = stop_result["stop_distance"]
            stop_percent = stop_result["stop_percent"]
        
        # ===== REGIME ADJUSTMENTS =====
        
        # Prepare risk parameters for regime adjustment
        risk_params = {
            "position_size": position_size,
            "position_fraction": position_fraction,
            "stop_distance": stop_distance,
            "stop_percent": stop_percent,
            "stop_price": stop_price,
            "risk_per_trade": stop_distance * (position_size / price) if price > 0 else 0
        }
        
        # Apply regime-based risk adjustments
        adjusted_params = self.regime_risk.adjust_risk(risk_params, self.current_regime)
        
        # ===== RISK SCORING =====
        
        # Calculate portfolio correlation
        avg_correlation = self._calculate_avg_correlation(symbol)
        
        # Score the trade
        risk_score_result = self.risk_scorer.score_trade({
            **adjusted_params,
            "volatility": volatility,
            "regime": self.current_regime,
            "volume": volume,
            "avg_volume": avg_volume,
            "bid_ask_spread": bid_ask_spread,
            "avg_portfolio_correlation": avg_correlation,
            "leverage": 1.0,  # No leverage by default
            "time_horizon_days": 5  # Default swing trading horizon
        })
        
        # ===== CREATE TRADE OBJECT =====
        
        trade = {
            "ticker": symbol,
            "analysis": analysis,
            "position_size": adjusted_params["position_size"],
            "position_fraction": adjusted_params["position_fraction"],
            "entry_price": price,
            "stop_price": adjusted_params["stop_price"],
            "stop_percent": adjusted_params["stop_percent"],
            "stop_distance": adjusted_params["stop_distance"],
            "risk_per_trade": adjusted_params["risk_per_trade"],
            "risk_score": risk_score_result["risk_score"],
            "risk_level": risk_score_result["risk_level"],
            "risk_components": risk_score_result["components"],
            "risk_warnings": risk_score_result["warnings"],
            "should_trade": risk_score_result["should_trade"],
            "regime": self.current_regime,
            "regime_description": self.regime_risk._get_regime_description(self.current_regime),
            "volatility": volatility,
            "atr": atr,
            "position_sizing_methods": {
                "kelly": kelly_size,
                "half_kelly": half_kelly_size,
                "fixed_fraction": fixed_size,
                "volatility_adjusted": vol_adjusted_size,
                "selected": adjusted_params["position_size"]
            },
            "stop_methods": {
                "atr_stop": stop_result["stop_price"],
                "volatility_stop": vol_stop_result.get("stop_price", 0),
                "selected": adjusted_params["stop_price"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # ===== QUEUE MANAGEMENT =====
        
        # Add to queue if approved
        if risk_score_result["should_trade"]:
            await self.approved_queue.add(trade)
            logging.info(f"✅ Trade approved for {symbol}: ${adjusted_params['position_size']:,.2f} position, "
                        f"stop ${adjusted_params['stop_price']:.2f} ({adjusted_params['stop_percent']:.1f}%), "
                        f"risk score: {risk_score_result['risk_score']:.2f}")
            message_type = "risk_approved"
        else:
            logging.info(f"❌ Trade rejected for {symbol} (risk score: {risk_score_result['risk_score']:.2f}, "
                        f"level: {risk_score_result['risk_level']})")
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
                "num_positions": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate portfolio returns from historical data
        portfolio_returns = self._calculate_portfolio_returns()
        
        # Calculate VaR
        var_result = self.var_calc.calculate(
            returns=portfolio_returns,
            portfolio_value=self.portfolio["capital"]
        )
        
        # Calculate Expected Shortfall (CVaR)
        es_result = self.es_calc.calculate(
            returns=portfolio_returns,
            portfolio_value=self.portfolio["capital"]
        )
        
        # Calculate diversification score
        diversification_result = self.diversification.calculate(self.portfolio["positions"])
        
        # Run stress tests
        stress_result = self.stress_tester.test_portfolio(self.portfolio["positions"])
        
        # Calculate portfolio risk score
        sector_exposures = self._calculate_sector_exposures()
        
        portfolio_for_scoring = {
            "positions": self.portfolio["positions"],
            "returns": portfolio_returns,
            "value": self.portfolio["capital"],
            "var_95": var_result.get("var_percent", 2) / 100,
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "sector_exposures": sector_exposures,
            "leverage": 1.0,  # No leverage by default
            "correlations": self._calculate_portfolio_correlations()
        }
        
        portfolio_risk_score = self.risk_scorer.score_portfolio(portfolio_for_scoring)
        
        # Calculate additional metrics
        invested = self.portfolio["capital"] - self.portfolio["cash"]
        invested_pct = (invested / self.portfolio["capital"] * 100) if self.portfolio["capital"] > 0 else 0
        
        return {
            "total_value": self.portfolio["capital"],
            "cash": self.portfolio["cash"],
            "invested": invested,
            "invested_pct": invested_pct,
            "num_positions": len(self.portfolio["positions"]),
            "var": var_result,
            "expected_shortfall": es_result,
            "diversification": diversification_result,
            "stress_test": stress_result,
            "portfolio_risk_score": portfolio_risk_score,
            "sector_exposures": sector_exposures,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_avg_correlation(self, new_symbol: str) -> float:
        """
        Calculate average correlation between new symbol and existing portfolio
        """
        if len(self.portfolio["positions"]) == 0:
            return 0.5  # Neutral correlation for first position
        
        if len(self.price_history) < 2:
            return 0.6  # Default moderate correlation
        
        try:
            # Get returns for existing positions
            existing_returns = []
            for position in self.portfolio["positions"]:
                symbol = position["symbol"]
                if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                    prices = [p["price"] for p in self.price_history[symbol][-20:]]
                    if len(prices) > 1:
                        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                                  for i in range(1, len(prices))]
                        existing_returns.append(returns)
            
            # Get returns for new symbol
            if new_symbol in self.price_history and len(self.price_history[new_symbol]) > 20:
                prices = [p["price"] for p in self.price_history[new_symbol][-20:]]
                new_returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                              for i in range(1, len(prices))]
                
                # Calculate correlations with existing positions
                correlations = []
                for existing in existing_returns:
                    if len(existing) == len(new_returns):
                        corr = np.corrcoef(existing, new_returns)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    return max(-1.0, min(1.0, avg_correlation))
            
            return 0.6  # Default moderate correlation
            
        except Exception as e:
            logging.warning(f"Error calculating correlation: {e}")
            return 0.6
    
    def _calculate_portfolio_returns(self) -> List[float]:
        """
        Calculate historical portfolio returns
        """
        if not self.portfolio["historical_returns"]:
            # Generate mock returns if no historical data
            return np.random.normal(0, 0.01, 252).tolist()
        
        return self.portfolio["historical_returns"][-252:]  # Last 252 days
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns
        """
        if not returns:
            return 0.0
        
        cumulative = 1
        peak = 1
        max_drawdown = 0
        
        for r in returns:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _calculate_portfolio_correlations(self) -> List[float]:
        """
        Calculate correlation matrix for portfolio positions
        """
        if len(self.portfolio["positions"]) < 2:
            return [1.0]
        
        correlations = []
        symbols = [p["symbol"] for p in self.portfolio["positions"]]
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if symbols[i] in self.price_history and symbols[j] in self.price_history:
                    # Get returns for both symbols
                    prices_i = [p["price"] for p in self.price_history[symbols[i]][-30:]]
                    prices_j = [p["price"] for p in self.price_history[symbols[j]][-30:]]
                    
                    if len(prices_i) > 1 and len(prices_j) > 1:
                        returns_i = [(prices_i[k] - prices_i[k-1]) / prices_i[k-1] 
                                    for k in range(1, len(prices_i))]
                        returns_j = [(prices_j[k] - prices_j[k-1]) / prices_j[k-1] 
                                    for k in range(1, len(prices_j))]
                        
                        min_len = min(len(returns_i), len(returns_j))
                        if min_len > 1:
                            corr = np.corrcoef(returns_i[:min_len], returns_j[:min_len])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
        
        return correlations if correlations else [0.5]
    
    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """
        Calculate sector exposures for current portfolio
        """
        sector_exposures = {}
        total_value = 0
        
        for position in self.portfolio["positions"]:
            sector = position.get("sector", "Unknown")
            value = position.get("value", 0)
            sector_exposures[sector] = sector_exposures.get(sector, 0) + value
            total_value += value
        
        # Normalize to percentages
        if total_value > 0:
            for sector in sector_exposures:
                sector_exposures[sector] = (sector_exposures[sector] / total_value) * 100
        
        return sector_exposures
    
    async def _update_portfolio(self, updates: Dict[str, Any]):
        """
        Update portfolio with new positions or changes
        """
        action = updates.get("action")
        symbol = updates.get("symbol")
        
        if action == "buy":
            # Check if position already exists
            existing = next((p for p in self.portfolio["positions"] if p["symbol"] == symbol), None)
            
            if existing:
                # Average in
                new_shares = existing["shares"] + updates.get("shares", 0)
                new_value = new_shares * updates.get("price", existing["current_price"])
                existing["shares"] = new_shares
                existing["value"] = new_value
                existing["current_price"] = updates.get("price", existing["current_price"])
                logging.info(f"📈 Averaged up position: {symbol} to {new_shares} shares")
            else:
                # Add new position
                position = {
                    "symbol": symbol,
                    "shares": updates.get("shares", 0),
                    "entry_price": updates.get("price", 0),
                    "current_price": updates.get("price", 0),
                    "value": updates.get("value", 0),
                    "stop_loss": updates.get("stop_loss", 0),
                    "take_profit": updates.get("take_profit", 0),
                    "sector": updates.get("sector", "Unknown"),
                    "entry_time": datetime.now().isoformat()
                }
                self.portfolio["positions"].append(position)
                self.portfolio["cash"] -= updates.get("value", 0)
                logging.info(f"➕ Added position: {symbol} for ${updates.get('value', 0):,.2f}")
            
        elif action == "sell":
            # Remove position
            position = next((p for p in self.portfolio["positions"] if p["symbol"] == symbol), None)
            if position:
                self.portfolio["cash"] += position["value"]
                self.portfolio["positions"] = [
                    p for p in self.portfolio["positions"] 
                    if p["symbol"] != symbol
                ]
                logging.info(f"➖ Sold position: {symbol} for ${position['value']:,.2f}")
        
        elif action == "update_price":
            # Update current prices
            for position in self.portfolio["positions"]:
                if position["symbol"] == symbol:
                    old_value = position["value"]
                    position["current_price"] = updates.get("price", position["current_price"])
                    position["value"] = position["shares"] * updates.get("price", position["current_price"])
                    
                    # Record return for historical tracking
                    if old_value > 0:
                        daily_return = (position["value"] - old_value) / old_value
                        self.portfolio["historical_returns"].append(daily_return)
                        # Keep last 500 returns
                        if len(self.portfolio["historical_returns"]) > 500:
                            self.portfolio["historical_returns"] = self.portfolio["historical_returns"][-500:]
                    break
        
        elif action == "update_portfolio_value":
            # Update total portfolio value
            total_invested = sum(p.get("value", 0) for p in self.portfolio["positions"])
            self.portfolio["capital"] = self.portfolio["cash"] + total_invested
        
        self.portfolio["last_updated"] = datetime.now().isoformat()
        
        # Recalculate total capital
        total_invested = sum(p.get("value", 0) for p in self.portfolio["positions"])
        self.portfolio["capital"] = self.portfolio["cash"] + total_invested
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get risk manager status
        """
        total_invested = sum(p.get("value", 0) for p in self.portfolio["positions"])
        invested_pct = (total_invested / self.portfolio["capital"] * 100) if self.portfolio["capital"] > 0 else 0
        
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "queue_size": self.message_queue.qsize(),
            "portfolio": {
                "capital": self.portfolio["capital"],
                "cash": self.portfolio["cash"],
                "invested": total_invested,
                "positions": len(self.portfolio["positions"]),
                "invested_pct": invested_pct
            },
            "queue": self.approved_queue.get_stats(),
            "current_regime": self.current_regime,
            "risk_appetite": self.regime_risk.get_risk_appetite(self.current_regime),
            "regime_risk_multiplier": self.regime_risk.risk_multipliers.get(self.current_regime, 1.0),
            "regime_stop_multiplier": self.regime_risk.stop_multipliers.get(self.current_regime, 2.0)
        }