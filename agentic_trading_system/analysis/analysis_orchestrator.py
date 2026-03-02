"""
Analysis Orchestrator - Receives fused trigger events and coordinates all analysis agents
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import yfinance as yf

from utils.logger import logger as logging
from agents.base_agent import BaseAgent, AgentMessage

# Import analysis components
from analysis.multi_timeframe_aggregator import MultiTimeframeAggregator, Timeframe
from analysis.weighted_score_engine import WeightedScoreEngine
from analysis.regime_detector import RegimeDetector
from analysis.technical.technical_analyzer import TechnicalAnalyzer
from analysis.fundamental.fundamental_analyzer import FundamentalAnalyzer
from analysis.sentiment.sentiment_analyzer import SentimentAnalyzer

class AnalysisOrchestrator(BaseAgent):
    """
    Orchestrates the entire analysis pipeline:
    1. Receives fused trigger events
    2. Calls Technical Analysis Agent
    3. Calls Fundamental Analysis Agent
    4. Calls Sentiment Analysis Agent
    5. Uses MultiTimeframeAggregator to combine timeframe signals
    6. Uses WeightedScoreEngine to combine all scores
    7. Sends final result to Risk Management
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Orchestrates multi-dimensional stock analysis",
            config=config
        )
        
        # Analysis agents (will be registered)
        self.technical_agent = None
        self.fundamental_agent = None
        self.sentiment_agent = None
        self.regime_detector = None
        
        # Initialize components
        self.timeframe_aggregator = MultiTimeframeAggregator(
            name="MultiTimeframeAggregator",
            config=config.get("timeframe_config", {})
        )
        
        self.score_engine = WeightedScoreEngine(
            name="WeightedScoreEngine",
            config=config.get("score_engine_config", {})
        )
        
        # Pending analyses
        self.pending_analyses = {}
        self.analysis_results = {}
        
        # Configuration
        self.analysis_timeout = config.get("analysis_timeout", 30)  # seconds
        self.min_confidence_to_proceed = config.get("min_confidence", 0.6)
        self.max_concurrent_analyses = config.get("max_concurrent_analyses", 10)
        
        # Semaphore for limiting concurrent analyses
        self.semaphore = asyncio.Semaphore(self.max_concurrent_analyses)
        
        logging.info(f"✅ AnalysisOrchestrator initialized")
        logging.info(f"   • TimeframeAggregator: {self.timeframe_aggregator.name}")
        logging.info(f"   • ScoreEngine: {self.score_engine.name}")
        logging.info(f"   • Max concurrent analyses: {self.max_concurrent_analyses}")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages
        """
        if message.message_type == "fused_signal":
            # New fused signal from trigger fusion
            return await self.handle_fused_signal(message.content)
            
        elif message.message_type == "analysis_result":
            # Result from one of the analysis agents
            return await self.handle_analysis_result(message.content)
            
        elif message.message_type == "regime_result":
            # Result from regime detector
            return await self.handle_regime_result(message.content)
            
        elif message.message_type == "aggregate_result":
            # Result from timeframe aggregator
            return await self.handle_aggregate_result(message.content)
            
        elif message.message_type == "combined_scores":
            # Result from score engine
            return await self.handle_combined_scores(message.content)
        
        elif message.message_type == "health_check":
            # Health check request
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="health_response",
                content=self.health_check()
            )
            
        return None
    
    async def handle_fused_signal(self, signal: Dict) -> Optional[AgentMessage]:
        """
        Handle a new fused signal from trigger fusion
        """
        symbol = signal.get("symbol")
        confidence = signal.get("confidence", 0.5)
        
        logging.info(f"📊 AnalysisOrchestrator received fused signal for {symbol} (conf: {confidence})")
        
        # Check if we're already analyzing this symbol
        for aid, pending in self.pending_analyses.items():
            if pending["symbol"] == symbol:
                logging.info(f"⏳ Already analyzing {symbol}, queuing this signal")
                # Could implement queueing logic here
                return None
        
        # Acquire semaphore to limit concurrent analyses
        if not self.semaphore.locked():
            await self.semaphore.acquire()
        
        try:
            # Create analysis request ID
            analysis_id = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store pending analysis
            self.pending_analyses[analysis_id] = {
                "symbol": symbol,
                "signal": signal,
                "start_time": datetime.now(),
                "results": {},
                "regime": None,
                "timeframe_result": None,
                "score_result": None,
                "status": "pending"
            }
            
            # Send requests to all analysis agents in parallel
            tasks = []
            
            # Request technical analysis
            if self.technical_agent:
                tasks.append(self._send_analysis_request(
                    analysis_id, symbol, signal, self.technical_agent.name, 3
                ))
            
            # Request fundamental analysis
            if self.fundamental_agent:
                tasks.append(self._send_analysis_request(
                    analysis_id, symbol, signal, self.fundamental_agent.name, 2
                ))
            
            # Request sentiment analysis
            if self.sentiment_agent:
                tasks.append(self._send_analysis_request(
                    analysis_id, symbol, signal, self.sentiment_agent.name, 2
                ))
            
            # Request regime detection
            if self.regime_detector:
                tasks.append(self._send_regime_request(analysis_id, symbol))
            
            # Wait for all requests to be sent
            if tasks:
                await asyncio.gather(*tasks)
            
            # Set timeout for this analysis
            asyncio.create_task(self._analysis_timeout(analysis_id))
            
            logging.info(f"✅ Dispatched analysis requests for {symbol} (ID: {analysis_id})")
            
        finally:
            # Release semaphore after requests are sent
            self.semaphore.release()
        
        return None
    
    async def _send_analysis_request(self, analysis_id: str, symbol: str, 
                                     signal: Dict, agent_name: str, priority: int):
        """Send analysis request to a specific agent"""
        await self.send_message(AgentMessage(
            sender=self.name,
            receiver=agent_name,
            message_type="analysis_request",
            content={
                "analysis_id": analysis_id,
                "symbol": symbol,
                "signal": signal
            },
            priority=priority,
            requires_response=True
        ))
    
    async def _send_regime_request(self, analysis_id: str, symbol: str):
        """Send regime detection request"""
        await self.send_message(AgentMessage(
            sender=self.name,
            receiver=self.regime_detector.name,
            message_type="regime_request",
            content={
                "analysis_id": analysis_id,
                "symbol": symbol
            },
            priority=3,
            requires_response=True
        ))
    
    async def handle_analysis_result(self, result: Dict) -> None:
        """
        Handle result from an analysis agent
        """
        analysis_id = result.get("analysis_id")
        agent_name = result.get("agent")
        score = result.get("score")
        details = result.get("details", {})
        
        if analysis_id not in self.pending_analyses:
            logging.warning(f"Received result for unknown analysis_id: {analysis_id}")
            return
        
        # Store result
        self.pending_analyses[analysis_id]["results"][agent_name] = {
            "score": score,
            "details": details,
            "timestamp": datetime.now()
        }
        
        logging.info(f"✅ Received {agent_name} result for {analysis_id}: score={score:.2f}")
        
        # Check if we should proceed to timeframe aggregation
        await self._check_and_aggregate_timeframes(analysis_id)
    
    async def handle_regime_result(self, result: Dict) -> None:
        """
        Handle result from regime detector
        """
        analysis_id = result.get("analysis_id")
        regime = result.get("regime")
        details = result.get("details", {})
        
        if analysis_id in self.pending_analyses:
            self.pending_analyses[analysis_id]["regime"] = {
                "regime": regime,
                "details": details
            }
            logging.info(f"✅ Received regime result for {analysis_id}: {regime}")
    
    async def handle_aggregate_result(self, result: Dict) -> None:
        """
        Handle result from timeframe aggregator
        """
        analysis_id = result.get("analysis_id")
        
        if analysis_id in self.pending_analyses:
            self.pending_analyses[analysis_id]["timeframe_result"] = result
            self.pending_analyses[analysis_id]["status"] = "timeframe_complete"
            logging.info(f"✅ Received timeframe aggregation for {analysis_id}")
            
            # Now proceed to score combination
            await self._combine_all_scores(analysis_id)
    
    async def handle_combined_scores(self, result: Dict) -> None:
        """
        Handle result from score engine
        """
        analysis_id = result.get("analysis_id")
        
        if analysis_id in self.pending_analyses:
            self.pending_analyses[analysis_id]["score_result"] = result
            self.pending_analyses[analysis_id]["status"] = "score_complete"
            
            # Analysis complete! Send to risk management
            await self._finalize_analysis(analysis_id)
    
    async def _check_and_aggregate_timeframes(self, analysis_id: str):
        """
        Check if we have technical data and trigger timeframe aggregation
        """
        pending = self.pending_analyses.get(analysis_id)
        if not pending:
            return
        
        # Check if we have technical analysis
        tech_agent_name = self.technical_agent.name if self.technical_agent else "TechnicalAnalyzer"
        if tech_agent_name in pending["results"]:
            tech_data = pending["results"][tech_agent_name]
            
            # Get data for different timeframes
            symbol = pending["symbol"]
            data_sets = await self._fetch_timeframe_data(symbol)
            
            if data_sets:
                # Request timeframe aggregation
                await self.send_message(AgentMessage(
                    sender=self.name,
                    receiver=self.timeframe_aggregator.name,
                    message_type="aggregate_request",
                    content={
                        "analysis_id": analysis_id,
                        "symbol": symbol,
                        "data_sets": data_sets
                    },
                    priority=3,
                    requires_response=True
                ))
                
                logging.info(f"🔄 Requested timeframe aggregation for {analysis_id}")
            else:
                logging.warning(f"⚠️ No timeframe data available for {symbol}")
    
    async def _fetch_timeframe_data(self, symbol: str) -> Dict:
        """
        Fetch data for all timeframes
        """
        data_sets = {}
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Intraday data
            try:
                data_sets[Timeframe.INTRADAY_1H.value] = ticker.history(period="1mo", interval="1h")
            except:
                logging.debug(f"Could not fetch 1h data for {symbol}")
            
            try:
                data_sets[Timeframe.INTRADAY_15M.value] = ticker.history(period="5d", interval="15m")
            except:
                logging.debug(f"Could not fetch 15m data for {symbol}")
            
            # Daily data (YOUR 60-DAY CORE)
            try:
                data_sets[Timeframe.DAILY.value] = ticker.history(period="6mo", interval="1d")
            except:
                logging.debug(f"Could not fetch daily data for {symbol}")
            
            # Weekly data
            try:
                data_sets[Timeframe.WEEKLY.value] = ticker.history(period="1y", interval="1wk")
            except:
                logging.debug(f"Could not fetch weekly data for {symbol}")
            
            # Monthly data
            try:
                data_sets[Timeframe.MONTHLY.value] = ticker.history(period="5y", interval="1mo")
            except:
                logging.debug(f"Could not fetch monthly data for {symbol}")
            
        except Exception as e:
            logging.error(f"Error fetching timeframe data for {symbol}: {e}")
        
        return data_sets
    
    async def _combine_all_scores(self, analysis_id: str):
        """
        Combine all scores using the weighted score engine
        """
        pending = self.pending_analyses[analysis_id]
        
        # Prepare scores for weighted engine
        scores = {}
        
        # Add technical score
        tech_agent_name = self.technical_agent.name if self.technical_agent else "TechnicalAnalyzer"
        if tech_agent_name in pending["results"]:
            scores["technical"] = {
                "score": pending["results"][tech_agent_name]["score"],
                "details": pending["results"][tech_agent_name]["details"],
                "confidence": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add fundamental score
        fund_agent_name = self.fundamental_agent.name if self.fundamental_agent else "FundamentalAnalyzer"
        if fund_agent_name in pending["results"]:
            scores["fundamental"] = {
                "score": pending["results"][fund_agent_name]["score"],
                "details": pending["results"][fund_agent_name]["details"],
                "confidence": 0.7,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add sentiment score
        sent_agent_name = self.sentiment_agent.name if self.sentiment_agent else "SentimentAnalyzer"
        if sent_agent_name in pending["results"]:
            scores["sentiment"] = {
                "score": pending["results"][sent_agent_name]["score"],
                "details": pending["results"][sent_agent_name]["details"],
                "confidence": 0.6,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add timeframe score
        if pending["timeframe_result"]:
            scores["timeframe"] = {
                "score": pending["timeframe_result"].get("weighted_score", 0.5),
                "alignment": pending["timeframe_result"].get("alignment", 0.5),
                "consensus": pending["timeframe_result"].get("consensus", {}),
                "details": pending["timeframe_result"],
                "confidence": 0.7,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add risk score (placeholder - would come from risk manager)
        scores["risk"] = {
            "score": 0.7,
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to weighted score engine
        await self.send_message(AgentMessage(
            sender=self.name,
            receiver=self.score_engine.name,
            message_type="combine_scores",
            content={
                "analysis_id": analysis_id,
                "scores": scores,
                "regime": pending["regime"]["regime"] if pending["regime"] else "unknown"
            },
            priority=4,
            requires_response=True
        ))
        
        logging.info(f"🧮 Requested score combination for {analysis_id}")
    
    async def _finalize_analysis(self, analysis_id: str):
        """
        Finalize analysis and send to risk management
        """
        pending = self.pending_analyses[analysis_id]
        score_result = pending["score_result"]
        
        # Create final analysis result
        analysis_result = {
            "analysis_id": analysis_id,
            "symbol": pending["symbol"],
            "final_score": score_result.get("final_score", 0.5),
            "risk_adjusted_score": score_result.get("risk_adjusted_score", 0.5),
            "confidence": score_result.get("confidence", 0.5),
            "action": score_result.get("recommendation", {}).get("action", "WATCH"),
            "action_strength": score_result.get("recommendation", {}).get("strength", 2),
            "reasons": score_result.get("recommendation", {}).get("reasons", []),
            "concerns": score_result.get("recommendation", {}).get("concerns", []),
            "breakdown": {
                "individual_scores": score_result.get("individual_scores", {}),
                "weights_used": score_result.get("weights_used", {}),
                "quality_scores": score_result.get("quality_scores", {})
            },
            "regime": pending["regime"],
            "timeframe_analysis": pending["timeframe_result"],
            "original_signal": pending["signal"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result
        self.analysis_results[analysis_id] = analysis_result
        pending["status"] = "complete"
        
        # Send to risk management
        await self.send_message(AgentMessage(
            sender=self.name,
            receiver="RiskManager",
            message_type="analysis_complete",
            content=analysis_result,
            priority=4
        ))
        
        # Clean up pending
        del self.pending_analyses[analysis_id]
        
        logging.info(f"✅ Analysis complete for {pending['symbol']}: {analysis_result['action']} "
                    f"(score={analysis_result['final_score']:.2f}, conf={analysis_result['confidence']:.2f})")
    
    async def _analysis_timeout(self, analysis_id: str):
        """
        Handle timeout for incomplete analyses
        """
        await asyncio.sleep(self.analysis_timeout)
        
        if analysis_id in self.pending_analyses:
            pending = self.pending_analyses[analysis_id]
            logging.warning(f"⚠️ Analysis {analysis_id} timed out after {self.analysis_timeout}s")
            
            # If we have timeframe result, combine what we have
            if pending.get("timeframe_result"):
                logging.info(f"🔄 Using partial results for {analysis_id}")
                await self._combine_all_scores(analysis_id)
            elif pending.get("results"):
                # Force completion with whatever we have
                logging.info(f"⚠️ Forcing completion with available results for {analysis_id}")
                await self._finalize_analysis(analysis_id)
            else:
                # No results at all, discard
                logging.error(f"❌ No results received for {analysis_id}, discarding")
                del self.pending_analyses[analysis_id]
    
    def register_analysis_agent(self, agent, agent_type: str):
        """Register an analysis agent"""
        if agent_type == "technical":
            self.technical_agent = agent
        elif agent_type == "fundamental":
            self.fundamental_agent = agent
        elif agent_type == "sentiment":
            self.sentiment_agent = agent
        elif agent_type == "regime":
            self.regime_detector = agent
        
        logging.info(f"📝 Registered {agent_type} agent: {agent.name}")
    
    def get_pending_count(self) -> int:
        """Get number of pending analyses"""
        return len(self.pending_analyses)
    
    def get_analysis_result(self, analysis_id: str) -> Optional[Dict]:
        """Get a specific analysis result"""
        return self.analysis_results.get(analysis_id)
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check"""
        base_health = super().health_check()
        base_health.update({
            "pending_analyses": len(self.pending_analyses),
            "completed_analyses": len(self.analysis_results),
            "timeout_setting": self.analysis_timeout,
            "max_concurrent": self.max_concurrent_analyses
        })
        return base_health