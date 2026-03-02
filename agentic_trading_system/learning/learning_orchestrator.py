"""
Learning Orchestrator - Main orchestrator for all learning components
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from utils.logger import logger as  logging
from agents.base_agent import BaseAgent, AgentMessage

# Import all learning components
from learning.feature_store import FeatureStore
from learning.attribution_engine import AttributionEngine
from learning.config_updater import ConfigUpdater
from learning.forward_tester import ForwardTester
from learning.models.weight_optimizer import WeightOptimizer
from learning.models.genetic_algorithm import GeneticAlgorithm
from learning.models.reinforcement_learning import ReinforcementLearning
from learning.models.ensemble_model import EnsembleModel
from learning.backtester.simulation_engine import SimulationEngine
from learning.backtester.monte_carlo import MonteCarlo
from learning.backtester.walk_forward import WalkForward

class LearningOrchestrator(BaseAgent):
    """
    Learning Orchestrator - Main orchestrator for all learning components
    
    Responsibilities:
    - Coordinate all learning activities
    - Schedule retraining
    - Monitor model performance
    - Manage learning workflows
    - Integrate with other agents
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Learning and adaptation orchestrator",
            config=config
        )
        
        # Initialize components
        self.feature_store = FeatureStore(config.get("feature_store_config", {}))
        self.attribution = AttributionEngine(config.get("attribution_config", {}))
        self.config_updater = ConfigUpdater(config.get("config_updater_config", {}))
        self.forward_tester = ForwardTester(config.get("forward_tester_config", {}))
        
        # Initialize models
        self.weight_optimizer = WeightOptimizer(config.get("weight_optimizer_config", {}))
        self.genetic_algorithm = GeneticAlgorithm(config.get("genetic_algorithm_config", {}))
        self.reinforcement_learning = ReinforcementLearning(config.get("rl_config", {}))
        self.ensemble_model = EnsembleModel(config.get("ensemble_config", {}))
        
        # Initialize backtester components
        self.simulation_engine = SimulationEngine(config.get("simulation_config", {}))
        self.monte_carlo = MonteCarlo(config.get("monte_carlo_config", {}))
        self.walk_forward = WalkForward(config.get("walk_forward_config", {}))
        
        # Learning state
        self.learning_enabled = config.get("learning_enabled", True)
        self.retrain_interval = config.get("retrain_interval_hours", 24)
        self.last_retrain = None
        
        # Performance tracking
        self.model_performance = {}
        self.learning_history = []
        
        # Start background tasks
        self._start_background_tasks()
        
        logging.info(f"✅ LearningOrchestrator initialized")
    
    def _start_background_tasks(self):
        """Start background learning tasks"""
        async def learning_loop():
            while self.learning_enabled:
                await asyncio.sleep(self.retrain_interval * 3600)
                await self.retrain_models()
        
        asyncio.create_task(learning_loop())
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process learning-related requests
        """
        msg_type = message.message_type
        
        if msg_type == "learn_from_trade":
            # Learn from completed trade
            trade_data = message.content
            await self.learn_from_trade(trade_data)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="learning_acknowledged",
                content={"status": "processed"}
            )
        
        elif msg_type == "optimize_weights":
            # Optimize signal weights
            signal_performance = message.content.get("signal_performance", {})
            weights = self.weight_optimizer.update_weights(signal_performance)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="optimized_weights",
                content={"weights": weights}
            )
        
        elif msg_type == "run_backtest":
            # Run backtest
            strategy = message.content.get("strategy")
            data = message.content.get("data")
            symbol = message.content.get("symbol")
            params = message.content.get("params", {})
            
            result = self.simulation_engine.run_backtest(strategy, data, symbol, params)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="backtest_result",
                content=result
            )
        
        elif msg_type == "get_attribution":
            # Get attribution report
            days = message.content.get("days", 30)
            report = self.attribution.generate_report(days)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="attribution_report",
                content=report
            )
        
        elif msg_type == "update_config":
            # Update configuration based on learning
            performance = message.content.get("performance", {})
            self.config_updater.update_trigger_thresholds(performance.get("signals", {}))
            self.config_updater.update_analysis_weights(performance.get("signals", {}))
            self.config_updater.update_risk_parameters(performance.get("risk", {}))
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="config_updated",
                content={"status": "success"}
            )
        
        elif msg_type == "get_learning_status":
            # Get learning status
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="learning_status",
                content=self.get_status()
            )
        
        return None
    
    async def learn_from_trade(self, trade_data: Dict[str, Any]):
        """
        Learn from a completed trade
        """
        trade_id = trade_data.get("trade_id")
        signals = trade_data.get("signals", [])
        outcome = trade_data.get("outcome")
        pnl = trade_data.get("pnl", 0)
        symbol = trade_data.get("symbol")
        
        # Record attribution
        self.attribution.record_trade_attribution(
            trade_id, symbol, signals, outcome, pnl
        )
        
        # Update weight optimizer
        signal_performance = {}
        for signal in signals:
            name = signal.get("name", signal.get("type", "unknown"))
            if name not in signal_performance:
                signal_performance[name] = []
            
            signal_performance[name].append({
                "outcome": outcome,
                "confidence": signal.get("confidence", 0.5)
            })
        
        self.weight_optimizer.update_weights(signal_performance)
        
        # Store in learning history
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "trade_learning",
            "trade_id": trade_id,
            "outcome": outcome,
            "pnl": pnl
        })
        
        logging.info(f"📚 Learned from trade {trade_id}: {outcome}")
    
    async def retrain_models(self):
        """
        Retrain all models
        """
        logging.info("🔄 Starting model retraining...")
        
        # Get latest data from feature store
        symbols = self.feature_store.get_all_symbols()
        
        if not symbols:
            logging.warning("No data available for retraining")
            return
        
        # Prepare training data
        training_data = self.feature_store.prepare_training_data(
            symbols=symbols,
            target_column="next_return",
            feature_columns=self._get_feature_columns(),
            lookback_days=60
        )
        
        if not training_data:
            logging.warning("Insufficient training data")
            return
        
        # Train ensemble model
        # This would require actual models to be added to ensemble
        # self.ensemble_model.fit(
        #     training_data["X_train"],
        #     training_data["y_train"],
        #     training_data["X_test"],
        #     training_data["y_test"]
        # )
        
        # Update model performance
        self.model_performance["last_retrain"] = datetime.now().isoformat()
        self.model_performance["training_samples"] = training_data.get("train_samples", 0)
        self.model_performance["test_samples"] = training_data.get("test_samples", 0)
        
        self.last_retrain = datetime.now()
        
        logging.info(f"✅ Model retraining complete")
    
    def _get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns for training
        """
        # This would come from feature store registration
        return ["rsi", "macd", "volume_ratio", "volatility", "trend_strength"]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get learning orchestrator status
        """
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "learning_enabled": self.learning_enabled,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "model_performance": self.model_performance,
            "learning_history_size": len(self.learning_history),
            "feature_stats": self.feature_store.get_stats(),
            "attribution_stats": len(self.attribution.trade_attributions),
            "components": {
                "feature_store": True,
                "attribution": True,
                "config_updater": True,
                "weight_optimizer": True,
                "simulation_engine": True
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        base_health = await super().health_check()
        
        base_health.update({
            "learning_enabled": self.learning_enabled,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "models_loaded": len(self.ensemble_model.models) if hasattr(self.ensemble_model, 'models') else 0
        })
        
        return base_health