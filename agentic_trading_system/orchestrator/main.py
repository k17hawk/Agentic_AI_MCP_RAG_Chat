#!/usr/bin/env python3
"""
Agentic AI Trading System - Main Orchestrator
with Integrated Continuous Learning Module
"""

import asyncio
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
import logging
from logging.handlers import RotatingFileHandler
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing modules
from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator
from agentic_trading_system.discovery.search_aggregator import SearchAggregator
from agentic_trading_system.prefilter.quality_gates import QualityGates
from agentic_trading_system.analysis.analysis_orchestrator import AnalysisOrchestrator
from agentic_trading_system.risk.risk_manager import RiskManager
from agentic_trading_system.portfolio.portfolio_optimizer import PortfolioOptimizer
from agentic_trading_system.hitl.alert_manager import AlertManager
from agentic_trading_system.execution.execution_engine import ExecutionEngine

# Import learning modules
from agentic_trading_system.learning.learning_orchestrator import LearningOrchestrator
from agentic_trading_system.learning.continuous_learner import ContinuousLearningService
from agentic_trading_system.learning.learning_trade_logger import TradeOutcomeLogger
from agentic_trading_system.learning.performance_analyzer import PerformanceAnalyzer
import os
# Import memory modules
from agentic_trading_system.memory.short_term.redis_client import RedisClient
from agentic_trading_system.memory.query_engine import  QueryEngine  as UnifiedMemoryQuery

# Import analytics
from analytics.metrics_engine import MetricsEngine

class AgenticTradingSystem:
    """Main trading system orchestrator with continuous learning"""
    
    def __init__(self, config_path: str = "config"):
        """Initialize the trading system with all components"""
        
        # Load configuration
        self.config_path = Path(config_path)
        self.settings = self._load_yaml("settings.yaml")
        self.learning_config = self._load_yaml("learning_config.yaml")
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize state
        self.running = True
        self.start_time = datetime.now()
        self.last_learning_run = datetime.now()
        self.last_metrics_update = datetime.now()
        
        # Initialize components
        self._init_components()
        
        # Initialize learning components
        self._init_learning()
        
        self.logger.info("🚀 Agentic Trading System Initialized")
        self.logger.info(f"📊 Learning Mode: {self.learning_config.get('mode', 'continuous')}")
        self.logger.info(f"🔄 Learning Interval: {self.learning_config.get('interval_hours', 24)} hours")
    
    def _load_yaml(self, filename: str) -> dict:
        """Load YAML configuration file"""
        try:
            with open(self.config_path / filename) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {filename} not found, using defaults")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with rotation"""
        logger = logging.getLogger("TradingSystem")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/trading_system.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_components(self):
        """Initialize all trading system components"""
        try:
            # Database connections
            self.redis_client = RedisClient(
                host=self.settings.get('redis_host', 'localhost'),
                port=self.settings.get('redis_port', 6379)
            )
            
            self.db_engine = create_engine(
                f"postgresql://{self.settings.get('db_user')}:{self.settings.get('db_password')}@"
                f"{self.settings.get('db_host', 'localhost')}/{self.settings.get('db_name', 'trading')}"
            )
            self.db_session = sessionmaker(bind=self.db_engine)
            
            # Memory layer
            self.memory = UnifiedMemoryQuery(
                redis_client=self.redis_client,
                db_session=self.db_session
            )
            
            # Core trading components
            self.trigger_orchestrator = TriggerOrchestrator()
            self.search_aggregator = SearchAggregator()
            self.quality_gates = QualityGates()
            self.analysis_orchestrator = AnalysisOrchestrator()
            self.risk_manager = RiskManager()
            self.portfolio_optimizer = PortfolioOptimizer()
            self.alert_manager = AlertManager()
            self.execution_engine = ExecutionEngine()
            
            # Analytics
            self.metrics_engine = MetricsEngine(self.memory)
            
            self.logger.info("✅ All trading components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _init_learning(self):
        """Initialize learning module components"""
        try:
            self.learning_orchestrator = LearningOrchestrator()
            self.trade_logger = TradeOutcomeLogger("discovery_outputs/")
            self.performance_analyzer = PerformanceAnalyzer()
            
            # Start continuous learning service if enabled
            if self.learning_config.get('mode') == 'continuous':
                self.learning_service = ContinuousLearningService(
                    config_path=str(self.config_path / "learning_config.yaml")
                )
                self.learning_thread = threading.Thread(
                    target=self._run_learning_service,
                    daemon=True
                )
                self.learning_thread.start()
                self.logger.info("✅ Continuous learning service started")
            
        except Exception as e:
            self.logger.warning(f"Learning module initialization failed: {e}")
            self.logger.warning("System will run without learning capabilities")
            self.learning_orchestrator = None
    
    def _run_learning_service(self):
        """Run learning service in background thread"""
        if hasattr(self, 'learning_service'):
            self.learning_service.run_continuous()
    
    async def process_market_cycle(self):
        """Execute one complete market analysis cycle"""
        cycle_start = datetime.now()
        self.logger.info("🔄 Starting market analysis cycle")
        
        try:
            # Stage 1: Trigger Detection
            triggers = await self.trigger_orchestrator.detect_triggers()
            if not triggers:
                self.logger.debug("No triggers detected")
                return
            
            self.logger.info(f"🎯 Detected {len(triggers)} triggers")
            
            # Stage 2: Discovery & Data Aggregation
            discovered_assets = await self.search_aggregator.aggregate(triggers)
            if not discovered_assets:
                self.logger.warning("No assets discovered from triggers")
                return
            
            # Stage 3: Prefilter (Quality Gates)
            filtered_assets = await self.quality_gates.filter(discovered_assets)
            if not filtered_assets:
                self.logger.info("All assets filtered out by quality gates")
                return
            
            # Stage 4: Multi-dimensional Analysis
            analyzed_assets = await self.analysis_orchestrator.analyze(filtered_assets)
            
            # Stage 5: Risk Assessment
            risk_assessed = await self.risk_manager.assess(analyzed_assets)
            
            # Stage 6: Portfolio Optimization
            portfolio_recommendations = await self.portfolio_optimizer.optimize(risk_assessed)
            
            # Stage 7: Human-in-the-Loop Approval
            approved_trades = await self.alert_manager.request_approval(portfolio_recommendations)
            
            # Stage 8: Execution
            if approved_trades:
                executed_trades = await self.execution_engine.execute(approved_trades)
                self.logger.info(f"✅ Executed {len(executed_trades)} trades")
                
                # Log trades for learning
                if self.learning_orchestrator:
                    for trade in executed_trades:
                        self.trade_logger.log_trade(trade)
            else:
                self.logger.info("No trades approved for execution")
            
            # Stage 9: Update Metrics
            await self._update_system_metrics()
            
            # Save cycle artifacts
            await self._save_cycle_artifacts({
                'triggers': triggers,
                'discovered': discovered_assets,
                'filtered': filtered_assets,
                'analyzed': analyzed_assets,
                'recommendations': portfolio_recommendations,
                'executed': approved_trades
            })
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"✅ Cycle completed in {cycle_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in market cycle: {e}", exc_info=True)
            await self._handle_cycle_error(e)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        now = datetime.now()
        
        # Update metrics every hour
        if (now - self.last_metrics_update).total_seconds() > 3600:
            metrics = await self.metrics_engine.calculate_all_metrics()
            
            # Store in memory
            self.redis_client.set("system:metrics", json.dumps(metrics))
            
            # Log key metrics
            self.logger.info(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"📊 Win Rate: {metrics.get('win_rate', 0):.1%}")
            self.logger.info(f"📊 Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
            
            self.last_metrics_update = now
    
    async def _save_cycle_artifacts(self, cycle_data: dict):
        """Save cycle artifacts for learning and debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = Path(f"discovery_outputs/{timestamp}")
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Save each stage output
        for stage, data in cycle_data.items():
            if data:
                with open(artifact_path / f"{stage}.json", 'w') as f:
                    json.dump(data, f, indent=2, default=str)
    
    async def _handle_cycle_error(self, error: Exception):
        """Handle errors in market cycle gracefully"""
        self.logger.error(f"Market cycle failed: {error}")
        
        # Check circuit breaker
        if await self._check_circuit_breaker():
            self.logger.warning("Circuit breaker triggered - pausing for 5 minutes")
            await asyncio.sleep(300)
    
    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trigger"""
        # Track consecutive failures
        if not hasattr(self, '_failure_count'):
            self._failure_count = 0
        
        self._failure_count += 1
        
        if self._failure_count >= 5:
            self._failure_count = 0
            return True
        
        return False
    
    async def run_learning_cycle(self):
        """Execute learning cycle to improve system performance"""
        if not self.learning_orchestrator:
            return
        
        self.logger.info("🧠 Starting learning cycle")
        
        try:
            # Get recent trades
            recent_trades = await self.memory.get_recent_trades(days=30)
            
            if len(recent_trades) < self.learning_config.get('min_trades_for_learning', 10):
                self.logger.info(f"Insufficient trades for learning (have {len(recent_trades)}, need 10)")
                return
            
            # Run learning tasks
            learning_results = {}
            
            # 1. Update analysis weights
            if self.learning_config.get('enable_weight_optimization', True):
                new_weights = await self.learning_orchestrator.optimize_weights(recent_trades)
                if new_weights:
                    learning_results['weights'] = new_weights
                    self.logger.info("✅ Analysis weights updated")
            
            # 2. Tune parameters
            if self.learning_config.get('enable_parameter_tuning', True):
                new_params = await self.learning_orchestrator.tune_parameters(recent_trades)
                if new_params:
                    learning_results['parameters'] = new_params
                    self.logger.info("✅ System parameters tuned")
            
            # 3. Update ML models
            if self.learning_config.get('enable_model_retraining', False):
                model_metrics = await self.learning_orchestrator.retrain_models(recent_trades)
                if model_metrics:
                    learning_results['models'] = model_metrics
                    self.logger.info("✅ ML models retrained")
            
            # Log learning results
            if learning_results:
                self.logger.info(f"🎓 Learning cycle completed: {list(learning_results.keys())}")
                await self._save_learning_results(learning_results)
            else:
                self.logger.info("No learning updates applied")
            
            self.last_learning_run = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {e}", exc_info=True)
    
    async def _save_learning_results(self, results: dict):
        """Save learning cycle results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"learning_results/{timestamp}")
        results_path.mkdir(parents=True, exist_ok=True)
        
        with open(results_path / "learning_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    async def run_continuous(self):
        """Main continuous execution loop"""
        self.logger.info("🚀 Starting Agentic Trading System")
        self.logger.info(f"📅 Start time: {self.start_time}")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Main loop counter
        cycle_count = 0
        
        while self.running:
            try:
                # Execute market cycle
                await self.process_market_cycle()
                cycle_count += 1
                
                # Check if learning cycle is due
                hours_since_learning = (datetime.now() - self.last_learning_run).total_seconds() / 3600
                if hours_since_learning >= self.learning_config.get('interval_hours', 24):
                    await self.run_learning_cycle()
                
                # Dynamic sleep based on market hours
                sleep_seconds = await self._calculate_sleep_interval()
                await asyncio.sleep(sleep_seconds)
                
            except asyncio.CancelledError:
                self.logger.info("System shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
        
        await self._graceful_shutdown()
    
    async def _calculate_sleep_interval(self) -> int:
        """Calculate sleep interval based on market hours"""
        now = datetime.now()
        
        # Check if market is open (simplified - adjust for your markets)
        is_market_hours = 9 <= now.hour <= 16 and now.weekday() < 5
        
        if is_market_hours:
            # During market hours: check every 60 seconds
            return 60
        else:
            # Outside market hours: check every 5 minutes
            return 300
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _graceful_shutdown(self):
        """Gracefully shutdown all components"""
        self.logger.info("🛑 Shutting down trading system...")
        
        # Close database connections
        if hasattr(self, 'db_engine'):
            self.db_engine.dispose()
        
        # Close Redis connection
        if hasattr(self, 'redis_client'):
            self.redis_client.close()
        
        # Save final metrics
        await self._update_system_metrics()
        
        # Stop learning service if running
        if hasattr(self, 'learning_service'):
            self.logger.info("Stopping learning service...")
        
        uptime = datetime.now() - self.start_time
        self.logger.info(f"✅ System shutdown complete. Uptime: {uptime}")
    
    def run_sync(self):
        """Synchronous wrapper for running the system"""
        asyncio.run(self.run_continuous())

def main():
    """Entry point for the trading system"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Agentic AI Trading System")
    parser.add_argument(
        '--mode',
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config",
        help="Configuration directory path"
    )
    parser.add_argument(
        '--learning-interval',
        type=int,
        default=24,
        help="Learning interval in hours (default: 24)"
    )
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.mode != 'paper':
        os.environ['TRADING_MODE'] = args.mode
    
    # Create and run system
    system = AgenticTradingSystem(config_path=args.config)
    
    # Override learning interval if specified
    if args.learning_interval:
        system.learning_config['interval_hours'] = args.learning_interval
    
    # Run the system
    try:
        system.run_sync()
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()