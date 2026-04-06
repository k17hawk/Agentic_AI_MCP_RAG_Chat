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
from typing import Dict, List, Optional, Any
import json
import logging
from logging.handlers import RotatingFileHandler
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import settings
from agentic_trading_system.config.settings import Settings, get_config

# Import existing modules
from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator
from agentic_trading_system.discovery.search_aggregator import SearchAggregator
from agentic_trading_system.prefilter.quality_gates import QualityGates
from agentic_trading_system.analysis.analysis_orchestrator import AnalysisOrchestrator
from agentic_trading_system.risk.risk_manager import RiskManager
from agentic_trading_system.portfolio.portfolio_optimizer import PortfolioOptimizer
from agentic_trading_system.hitl.alert_manager import AlertManager
from agentic_trading_system.execution.execution_engine import ExecutionEngine


class AgenticTradingSystem:
    """Main trading system orchestrator with continuous learning"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the trading system with all components"""
        
        # ✅ Initialize settings FIRST
        self.settings = Settings()
        
        # ✅ Setup logging SECOND (needs settings)
        self.logger = self._setup_logging()
        
        # ✅ Initialize other attributes
        self.config_dir = config_dir
        self.start_time = datetime.now()
        self.running = True
        self.last_metrics_update = datetime.now()
        self.last_learning_run = datetime.now()
        
        # ✅ Initialize components THIRD (needs logger and settings)
        self._init_components()
        
        # ✅ Load learning config LAST
        self.learning_config = self._load_or_create_learning_config()
    


    def _load_or_create_learning_config(self) -> dict:
        """Load learning config or create default one"""
        learning_config_path = self.settings.config_dir / "learning_config.yaml"
        
        # Default configuration
        default_config = {
            'mode': 'continuous',
            'interval_hours': 24,
            'min_trades_for_learning': 10,
            'enable_weight_optimization': True,
            'enable_parameter_tuning': True,
            'enable_model_retraining': False,
            'max_weight_change_percent': 15.0,
            'min_improvement_threshold': 0.05,
            'learning_rate': 0.1,
            'genetic_generations': 10,
            'population_size': 20
        }
        
        try:
            if learning_config_path.exists():
                import yaml
                with open(learning_config_path) as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        default_config.update(user_config)
                        self.logger.info(f"✅ Loaded learning config from {learning_config_path}")
                    else:
                        self.logger.warning(f"Learning config file exists but is empty, using defaults")
                        # Create default config file
                        self._create_default_learning_config(learning_config_path, default_config)
            else:
                self.logger.warning(f"Learning config not found at {learning_config_path}, creating default")
                self._create_default_learning_config(learning_config_path, default_config)
        except Exception as e:
            self.logger.warning(f"Error loading learning config: {e}, using defaults")
        
        return default_config
    
    def _create_default_learning_config(self, path: Path, config: dict):
        """Create default learning config file"""
        try:
            import yaml
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"📝 Created default learning config at {path}")
        except Exception as e:
            self.logger.warning(f"Could not create learning config file: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with rotation"""
        # Get logging config from settings
        log_config = self.settings.get_logging_config()
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        logger = logging.getLogger("TradingSystem")
        log_level = log_config.get('level', 'INFO') if log_config else 'INFO'
        logger.setLevel(getattr(logging, log_level))
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        try:
            file_handler = RotatingFileHandler(
                'logs/trading_system.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(console_format)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
        
        return logger
    
    def _init_components(self):
        """Initialize all trading system components"""
        
        # Initialize Trigger Orchestrator
        try:
            # Get trigger config from settings
            trigger_config = self.settings.get_triggers()
            
            # Create trigger orchestrator
            self.trigger_orchestrator = TriggerOrchestrator(
                memory_agent=None,
                message_bus=None,
                max_concurrent_triggers=trigger_config.get('max_concurrent', 5) if trigger_config else 5
            )
            self.logger.info("✅ TriggerOrchestrator initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize TriggerOrchestrator: {e}")
            self.trigger_orchestrator = None
        
        # Initialize other components
        components = {
            'search_aggregator': SearchAggregator,
            'quality_gates': QualityGates,
            'analysis_orchestrator': AnalysisOrchestrator,
            'risk_manager': RiskManager,
            'portfolio_optimizer': PortfolioOptimizer,
            'alert_manager': AlertManager,
            'execution_engine': ExecutionEngine
        }
        
        for name, component_class in components.items():
            try:
                # Try to initialize with name and config
                instance = component_class(
                    name=name,
                    config={'enabled': True, 'log_level': 'INFO'}
                )
                setattr(self, name, instance)
                self.logger.info(f"✅ {name} initialized")
            except TypeError as e:
                # Try without arguments
                if "required positional arguments" in str(e):
                    try:
                        instance = component_class()
                        setattr(self, name, instance)
                        self.logger.info(f"✅ {name} initialized (no args)")
                    except Exception as e2:
                        self.logger.debug(f"Could not initialize {name} without args: {e2}")
                        setattr(self, name, None)
                else:
                    self.logger.debug(f"Could not initialize {name}: {e}")
                    setattr(self, name, None)
            except Exception as e:
                self.logger.debug(f"Could not initialize {name}: {e}")
                setattr(self, name, None)
        
        # Create memory stub
        self.memory = None
        
        # Count successfully initialized components
        initialized = sum(1 for name in components if getattr(self, name, None) is not None)
        self.logger.info(f"✅ Component initialization completed ({initialized}/{len(components)} components ready)")
    
    async def process_market_cycle(self):
        """Execute one complete market analysis cycle"""
        cycle_start = datetime.now()
        self.logger.info("🔄 Starting market analysis cycle")
        
        try:
            # Check if trigger orchestrator is available
            if not self.trigger_orchestrator:
                self.logger.warning("Trigger orchestrator not available, skipping cycle")
                return
            
            # Get trigger status
            try:
                status = self.trigger_orchestrator.get_status()
                self.logger.info(f"📊 Trigger status: {status.get('active_triggers', 0)} active triggers, "
                               f"{status.get('total_events', 0)} total events")
            except Exception as e:
                self.logger.debug(f"Could not get trigger status: {e}")
            
            # Get real triggers from orchestrator if available
            # Note: TriggerOrchestrator doesn't have detect_triggers method
            # So we'll use the event queue or just log status
            
            # For now, just save status as artifact
            await self._save_cycle_artifacts({
                'timestamp': datetime.now().isoformat(),
                'orchestrator_status': status if 'status' in locals() else {},
                'status': 'running'
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
            # Get configs from settings
            risk_config = self.settings.get_risk_config()
            analysis_weights = self.settings.get_analysis_weights()
            triggers = self.settings.get_triggers()
            
            self.logger.info("📊 Updating system metrics")
            if risk_config:
                self.logger.debug(f"  Risk: Max Drawdown {risk_config.get('max_drawdown', 0.2)*100:.0f}%")
            if analysis_weights:
                weights = analysis_weights.get('regime_weights', {}).get('default', {})
                self.logger.debug(f"  Weights: T={weights.get('technical', 33):.0f}, "
                                f"S={weights.get('sentiment', 33):.0f}, "
                                f"F={weights.get('fundamental', 34):.0f}")
            if triggers:
                self.logger.debug(f"  Triggers: {len(triggers.get('triggers', {}))} configured")
            
            self.last_metrics_update = now
    
    async def _save_cycle_artifacts(self, cycle_data: dict):
        """Save cycle artifacts for learning and debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_path = Path(f"discovery_outputs/{timestamp}")
            artifact_path.mkdir(parents=True, exist_ok=True)
            
            # Save data
            with open(artifact_path / "cycle_data.json", 'w') as f:
                json.dump(cycle_data, f, indent=2, default=str)
            
            self.logger.debug(f"📁 Saved artifacts to {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save artifacts: {e}")
    
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
        self.logger.info("🧠 Starting learning cycle")
        
        try:
            # Get current configuration for analysis
            current_weights = self.settings.get_analysis_weights()
            risk_config = self.settings.get_risk_config()
            
            # Analyze current performance
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'current_weights': current_weights.get('regime_weights', {}).get('default', {}) if current_weights else {},
                'risk_settings': {
                    'max_drawdown': risk_config.get('max_drawdown', 0.2) if risk_config else 0.2,
                    'position_size': risk_config.get('position_size', 0.02) if risk_config else 0.02
                } if risk_config else {},
                'recommendations': []
            }
            
            # Generate simple recommendations based on current config
            if current_weights:
                weights = current_weights.get('regime_weights', {}).get('default', {})
                technical = weights.get('technical', 33)
                sentiment = weights.get('sentiment', 33)
                fundamental = weights.get('fundamental', 34)
                
                # Check if weights are balanced (all around 33%)
                if abs(technical - sentiment) < 10 and abs(technical - fundamental) < 10:
                    analysis_results['recommendations'].append({
                        'type': 'weight_adjustment',
                        'message': 'Weights are very balanced. Consider optimizing based on recent performance.',
                        'priority': 'low'
                    })
            
            # Save learning results
            await self._save_learning_results(analysis_results)
            
            self.logger.info(f"✅ Learning cycle completed. Recommendations: {len(analysis_results['recommendations'])}")
            self.last_learning_run = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {e}", exc_info=True)
    
    async def _save_learning_results(self, results: dict):
        """Save learning cycle results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = Path(f"learning_results/{timestamp}")
            results_path.mkdir(parents=True, exist_ok=True)
            
            with open(results_path / "learning_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"📁 Saved learning results to {results_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save learning results: {e}")
    
    async def run_continuous(self):
        """Main continuous execution loop"""
        self.logger.info("🚀 Starting Agentic Trading System")
        self.logger.info(f"📅 Start time: {self.start_time}")
        
        # Print configuration summary
        self._print_config_summary()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start the trigger orchestrator if available
        orchestrator_task = None
        if self.trigger_orchestrator:
            try:
                # Run orchestrator in background
                orchestrator_task = asyncio.create_task(
                    self.trigger_orchestrator.start()
                )
                self.logger.info("✅ Trigger orchestrator started")
            except Exception as e:
                self.logger.error(f"Failed to start orchestrator: {e}")
        
        # Main loop counter
        cycle_count = 0
        
        while self.running:
            try:
                # Execute market cycle
                await self.process_market_cycle()
                cycle_count += 1
                
                # Log status every 10 cycles
                if cycle_count % 10 == 0:
                    self.logger.info(f"📊 System status: {cycle_count} cycles completed, "
                                   f"uptime: {datetime.now() - self.start_time}")
                
                # Check if learning cycle is due
                hours_since_learning = (datetime.now() - self.last_learning_run).total_seconds() / 3600
                if hours_since_learning >= self.learning_config.get('interval_hours', 24):
                    await self.run_learning_cycle()
                
                # Sleep for 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                self.logger.info("System shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
        
        # Cleanup
        if orchestrator_task:
            orchestrator_task.cancel()
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
        
        await self._graceful_shutdown()
    
    def _print_config_summary(self):
        """Print configuration summary"""
        self.logger.info("📋 Configuration Summary:")
        
        # Trigger config
        triggers = self.settings.get_triggers()
        if triggers:
            trigger_count = len(triggers.get('triggers', {}))
            self.logger.info(f"  - Triggers: {trigger_count} configured")
        else:
            self.logger.info(f"  - Triggers: Using defaults")
        
        # Risk config
        risk = self.settings.get_risk_config()
        if risk:
            self.logger.info(f"  - Risk: Max Drawdown {risk.get('max_drawdown', 0.2)*100:.0f}%, "
                           f"Position Size {risk.get('position_size', 0.02)*100:.1f}%")
        else:
            self.logger.info(f"  - Risk: Using defaults")
        
        # Analysis weights
        weights = self.settings.get_analysis_weights()
        if weights and 'regime_weights' in weights:
            default_weights = weights['regime_weights'].get('default', {})
            if default_weights:
                self.logger.info(f"  - Analysis Weights: T={default_weights.get('technical', 33):.0f}%, "
                               f"S={default_weights.get('sentiment', 33):.0f}%, "
                               f"F={default_weights.get('fundamental', 34):.0f}%")
            else:
                self.logger.info(f"  - Analysis Weights: Using defaults (33/33/34)")
        else:
            self.logger.info(f"  - Analysis Weights: Using defaults (33/33/34)")
        
        # Learning config
        self.logger.info(f"  - Learning: {self.learning_config.get('mode', 'continuous')} mode, "
                        f"every {self.learning_config.get('interval_hours', 24)} hours")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _graceful_shutdown(self):
        """Gracefully shutdown all components"""
        self.logger.info("🛑 Shutting down trading system...")
        
        # Stop trigger orchestrator
        if self.trigger_orchestrator:
            try:
                await self.trigger_orchestrator.stop()
                self.logger.info("Trigger orchestrator stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping orchestrator: {e}")
        
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
    system = AgenticTradingSystem(config_dir=args.config)
    
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

