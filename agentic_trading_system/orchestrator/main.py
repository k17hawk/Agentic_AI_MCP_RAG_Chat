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
from agentic_trading_system.triggers.base_trigger import TriggerEvent, TriggerPriority
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
        
        # Initialize settings FIRST
        self.settings = Settings()
        
        # Setup logging SECOND (needs settings)
        self.logger = self._setup_logging()
        
        # Initialize other attributes
        self.config_dir = config_dir
        self.start_time = datetime.now()
        self.running = True
        self.last_metrics_update = datetime.now()
        self.last_learning_run = datetime.now()
        
        # Initialize components as None (will be initialized in _init_components)
        self.trigger_orchestrator = None
        self.search_aggregator = None
        self.quality_gates = None
        self.analysis_orchestrator = None
        self.risk_manager = None
        self.portfolio_optimizer = None
        self.alert_manager = None
        self.execution_engine = None
        self.memory = None
        self.rejected_logger = None
        self.passed_queue = None
        self.risk_approved_queue = None
        self.timeout_manager = None
        self.order_manager = None
        
        # Pipeline statistics
        self.pipeline_stats = {
            "events_processed": 0,
            "events_passed_quality": 0,
            "events_approved_risk": 0,
            "events_executed": 0,
            "events_rejected": 0,
            "pipeline_errors": 0
        }
        
        # Initialize components synchronously
        self._init_components()
        
        # Load learning config LAST
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
        """Initialize all trading system components synchronously"""
        
        # Initialize Trigger Orchestrator
        try:
            trigger_config = self.settings.get_triggers()
            
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
                # Try initializing with name and config
                instance = component_class(
                    name=name,
                    config={'enabled': True, 'log_level': 'INFO'}
                )
                setattr(self, name, instance)
                self.logger.info(f"✅ {name} initialized")
            except TypeError:
                try:
                    # Try initializing without arguments
                    instance = component_class()
                    setattr(self, name, instance)
                    self.logger.info(f"✅ {name} initialized (no args)")
                except Exception as e2:
                    self.logger.debug(f"Could not initialize {name}: {e2}")
                    setattr(self, name, None)
            except Exception as e:
                self.logger.debug(f"Could not initialize {name}: {e}")
                setattr(self, name, None)
        
        # Get references to sub-components for background tasks
        if self.risk_manager and hasattr(self.risk_manager, 'approved_queue'):
            self.risk_approved_queue = self.risk_manager.approved_queue
        
        if self.alert_manager and hasattr(self.alert_manager, 'timeout_manager'):
            self.timeout_manager = self.alert_manager.timeout_manager
        
        if self.execution_engine and hasattr(self.execution_engine, 'order_manager'):
            self.order_manager = self.execution_engine.order_manager
        
        if self.quality_gates and hasattr(self.quality_gates, 'passed_queue'):
            self.passed_queue = self.quality_gates.passed_queue
        
        # Count components
        component_names = list(components.keys())
        initialized = sum(1 for name in component_names if getattr(self, name, None) is not None)
        self.logger.info(f"✅ Component initialization completed ({initialized}/{len(component_names)} components ready)")
    
    async def _start_background_tasks(self):
        """Start background tasks for components that need them"""
        
        background_components = [
            ('rejected_logger', getattr(self.quality_gates, 'rejected_logger', None) if self.quality_gates else None),
            ('risk_approved_queue', self.risk_approved_queue),
            ('timeout_manager', self.timeout_manager),
            ('order_manager', self.order_manager),
            ('passed_queue', self.passed_queue),
        ]
        
        for name, component in background_components:
            if component:
                # Try each possible startup method
                if hasattr(component, 'ensure_expiry_started'):
                    try:
                        await component.ensure_expiry_started()
                        self.logger.debug(f"Started expiry checker for {name}")
                    except Exception as e:
                        self.logger.debug(f"Could not start expiry checker for {name}: {e}")
                elif hasattr(component, 'ensure_cleanup_started'):
                    try:
                        await component.ensure_cleanup_started()
                        self.logger.debug(f"Started cleanup for {name}")
                    except Exception as e:
                        self.logger.debug(f"Could not start cleanup for {name}: {e}")
                elif hasattr(component, 'ensure_checker_started'):
                    try:
                        await component.ensure_checker_started()
                        self.logger.debug(f"Started checker for {name}")
                    except Exception as e:
                        self.logger.debug(f"Could not start checker for {name}: {e}")
    
    async def _process_trigger_events(self):
        """Process events from trigger orchestrator through the pipeline"""
        
        self.logger.info("🔄 Starting event processor pipeline")
        
        while self.running:
            try:
                # Get events from orchestrator's priority queues
                if not self.trigger_orchestrator or not hasattr(self.trigger_orchestrator, 'priority_queues'):
                    await asyncio.sleep(0.1)
                    continue
                
                if not self.trigger_orchestrator.priority_queues:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check all priority queues for events
                event = None
                event_priority = None
                
                for priority in [TriggerPriority.CRITICAL, TriggerPriority.HIGH, 
                               TriggerPriority.MEDIUM, TriggerPriority.LOW]:
                    queue = self.trigger_orchestrator.priority_queues.get(priority)
                    if queue and not queue.empty():
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=0.1)
                            event_priority = priority
                            break
                        except asyncio.TimeoutError:
                            continue
                
                if event is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process through pipeline
                self.logger.info(f"📨 Processing {event_priority.name if event_priority else 'UNKNOWN'} priority event "
                               f"for {event.symbol} from {event.source_trigger} (confidence: {event.confidence:.2f})")
                
                await self._run_event_pipeline(event)
                
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}", exc_info=True)
                self.pipeline_stats["pipeline_errors"] += 1
                await asyncio.sleep(1)
        
        self.logger.info("🛑 Event processor stopped")
    
    async def _run_event_pipeline(self, event: TriggerEvent):
        """Run event through all processing stages"""
        
        self.logger.debug(f"🔄 Running pipeline for {event.symbol}")
        
        try:
            # Stage 1: Search & Discovery
            search_results = {}
            if self.search_aggregator:
                try:
                    if hasattr(self.search_aggregator, 'process'):
                        search_results = await self.search_aggregator.process(event)
                    elif hasattr(self.search_aggregator, 'search'):
                        search_results = await self.search_aggregator.search(event.symbol)
                    else:
                        search_results = {'signals': [], 'related_news': []}
                    self.logger.debug(f"  ✅ Search aggregator: {len(search_results.get('signals', []))} signals")
                except Exception as e:
                    self.logger.warning(f"Search aggregator error: {e}")
                    search_results = {}
            
            # Stage 2: Quality Gates
            quality_passed = True
            quality_score = 0
            if self.quality_gates:
                try:
                    if hasattr(self.quality_gates, 'evaluate'):
                        quality_result = await self.quality_gates.evaluate(event, search_results)
                    elif hasattr(self.quality_gates, 'check'):
                        quality_result = await self.quality_gates.check(event)
                    else:
                        quality_result = {'passed': True, 'score': event.confidence}
                    
                    quality_passed = quality_result.get('passed', False)
                    quality_score = quality_result.get('score', 0)
                    
                    if not quality_passed:
                        self.logger.info(f"  ❌ Event failed quality gates: {quality_result.get('reason', 'Unknown')}")
                        self.pipeline_stats["events_rejected"] += 1
                        return
                    
                    self.logger.debug(f"  ✅ Quality gates passed: score {quality_score:.2f}")
                    self.pipeline_stats["events_passed_quality"] += 1
                except Exception as e:
                    self.logger.warning(f"Quality gates error: {e}")
                    quality_passed = True
            
            # Stage 3: Analysis
            analysis = {'confidence': event.confidence, 'technical_score': 0.5, 'sentiment_score': 0.5}
            if self.analysis_orchestrator:
                try:
                    if hasattr(self.analysis_orchestrator, 'analyze'):
                        analysis = await self.analysis_orchestrator.analyze(event, search_results)
                    elif hasattr(self.analysis_orchestrator, 'process'):
                        analysis = await self.analysis_orchestrator.process(event)
                    else:
                        analysis = {'confidence': event.confidence, 'signals': []}
                    
                    self.logger.debug(f"  ✅ Analysis complete: confidence {analysis.get('confidence', 0):.2f}")
                except Exception as e:
                    self.logger.warning(f"Analysis error: {e}")
            
            # Stage 4: Risk Management
            risk_approved = True
            position_size = 0.02
            if self.risk_manager:
                try:
                    if hasattr(self.risk_manager, 'assess'):
                        risk_assessment = await self.risk_manager.assess(event, analysis)
                    elif hasattr(self.risk_manager, 'evaluate'):
                        risk_assessment = await self.risk_manager.evaluate(event)
                    else:
                        risk_assessment = {'approved': True, 'position_size': 0.02}
                    
                    risk_approved = risk_assessment.get('approved', False)
                    position_size = risk_assessment.get('position_size', 0.02)
                    
                    if not risk_approved:
                        self.logger.info(f"  ❌ Risk rejected: {risk_assessment.get('reason', 'Risk limit exceeded')}")
                        self.pipeline_stats["events_rejected"] += 1
                        return
                    
                    self.logger.debug(f"  ✅ Risk approved: position size {position_size:.2%}")
                    self.pipeline_stats["events_approved_risk"] += 1
                except Exception as e:
                    self.logger.warning(f"Risk manager error: {e}")
                    risk_approved = True
            
            # Stage 5: Portfolio Optimization
            portfolio_signal = {'action': 'hold', 'confidence': analysis.get('confidence', event.confidence)}
            if self.portfolio_optimizer:
                try:
                    if hasattr(self.portfolio_optimizer, 'optimize'):
                        portfolio_signal = await self.portfolio_optimizer.optimize(event, analysis, {'position_size': position_size})
                    elif hasattr(self.portfolio_optimizer, 'process'):
                        portfolio_signal = await self.portfolio_optimizer.process(event)
                    else:
                        portfolio_signal = {'action': 'hold', 'reason': 'No optimizer logic'}
                    
                    self.logger.debug(f"  ✅ Portfolio optimized: action {portfolio_signal.get('action', 'hold')}")
                except Exception as e:
                    self.logger.warning(f"Portfolio optimizer error: {e}")
            
            # Stage 6: Alert Management
            if self.alert_manager and portfolio_signal.get('action') != 'hold':
                try:
                    if hasattr(self.alert_manager, 'send_alert'):
                        await self.alert_manager.send_alert(event, analysis, portfolio_signal)
                    elif hasattr(self.alert_manager, 'notify'):
                        await self.alert_manager.notify(f"Signal: {portfolio_signal.get('action')} {event.symbol}")
                    else:
                        self.logger.info(f"📢 Alert: {portfolio_signal.get('action').upper()} {event.symbol} - Confidence: {analysis.get('confidence', 0):.2f}")
                    
                    self.logger.debug(f"  ✅ Alert sent")
                except Exception as e:
                    self.logger.warning(f"Alert manager error: {e}")
            
            # Stage 7: Execution
            if self.execution_engine and portfolio_signal.get('action') in ['buy', 'sell']:
                try:
                    if hasattr(self.execution_engine, 'execute'):
                        execution_result = await self.execution_engine.execute(portfolio_signal)
                    elif hasattr(self.execution_engine, 'place_order'):
                        execution_result = await self.execution_engine.place_order(
                            symbol=event.symbol,
                            action=portfolio_signal.get('action'),
                            quantity=portfolio_signal.get('quantity', 100)
                        )
                    else:
                        execution_result = {'status': 'simulated', 'order_id': 'sim_' + event.symbol}
                    
                    if execution_result.get('status') in ['completed', 'filled', 'simulated']:
                        self.logger.info(f"  ✅ Execution: {execution_result.get('status')} - Order ID: {execution_result.get('order_id', 'N/A')}")
                        self.pipeline_stats["events_executed"] += 1
                    else:
                        self.logger.warning(f"  ⚠️ Execution failed: {execution_result.get('message', 'Unknown error')}")
                except Exception as e:
                    self.logger.warning(f"Execution engine error: {e}")
            
            # Update pipeline stats
            self.pipeline_stats["events_processed"] += 1
            
            # Log final decision
            action = portfolio_signal.get('action', 'hold')
            confidence = analysis.get('confidence', event.confidence)
            
            if action != 'hold':
                self.logger.info(f"🎯 DECISION: {action.upper()} {event.symbol} | Confidence: {confidence:.2f} | Position: {position_size:.2%}")
            else:
                self.logger.debug(f"🎯 DECISION: HOLD {event.symbol} | Confidence: {confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Pipeline error for {event.symbol}: {e}", exc_info=True)
            self.pipeline_stats["pipeline_errors"] += 1
    
    async def process_market_cycle(self):
        """Execute one complete market analysis cycle"""
        cycle_start = datetime.now()
        self.logger.info("🔄 Starting market analysis cycle")
        
        try:
            # Check if trigger orchestrator is available
            if not self.trigger_orchestrator:
                self.logger.warning("Trigger orchestrator not available, skipping cycle")
                return
            
            # Get pipeline statistics
            self.logger.info(f"📊 Pipeline stats: {self.pipeline_stats['events_processed']} processed, "
                           f"{self.pipeline_stats['events_executed']} executed, "
                           f"{self.pipeline_stats['pipeline_errors']} errors")
            
            # Get orchestrator stats if available
            if hasattr(self.trigger_orchestrator, 'stats'):
                stats = self.trigger_orchestrator.stats
                self.logger.info(f"📊 Trigger stats: {stats.get('total_events', 0)} total events, "
                               f"{stats.get('errors', 0)} errors")
            
            # Check queue sizes
            if hasattr(self.trigger_orchestrator, 'priority_queues') and self.trigger_orchestrator.priority_queues:
                queue_sizes = {}
                for priority, queue in self.trigger_orchestrator.priority_queues.items():
                    if queue:
                        queue_sizes[priority.name if hasattr(priority, 'name') else str(priority)] = queue.qsize()
                
                if any(qsize > 0 for qsize in queue_sizes.values()):
                    self.logger.info(f"📨 Events in queues: {queue_sizes}")
            
            # Save cycle artifacts
            await self._save_cycle_artifacts({
                'timestamp': datetime.now().isoformat(),
                'status': 'running',
                'pipeline_stats': self.pipeline_stats,
                'orchestrator_stats': self.trigger_orchestrator.stats if hasattr(self.trigger_orchestrator, 'stats') else {}
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
            self.logger.info(f"  Pipeline: {self.pipeline_stats['events_processed']} events, "
                           f"{self.pipeline_stats['events_executed']} executed")
            
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
                'pipeline_stats': self.pipeline_stats,
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
            
            # Check pipeline performance
            if self.pipeline_stats['events_processed'] > 0:
                execution_rate = self.pipeline_stats['events_executed'] / self.pipeline_stats['events_processed']
                if execution_rate < 0.3:
                    analysis_results['recommendations'].append({
                        'type': 'pipeline_optimization',
                        'message': f'Low execution rate: {execution_rate:.1%}. Check risk and quality gates.',
                        'priority': 'medium'
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
        
        # Start background tasks for components
        await self._start_background_tasks()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start the trigger orchestrator if available
        orchestrator_task = None
        event_processor_task = None
        
        if self.trigger_orchestrator:
            try:
                orchestrator_task = asyncio.create_task(self.trigger_orchestrator.start())
                self.logger.info("✅ Trigger orchestrator started")
                
                # Start event processor
                event_processor_task = asyncio.create_task(self._process_trigger_events())
                self.logger.info("✅ Event processor started")
                
            except Exception as e:
                self.logger.error(f"Failed to start orchestrator: {e}")
        
        # Main loop (heartbeat and learning)
        cycle_count = 0
        while self.running:
            try:
                await self.process_market_cycle()
                cycle_count += 1
                
                if cycle_count % 10 == 0:
                    self.logger.info(f"📊 System status: {cycle_count} cycles, "
                                   f"{self.pipeline_stats['events_processed']} events processed, "
                                   f"uptime: {datetime.now() - self.start_time}")
                
                # Run learning cycle at configured interval
                hours_since_learning = (datetime.now() - self.last_learning_run).total_seconds() / 3600
                if hours_since_learning >= self.learning_config.get('interval_hours', 24):
                    await self.run_learning_cycle()
                
                # Update metrics periodically
                await self._update_system_metrics()
                
                await asyncio.sleep(60)  # Heartbeat every minute
                
            except asyncio.CancelledError:
                self.logger.info("System shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)
        
        # Cleanup
        if orchestrator_task:
            orchestrator_task.cancel()
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
        
        if event_processor_task:
            event_processor_task.cancel()
            try:
                await event_processor_task
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
        
        # Pipeline info
        self.logger.info(f"  - Pipeline: Search → Quality → Analysis → Risk → Portfolio → Alert → Execution")
    
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
        
        # Print final stats
        self.logger.info("📊 Final Pipeline Statistics:")
        self.logger.info(f"  Events Processed: {self.pipeline_stats['events_processed']}")
        self.logger.info(f"  Events Passed Quality: {self.pipeline_stats['events_passed_quality']}")
        self.logger.info(f"  Events Approved Risk: {self.pipeline_stats['events_approved_risk']}")
        self.logger.info(f"  Events Executed: {self.pipeline_stats['events_executed']}")
        self.logger.info(f"  Events Rejected: {self.pipeline_stats['events_rejected']}")
        self.logger.info(f"  Pipeline Errors: {self.pipeline_stats['pipeline_errors']}")
        
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