#!/usr/bin/env python3
"""
Continuous Learning Service - Runs learning tasks in background
"""

import time
import yaml
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

# Import learning components
from agentic_trading_system.learning.learning_orchestrator import LearningOrchestrator
from agentic_trading_system.learning.learning_trade_logger import TradeOutcomeLogger
from agentic_trading_system.learning.models.weight_optimizer import WeightOptimizer
from agentic_trading_system.learning.models.genetic_algorithm  import GeneticAlgorithm as GeneticTuner

# Import memory and analytics
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from memory.query_engine import UnifiedMemoryQuery
    from analytics.metrics_engine import MetricsEngine
except ImportError:
    # Create mock versions if modules don't exist yet
    class UnifiedMemoryQuery:
        def get_trades(self, **kwargs):
            return []
        def get_performance_data(self, **kwargs):
            return {}
    
    class MetricsEngine:
        def get_current_sharpe_ratio(self):
            return 0.0

class ContinuousLearningService:
    """Background service that continuously learns from trading outcomes"""
    
    def __init__(self, config_path: str = "config/learning_config.yaml"):
        """Initialize the continuous learning service"""
        
        # Load configuration
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.learning_orchestrator = LearningOrchestrator()
        self.trade_logger = TradeOutcomeLogger()
        self.weight_optimizer = WeightOptimizer()
        self.genetic_tuner = GeneticTuner()
        
        # Initialize memory and metrics
        self.memory = UnifiedMemoryQuery()
        self.metrics = MetricsEngine()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Tracking variables
        self.last_run = {}
        self.learning_history = []
        
        # Learning schedules
        self.schedules = {
            'weight_optimization': {
                'interval': self.config.get('weight_optimization_interval_hours', 24),
                'last_run': None,
                'min_data_points': self.config.get('min_trades_for_learning', 10)
            },
            'parameter_tuning': {
                'interval': self.config.get('parameter_tuning_interval_hours', 168),  # Weekly
                'last_run': None,
                'min_data_points': self.config.get('min_trades_for_tuning', 50)
            },
            'performance_review': {
                'interval': self.config.get('performance_review_interval_hours', 6),
                'last_run': None,
                'min_data_points': 1
            }
        }
        
        self.logger.info("Continuous Learning Service initialized")
    
    def _load_config(self) -> dict:
        """Load learning configuration"""
        default_config = {
            'mode': 'continuous',
            'weight_optimization_interval_hours': 24,
            'parameter_tuning_interval_hours': 168,
            'performance_review_interval_hours': 6,
            'min_trades_for_learning': 10,
            'min_trades_for_tuning': 50,
            'max_weight_change_percent': 15.0,
            'min_improvement_threshold': 0.05,
            'genetic_generations': 10,
            'population_size': 20,
            'learning_rate': 0.1,
            'enable_auto_update': True
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                    self.logger.info(f"Loaded config from {self.config_path}")
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the learning service"""
        logger = logging.getLogger("ContinuousLearning")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/learning_service.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_weight_optimization(self) -> Dict[str, Any]:
        """Optimize analysis weights based on recent performance"""
        self.logger.info("Starting weight optimization cycle")
        result = {'status': 'skipped', 'reason': None, 'changes': {}}
        
        try:
            # Get recent trades
            recent_trades = self._get_recent_trades(days=30)
            
            if len(recent_trades) < self.schedules['weight_optimization']['min_data_points']:
                result['reason'] = f"Insufficient trades: {len(recent_trades)} < {self.schedules['weight_optimization']['min_data_points']}"
                self.logger.info(result['reason'])
                return result
            
            # Run Bayesian weight update
            new_weights = self.weight_optimizer.optimize_weights(recent_trades)
            
            if new_weights:
                # Validate weight update
                if self._validate_weight_update(new_weights):
                    # Update configuration
                    if self.config.get('enable_auto_update', True):
                        self._update_config_weights(new_weights)
                        result['status'] = 'success'
                        result['changes'] = new_weights
                        self.logger.info(f"Weight optimization completed: {new_weights}")
                    else:
                        result['status'] = 'dry_run'
                        result['changes'] = new_weights
                        self.logger.info(f"Weight optimization dry run: {new_weights}")
                else:
                    result['status'] = 'rejected'
                    result['reason'] = "Weight changes exceeded safety limits"
                    self.logger.warning(result['reason'])
            
            # Record this run
            self.schedules['weight_optimization']['last_run'] = datetime.now()
            
        except Exception as e:
            result['status'] = 'error'
            result['reason'] = str(e)
            self.logger.error(f"Weight optimization failed: {e}", exc_info=True)
        
        return result
    
    def run_parameter_tuning(self) -> Dict[str, Any]:
        """Evolve trigger thresholds and risk parameters"""
        self.logger.info("Starting genetic parameter tuning")
        result = {'status': 'skipped', 'reason': None, 'parameters': {}}
        
        try:
            # Get last month's performance data
            monthly_data = self._get_performance_data(days=30)
            
            if len(monthly_data.get('trades', [])) < self.schedules['parameter_tuning']['min_data_points']:
                result['reason'] = f"Insufficient data: {len(monthly_data.get('trades', []))} trades"
                self.logger.info(result['reason'])
                return result
            
            # Run genetic algorithm
            best_params = self.genetic_tuner.tune_parameters(
                monthly_data,
                generations=self.config.get('genetic_generations', 10),
                population_size=self.config.get('population_size', 20)
            )
            
            if best_params:
                # Calculate expected improvement
                current_sharpe = self._get_current_sharpe_ratio()
                expected_sharpe = best_params.get('expected_sharpe', current_sharpe)
                improvement = (expected_sharpe - current_sharpe) / max(abs(current_sharpe), 0.01)
                
                if improvement > self.config.get('min_improvement_threshold', 0.05):
                    # Update configuration
                    if self.config.get('enable_auto_update', True):
                        self._update_trigger_configs(best_params)
                        result['status'] = 'success'
                        result['parameters'] = best_params
                        result['improvement'] = improvement
                        self.logger.info(f"Parameter tuning applied - expected improvement: {improvement:.2%}")
                    else:
                        result['status'] = 'dry_run'
                        result['parameters'] = best_params
                        self.logger.info(f"Parameter tuning dry run - improvement: {improvement:.2%}")
                else:
                    result['status'] = 'rejected'
                    result['reason'] = f"Improvement {improvement:.2%} below threshold {self.config.get('min_improvement_threshold', 0.05):.2%}"
                    self.logger.info(result['reason'])
            
            # Record this run
            self.schedules['parameter_tuning']['last_run'] = datetime.now()
            
        except Exception as e:
            result['status'] = 'error'
            result['reason'] = str(e)
            self.logger.error(f"Parameter tuning failed: {e}", exc_info=True)
        
        return result
    
    def run_performance_review(self) -> Dict[str, Any]:
        """Review recent performance and generate insights"""
        self.logger.info("Starting performance review")
        result = {
            'status': 'success',
            'metrics': {},
            'insights': [],
            'recommendations': []
        }
        
        try:
            # Get performance metrics
            trades = self._get_recent_trades(days=7)
            
            if len(trades) == 0:
                result['insights'].append("No trades in the last 7 days")
                return result
            
            # Calculate key metrics
            df = pd.DataFrame(trades)
            
            win_rate = (df['pnl'] > 0).mean() if 'pnl' in df else 0
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if any(df['pnl'] > 0) else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if any(df['pnl'] < 0) else 0
            
            result['metrics'] = {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win * win_rate / (avg_loss * (1 - win_rate))) if avg_loss != 0 else 0,
                'total_pnl': df['pnl'].sum() if 'pnl' in df else 0
            }
            
            # Generate insights
            if win_rate < 0.4:
                result['insights'].append("Win rate is below 40% - consider adjusting analysis weights")
            elif win_rate > 0.6:
                result['insights'].append("Excellent win rate! Consider increasing position sizes")
            
            if result['metrics']['profit_factor'] < 1.0:
                result['insights'].append("Profit factor below 1.0 - strategy is losing money overall")
            
            # Generate recommendations
            if win_rate < 0.5:
                result['recommendations'].append({
                    'action': 'update_weights',
                    'priority': 'high',
                    'reason': 'Low win rate suggests weight optimization needed'
                })
            
            # Save performance report
            self._save_performance_report(result)
            
            self.schedules['performance_review']['last_run'] = datetime.now()
            
        except Exception as e:
            result['status'] = 'error'
            result['reason'] = str(e)
            self.logger.error(f"Performance review failed: {e}", exc_info=True)
        
        return result
    
    def _get_recent_trades(self, days: int = 30) -> List[Dict]:
        """Get recent trades from memory"""
        try:
            # Try to get from memory module
            trades = self.memory.get_trades(
                start_date=datetime.now() - timedelta(days=days)
            )
            return trades if trades else []
        except Exception as e:
            self.logger.warning(f"Could not get trades from memory: {e}")
            # Fallback to reading from discovery_outputs
            return self._read_trades_from_artifacts(days)
    
    def _read_trades_from_artifacts(self, days: int) -> List[Dict]:
        """Fallback: read trades from discovery_output artifacts"""
        trades = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        artifacts_dir = Path("discovery_outputs")
        if not artifacts_dir.exists():
            return trades
        
        for artifact in artifacts_dir.rglob("*.json"):
            try:
                with open(artifact) as f:
                    data = json.load(f)
                
                # Check if this is a trade artifact
                if 'executed' in data or 'recommendations' in data:
                    timestamp_str = artifact.parent.name if artifact.parent.name != "discovery_outputs" else artifact.stem
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        if timestamp >= cutoff_date:
                            trades.append({
                                'timestamp': timestamp,
                                'ticker': data.get('ticker', 'UNKNOWN'),
                                'pnl': data.get('pnl', 0),
                                'signal_scores': data.get('analysis', {})
                            })
                    except:
                        pass
            except Exception as e:
                self.logger.debug(f"Could not read {artifact}: {e}")
        
        return trades
    
    def _get_performance_data(self, days: int) -> Dict:
        """Get comprehensive performance data"""
        trades = self._get_recent_trades(days)
        
        return {
            'trades': trades,
            'start_date': datetime.now() - timedelta(days=days),
            'end_date': datetime.now(),
            'total_trades': len(trades)
        }
    
    def _get_current_sharpe_ratio(self) -> float:
        """Get current Sharpe ratio from metrics"""
        try:
            return self.metrics.get_current_sharpe_ratio()
        except:
            return 0.5  # Default fallback
    
    def _validate_weight_update(self, new_weights: Dict) -> bool:
        """Validate that weight changes are within safe limits"""
        try:
            # Load current weights
            weights_path = Path("config/analysis_weights.yaml")
            if not weights_path.exists():
                return True  # No existing weights to compare
            
            with open(weights_path) as f:
                current = yaml.safe_load(f)
            
            max_change = self.config.get('max_weight_change_percent', 15.0)
            
            for factor, new_weight in new_weights.items():
                if factor in current.get('regime_weights', {}).get('default', {}):
                    current_weight = current['regime_weights']['default'][factor]
                    change_pct = abs(new_weight - current_weight) / max(current_weight, 0.01) * 100
                    
                    if change_pct > max_change:
                        self.logger.warning(f"Factor {factor} changed by {change_pct:.1f}% (limit: {max_change}%)")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False  # Safe to reject on error
    
    def _update_config_weights(self, new_weights: Dict):
        """Update analysis_weights.yaml with new weights"""
        weights_path = Path("config/analysis_weights.yaml")
        
        try:
            # Load existing config
            if weights_path.exists():
                with open(weights_path) as f:
                    config = yaml.safe_load(f)
            else:
                config = {'regime_weights': {'default': {}}}
            
            # Update weights
            for factor, weight in new_weights.items():
                if 'regime_weights' not in config:
                    config['regime_weights'] = {}
                if 'default' not in config['regime_weights']:
                    config['regime_weights']['default'] = {}
                config['regime_weights']['default'][factor] = weight
            
            # Save backup
            backup_path = weights_path.with_suffix(f".yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if weights_path.exists():
                import shutil
                shutil.copy(weights_path, backup_path)
            
            # Save new config
            with open(weights_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated weights in {weights_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to update weights config: {e}")
    
    def _update_trigger_configs(self, new_params: Dict):
        """Update trigger configuration with tuned parameters"""
        triggers_path = Path("config/triggers.yaml")
        
        try:
            if triggers_path.exists():
                with open(triggers_path) as f:
                    config = yaml.safe_load(f)
                
                # Update parameters
                if 'price_thresholds' in new_params:
                    config['price_alert_trigger'] = config.get('price_alert_trigger', {})
                    config['price_alert_trigger']['threshold'] = new_params['price_thresholds']
                
                if 'volume_thresholds' in new_params:
                    config['volume_spike_trigger'] = config.get('volume_spike_trigger', {})
                    config['volume_spike_trigger']['threshold'] = new_params['volume_thresholds']
                
                # Save backup
                backup_path = triggers_path.with_suffix(f".yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                import shutil
                shutil.copy(triggers_path, backup_path)
                
                # Save new config
                with open(triggers_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.logger.info(f"Updated parameters in {triggers_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to update trigger configs: {e}")
    
    def _save_performance_report(self, report: Dict):
        """Save performance review report"""
        reports_dir = Path("learning_results/performance_reviews")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"performance_review_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {report_path}")
    
    def run_continuous(self):
        """Main loop for continuous learning service"""
        self.logger.info("=" * 60)
        self.logger.info("Continuous Learning Service Started")
        self.logger.info(f"Configuration: {self.config}")
        self.logger.info("=" * 60)
        
        # Run initial learning cycles
        self.logger.info("Running initial learning cycles...")
        self.run_weight_optimization()
        self.run_performance_review()
        
        # Main loop
        while True:
            try:
                now = datetime.now()
                
                # Check each scheduled task
                for task_name, schedule_info in self.schedules.items():
                    if schedule_info['last_run'] is None:
                        continue
                    
                    hours_since_last = (now - schedule_info['last_run']).total_seconds() / 3600
                    
                    if hours_since_last >= schedule_info['interval']:
                        self.logger.info(f"Running scheduled task: {task_name}")
                        
                        if task_name == 'weight_optimization':
                            self.run_weight_optimization()
                        elif task_name == 'parameter_tuning':
                            self.run_parameter_tuning()
                        elif task_name == 'performance_review':
                            self.run_performance_review()
                
                # Sleep for 1 hour between checks
                time.sleep(3600)
                
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(300)  # Sleep 5 minutes on error
        
        self.logger.info("Continuous Learning Service Stopped")

# For running as standalone service
if __name__ == "__main__":
    service = ContinuousLearningService()
    service.run_continuous()