"""
Walk Forward - Walk-forward analysis for strategy validation
"""
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from utils.logger import logger as logging
from learning.backtester.simulation_engine import SimulationEngine

class WalkForward:
    """
    Walk Forward - Walk-forward analysis for strategy validation
    
    Features:
    - Rolling window optimization
    - Out-of-sample testing
    - Parameter stability analysis
    - Performance decay measurement
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Walk-forward parameters
        self.train_size = config.get("train_size", 252)  # 1 year
        self.test_size = config.get("test_size", 63)  # 3 months
        self.step_size = config.get("step_size", 63)  # 3 months
        
        # Optimization parameters
        self.optimization_method = config.get("optimization_method", "grid")  # grid, random, genetic
        
        # Storage
        self.data_dir = config.get("data_dir", "data/walkforward")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Results
        self.results = []
        
        logging.info(f"✅ WalkForward initialized")
    
    def run(self, strategy_func: Callable,
           data: pd.DataFrame,
           symbol: str,
           param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        """
        engine = SimulationEngine(self.config)
        
        results = []
        all_params = []
        oos_returns = []
        
        # Create windows
        total_days = len(data)
        start_idx = 0
        
        while start_idx + self.train_size + self.test_size <= total_days:
            train_end = start_idx + self.train_size
            test_end = train_end + self.test_size
            
            # Split data
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Optimize on training data
            opt_result = engine.optimize_parameters(
                strategy_func, train_data, symbol, param_grid
            )
            
            best_params = opt_result['best_params']
            all_params.append(best_params)
            
            # Test on out-of-sample data
            test_result = engine.run_backtest(
                strategy_func, test_data, symbol, best_params
            )
            
            oos_return = test_result['total_return']
            oos_returns.append(oos_return)
            
            results.append({
                'window': len(results) + 1,
                'train_start': data.index[start_idx].strftime('%Y-%m-%d'),
                'train_end': data.index[train_end-1].strftime('%Y-%m-%d'),
                'test_start': data.index[train_end].strftime('%Y-%m-%d'),
                'test_end': data.index[test_end-1].strftime('%Y-%m-%d'),
                'best_params': best_params,
                'train_return': opt_result['best_return'],
                'test_return': oos_return,
                'test_metrics': test_result['metrics']
            })
            
            # Move window
            start_idx += self.step_size
        
        # Calculate aggregate statistics
        avg_oos_return = np.mean(oos_returns)
        std_oos_return = np.std(oos_returns)
        worst_oos_return = np.min(oos_returns)
        best_oos_return = np.max(oos_returns)
        
        # Parameter stability analysis
        param_stability = self._analyze_parameter_stability(all_params)
        
        # Performance decay
        decay = self._analyze_performance_decay(results)
        
        summary = {
            'symbol': symbol,
            'num_windows': len(results),
            'train_size_days': self.train_size,
            'test_size_days': self.test_size,
            'step_size_days': self.step_size,
            'results': results,
            'statistics': {
                'avg_oos_return': float(avg_oos_return),
                'std_oos_return': float(std_oos_return),
                'worst_oos_return': float(worst_oos_return),
                'best_oos_return': float(best_oos_return),
                'avg_train_return': float(np.mean([r['train_return'] for r in results])),
                'avg_test_win_rate': float(np.mean([r['test_metrics'].get('win_rate', 0) for r in results])),
                'avg_test_sharpe': float(np.mean([r['test_metrics'].get('sharpe_ratio', 0) for r in results]))
            },
            'parameter_stability': param_stability,
            'performance_decay': decay,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(summary)
        
        return summary
    
    def _analyze_parameter_stability(self, all_params: List[Dict]) -> Dict[str, Any]:
        """
        Analyze stability of parameters across windows
        """
        if not all_params:
            return {}
        
        stability = {}
        
        # Get all parameter names
        param_names = all_params[0].keys()
        
        for param in param_names:
            values = [p[param] for p in all_params]
            
            if isinstance(values[0], (int, float)):
                stability[param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
                }
            else:
                # Categorical parameter
                unique, counts = np.unique(values, return_counts=True)
                stability[param] = {
                    'unique_values': unique.tolist(),
                    'frequencies': (counts / len(values)).tolist(),
                    'most_common': unique[np.argmax(counts)]
                }
        
        return stability
    
    def _analyze_performance_decay(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze performance decay over time
        """
        if len(results) < 3:
            return {}
        
        train_returns = [r['train_return'] for r in results]
        test_returns = [r['test_return'] for r in results]
        
        # Calculate decay rate
        x = np.arange(len(results))
        
        # Linear regression for test returns
        z = np.polyfit(x, test_returns, 1)
        decay_slope = z[0]
        
        # Correlation between train and test
        correlation = np.corrcoef(train_returns, test_returns)[0, 1] if len(results) > 1 else 0
        
        # Overfitting indicator (large gap between train and test)
        avg_train = np.mean(train_returns)
        avg_test = np.mean(test_returns)
        overfitting_gap = avg_train - avg_test
        
        return {
            'decay_slope': float(decay_slope),
            'train_test_correlation': float(correlation),
            'avg_train_return': float(avg_train),
            'avg_test_return': float(avg_test),
            'overfitting_gap': float(overfitting_gap),
            'overfitting_ratio': float(overfitting_gap / avg_train) if avg_train != 0 else float('inf')
        }
    
    def compare_strategies(self, strategies: Dict[str, Callable],
                          data: pd.DataFrame,
                          symbol: str,
                          param_grids: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple strategies using walk-forward
        """
        results = {}
        
        for name, strategy in strategies.items():
            param_grid = param_grids.get(name, {})
            result = self.run(strategy, data, symbol, param_grid)
            results[name] = {
                'avg_oos_return': result['statistics']['avg_oos_return'],
                'avg_test_sharpe': result['statistics']['avg_test_sharpe'],
                'parameter_stability': result['parameter_stability'],
                'performance_decay': result['performance_decay']
            }
        
        # Rank strategies
        ranked_by_return = sorted(
            results.items(),
            key=lambda x: x[1]['avg_oos_return'],
            reverse=True
        )
        
        ranked_by_sharpe = sorted(
            results.items(),
            key=lambda x: x[1]['avg_test_sharpe'],
            reverse=True
        )
        
        return {
            'results': results,
            'ranked_by_return': ranked_by_return,
            'ranked_by_sharpe': ranked_by_sharpe,
            'best_strategy': ranked_by_return[0][0] if ranked_by_return else None
        }
    
    def save_results(self, filename: str = None):
        """
        Save walk-forward results to disk
        """
        if filename is None:
            filename = f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logging.info(f"💾 Saved walk-forward results to {filepath}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")