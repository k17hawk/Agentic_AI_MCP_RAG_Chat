"""
Backtester - Main backtesting module for strategy validation
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
from utils.logger import logging

# Import backtester components
from learning.backtester.simulation_engine import SimulationEngine
from learning.backtester.monte_carlo import MonteCarlo
from learning.backtester.walk_forward import WalkForward

class Backtester:
    """
    Backtester - Main backtesting module for strategy validation
    
    This is the main entry point for backtesting functionality,
    coordinating simulation, Monte Carlo, and walk-forward analysis.
    
    Features:
    - Historical strategy testing
    - Monte Carlo risk simulation
    - Walk-forward validation
    - Parameter optimization
    - Performance metrics
    - Strategy comparison
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.simulation = SimulationEngine(config.get("simulation_config", {}))
        self.monte_carlo = MonteCarlo(config.get("monte_carlo_config", {}))
        self.walk_forward = WalkForward(config.get("walk_forward_config", {}))
        
        # Storage
        self.data_dir = config.get("data_dir", "data/backtests")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Results cache
        self.results_cache = {}
        
        logging.info(f"✅ Backtester initialized")
    
    def run_backtest(self, strategy_func: Callable, 
                    data: pd.DataFrame,
                    symbol: str,
                    params: Dict[str, Any] = None,
                    use_walk_forward: bool = False,
                    param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Run a backtest with optional walk-forward validation
        """
        if use_walk_forward and param_grid:
            # Run walk-forward analysis
            result = self.walk_forward.run(strategy_func, data, symbol, param_grid)
            
            # Add simulation of best parameters on full data
            if result['results']:
                best_params = result['results'][-1]['best_params']  # Last window's best
                full_result = self.simulation.run_backtest(
                    strategy_func, data, symbol, best_params
                )
                result['full_data_result'] = full_result
        else:
            # Simple backtest
            result = self.simulation.run_backtest(strategy_func, data, symbol, params)
        
        # Cache result
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_cache[cache_key] = result
        
        return result
    
    def optimize_parameters(self, strategy_func: Callable,
                           data: pd.DataFrame,
                           symbol: str,
                           param_grid: Dict[str, List],
                           method: str = "grid") -> Dict[str, Any]:
        """
        Optimize strategy parameters
        """
        if method == "grid":
            result = self.simulation.optimize_parameters(
                strategy_func, data, symbol, param_grid
            )
        elif method == "genetic":
            # Use genetic algorithm for optimization
            # This would integrate with the genetic algorithm model
            result = {"method": "genetic", "status": "not_implemented"}
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return result
    
    def compare_strategies(self, strategies: Dict[str, Callable],
                          data: pd.DataFrame,
                          symbol: str,
                          params: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Compare multiple strategies
        """
        return self.simulation.compare_strategies(strategies, data, symbol)
    
    def monte_carlo_risk(self, returns: np.ndarray,
                         initial_price: float,
                         method: str = "bootstrap",
                         num_simulations: int = 10000) -> Dict[str, Any]:
        """
        Perform Monte Carlo risk analysis
        """
        if method == "bootstrap":
            result = self.monte_carlo.run_simulation(
                initial_price=initial_price,
                historical_returns=returns,
                method="bootstrap",
                num_simulations=num_simulations
            )
        else:
            # Calculate mu and sigma from returns
            mu = np.mean(returns) * 252
            sigma = np.std(returns) * np.sqrt(252)
            
            result = self.monte_carlo.run_simulation(
                initial_price=initial_price,
                mu=mu,
                sigma=sigma,
                method="random_walk",
                num_simulations=num_simulations
            )
        
        return result
    
    def walk_forward_validate(self, strategy_func: Callable,
                             data: pd.DataFrame,
                             symbol: str,
                             param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform walk-forward validation
        """
        return self.walk_forward.run(strategy_func, data, symbol, param_grid)
    
    def stress_test(self, strategy_func: Callable,
                   data: pd.DataFrame,
                   scenarios: Dict[str, float]) -> Dict[str, Any]:
        """
        Stress test strategy under various scenarios
        """
        return self.monte_carlo.stress_test(strategy_func, data, scenarios)
    
    def calculate_metrics(self, equity_curve: List[float],
                         trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance metrics
        """
        if not equity_curve or len(equity_curve) < 2:
            return {}
        
        # Convert to array
        equity = np.array(equity_curve)
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        
        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_drawdown = np.max(drawdown) * 100
        
        # Win/Loss analysis
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                               sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'num_trades': len(trades)
        }
    
    def generate_report(self, backtest_id: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report
        """
        if backtest_id and backtest_id in self.results_cache:
            results = [self.results_cache[backtest_id]]
        else:
            results = list(self.results_cache.values())
        
        if not results:
            return {"error": "No backtest results found"}
        
        # Aggregate statistics
        avg_return = np.mean([r.get('total_return', 0) for r in results])
        avg_sharpe = np.mean([r.get('metrics', {}).get('sharpe_ratio', 0) for r in results])
        avg_win_rate = np.mean([r.get('metrics', {}).get('win_rate', 0) for r in results])
        
        best_return = max(results, key=lambda x: x.get('total_return', 0))
        best_sharpe = max(results, key=lambda x: x.get('metrics', {}).get('sharpe_ratio', 0))
        
        return {
            'num_backtests': len(results),
            'average_metrics': {
                'avg_return': float(avg_return),
                'avg_sharpe': float(avg_sharpe),
                'avg_win_rate': float(avg_win_rate)
            },
            'best_by_return': {
                'id': best_return.get('backtest_id', 'unknown'),
                'return': best_return.get('total_return', 0),
                'symbol': best_return.get('symbol', 'unknown')
            },
            'best_by_sharpe': {
                'id': best_sharpe.get('backtest_id', 'unknown'),
                'sharpe': best_sharpe.get('metrics', {}).get('sharpe_ratio', 0),
                'symbol': best_sharpe.get('symbol', 'unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, backtest_id: str, format: str = "json") -> str:
        """
        Export backtest results to file
        """
        if backtest_id not in self.results_cache:
            raise ValueError(f"Backtest ID not found: {backtest_id}")
        
        result = self.results_cache[backtest_id]
        
        filename = f"backtest_{backtest_id}_{datetime.now().strftime('%Y%m%d')}.{format}"
        filepath = os.path.join(self.data_dir, filename)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif format == "csv":
            # Convert equity curve to CSV
            if 'equity_curve' in result:
                df = pd.DataFrame(result['equity_curve'])
                df.to_csv(filepath, index=False)
        
        logging.info(f"💾 Exported backtest results to {filepath}")
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load backtest results from file
        """
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
            else:
                # Assume CSV
                data = pd.read_csv(filepath).to_dict('records')
        
        return data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get backtester status
        """
        return {
            'initialized': True,
            'cached_results': len(self.results_cache),
            'components': {
                'simulation': True,
                'monte_carlo': True,
                'walk_forward': True
            },
            'config': {
                'data_dir': self.data_dir
            }
        }