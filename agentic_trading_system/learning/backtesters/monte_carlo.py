"""
Monte Carlo - Monte Carlo simulation for risk analysis
"""
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from utils.logger import logger as  logging
from learning.backtesters.simulation_engine import SimulationEngine
class MonteCarlo:
    """
    Monte Carlo - Monte Carlo simulation for risk analysis
    
    Features:
    - Random walk simulations
    - Bootstrapping from historical returns
    - VaR and CVaR calculation
    - Confidence intervals
    - Stress testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Simulation parameters
        self.num_simulations = config.get("num_simulations", 10000)
        self.num_days = config.get("num_days", 252)  # 1 trading year
        self.confidence_level = config.get("confidence_level", 0.95)
        
        # Random seed for reproducibility
        self.random_seed = config.get("random_seed", 42)
        np.random.seed(self.random_seed)
        
        # Storage
        self.data_dir = config.get("data_dir", "data/montecarlo")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logging.info(f"✅ MonteCarlo initialized")
    
    def simulate_random_walk(self, initial_price: float, 
                            mu: float, sigma: float,
                            days: int = None) -> np.ndarray:
        """
        Simulate geometric Brownian motion
        """
        if days is None:
            days = self.num_days
        
        dt = 1/252  # Daily time step
        prices = [initial_price]
        
        for _ in range(days):
            epsilon = np.random.normal(0, 1)
            price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)
            prices.append(price)
        
        return np.array(prices)
    
    def simulate_bootstrap(self, returns: np.ndarray, 
                          initial_price: float,
                          days: int = None) -> np.ndarray:
        """
        Bootstrap from historical returns
        """
        if days is None:
            days = self.num_days
        
        prices = [initial_price]
        
        for _ in range(days):
            # Sample random return from history
            r = np.random.choice(returns)
            price = prices[-1] * (1 + r)
            prices.append(price)
        
        return np.array(prices)
    
    def run_simulation(self, initial_price: float, 
                      historical_returns: np.ndarray = None,
                      mu: float = None, sigma: float = None,
                      method: str = "bootstrap") -> Dict[str, Any]:
        """
        Run Monte Carlo simulation
        """
        all_paths = []
        final_prices = []
        
        for i in range(self.num_simulations):
            if method == "bootstrap" and historical_returns is not None:
                path = self.simulate_bootstrap(historical_returns, initial_price)
            elif method == "random_walk" and mu is not None and sigma is not None:
                path = self.simulate_random_walk(initial_price, mu, sigma)
            else:
                raise ValueError("Insufficient parameters for simulation method")
            
            all_paths.append(path)
            final_prices.append(path[-1])
        
        final_prices = np.array(final_prices)
        
        # Calculate statistics
        mean_final = np.mean(final_prices)
        median_final = np.median(final_prices)
        std_final = np.std(final_prices)
        
        # Percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(final_prices, p)
        
        # VaR and CVaR
        var = self.calculate_var(final_prices, initial_price)
        cvar = self.calculate_cvar(final_prices, initial_price)
        
        # Probability of loss/profit
        prob_loss = np.mean(final_prices < initial_price)
        prob_profit = np.mean(final_prices > initial_price)
        
        return {
            "initial_price": initial_price,
            "num_simulations": self.num_simulations,
            "num_days": self.num_days,
            "method": method,
            "statistics": {
                "mean": float(mean_final),
                "median": float(median_final),
                "std": float(std_final),
                "min": float(np.min(final_prices)),
                "max": float(np.max(final_prices))
            },
            "percentiles": percentiles,
            "risk_metrics": {
                "var_95": var,
                "cvar_95": cvar,
                "prob_loss": float(prob_loss),
                "prob_profit": float(prob_profit),
                "expected_return": float((mean_final - initial_price) / initial_price * 100)
            },
            "all_paths": all_paths[:100],  # Store first 100 for visualization
            "final_prices": final_prices.tolist()
        }
    
    def calculate_var(self, final_prices: np.ndarray, 
                     initial_price: float,
                     confidence: float = None) -> float:
        """
        Calculate Value at Risk
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns = (final_prices - initial_price) / initial_price
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return float(var * 100)  # Return as percentage
    
    def calculate_cvar(self, final_prices: np.ndarray,
                      initial_price: float,
                      confidence: float = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns = (final_prices - initial_price) / initial_price
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        
        return float(cvar * 100)  # Return as percentage
    
    def stress_test(self, strategy_func: Callable,
                   data: pd.DataFrame,
                   shock_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """
        Stress test strategy under various shock scenarios
        """
        results = {}
        
        for scenario_name, shock_pct in shock_scenarios.items():
            # Apply shock to data
            shocked_data = data.copy()
            shocked_data['Close'] = shocked_data['Close'] * (1 + shock_pct / 100)
            
            # Run simulation
            
            engine = SimulationEngine(self.config)
            result = engine.run_backtest(strategy_func, shocked_data, "STRESS_TEST")
            
            results[scenario_name] = {
                "shock_pct": shock_pct,
                "total_return": result["total_return"],
                "max_drawdown": result["metrics"].get("max_drawdown", 0),
                "sharpe": result["metrics"].get("sharpe_ratio", 0)
            }
        
        return results
    
    def confidence_interval(self, metric_values: List[float],
                           confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence interval for a metric
        """
        values = np.array(metric_values)
        mean = np.mean(values)
        std = np.std(values)
        
        from scipy import stats
        ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
        
        return {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "confidence": confidence
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save simulation results to disk
        """
        if filename is None:
            filename = f"montecarlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logging.info(f"💾 Saved Monte Carlo results to {filepath}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")