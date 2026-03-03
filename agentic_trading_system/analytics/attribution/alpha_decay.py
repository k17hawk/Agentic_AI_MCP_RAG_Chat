"""
Alpha Decay - Analyzes how signal alpha decays over time
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from utils.logger import logger as logging

class AlphaDecay:
    """
    Alpha Decay - Analyzes how signal predictive power decays over time
    
    Features:
    - Half-life calculation
    - Decay curves
    - Signal persistence
    - Optimal holding periods
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.max_lag = config.get("max_lag", 30)  # Maximum lag days to analyze
        self.min_samples = config.get("min_samples", 20)
        
        logging.info(f"✅ AlphaDecay initialized")
    
    def calculate_decay_curve(self, signals: List[Dict], 
                             subsequent_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate alpha decay curve for signals
        """
        if len(signals) < self.min_samples:
            return {"error": "Insufficient samples"}
        
        # Align signals with subsequent returns
        decay_rates = []
        
        for lag in range(1, self.max_lag + 1):
            lagged_returns = []
            
            for i, signal in enumerate(signals):
                if i + lag < len(subsequent_returns):
                    lagged_returns.append(subsequent_returns[i + lag])
            
            if lagged_returns:
                avg_return = np.mean(lagged_returns)
                decay_rates.append({
                    "lag": lag,
                    "avg_return": float(avg_return),
                    "samples": len(lagged_returns)
                })
        
        if not decay_rates:
            return {"error": "No decay data"}
        
        # Fit exponential decay model
        lags = np.array([d["lag"] for d in decay_rates])
        returns = np.array([d["avg_return"] for d in decay_rates])
        
        # Simple exponential decay: y = a * exp(-b * x)
        # Take log for linear regression
        log_returns = np.log(np.abs(returns) + 1e-10)  # Avoid log(0)
        
        # Linear regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(lags, log_returns)
        
        # Calculate half-life
        half_life = np.log(2) / (-slope) if slope < 0 else float('inf')
        
        # Calculate decay rate per day
        decay_rate = -slope if slope < 0 else 0
        
        return {
            "decay_curve": decay_rates,
            "half_life_days": float(half_life),
            "decay_rate_per_day": float(decay_rate),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "initial_alpha": float(np.exp(intercept)),
            "model_params": {
                "slope": float(slope),
                "intercept": float(intercept)
            }
        }
    
    def calculate_signal_half_life(self, signal_name: str,
                                  trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate half-life for a specific signal
        """
        # Extract signal occurrences and subsequent returns
        signal_events = []
        
        for trade in trades:
            signals = trade.get('signals', [])
            for signal in signals:
                if signal.get('name') == signal_name or signal.get('type') == signal_name:
                    # Record signal time and subsequent returns
                    exit_time = datetime.fromisoformat(trade.get('exit_time', datetime.now().isoformat()))
                    pnl = trade.get('pnl', 0)
                    
                    signal_events.append({
                        'time': exit_time,
                        'pnl': pnl,
                        'return': pnl / trade.get('position_value', 1) if trade.get('position_value') else 0
                    })
                    break
        
        if len(signal_events) < self.min_samples:
            return {"error": f"Insufficient samples for {signal_name}"}
        
        # Sort by time
        signal_events.sort(key=lambda x: x['time'])
        
        # Calculate returns at different lags
        returns_by_lag = defaultdict(list)
        
        for i, event in enumerate(signal_events):
            for j in range(i + 1, min(i + self.max_lag + 1, len(signal_events))):
                lag = j - i
                returns_by_lag[lag].append(event['return'])
        
        # Calculate average return for each lag
        decay_data = []
        for lag in range(1, self.max_lag + 1):
            if lag in returns_by_lag and len(returns_by_lag[lag]) >= 5:
                avg_return = np.mean(returns_by_lag[lag])
                decay_data.append({
                    "lag": lag,
                    "avg_return": float(avg_return),
                    "samples": len(returns_by_lag[lag])
                })
        
        if len(decay_data) < 3:
            return {"error": "Insufficient decay data"}
        
        # Fit decay model
        lags = np.array([d["lag"] for d in decay_data])
        returns = np.array([d["avg_return"] for d in decay_data])
        
        # Exponential fit
        from scipy.optimize import curve_fit
        
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        
        try:
            popt, pcov = curve_fit(exp_decay, lags, returns, p0=[returns[0], 0.1])
            a, b = popt
            
            half_life = np.log(2) / b if b > 0 else float('inf')
            
            # Calculate fit quality
            fitted = exp_decay(lags, a, b)
            residuals = returns - fitted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((returns - np.mean(returns))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                "signal": signal_name,
                "half_life_days": float(half_life),
                "decay_rate": float(b),
                "initial_alpha": float(a),
                "r_squared": float(r_squared),
                "decay_curve": decay_data,
                "fitted_curve": [
                    {"lag": lag, "fitted_return": float(exp_decay(lag, a, b))}
                    for lag in lags
                ],
                "samples": len(signal_events)
            }
            
        except Exception as e:
            logging.error(f"Error fitting decay curve: {e}")
            return {"error": str(e)}
    
    def compare_signal_decay(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare decay rates across different signals
        """
        # Get all unique signals
        all_signals = set()
        for trade in trades:
            for signal in trade.get('signals', []):
                name = signal.get('name', signal.get('type', 'unknown'))
                all_signals.add(name)
        
        # Calculate half-life for each signal
        results = {}
        for signal in all_signals:
            decay = self.calculate_signal_half_life(signal, trades)
            if "half_life_days" in decay:
                results[signal] = decay
        
        # Sort by half-life
        sorted_signals = sorted(
            results.items(),
            key=lambda x: x[1].get("half_life_days", float('inf'))
        )
        
        return {
            "signal_decay": results,
            "longest_lived": sorted_signals[0][0] if sorted_signals else None,
            "shortest_lived": sorted_signals[-1][0] if sorted_signals else None,
            "average_half_life": float(np.mean([r.get("half_life_days", 0) for r in results.values()])),
            "decay_comparison": [
                {
                    "signal": s,
                    "half_life": data.get("half_life_days", 0),
                    "decay_rate": data.get("decay_rate", 0)
                }
                for s, data in sorted_signals
            ]
        }
    
    def optimal_holding_period(self, decay_curve: List[Dict]) -> Dict[str, Any]:
        """
        Calculate optimal holding period based on decay curve
        """
        if not decay_curve:
            return {}
        
        # Find peak return
        peak_return = max(decay_curve, key=lambda x: x["avg_return"])
        
        # Find where return drops below 50% of peak
        half_peak = peak_return["avg_return"] / 2
        
        optimal_period = peak_return["lag"]
        for point in decay_curve:
            if point["avg_return"] < half_peak:
                break
            optimal_period = point["lag"]
        
        return {
            "peak_return_lag": peak_return["lag"],
            "peak_return": float(peak_return["avg_return"]),
            "optimal_holding_period": optimal_period,
            "half_life": optimal_period - peak_return["lag"],
            "recommendation": f"Hold for {optimal_period} days for optimal returns"
        }
    
    def calculate_alpha_persistence(self, strategy_returns: List[float],
                                   benchmark_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate how long alpha persists
        """
        if len(strategy_returns) < 60:
            return {"error": "Insufficient data"}
        
        # Calculate rolling alpha (simplified)
        window = 20
        alphas = []
        
        for i in range(window, len(strategy_returns)):
            strat_window = strategy_returns[i-window:i]
            bench_window = benchmark_returns[i-window:i]
            
            # Simple alpha = strategy return - benchmark return
            alpha = np.mean(strat_window) - np.mean(bench_window)
            alphas.append(alpha)
        
        if len(alphas) < 10:
            return {"error": "Insufficient alpha data"}
        
        # Calculate autocorrelation of alpha
        alphas = np.array(alphas)
        
        # Calculate autocorrelation at different lags
        autocorrelations = []
        for lag in range(1, min(20, len(alphas) // 2)):
            autocorr = np.corrcoef(alphas[:-lag], alphas[lag:])[0, 1]
            if not np.isnan(autocorr):
                autocorrelations.append({
                    "lag": lag,
                    "autocorrelation": float(autocorr)
                })
        
        # Find where autocorrelation becomes insignificant
        persistence = 0
        for ac in autocorrelations:
            if abs(ac["autocorrelation"]) > 0.3:
                persistence = ac["lag"]
            else:
                break
        
        return {
            "alpha_persistence_days": persistence,
            "alpha_autocorrelation": autocorrelations,
            "alpha_volatility": float(np.std(alphas)),
            "alpha_series": alphas.tolist()
        }