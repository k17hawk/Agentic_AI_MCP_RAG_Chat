"""
Attribution Engine - Analyzes which signals contributed to trade outcomes
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json
from utils.logger import logger as logging

class AttributionEngine:
    """
    Attribution Engine - Analyzes which signals and factors contributed to trade outcomes
    
    Responsibilities:
    - Track which signals led to each trade
    - Calculate contribution percentages
    - Identify most predictive signals
    - Measure signal decay over time
    - Generate attribution reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/attribution")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Attribution tracking
        self.trade_attributions = {}  # trade_id -> attribution data
        self.signal_performance = defaultdict(lambda: {
            "appearances": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "avg_confidence": 0.0
        })
        
        # Factor analysis
        self.factor_loadings = defaultdict(dict)
        self.factor_performance = defaultdict(lambda: defaultdict(float))
        
        # Decay parameters
        self.decay_half_life = config.get("decay_half_life_days", 30)
        
        logging.info(f"✅ AttributionEngine initialized")
    
    def record_trade_attribution(self, trade_id: str, symbol: str,
                                 signals: List[Dict[str, Any]],
                                 outcome: str, pnl: float,
                                 confidence: float = None) -> bool:
        """
        Record which signals contributed to a trade
        """
        attribution = {
            "trade_id": trade_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "pnl": pnl,
            "confidence": confidence,
            "signals": signals,
            "contributions": self._calculate_contributions(signals, outcome, pnl)
        }
        
        self.trade_attributions[trade_id] = attribution
        
        # Update signal performance
        for signal in signals:
            signal_name = signal.get("name", signal.get("type", "unknown"))
            signal_conf = signal.get("confidence", 0.5)
            
            perf = self.signal_performance[signal_name]
            perf["appearances"] += 1
            perf["avg_confidence"] = (
                (perf["avg_confidence"] * (perf["appearances"] - 1) + signal_conf) /
                perf["appearances"]
            )
            perf["total_pnl"] += pnl
            
            if outcome == "win":
                perf["wins"] += 1
            elif outcome == "loss":
                perf["losses"] += 1
        
        logging.info(f"📊 Recorded attribution for trade {trade_id}")
        
        return True
    
    def _calculate_contributions(self, signals: List[Dict], 
                                outcome: str, pnl: float) -> Dict[str, float]:
        """
        Calculate contribution percentages for each signal
        """
        total_confidence = sum(s.get("confidence", 0.5) for s in signals)
        
        if total_confidence == 0:
            return {s.get("name", s.get("type", f"signal_{i}")): 0 for i, s in enumerate(signals)}
        
        contributions = {}
        for signal in signals:
            name = signal.get("name", signal.get("type", "unknown"))
            weight = signal.get("confidence", 0.5) / total_confidence
            contributions[name] = weight
        
        return contributions
    
    def get_best_signals(self, min_appearances: int = 10) -> List[Dict[str, Any]]:
        """
        Get best performing signals
        """
        results = []
        
        for signal_name, perf in self.signal_performance.items():
            if perf["appearances"] < min_appearances:
                continue
            
            total_trades = perf["wins"] + perf["losses"]
            win_rate = perf["wins"] / total_trades if total_trades > 0 else 0
            
            results.append({
                "signal": signal_name,
                "appearances": perf["appearances"],
                "wins": perf["wins"],
                "losses": perf["losses"],
                "win_rate": win_rate,
                "total_pnl": perf["total_pnl"],
                "avg_pnl": perf["total_pnl"] / perf["appearances"] if perf["appearances"] > 0 else 0,
                "avg_confidence": perf["avg_confidence"]
            })
        
        # Sort by win rate
        results.sort(key=lambda x: (x["win_rate"], x["total_pnl"]), reverse=True)
        
        return results
    
    def get_worst_signals(self, min_appearances: int = 5) -> List[Dict[str, Any]]:
        """
        Get worst performing signals
        """
        results = self.get_best_signals(min_appearances)
        return results[::-1]  # Reverse order
    
    def analyze_signal_decay(self, signal_name: str, 
                            time_windows: List[int] = None) -> Dict[str, Any]:
        """
        Analyze how signal performance decays over time
        """
        if time_windows is None:
            time_windows = [7, 30, 60, 90]
        
        # Filter trades with this signal
        relevant_trades = []
        for trade_id, attribution in self.trade_attributions.items():
            for signal in attribution["signals"]:
                if signal.get("name") == signal_name or signal.get("type") == signal_name:
                    relevant_trades.append(attribution)
                    break
        
        if not relevant_trades:
            return {"error": f"No trades found with signal {signal_name}"}
        
        # Sort by time
        relevant_trades.sort(key=lambda x: x["timestamp"])
        
        # Calculate performance over time windows
        decay_analysis = {}
        now = datetime.now()
        
        for days in time_windows:
            cutoff = now - timedelta(days=days)
            recent_trades = [
                t for t in relevant_trades 
                if datetime.fromisoformat(t["timestamp"]) > cutoff
            ]
            
            if recent_trades:
                wins = sum(1 for t in recent_trades if t["outcome"] == "win")
                losses = sum(1 for t in recent_trades if t["outcome"] == "loss")
                total = wins + losses
                
                decay_analysis[f"last_{days}d"] = {
                    "trades": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / total if total > 0 else 0,
                    "total_pnl": sum(t["pnl"] for t in recent_trades)
                }
        
        # Calculate decay rate
        if len(decay_analysis) >= 2:
            # Simple linear decay estimate
            win_rates = [v["win_rate"] for v in decay_analysis.values()]
            decay_analysis["decay_rate"] = (win_rates[-1] - win_rates[0]) / len(win_rates)
        
        return decay_analysis
    
    def factor_attribution(self, returns: np.ndarray, 
                          factor_returns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform factor attribution analysis
        """
        results = {}
        
        # Simple linear regression for each factor
        import statsmodels.api as sm
        
        for factor_name, factor_ret in factor_returns.items():
            # Align lengths
            min_len = min(len(returns), len(factor_ret))
            y = returns[-min_len:]
            X = factor_ret[-min_len:]
            
            # Add constant
            X = sm.add_constant(X)
            
            try:
                model = sm.OLS(y, X).fit()
                
                results[factor_name] = {
                    "coefficient": model.params[1] if len(model.params) > 1 else 0,
                    "p_value": model.pvalues[1] if len(model.pvalues) > 1 else 1,
                    "r_squared": model.rsquared,
                    "t_stat": model.tvalues[1] if len(model.tvalues) > 1 else 0,
                    "significant": model.pvalues[1] < 0.05 if len(model.pvalues) > 1 else False
                }
            except Exception as e:
                logging.error(f"Factor analysis error for {factor_name}: {e}")
        
        return results
    
    def get_signal_correlation(self) -> Dict[str, Any]:
        """
        Calculate correlation between different signals
        """
        # Group trades by signal
        signal_trades = defaultdict(list)
        
        for trade_id, attribution in self.trade_attributions.items():
            for signal in attribution["signals"]:
                name = signal.get("name", signal.get("type", "unknown"))
                signal_trades[name].append({
                    "outcome": 1 if attribution["outcome"] == "win" else -1 if attribution["outcome"] == "loss" else 0,
                    "pnl": attribution["pnl"],
                    "confidence": signal.get("confidence", 0.5)
                })
        
        # Calculate pairwise correlations
        signals = list(signal_trades.keys())
        n = len(signals)
        
        if n < 2:
            return {}
        
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Get outcomes for signals that appeared together
                # This is simplified - would need proper alignment in production
                outcomes_i = [t["outcome"] for t in signal_trades[signals[i]]]
                outcomes_j = [t["outcome"] for t in signal_trades[signals[j]]]
                
                min_len = min(len(outcomes_i), len(outcomes_j))
                if min_len >= 5:
                    corr = np.corrcoef(outcomes_i[:min_len], outcomes_j[:min_len])[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        return {
            "signals": signals,
            "correlation_matrix": correlation_matrix.tolist(),
            "positive_correlations": np.sum(correlation_matrix > 0.3) / 2,
            "negative_correlations": np.sum(correlation_matrix < -0.3) / 2
        }
    
    def generate_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate attribution report
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_trades = [
            t for t in self.trade_attributions.values()
            if datetime.fromisoformat(t["timestamp"]) > cutoff
        ]
        
        if not recent_trades:
            return {"message": f"No trades in last {days} days"}
        
        # Overall statistics
        total_trades = len(recent_trades)
        wins = sum(1 for t in recent_trades if t["outcome"] == "win")
        losses = sum(1 for t in recent_trades if t["outcome"] == "loss")
        
        # Signal usage
        signal_usage = defaultdict(int)
        for trade in recent_trades:
            for signal in trade["signals"]:
                name = signal.get("name", signal.get("type", "unknown"))
                signal_usage[name] += 1
        
        # Best signals
        best_signals = self.get_best_signals(min_appearances=5)
        
        return {
            "period_days": days,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "most_used_signals": dict(sorted(signal_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "best_signals": best_signals[:10],
            "worst_signals": self.get_worst_signals(min_appearances=5)[:5],
            "signal_correlation": self.get_signal_correlation()
        }
    
    def save_attribution_data(self):
        """Save attribution data to disk"""
        try:
            filename = f"{self.data_dir}/attribution_{datetime.now().strftime('%Y%m%d')}.json"
            data = {
                "trade_attributions": self.trade_attributions,
                "signal_performance": dict(self.signal_performance),
                "updated_at": datetime.now().isoformat()
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"💾 Saved attribution data to {filename}")
        except Exception as e:
            logging.error(f"Error saving attribution data: {e}")