"""
Signal Attribution - Analyzes which signals contributed to performance
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from utils.logger import logger as  logging

class SignalAttribution:
    """
    Signal Attribution - Analyzes contribution of different signals to performance
    
    Features:
    - Signal contribution analysis
    - Signal performance tracking
    - Signal interaction effects
    - Rolling attribution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.min_samples = config.get("min_samples", 10)
        self.decay_factor = config.get("decay_factor", 0.95)  # Exponential decay
        
        logging.info(f"✅ SignalAttribution initialized")
    
    def calculate_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate signal attribution for trades
        """
        if not trades:
            return {}
        
        # Initialize containers
        signal_contributions = defaultdict(lambda: {
            "total_pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "avg_confidence": 0.0,
            "avg_contribution": 0.0
        })
        
        trade_signals = defaultdict(list)  # trade_id -> signals
        
        # Process each trade
        for trade in trades:
            trade_id = trade.get('trade_id', str(id(trade)))
            signals = trade.get('signals', [])
            pnl = trade.get('pnl', 0)
            outcome = trade.get('outcome', 'unknown')
            
            if not signals:
                continue
            
            # Calculate signal weights
            total_confidence = sum(s.get('confidence', 0.5) for s in signals)
            
            if total_confidence == 0:
                continue
            
            # Attribute P&L to signals
            for signal in signals:
                signal_name = signal.get('name', signal.get('type', 'unknown'))
                confidence = signal.get('confidence', 0.5)
                weight = confidence / total_confidence
                
                # Contribution is weighted by confidence
                contribution = pnl * weight
                
                # Update signal stats
                signal_contributions[signal_name]["total_pnl"] += contribution
                signal_contributions[signal_name]["trades"] += 1
                signal_contributions[signal_name]["avg_confidence"] += confidence
                
                if pnl > 0:
                    signal_contributions[signal_name]["wins"] += 1
                elif pnl < 0:
                    signal_contributions[signal_name]["losses"] += 1
                
                # Store for trade-level analysis
                trade_signals[trade_id].append({
                    "signal": signal_name,
                    "confidence": confidence,
                    "weight": weight,
                    "contribution": contribution
                })
        
        # Calculate averages and win rates
        for signal, data in signal_contributions.items():
            if data["trades"] > 0:
                data["avg_confidence"] /= data["trades"]
                data["avg_contribution"] = data["total_pnl"] / data["trades"]
                data["win_rate"] = data["wins"] / (data["wins"] + data["losses"]) if (data["wins"] + data["losses"]) > 0 else 0
        
        # Calculate total attribution
        total_pnl = sum(d["total_pnl"] for d in signal_contributions.values())
        
        # Calculate percentages
        signal_percentages = {}
        for signal, data in signal_contributions.items():
            if total_pnl != 0:
                signal_percentages[signal] = (data["total_pnl"] / total_pnl) * 100
        
        # Identify top contributors
        sorted_signals = sorted(
            signal_contributions.items(),
            key=lambda x: abs(x[1]["total_pnl"]),
            reverse=True
        )
        
        return {
            "total_attributed_pnl": float(total_pnl),
            "num_signals": len(signal_contributions),
            "signal_contributions": {
                signal: {
                    "total_pnl": float(data["total_pnl"]),
                    "percentage": float(signal_percentages.get(signal, 0)),
                    "trades": data["trades"],
                    "win_rate": float(data.get("win_rate", 0)),
                    "avg_confidence": float(data["avg_confidence"]),
                    "avg_contribution": float(data["avg_contribution"])
                }
                for signal, data in signal_contributions.items()
            },
            "top_contributors": [
                {
                    "signal": signal,
                    "pnl": float(data["total_pnl"]),
                    "percentage": float(signal_percentages.get(signal, 0)),
                    "trades": data["trades"]
                }
                for signal, data in sorted_signals[:10]
            ],
            "trade_attributions": dict(trade_signals)
        }
    
    def calculate_rolling_attribution(self, trades: List[Dict[str, Any]], 
                                     window_days: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate rolling signal attribution over time
        """
        if len(trades) < 2:
            return []
        
        # Sort trades by date
        sorted_trades = sorted(
            [t for t in trades if t.get('exit_time')],
            key=lambda x: x['exit_time']
        )
        
        if len(sorted_trades) < 2:
            return []
        
        # Create rolling windows
        start_date = datetime.fromisoformat(sorted_trades[0]['exit_time'])
        end_date = datetime.fromisoformat(sorted_trades[-1]['exit_time'])
        
        current_date = start_date
        rolling_results = []
        
        while current_date <= end_date:
            window_start = current_date - timedelta(days=window_days)
            
            # Get trades in window
            window_trades = [
                t for t in sorted_trades
                if window_start <= datetime.fromisoformat(t['exit_time']) <= current_date
            ]
            
            if window_trades:
                attribution = self.calculate_attribution(window_trades)
                rolling_results.append({
                    "date": current_date.isoformat(),
                    "attribution": attribution
                })
            
            current_date += timedelta(days=1)
        
        return rolling_results
    
    def calculate_signal_correlation(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate correlation between different signals
        """
        # Build signal occurrence matrix
        signal_trades = defaultdict(list)
        
        for trade in trades:
            signals = trade.get('signals', [])
            for signal in signals:
                signal_name = signal.get('name', signal.get('type', 'unknown'))
                signal_trades[signal_name].append({
                    'pnl': trade.get('pnl', 0),
                    'outcome': 1 if trade.get('pnl', 0) > 0 else 0
                })
        
        signal_names = list(signal_trades.keys())
        n = len(signal_names)
        
        if n < 2:
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Get P&L for trades where both signals appeared
                # This is simplified - would need proper alignment
                common_pnl_i = [t['pnl'] for t in signal_trades[signal_names[i]]]
                common_pnl_j = [t['pnl'] for t in signal_trades[signal_names[j]]]
                
                min_len = min(len(common_pnl_i), len(common_pnl_j))
                if min_len >= 5:
                    corr = np.corrcoef(common_pnl_i[:min_len], common_pnl_j[:min_len])[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        return {
            "signals": signal_names,
            "correlation_matrix": correlation_matrix.tolist(),
            "positive_correlations": int(np.sum(correlation_matrix > 0.3) / 2),
            "negative_correlations": int(np.sum(correlation_matrix < -0.3) / 2)
        }
    
    def get_signal_performance_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get performance summary by signal
        """
        attribution = self.calculate_attribution(trades)
        
        # Calculate additional metrics
        signal_metrics = {}
        
        for signal, data in attribution.get("signal_contributions", {}).items():
            # Calculate risk-adjusted metrics
            if data["trades"] > 0:
                # Simple Sharpe-like metric
                avg_pnl = data["avg_contribution"]
                # Would need std dev for proper Sharpe
                
                signal_metrics[signal] = {
                    **data,
                    "score": data["win_rate"] * abs(avg_pnl) if avg_pnl else 0
                }
        
        # Rank signals
        ranked_signals = sorted(
            signal_metrics.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return {
            "signal_metrics": signal_metrics,
            "top_signals": [s for s, _ in ranked_signals[:5]],
            "bottom_signals": [s for s, _ in ranked_signals[-5:]],
            "best_signal": ranked_signals[0][0] if ranked_signals else None
        }