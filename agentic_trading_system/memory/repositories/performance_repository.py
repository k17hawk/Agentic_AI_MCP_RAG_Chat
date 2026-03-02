"""
Performance Repository - Data access for performance metrics
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from utils.logger import logger as logging
from memory.models import PerformanceMetrics

class PerformanceRepository:
    """
    Performance Repository - Handles CRUD operations for performance metrics
    
    Storage: JSON file (can be upgraded to PostgreSQL)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/performance")
        self.daily_file = os.path.join(self.data_dir, "daily_metrics.json")
        self.aggregate_file = os.path.join(self.data_dir, "aggregate_metrics.json")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # In-memory cache
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.daily_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Aggregate statistics
        self.aggregate = {
            "all_time": PerformanceMetrics(
                start_date=datetime.now(),
                end_date=datetime.now()
            ),
            "last_30_days": None,
            "last_90_days": None,
            "last_year": None
        }
        
        # Load existing metrics
        self._load()
        
        logging.info(f"✅ PerformanceRepository initialized")
    
    def save_daily(self, metrics: PerformanceMetrics) -> str:
        """
        Save daily performance metrics
        """
        period_id = metrics.period_id
        self.daily_metrics[period_id] = metrics
        
        # Update aggregates
        self._update_aggregates()
        
        self._save_daily()
        
        logging.info(f"✅ Daily metrics saved: {period_id}")
        
        return period_id
    
    def get_daily(self, date: str) -> Optional[PerformanceMetrics]:
        """
        Get metrics for a specific date
        """
        return self.daily_metrics.get(date)
    
    def get_date_range(self, start_date: datetime, end_date: datetime) -> List[PerformanceMetrics]:
        """
        Get metrics for a date range
        """
        metrics = []
        for metric in self.daily_metrics.values():
            if start_date <= metric.start_date <= end_date:
                metrics.append(metric)
        
        return sorted(metrics, key=lambda x: x.start_date)
    
    def get_aggregate(self, period: str = "all_time") -> Optional[PerformanceMetrics]:
        """
        Get aggregate metrics for a period
        """
        return self.aggregate.get(period)
    
    def calculate_performance(self, trades: List[Any], 
                             start_date: datetime,
                             end_date: datetime) -> PerformanceMetrics:
        """
        Calculate performance metrics from trades
        """
        metrics = PerformanceMetrics(
            start_date=start_date,
            end_date=end_date
        )
        
        # Filter trades in period
        period_trades = [
            t for t in trades 
            if start_date <= t.entry_time <= end_date
        ]
        
        if not period_trades:
            return metrics
        
        # Calculate basic stats
        metrics.total_trades = len(period_trades)
        
        for trade in period_trades:
            if trade.pnl is None:
                continue
            
            metrics.total_turnover += abs(trade.total_value)
            metrics.total_volume += trade.quantity
            
            if trade.pnl > 0:
                metrics.winning_trades += 1
                metrics.gross_profit += trade.pnl
                metrics.average_win = (
                    (metrics.average_win * (metrics.winning_trades - 1) + trade.pnl) /
                    metrics.winning_trades
                )
                if trade.pnl > metrics.largest_win:
                    metrics.largest_win = trade.pnl
                    
            elif trade.pnl < 0:
                metrics.losing_trades += 1
                metrics.gross_loss += abs(trade.pnl)
                metrics.average_loss = (
                    (metrics.average_loss * (metrics.losing_trades - 1) + abs(trade.pnl)) /
                    metrics.losing_trades
                )
                if abs(trade.pnl) > metrics.largest_loss:
                    metrics.largest_loss = abs(trade.pnl)
            else:
                metrics.breakeven_trades += 1
        
        metrics.net_pnl = metrics.gross_profit - metrics.gross_loss
        
        # Calculate ratios
        total_trades = metrics.winning_trades + metrics.losing_trades
        if total_trades > 0:
            metrics.win_rate = metrics.winning_trades / total_trades
        
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        # Strategy breakdown
        by_strategy = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        for trade in period_trades:
            strategy = trade.strategy
            by_strategy[strategy]["trades"] += 1
            if trade.pnl:
                by_strategy[strategy]["pnl"] += trade.pnl
                if trade.pnl > 0:
                    by_strategy[strategy]["wins"] += 1
        
        metrics.by_strategy = dict(by_strategy)
        
        return metrics
    
    def _update_aggregates(self):
        """Update aggregate statistics"""
        all_trades = []
        for daily in self.daily_metrics.values():
            # This would need trade data from trade repository
            pass
    
    def _save_daily(self):
        """Save daily metrics to disk"""
        try:
            data = {
                "daily_metrics": {k: v.dict() for k, v in self.daily_metrics.items()},
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.daily_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving daily metrics: {e}")
    
    def _load(self):
        """Load metrics from disk"""
        try:
            if os.path.exists(self.daily_file):
                with open(self.daily_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.get("daily_metrics", {}).items():
                        self.daily_metrics[k] = PerformanceMetrics(**v)
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")