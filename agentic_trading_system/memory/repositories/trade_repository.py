"""
Trade Repository - Data access for trades
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from utils.logger import logger as logging
from memory.models import Trade

class TradeRepository:
    """
    Trade Repository - Handles CRUD operations for trades
    
    Storage: JSON file (can be upgraded to PostgreSQL)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/trades")
        self.current_file = os.path.join(self.data_dir, "current_trades.json")
        self.archive_dir = os.path.join(self.data_dir, "archive")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # In-memory cache
        self.trades: Dict[str, Trade] = {}
        self.by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.by_date: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing trades
        self._load()
        
        logging.info(f"✅ TradeRepository initialized with {len(self.trades)} trades")
    
    def save(self, trade: Trade) -> str:
        """
        Save a trade
        """
        # Store in memory
        self.trades[trade.trade_id] = trade
        
        # Update indexes
        self.by_symbol[trade.symbol].append(trade.trade_id)
        date_key = trade.entry_time.strftime("%Y-%m-%d")
        self.by_date[date_key].append(trade.trade_id)
        
        # Persist to disk
        self._save()
        
        logging.info(f"✅ Trade saved: {trade.trade_id} - {trade.symbol} {trade.action} {trade.quantity}")
        
        return trade.trade_id
    
    def save_many(self, trades: List[Trade]) -> List[str]:
        """
        Save multiple trades
        """
        trade_ids = []
        for trade in trades:
            trade_ids.append(self.save(trade))
        return trade_ids
    
    def get(self, trade_id: str) -> Optional[Trade]:
        """
        Get a trade by ID
        """
        return self.trades.get(trade_id)
    
    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get trades for a symbol
        """
        trade_ids = self.by_symbol.get(symbol, [])[-limit:]
        return [self.trades[tid] for tid in trade_ids if tid in self.trades]
    
    def get_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Trade]:
        """
        Get trades within a date range
        """
        trades = []
        for trade in self.trades.values():
            if start_date <= trade.entry_time <= end_date:
                trades.append(trade)
        
        # Sort by date
        trades.sort(key=lambda x: x.entry_time, reverse=True)
        return trades
    
    def get_recent(self, limit: int = 100) -> List[Trade]:
        """
        Get most recent trades
        """
        sorted_trades = sorted(
            self.trades.values(),
            key=lambda x: x.entry_time,
            reverse=True
        )
        return sorted_trades[:limit]
    
    def get_open_trades(self) -> List[Trade]:
        """
        Get trades without exit time (still open)
        """
        return [t for t in self.trades.values() if t.exit_time is None]
    
    def update_outcome(self, trade_id: str, pnl: float, exit_time: datetime) -> bool:
        """
        Update trade outcome (close position)
        """
        if trade_id not in self.trades:
            return False
        
        trade = self.trades[trade_id]
        trade.pnl = pnl
        trade.exit_time = exit_time
        
        if trade.entry_time:
            hold_seconds = (exit_time - trade.entry_time).total_seconds()
            trade.hold_time_seconds = int(hold_seconds)
        
        if pnl > 0:
            trade.outcome = "win"
        elif pnl < 0:
            trade.outcome = "loss"
        else:
            trade.outcome = "breakeven"
        
        trade.updated_at = datetime.now()
        trade.version += 1
        
        self._save()
        
        return True
    
    def delete(self, trade_id: str) -> bool:
        """
        Delete a trade
        """
        if trade_id not in self.trades:
            return False
        
        trade = self.trades[trade_id]
        
        # Remove from indexes
        if trade.symbol in self.by_symbol:
            if trade_id in self.by_symbol[trade.symbol]:
                self.by_symbol[trade.symbol].remove(trade_id)
        
        date_key = trade.entry_time.strftime("%Y-%m-%d")
        if date_key in self.by_date:
            if trade_id in self.by_date[date_key]:
                self.by_date[date_key].remove(trade_id)
        
        # Remove from memory
        del self.trades[trade_id]
        
        # Archive to separate file before deleting
        self._archive_trade(trade)
        
        self._save()
        
        return True
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trading statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trades.values() if t.entry_time > cutoff]
        
        if not recent_trades:
            return {}
        
        wins = [t for t in recent_trades if t.outcome == "win"]
        losses = [t for t in recent_trades if t.outcome == "loss"]
        
        total_pnl = sum(t.pnl for t in recent_trades if t.pnl is not None)
        
        return {
            "period_days": days,
            "total_trades": len(recent_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(recent_trades) if recent_trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(recent_trades) if recent_trades else 0,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        }
    
    def get_performance_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance breakdown by strategy
        """
        by_strategy = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        
        for trade in self.trades.values():
            if trade.outcome is None:
                continue
            
            strategy = trade.strategy
            by_strategy[strategy]["trades"] += 1
            by_strategy[strategy]["pnl"] += trade.pnl or 0
            
            if trade.outcome == "win":
                by_strategy[strategy]["wins"] += 1
            elif trade.outcome == "loss":
                by_strategy[strategy]["losses"] += 1
        
        # Calculate win rates
        for strategy in by_strategy:
            total = by_strategy[strategy]["wins"] + by_strategy[strategy]["losses"]
            if total > 0:
                by_strategy[strategy]["win_rate"] = by_strategy[strategy]["wins"] / total
            else:
                by_strategy[strategy]["win_rate"] = 0
        
        return dict(by_strategy)
    
    def count(self) -> int:
        """Get total number of trades"""
        return len(self.trades)
    
    def _save(self):
        """Save trades to disk"""
        try:
            data = {
                "trades": {tid: trade.dict() for tid, trade in self.trades.items()},
                "indexes": {
                    "by_symbol": dict(self.by_symbol),
                    "by_date": dict(self.by_date)
                },
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.current_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving trades: {e}")
    
    def _load(self):
        """Load trades from disk"""
        try:
            if os.path.exists(self.current_file):
                with open(self.current_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load trades
                    for tid, trade_data in data.get("trades", {}).items():
                        self.trades[tid] = Trade(**trade_data)
                    
                    # Load indexes
                    self.by_symbol = defaultdict(list, data.get("indexes", {}).get("by_symbol", {}))
                    self.by_date = defaultdict(list, data.get("indexes", {}).get("by_date", {}))
                    
        except Exception as e:
            logging.error(f"Error loading trades: {e}")
    
    def _archive_trade(self, trade: Trade):
        """Archive a deleted trade"""
        try:
            archive_file = os.path.join(
                self.archive_dir,
                f"trade_{trade.trade_id}_{datetime.now().strftime('%Y%m%d')}.json"
            )
            with open(archive_file, 'w') as f:
                json.dump(trade.dict(), f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error archiving trade: {e}")