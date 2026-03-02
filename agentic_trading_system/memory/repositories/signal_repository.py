"""
Signal Repository - Data access for trading signals
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from utils.logger import logger as logging
from memory.models import Signal

class SignalRepository:
    """
    Signal Repository - Handles CRUD operations for signals
    
    Storage: JSON file (can be upgraded to PostgreSQL)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/signals")
        self.current_file = os.path.join(self.data_dir, "signals.json")
        self.stats_file = os.path.join(self.data_dir, "signal_stats.json")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # In-memory cache
        self.signals: Dict[str, Signal] = {}
        self.by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.by_type: Dict[str, List[str]] = defaultdict(list)
        self.by_source: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_signals": 0,
            "by_type": defaultdict(int),
            "by_source": defaultdict(int),
            "accuracy": defaultdict(lambda: {"correct": 0, "total": 0})
        }
        
        # Load existing signals
        self._load()
        
        logging.info(f"✅ SignalRepository initialized with {len(self.signals)} signals")
    
    def save(self, signal: Signal) -> str:
        """
        Save a signal
        """
        # Store in memory
        self.signals[signal.signal_id] = signal
        
        # Update indexes
        self.by_symbol[signal.symbol].append(signal.signal_id)
        self.by_type[signal.signal_type].append(signal.signal_id)
        self.by_source[signal.source].append(signal.signal_id)
        
        # Update stats
        self.stats["total_signals"] += 1
        self.stats["by_type"][signal.signal_type] += 1
        self.stats["by_source"][signal.source] += 1
        
        # Persist to disk
        self._save()
        
        logging.debug(f"✅ Signal saved: {signal.signal_id} - {signal.symbol} ({signal.signal_type})")
        
        return signal.signal_id
    
    def save_many(self, signals: List[Signal]) -> List[str]:
        """
        Save multiple signals
        """
        signal_ids = []
        for signal in signals:
            signal_ids.append(self.save(signal))
        return signal_ids
    
    def get(self, signal_id: str) -> Optional[Signal]:
        """
        Get a signal by ID
        """
        return self.signals.get(signal_id)
    
    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Signal]:
        """
        Get signals for a symbol
        """
        signal_ids = self.by_symbol.get(symbol, [])[-limit:]
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]
    
    def get_by_type(self, signal_type: str, limit: int = 100) -> List[Signal]:
        """
        Get signals by type
        """
        signal_ids = self.by_type.get(signal_type, [])[-limit:]
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]
    
    def get_by_source(self, source: str, limit: int = 100) -> List[Signal]:
        """
        Get signals by source
        """
        signal_ids = self.by_source.get(source, [])[-limit:]
        return [self.signals[sid] for sid in signal_ids if sid in self.signals]
    
    def get_recent(self, limit: int = 100) -> List[Signal]:
        """
        Get most recent signals
        """
        sorted_signals = sorted(
            self.signals.values(),
            key=lambda x: x.generated_at,
            reverse=True
        )
        return sorted_signals[:limit]
    
    def get_active_signals(self) -> List[Signal]:
        """
        Get signals that haven't expired
        """
        now = datetime.now()
        return [s for s in self.signals.values() if s.is_active and s.expires_at > now]
    
    def update_outcome(self, signal_id: str, led_to_trade: bool, 
                      trade_id: Optional[str] = None) -> bool:
        """
        Update signal outcome (whether it led to a trade)
        """
        if signal_id not in self.signals:
            return False
        
        signal = self.signals[signal_id]
        signal.led_to_trade = led_to_trade
        if trade_id:
            signal.trade_id = trade_id
        
        # Update accuracy stats
        signal_type = signal.signal_type
        self.stats["accuracy"][signal_type]["total"] += 1
        if led_to_trade:
            self.stats["accuracy"][signal_type]["correct"] += 1
        
        # Calculate accuracy
        total = self.stats["accuracy"][signal_type]["total"]
        correct = self.stats["accuracy"][signal_type]["correct"]
        if total > 0:
            signal.accuracy = correct / total
        
        signal.updated_at = datetime.now()
        
        self._save()
        self._save_stats()
        
        return True
    
    def get_accuracy_by_type(self) -> Dict[str, float]:
        """
        Get accuracy statistics by signal type
        """
        accuracy = {}
        for signal_type, stats in self.stats["accuracy"].items():
            if stats["total"] > 0:
                accuracy[signal_type] = stats["correct"] / stats["total"]
        return accuracy
    
    def get_best_performing_signals(self, min_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Get best performing signal types
        """
        performance = []
        for signal_type, stats in self.stats["accuracy"].items():
            if stats["total"] >= min_samples:
                performance.append({
                    "signal_type": signal_type,
                    "accuracy": stats["correct"] / stats["total"],
                    "samples": stats["total"],
                    "correct": stats["correct"]
                })
        
        return sorted(performance, key=lambda x: x["accuracy"], reverse=True)
    
    def get_recent_accuracy(self, days: int = 7) -> Dict[str, float]:
        """
        Get accuracy for recent signals
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_signals = [s for s in self.signals.values() if s.generated_at > cutoff]
        
        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for signal in recent_signals:
            if signal.led_to_trade is not None:
                by_type[signal.signal_type]["total"] += 1
                if signal.led_to_trade:
                    by_type[signal.signal_type]["correct"] += 1
        
        accuracy = {}
        for signal_type, stats in by_type.items():
            if stats["total"] > 0:
                accuracy[signal_type] = stats["correct"] / stats["total"]
        
        return accuracy
    
    def cleanup_expired(self) -> int:
        """
        Mark expired signals as inactive
        """
        now = datetime.now()
        count = 0
        
        for signal in self.signals.values():
            if signal.is_active and signal.expires_at <= now:
                signal.is_active = False
                count += 1
        
        if count > 0:
            self._save()
            logging.info(f"🧹 Marked {count} expired signals as inactive")
        
        return count
    
    def delete_old_signals(self, days: int = 30) -> int:
        """
        Delete signals older than specified days
        """
        cutoff = datetime.now() - timedelta(days=days)
        to_delete = []
        
        for signal_id, signal in self.signals.items():
            if signal.generated_at < cutoff:
                to_delete.append(signal_id)
        
        for signal_id in to_delete:
            self.delete(signal_id)
        
        return len(to_delete)
    
    def delete(self, signal_id: str) -> bool:
        """
        Delete a signal
        """
        if signal_id not in self.signals:
            return False
        
        signal = self.signals[signal_id]
        
        # Remove from indexes
        if signal.symbol in self.by_symbol:
            if signal_id in self.by_symbol[signal.symbol]:
                self.by_symbol[signal.symbol].remove(signal_id)
        
        if signal.signal_type in self.by_type:
            if signal_id in self.by_type[signal.signal_type]:
                self.by_type[signal.signal_type].remove(signal_id)
        
        if signal.source in self.by_source:
            if signal_id in self.by_source[signal.source]:
                self.by_source[signal.source].remove(signal_id)
        
        # Remove from memory
        del self.signals[signal_id]
        
        self._save()
        
        return True
    
    def count(self) -> int:
        """Get total number of signals"""
        return len(self.signals)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal statistics"""
        return {
            "total_signals": self.stats["total_signals"],
            "by_type": dict(self.stats["by_type"]),
            "by_source": dict(self.stats["by_source"]),
            "accuracy_by_type": self.get_accuracy_by_type(),
            "active_signals": len(self.get_active_signals()),
            "recent_accuracy": self.get_recent_accuracy()
        }
    
    def _save(self):
        """Save signals to disk"""
        try:
            data = {
                "signals": {sid: signal.dict() for sid, signal in self.signals.items()},
                "indexes": {
                    "by_symbol": dict(self.by_symbol),
                    "by_type": dict(self.by_type),
                    "by_source": dict(self.by_source)
                },
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.current_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving signals: {e}")
    
    def _save_stats(self):
        """Save statistics separately"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error saving signal stats: {e}")
    
    def _load(self):
        """Load signals from disk"""
        try:
            if os.path.exists(self.current_file):
                with open(self.current_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load signals
                    for sid, signal_data in data.get("signals", {}).items():
                        self.signals[sid] = Signal(**signal_data)
                    
                    # Load indexes
                    self.by_symbol = defaultdict(list, data.get("indexes", {}).get("by_symbol", {}))
                    self.by_type = defaultdict(list, data.get("indexes", {}).get("by_type", {}))
                    self.by_source = defaultdict(list, data.get("indexes", {}).get("by_source", {}))
            
            # Load stats
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.stats.update(json.load(f))
                    
        except Exception as e:
            logging.error(f"Error loading signals: {e}")