"""
Decision Tracker - Tracks human decisions for learning
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from agentic_trading_system.utils.logger import logger as  logging

class DecisionTracker:
    """
    Decision Tracker - Tracks human decisions for learning and analysis
    
    Tracks:
    - Approval rates
    - Response times
    - Decision patterns
    - Performance by decision type
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.log_file = config.get("log_file", "data/decisions.json")
        self.max_history = config.get("max_history", 1000)
        
        # In-memory storage
        self.decisions = []
        self.stats = defaultdict(lambda: defaultdict(int))
        
        # Load existing data
        self._load_decisions()
        
        logging.info(f"✅ DecisionTracker initialized")
    
    async def track_decision(self, decision: Dict[str, Any]) -> str:
        """
        Track a human decision
        """
        decision_id = f"dec_{datetime.now().timestamp()}"
        
        entry = {
            "id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": decision.get("symbol"),
            "decision": decision.get("decision"),  # approve, reject, modify
            "response_time": decision.get("response_time"),  # seconds
            "confidence": decision.get("confidence"),
            "risk_score": decision.get("risk_score"),
            "action": decision.get("action"),  # BUY, SELL, etc.
            "source": decision.get("source"),  # whatsapp, email, etc.
            "notes": decision.get("notes"),
            "outcome": None  # Will be filled later
        }
        
        self.decisions.append(entry)
        
        # Update stats
        self.stats["total_decisions"][entry["decision"]] += 1
        self.stats["by_symbol"][entry["symbol"]][entry["decision"]] += 1
        
        if entry["response_time"]:
            self.stats["response_times"][entry["decision"]].append(entry["response_time"])
        
        # Trim if needed
        if len(self.decisions) > self.max_history:
            self.decisions = self.decisions[-self.max_history:]
        
        # Save to file
        self._save_decisions()
        
        logging.info(f"📝 Tracked decision: {entry['decision']} for {entry['symbol']}")
        
        return decision_id
    
    async def update_outcome(self, decision_id: str, outcome: str, 
                            pnl: float = None) -> bool:
        """
        Update the outcome of a decision
        """
        for decision in self.decisions:
            if decision["id"] == decision_id:
                decision["outcome"] = outcome
                decision["pnl"] = pnl
                
                # Update performance stats
                if outcome == "win":
                    self.stats["wins"][decision["decision"]] += 1
                elif outcome == "loss":
                    self.stats["losses"][decision["decision"]] += 1
                
                self._save_decisions()
                logging.info(f"📊 Updated outcome for {decision_id}: {outcome}")
                return True
        
        return False
    
    def get_approval_rate(self, days: int = 30) -> Dict[str, float]:
        """Get approval rate over specified period"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [
            d for d in self.decisions 
            if datetime.fromisoformat(d["timestamp"]) > cutoff
        ]
        
        if not recent:
            return {}
        
        total = len(recent)
        approvals = len([d for d in recent if d["decision"] == "approve"])
        rejections = len([d for d in recent if d["decision"] == "reject"])
        modifications = len([d for d in recent if d["decision"] == "modify"])
        
        return {
            "approval_rate": approvals / total if total > 0 else 0,
            "rejection_rate": rejections / total if total > 0 else 0,
            "modification_rate": modifications / total if total > 0 else 0,
            "total_decisions": total
        }
    
    def get_average_response_time(self, decision_type: str = None) -> float:
        """Get average response time in seconds"""
        times = []
        
        for decision in self.decisions:
            if decision.get("response_time"):
                if not decision_type or decision["decision"] == decision_type:
                    times.append(decision["response_time"])
        
        if times:
            return sum(times) / len(times)
        
        return 0
    
    def get_decision_by_symbol(self, symbol: str) -> List[Dict]:
        """Get all decisions for a symbol"""
        return [d for d in self.decisions if d["symbol"] == symbol]
    
    def get_performance_by_decision_type(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by decision type"""
        result = {}
        
        for decision_type in ["approve", "reject", "modify"]:
            wins = self.stats["wins"].get(decision_type, 0)
            losses = self.stats["losses"].get(decision_type, 0)
            total = wins + losses
            
            if total > 0:
                result[decision_type] = {
                    "win_rate": wins / total,
                    "wins": wins,
                    "losses": losses,
                    "total": total
                }
        
        return result
    
    def get_decision_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent decisions"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [
            d for d in self.decisions 
            if datetime.fromisoformat(d["timestamp"]) > cutoff
        ]
        
        if not recent:
            return {}
        
        # Group by day
        by_day = defaultdict(list)
        for decision in recent:
            day = decision["timestamp"][:10]
            by_day[day].append(decision)
        
        return {
            "period_days": days,
            "total_decisions": len(recent),
            "by_day": {
                day: {
                    "count": len(decisions),
                    "approvals": len([d for d in decisions if d["decision"] == "approve"]),
                    "rejections": len([d for d in decisions if d["decision"] == "reject"])
                }
                for day, decisions in by_day.items()
            },
            "avg_response_time": self.get_average_response_time(),
            "approval_rate": self.get_approval_rate(days)["approval_rate"]
        }
    
    def _save_decisions(self):
        """Save decisions to file"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump({
                    "decisions": self.decisions,
                    "stats": dict(self.stats),
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving decisions: {e}")
    
    def _load_decisions(self):
        """Load decisions from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.decisions = data.get("decisions", [])
                    
                    # Rebuild stats
                    self.stats = defaultdict(lambda: defaultdict(int))
                    for decision in self.decisions:
                        self.stats["total_decisions"][decision["decision"]] += 1
                        self.stats["by_symbol"][decision["symbol"]][decision["decision"]] += 1
                        
                        if decision.get("response_time"):
                            self.stats["response_times"][decision["decision"]].append(
                                decision["response_time"]
                            )
                        
                        if decision.get("outcome") == "win":
                            self.stats["wins"][decision["decision"]] += 1
                        elif decision.get("outcome") == "loss":
                            self.stats["losses"][decision["decision"]] += 1
                    
                    logging.info(f"📂 Loaded {len(self.decisions)} decisions")
        except Exception as e:
            logging.error(f"Error loading decisions: {e}")