"""
Feedback Logger - Logs feedback from human decisions to improve the system
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from agentic_trading_system.utils.logger import logger as logging

class FeedbackLogger:
    """
    Feedback Logger - Logs feedback to improve the system
    
    Uses human decisions to:
    - Adjust trigger thresholds
    - Tune analysis weights
    - Improve risk parameters
    - Train ML models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.log_file = config.get("log_file", "data/feedback.json")
        self.max_history = config.get("max_history", 1000)
        
        # Feedback categories
        self.feedback = []
        self.trigger_feedback = defaultdict(list)
        self.analysis_feedback = defaultdict(list)
        self.risk_feedback = defaultdict(list)
        
        # Load existing data
        self._load_feedback()
        
        logging.info(f"✅ FeedbackLogger initialized")
    
    async def log_feedback(self, feedback_type: str, data: Dict[str, Any]) -> str:
        """
        Log feedback from a human decision
        """
        feedback_id = f"fb_{datetime.now().timestamp()}"
        
        entry = {
            "id": feedback_id,
            "type": feedback_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "processed": False
        }
        
        self.feedback.append(entry)
        
        # Categorize
        if feedback_type == "trigger":
            trigger_name = data.get("trigger_name", "unknown")
            self.trigger_feedback[trigger_name].append(data)
        elif feedback_type == "analysis":
            analysis_type = data.get("analysis_type", "unknown")
            self.analysis_feedback[analysis_type].append(data)
        elif feedback_type == "risk":
            risk_type = data.get("risk_type", "unknown")
            self.risk_feedback[risk_type].append(data)
        
        # Trim if needed
        if len(self.feedback) > self.max_history:
            self.feedback = self.feedback[-self.max_history:]
        
        # Save to file
        self._save_feedback()
        
        logging.info(f"📝 Logged {feedback_type} feedback: {feedback_id}")
        
        return feedback_id
    
    async def log_trigger_feedback(self, trigger_name: str, triggered: bool,
                                   outcome: str, metadata: Dict = None) -> str:
        """Log feedback about a trigger"""
        data = {
            "trigger_name": trigger_name,
            "triggered": triggered,
            "outcome": outcome,
            "metadata": metadata or {}
        }
        return await self.log_feedback("trigger", data)
    
    async def log_analysis_feedback(self, symbol: str, analysis_type: str,
                                   predicted_score: float, actual_outcome: str,
                                   metadata: Dict = None) -> str:
        """Log feedback about analysis accuracy"""
        data = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "predicted_score": predicted_score,
            "actual_outcome": actual_outcome,
            "metadata": metadata or {}
        }
        return await self.log_feedback("analysis", data)
    
    async def log_risk_feedback(self, symbol: str, risk_score: float,
                               stop_loss_hit: bool, max_drawdown: float,
                               metadata: Dict = None) -> str:
        """Log feedback about risk parameters"""
        data = {
            "symbol": symbol,
            "risk_score": risk_score,
            "stop_loss_hit": stop_loss_hit,
            "max_drawdown": max_drawdown,
            "metadata": metadata or {}
        }
        return await self.log_feedback("risk", data)
    
    def get_trigger_performance(self, trigger_name: str = None) -> Dict[str, Any]:
        """Get performance metrics for triggers"""
        result = {}
        
        triggers = [trigger_name] if trigger_name else self.trigger_feedback.keys()
        
        for name in triggers:
            feedback = self.trigger_feedback.get(name, [])
            
            if not feedback:
                continue
            
            total = len(feedback)
            correct = len([f for f in feedback if self._was_correct(f)])
            
            result[name] = {
                "total": total,
                "accuracy": correct / total if total > 0 else 0,
                "true_positives": len([f for f in feedback if f["triggered"] and f["outcome"] == "win"]),
                "false_positives": len([f for f in feedback if f["triggered"] and f["outcome"] == "loss"]),
                "true_negatives": len([f for f in feedback if not f["triggered"] and f["outcome"] == "loss"]),
                "false_negatives": len([f for f in feedback if not f["triggered"] and f["outcome"] == "win"])
            }
        
        return result
    
    def get_analysis_accuracy(self, analysis_type: str = None) -> Dict[str, Any]:
        """Get accuracy metrics for analysis"""
        result = {}
        
        types = [analysis_type] if analysis_type else self.analysis_feedback.keys()
        
        for atype in types:
            feedback = self.analysis_feedback.get(atype, [])
            
            if not feedback:
                continue
            
            # Calculate correlation between predicted score and actual outcome
            scores = []
            outcomes = []
            
            for f in feedback:
                scores.append(f["predicted_score"])
                outcomes.append(1 if f["actual_outcome"] == "win" else 0)
            
            if len(scores) > 1:
                import numpy as np
                correlation = np.corrcoef(scores, outcomes)[0, 1]
            else:
                correlation = 0
            
            result[atype] = {
                "total": len(feedback),
                "avg_predicted_score": sum(scores) / len(scores),
                "correlation": float(correlation) if not np.isnan(correlation) else 0,
                "by_outcome": {
                    "wins": len([f for f in feedback if f["actual_outcome"] == "win"]),
                    "losses": len([f for f in feedback if f["actual_outcome"] == "loss"])
                }
            }
        
        return result
    
    def get_risk_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness metrics for risk parameters"""
        result = {}
        
        for risk_type, feedback in self.risk_feedback.items():
            if not feedback:
                continue
            
            stop_loss_hits = len([f for f in feedback if f["stop_loss_hit"]])
            avg_drawdown = sum(f["max_drawdown"] for f in feedback) / len(feedback)
            
            result[risk_type] = {
                "total": len(feedback),
                "stop_loss_hit_rate": stop_loss_hits / len(feedback),
                "avg_max_drawdown": avg_drawdown,
                "risk_scores": [f["risk_score"] for f in feedback]
            }
        
        return result
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for system improvement based on feedback"""
        suggestions = []
        
        # Check trigger performance
        trigger_perf = self.get_trigger_performance()
        for trigger, metrics in trigger_perf.items():
            if metrics["accuracy"] < 0.6:
                suggestions.append({
                    "component": "trigger",
                    "name": trigger,
                    "issue": "Low accuracy",
                    "suggestion": "Consider adjusting thresholds",
                    "metrics": metrics
                })
        
        # Check analysis correlation
        analysis_acc = self.get_analysis_accuracy()
        for atype, metrics in analysis_acc.items():
            if metrics["correlation"] < 0.3:
                suggestions.append({
                    "component": "analysis",
                    "name": atype,
                    "issue": "Poor prediction correlation",
                    "suggestion": "Review analysis methodology",
                    "metrics": metrics
                })
        
        # Check stop loss effectiveness
        risk_effect = self.get_risk_effectiveness()
        for rtype, metrics in risk_effect.items():
            if metrics["stop_loss_hit_rate"] > 0.7:
                suggestions.append({
                    "component": "risk",
                    "name": rtype,
                    "issue": "High stop loss hit rate",
                    "suggestion": "Consider widening stops",
                    "metrics": metrics
                })
        
        return suggestions
    
    def _was_correct(self, feedback: Dict) -> bool:
        """Determine if trigger was correct"""
        triggered = feedback["triggered"]
        outcome = feedback["outcome"]
        
        return (triggered and outcome == "win") or (not triggered and outcome == "loss")
    
    def _save_feedback(self):
        """Save feedback to file"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump({
                    "feedback": self.feedback,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving feedback: {e}")
    
    def _load_feedback(self):
        """Load feedback from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.feedback = data.get("feedback", [])
                    
                    # Rebuild categorized feedback
                    for entry in self.feedback:
                        if entry["type"] == "trigger":
                            name = entry["data"].get("trigger_name", "unknown")
                            self.trigger_feedback[name].append(entry["data"])
                        elif entry["type"] == "analysis":
                            atype = entry["data"].get("analysis_type", "unknown")
                            self.analysis_feedback[atype].append(entry["data"])
                        elif entry["type"] == "risk":
                            rtype = entry["data"].get("risk_type", "unknown")
                            self.risk_feedback[rtype].append(entry["data"])
                    
                    logging.info(f"📂 Loaded {len(self.feedback)} feedback entries")
        except Exception as e:
            logging.error(f"Error loading feedback: {e}")