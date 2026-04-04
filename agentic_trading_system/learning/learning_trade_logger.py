import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class TradeOutcomeLogger:
    def __init__(self, discovery_output_path="discovery_outputs/"):
        self.output_path = Path(discovery_output_path)
        self.trades_log = []
    
    def parse_artifact(self, artifact_path):
        """Extract signal, recommendation, and actual outcome from your artifacts"""
        with open(artifact_path) as f:
            data = json.load(f)  # or however your artifacts are structured
        
        return {
            "timestamp": data.get("timestamp"),
            "ticker": data.get("ticker"),
            "signal_strength": data.get("analysis", {}).get("final_score"),
            "recommendation": data.get("portfolio", {}).get("action"),  # BUY/SELL/HOLD
            "confidence": data.get("analysis", {}).get("confidence"),
            "technical_score": data.get("analysis", {}).get("technical_score"),
            "sentiment_score": data.get("analysis", {}).get("sentiment_score"),
            "fundamental_score": data.get("analysis", {}).get("fundamental_score"),
            "risk_score": data.get("risk", {}).get("risk_score"),
            "actual_outcome": None,  # To be filled after trade completes
            "actual_pnl": None,
            "exit_reason": None
        }
    
    def update_with_outcome(self, trade_id, final_pnl, exit_reason):
        """After trade closes, update with actual results"""
        for trade in self.trades_log:
            if trade["trade_id"] == trade_id:
                trade["actual_pnl"] = final_pnl
                trade["actual_outcome"] = "PROFIT" if final_pnl > 0 else "LOSS"
                trade["exit_reason"] = exit_reason
                break
    
    def create_learning_dataset(self):
        """Convert your logs into a format the learning module can use"""
        df = pd.DataFrame(self.trades_log)
        
        # Create simple labels for supervised learning
        df["was_correct"] = (
            (df["recommendation"] == "BUY") & (df["actual_pnl"] > 0)
        ) | (
            (df["recommendation"] == "SELL") & (df["actual_pnl"] < 0)
        )
        
        return df