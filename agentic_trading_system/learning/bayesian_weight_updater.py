import yaml
import numpy as np
from pathlib import Path

class SimpleWeightUpdater:
    def __init__(self, config_path="config/analysis_weights.yaml"):
        self.config_path = Path(config_path)
        self.weights = self.load_current_weights()
        
    def load_current_weights(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def update_from_trades(self, trade_dataframe, learning_rate=0.1):
        """
        trade_dataframe: columns = ['technical_score', 'sentiment_score', 
                                    'fundamental_score', 'was_correct']
        """
        # For each factor, calculate how well it predicted success
        factor_performance = {}
        
        for factor in ['technical_score', 'sentiment_score', 'fundamental_score']:
            # Correlation between factor score and trade success
            successful_trades = trade_dataframe[trade_dataframe['was_correct']]
            unsuccessful_trades = trade_dataframe[~trade_dataframe['was_correct']]
            
            if len(successful_trades) > 0:
                avg_success_score = successful_trades[factor].mean()
            else:
                avg_success_score = 0.5  # neutral prior
            
            if len(unsuccessful_trades) > 0:
                avg_failure_score = unsuccessful_trades[factor].mean()
            else:
                avg_failure_score = 0.5
            
            # Performance = success score - failure score
            factor_performance[factor] = avg_success_score - avg_failure_score
        
        # Apply Laplace smoothing (adds 1 pseudo-success and 1 pseudo-failure)
        # This prevents extreme weights with small sample sizes
        total_performance = sum(abs(v) for v in factor_performance.values())
        
        if total_performance > 0:
            new_weights = {}
            for factor, performance in factor_performance.items():
                # Normalized performance + smoothing
                raw_weight = max(0.1, (performance + 0.5) / (total_performance + 1))
                new_weights[factor] = raw_weight * 100  # convert to percentage
        
            # Blend with old weights (conservative update)
            for factor in self.weights['regime_weights']['default']:
                old = self.weights['regime_weights']['default'][factor]
                new = new_weights.get(factor, old)
                self.weights['regime_weights']['default'][factor] = (
                    (1 - learning_rate) * old + learning_rate * new
                )
            
            self.save_weights()
            return new_weights
        
        return factor_performance
    
    def save_weights(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.weights, f)