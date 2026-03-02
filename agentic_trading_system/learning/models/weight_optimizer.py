"""
Weight Optimizer - Optimizes weights for different signals using Bayesian approach
"""
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import stats
from collections import defaultdict
import json
from utils.logger import logger as logging

class WeightOptimizer:
    """
    Weight Optimizer - Optimizes weights for different signals using Bayesian updating
    
    Features:
    - Bayesian weight updating based on performance
    - Prior knowledge incorporation
    - Uncertainty quantification
    - Adaptive learning rate
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Prior parameters
        self.default_weight = config.get("default_weight", 0.5)
        self.prior_strength = config.get("prior_strength", 10)  # Higher = stronger prior
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.min_weight = config.get("min_weight", 0.05)
        self.max_weight = config.get("max_weight", 0.95)
        
        # Weight storage
        self.weights = {}  # signal_name -> current weight
        self.weight_history = defaultdict(list)  # signal_name -> list of (timestamp, weight)
        self.performance_history = defaultdict(list)  # signal_name -> list of outcomes
        
        # Uncertainty tracking
        self.weight_uncertainty = {}  # signal_name -> uncertainty (0-1)
        
        logging.info(f"✅ WeightOptimizer initialized")
    
    def initialize_weights(self, signal_names: List[str], 
                          initial_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Initialize weights for signals
        """
        if initial_weights:
            for name, weight in initial_weights.items():
                self.weights[name] = max(self.min_weight, min(self.max_weight, weight))
                self.weight_uncertainty[name] = 0.5  # Start with medium uncertainty
        
        # Initialize remaining with default
        for name in signal_names:
            if name not in self.weights:
                self.weights[name] = self.default_weight
                self.weight_uncertainty[name] = 0.5
        
        return self.weights.copy()
    
    def update_weights(self, signal_performance: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Update weights based on recent performance
        """
        for signal_name, performances in signal_performance.items():
            if signal_name not in self.weights:
                self.weights[signal_name] = self.default_weight
                self.weight_uncertainty[signal_name] = 0.5
            
            # Calculate performance metrics
            outcomes = [p["outcome"] for p in performances if "outcome" in p]
            confidences = [p.get("confidence", 0.5) for p in performances]
            
            if not outcomes:
                continue
            
            # Convert outcomes to numeric (win=1, loss=0)
            numeric_outcomes = [1 if o == "win" else 0 if o == "loss" else 0.5 for o in outcomes]
            
            # Bayesian update
            new_weight = self._bayesian_update(
                self.weights[signal_name],
                numeric_outcomes,
                confidences,
                self.weight_uncertainty[signal_name]
            )
            
            # Apply constraints
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Update uncertainty (decreases with more evidence)
            n = len(outcomes)
            self.weight_uncertainty[signal_name] = max(0.1, 1.0 / (1 + n / 10))
            
            # Store
            old_weight = self.weights[signal_name]
            self.weights[signal_name] = new_weight
            
            logging.debug(f"📊 Updated {signal_name}: {old_weight:.3f} -> {new_weight:.3f}")
        
        return self.weights.copy()
    
    def _bayesian_update(self, current_weight: float, outcomes: List[float],
                        confidences: List[float], uncertainty: float) -> float:
        """
        Perform Bayesian update of weight
        """
        # Convert to Beta distribution parameters
        # Beta(α, β) where α = weight * strength, β = (1-weight) * strength
        
        # Prior strength based on uncertainty
        prior_strength = self.prior_strength * (1 - uncertainty)
        
        alpha_prior = current_weight * prior_strength
        beta_prior = (1 - current_weight) * prior_strength
        
        # Update with evidence
        alpha_post = alpha_prior
        beta_post = beta_prior
        
        for outcome, confidence in zip(outcomes, confidences):
            # Weight evidence by confidence
            effective_observations = confidence * self.learning_rate
            
            if outcome > 0.5:  # Win
                alpha_post += effective_observations
            elif outcome < 0.5:  # Loss
                beta_post += effective_observations
            else:  # Neutral
                alpha_post += effective_observations * 0.5
                beta_post += effective_observations * 0.5
        
        # Posterior mean
        if alpha_post + beta_post > 0:
            posterior_mean = alpha_post / (alpha_post + beta_post)
        else:
            posterior_mean = current_weight
        
        return posterior_mean
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights
        """
        return self.weights.copy()
    
    def get_weight_with_uncertainty(self, signal_name: str) -> Dict[str, float]:
        """
        Get weight with uncertainty estimate
        """
        if signal_name not in self.weights:
            return {"weight": self.default_weight, "uncertainty": 1.0}
        
        return {
            "weight": self.weights[signal_name],
            "uncertainty": self.weight_uncertainty.get(signal_name, 0.5)
        }
    
    def calculate_ensemble_weight(self, signal_weights: Dict[str, float]) -> float:
        """
        Calculate ensemble weight from multiple signals
        """
        if not signal_weights:
            return 0.5
        
        # Weighted average with uncertainty adjustment
        total_weight = 0
        total_confidence = 0
        
        for signal, weight in signal_weights.items():
            if signal in self.weights:
                signal_weight = self.weights[signal]
                uncertainty = self.weight_uncertainty.get(signal, 0.5)
                
                # Adjust by confidence (1 - uncertainty)
                confidence = 1 - uncertainty
                total_weight += signal_weight * weight * confidence
                total_confidence += weight * confidence
        
        if total_confidence > 0:
            return total_weight / total_confidence
        
        return 0.5
    
    def reset_weights(self, signal_names: List[str] = None):
        """
        Reset weights to default
        """
        if signal_names:
            for name in signal_names:
                self.weights[name] = self.default_weight
                self.weight_uncertainty[name] = 0.5
        else:
            self.weights = {}
            self.weight_uncertainty = {}
    
    def get_weight_history(self, signal_name: str) -> List[Dict]:
        """
        Get weight history for a signal
        """
        return self.weight_history.get(signal_name, [])