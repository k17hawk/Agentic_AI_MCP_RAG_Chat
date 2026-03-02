"""
Ensemble Model - Combines multiple models for better predictions
"""
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from collections import defaultdict
from datetime import datetime
import json
import os
from utils.logger import logger as  logging

class EnsembleModel:
    """
    Ensemble Model - Combines multiple models for better predictions
    
    Techniques:
    - Voting (hard/soft)
    - Stacking
    - Bagging
    - Boosting
    - Weighted averaging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Ensemble parameters
        self.ensemble_method = config.get("ensemble_method", "voting")  # voting, stacking, weighted
        self.voting_type = config.get("voting_type", "soft")  # hard, soft
        
        # Models
        self.models = []  # List of (model, weight)
        self.meta_model = None  # For stacking
        
        # Performance tracking
        self.model_performance = defaultdict(dict)
        self.model_weights = {}
        
        # Storage
        self.data_dir = config.get("data_dir", "data/ensemble")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logging.info(f"✅ EnsembleModel initialized with method: {self.ensemble_method}")
    
    def add_model(self, model: Any, weight: float = 1.0, name: str = None):
        """
        Add a model to the ensemble
        """
        if name is None:
            name = f"model_{len(self.models)}"
        
        self.models.append({
            "model": model,
            "weight": weight,
            "name": name
        })
        
        logging.info(f"➕ Added model {name} with weight {weight}")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set weights for models
        """
        for model_info in self.models:
            name = model_info["name"]
            if name in weights:
                model_info["weight"] = weights[name]
        
        logging.info(f"📊 Updated model weights")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train all models in the ensemble
        """
        for model_info in self.models:
            model = model_info["model"]
            name = model_info["name"]
            
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                    logging.info(f"✅ Trained model: {name}")
                    
                    # Evaluate on validation set if provided
                    if X_val is not None and y_val is not None:
                        if hasattr(model, 'predict'):
                            predictions = model.predict(X_val)
                            accuracy = np.mean(predictions == y_val)
                            self.model_performance[name]["validation_accuracy"] = accuracy
                
            except Exception as e:
                logging.error(f"Error training model {name}: {e}")
        
        # Train meta-model for stacking
        if self.ensemble_method == "stacking" and X_val is not None:
            self._train_meta_model(X_val, y_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble
        """
        if self.ensemble_method == "voting":
            return self._predict_voting(X)
        elif self.ensemble_method == "stacking":
            return self._predict_stacking(X)
        elif self.ensemble_method == "weighted":
            return self._predict_weighted(X)
        else:
            return self._predict_voting(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        """
        if self.ensemble_method == "voting" and self.voting_type == "soft":
            return self._predict_proba_voting(X)
        elif self.ensemble_method == "weighted":
            return self._predict_proba_weighted(X)
        else:
            # Default to voting
            return self._predict_proba_voting(X)
    
    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """
        Voting-based prediction
        """
        all_predictions = []
        
        for model_info in self.models:
            model = model_info["model"]
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X)
                    all_predictions.append(pred)
                except Exception as e:
                    logging.error(f"Prediction error for {model_info['name']}: {e}")
        
        if not all_predictions:
            return np.zeros(len(X))
        
        all_predictions = np.array(all_predictions)
        
        if self.voting_type == "hard":
            # Hard voting: majority vote
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=all_predictions
            )
        else:
            # Soft voting: average probabilities
            final_predictions = np.mean(all_predictions, axis=0)
        
        return final_predictions
    
    def _predict_proba_voting(self, X: np.ndarray) -> np.ndarray:
        """
        Soft voting with probabilities
        """
        all_probas = []
        
        for model_info in self.models:
            model = model_info["model"]
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    all_probas.append(proba)
                except Exception as e:
                    logging.error(f"Probability error for {model_info['name']}: {e}")
        
        if not all_probas:
            return np.zeros((len(X), 2))
        
        # Average probabilities
        return np.mean(all_probas, axis=0)
    
    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average prediction
        """
        weighted_sum = None
        total_weight = 0
        
        for model_info in self.models:
            model = model_info["model"]
            weight = model_info["weight"]
            
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X)
                    
                    if weighted_sum is None:
                        weighted_sum = pred * weight
                    else:
                        weighted_sum += pred * weight
                    
                    total_weight += weight
                except Exception as e:
                    logging.error(f"Prediction error for {model_info['name']}: {e}")
        
        if weighted_sum is None or total_weight == 0:
            return np.zeros(len(X))
        
        return weighted_sum / total_weight
    
    def _predict_proba_weighted(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average probability
        """
        weighted_sum = None
        total_weight = 0
        
        for model_info in self.models:
            model = model_info["model"]
            weight = model_info["weight"]
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    
                    if weighted_sum is None:
                        weighted_sum = proba * weight
                    else:
                        weighted_sum += proba * weight
                    
                    total_weight += weight
                except Exception as e:
                    logging.error(f"Probability error for {model_info['name']}: {e}")
        
        if weighted_sum is None or total_weight == 0:
            return np.zeros((len(X), 2))
        
        return weighted_sum / total_weight
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """
        Stacking-based prediction using meta-model
        """
        if self.meta_model is None:
            return self._predict_voting(X)
        
        # Get base model predictions
        base_predictions = []
        for model_info in self.models:
            model = model_info["model"]
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X)
                    base_predictions.append(pred)
                except Exception as e:
                    logging.error(f"Prediction error for {model_info['name']}: {e}")
        
        if not base_predictions:
            return np.zeros(len(X))
        
        # Stack predictions
        stacked = np.column_stack(base_predictions)
        
        # Meta-model prediction
        try:
            return self.meta_model.predict(stacked)
        except Exception as e:
            logging.error(f"Meta-model prediction error: {e}")
            return np.zeros(len(X))
    
    def _train_meta_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train meta-model for stacking
        """
        # Get base model predictions on validation set
        base_predictions = []
        for model_info in self.models:
            model = model_info["model"]
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_val)
                    base_predictions.append(pred)
                except Exception as e:
                    logging.error(f"Prediction error for {model_info['name']}: {e}")
        
        if not base_predictions:
            return
        
        # Stack predictions
        stacked = np.column_stack(base_predictions)
        
        # Train meta-model (simple logistic regression)
        try:
            from sklearn.linear_model import LogisticRegression
            self.meta_model = LogisticRegression()
            self.meta_model.fit(stacked, y_val)
            logging.info(f"✅ Trained meta-model for stacking")
        except Exception as e:
            logging.error(f"Error training meta-model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance across ensemble
        """
        importance = defaultdict(float)
        count = 0
        
        for model_info in self.models:
            model = model_info["model"]
            name = model_info["name"]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, imp in enumerate(importances):
                    importance[f"feature_{i}"] += imp
                count += 1
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0]
                for i, c in enumerate(coef):
                    importance[f"feature_{i}"] += abs(c)
                count += 1
        
        if count > 0:
            for key in importance:
                importance[key] /= count
        
        return dict(importance)
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights
        """
        return {info["name"]: info["weight"] for info in self.models}
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize model weights using validation performance
        """
        from scipy.optimize import minimize
        
        n_models = len(self.models)
        
        def objective(weights):
            # Set weights
            for i, model_info in enumerate(self.models):
                model_info["weight"] = weights[i]
            
            # Predict
            predictions = self._predict_weighted(X_val)
            
            # Calculate error (negative accuracy for minimization)
            if self.voting_type == "soft":
                # For regression/probabilities, use MSE
                error = np.mean((predictions - y_val) ** 2)
            else:
                # For classification, use 1 - accuracy
                accuracy = np.mean(predictions == y_val)
                error = 1 - accuracy
            
            return error
        
        # Constraints: weights sum to 1, all >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial weights
        initial_weights = [info["weight"] for info in self.models]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            for i, model_info in enumerate(self.models):
                model_info["weight"] = result.x[i]
            logging.info(f"✅ Optimized weights: {result.x}")
        
        return result.x
    
    def save_model(self, filename: str = None):
        """
        Save ensemble model to disk
        """
        if filename is None:
            filename = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Save model metadata (can't save actual model objects)
        data = {
            "ensemble_method": self.ensemble_method,
            "voting_type": self.voting_type,
            "models": [
                {
                    "name": info["name"],
                    "weight": info["weight"],
                    "model_type": type(info["model"]).__name__
                }
                for info in self.models
            ],
            "model_weights": self.get_model_weights(),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"💾 Saved ensemble model to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")