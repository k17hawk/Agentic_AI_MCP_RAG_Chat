"""
Model Weights Repository - Data access for model weights
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from utils.logger import logger as logging
from memory.models import ModelWeights

class ModelWeightsRepository:
    """
    Model Weights Repository - Handles CRUD operations for model weights
    
    Storage: JSON file (can be upgraded to PostgreSQL)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/models")
        self.weights_file = os.path.join(self.data_dir, "model_weights.json")
        self.active_weights_file = os.path.join(self.data_dir, "active_weights.json")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # In-memory cache
        self.weights: Dict[str, ModelWeights] = {}
        self.by_model: Dict[str, List[str]] = {}
        self.active_weights: Dict[str, str] = {}  # model_name -> weights_id
        
        # Load existing weights
        self._load()
        
        logging.info(f"✅ ModelWeightsRepository initialized with {len(self.weights)} weight sets")
    
    def save(self, weights: ModelWeights) -> str:
        """
        Save model weights
        """
        weights_id = weights.weights_id
        self.weights[weights_id] = weights
        
        # Update by_model index
        model_name = weights.model_name
        if model_name not in self.by_model:
            self.by_model[model_name] = []
        self.by_model[model_name].append(weights_id)
        
        self._save()
        
        logging.info(f"✅ Model weights saved: {weights_id} for {model_name} v{weights.model_version}")
        
        return weights_id
    
    def get(self, weights_id: str) -> Optional[ModelWeights]:
        """
        Get weights by ID
        """
        return self.weights.get(weights_id)
    
    def get_by_model(self, model_name: str, version: str = None) -> Optional[ModelWeights]:
        """
        Get weights for a model, optionally specific version
        """
        if model_name not in self.by_model:
            return None
        
        weight_ids = self.by_model[model_name]
        
        if version:
            # Find specific version
            for wid in weight_ids:
                w = self.weights.get(wid)
                if w and w.model_version == version:
                    return w
            return None
        else:
            # Return latest
            latest_id = weight_ids[-1] if weight_ids else None
            return self.weights.get(latest_id) if latest_id else None
    
    def get_active(self, model_name: str) -> Optional[ModelWeights]:
        """
        Get active weights for a model
        """
        weights_id = self.active_weights.get(model_name)
        if weights_id:
            return self.weights.get(weights_id)
        return None
    
    def set_active(self, model_name: str, weights_id: str) -> bool:
        """
        Set active weights for a model
        """
        if weights_id not in self.weights:
            return False
        
        weights = self.weights[weights_id]
        if weights.model_name != model_name:
            return False
        
        self.active_weights[model_name] = weights_id
        self._save_active()
        
        logging.info(f"✅ Active weights set for {model_name}: {weights_id}")
        
        return True
    
    def get_all_versions(self, model_name: str) -> List[ModelWeights]:
        """
        Get all versions of a model
        """
        weight_ids = self.by_model.get(model_name, [])
        versions = []
        for wid in weight_ids:
            w = self.weights.get(wid)
            if w:
                versions.append(w)
        
        return sorted(versions, key=lambda x: x.training_date, reverse=True)
    
    def delete(self, weights_id: str) -> bool:
        """
        Delete weights
        """
        if weights_id not in self.weights:
            return False
        
        weights = self.weights[weights_id]
        
        # Remove from by_model index
        model_name = weights.model_name
        if model_name in self.by_model:
            if weights_id in self.by_model[model_name]:
                self.by_model[model_name].remove(weights_id)
        
        # Remove from active if active
        if self.active_weights.get(model_name) == weights_id:
            del self.active_weights[model_name]
        
        # Remove from memory
        del self.weights[weights_id]
        
        self._save()
        self._save_active()
        
        return True
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model
        """
        v1 = self.get_by_model(model_name, version1)
        v2 = self.get_by_model(model_name, version2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        comparison = {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "version1_date": v1.training_date.isoformat(),
            "version2_date": v2.training_date.isoformat(),
            "version1_score": v1.validation_score,
            "version2_score": v2.validation_score,
            "score_difference": (v2.validation_score - v1.validation_score) if v1.validation_score and v2.validation_score else None
        }
        
        # Compare weights
        weight_diffs = {}
        all_keys = set(v1.technical_weights.keys()) | set(v2.technical_weights.keys())
        for key in all_keys:
            w1 = v1.technical_weights.get(key, 0)
            w2 = v2.technical_weights.get(key, 0)
            if abs(w2 - w1) > 0.01:
                weight_diffs[key] = {"old": w1, "new": w2, "diff": w2 - w1}
        
        comparison["weight_changes"] = weight_diffs
        
        return comparison
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get latest version number for a model
        """
        versions = self.get_all_versions(model_name)
        if versions:
            return versions[0].model_version
        return None
    
    def count(self) -> int:
        """Get total number of weight sets"""
        return len(self.weights)
    
    def _save(self):
        """Save weights to disk"""
        try:
            data = {
                "weights": {wid: w.dict() for wid, w in self.weights.items()},
                "by_model": self.by_model,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.weights_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Error saving model weights: {e}")
    
    def _save_active(self):
        """Save active weights separately"""
        try:
            with open(self.active_weights_file, 'w') as f:
                json.dump(self.active_weights, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error saving active weights: {e}")
    
    def _load(self):
        """Load weights from disk"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load weights
                    for wid, w_data in data.get("weights", {}).items():
                        self.weights[wid] = ModelWeights(**w_data)
                    
                    # Load by_model index
                    self.by_model = data.get("by_model", {})
            
            # Load active weights
            if os.path.exists(self.active_weights_file):
                with open(self.active_weights_file, 'r') as f:
                    self.active_weights = json.load(f)
                    
        except Exception as e:
            logging.error(f"Error loading model weights: {e}")