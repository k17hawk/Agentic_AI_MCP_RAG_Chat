"""
Feature Store - Stores and manages features for machine learning
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from utils.logger import logger as  logging

class FeatureStore:
    """
    Feature Store - Central repository for all features used in ML models
    
    Responsibilities:
    - Store features by symbol and timestamp
    - Normalize/standardize features
    - Handle missing values
    - Provide training/validation splits
    - Feature versioning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.data_dir = config.get("data_dir", "data/features")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Feature metadata
        self.feature_definitions = {}
        self.feature_stats = {}
        self.categorical_encodings = {}
        
        # In-memory cache
        self.feature_cache = defaultdict(dict)
        self.latest_features = {}
        
        # Normalization parameters
        self.normalization = config.get("normalization", "standard")  # standard, minmax, robust
        self.normalization_params = {}
        
        # Version
        self.version = config.get("version", "1.0.0")
        
        logging.info(f"✅ FeatureStore initialized (v{self.version})")
    
    def register_feature(self, name: str, feature_type: str, 
                        description: str = "", 
                        source: str = None,
                        category: str = "technical") -> bool:
        """
        Register a new feature definition
        """
        if name in self.feature_definitions:
            logging.warning(f"Feature {name} already registered")
            return False
        
        self.feature_definitions[name] = {
            "name": name,
            "type": feature_type,  # continuous, categorical, binary
            "description": description,
            "source": source,
            "category": category,
            "created_at": datetime.now().isoformat(),
            "version": self.version
        }
        
        logging.info(f"✅ Registered feature: {name} ({feature_type})")
        return True
    
    def store_features(self, symbol: str, timestamp: datetime, 
                      features: Dict[str, Any]) -> bool:
        """
        Store features for a symbol at a specific time
        """
        date_key = timestamp.strftime("%Y%m%d")
        
        # Store in cache
        if symbol not in self.feature_cache:
            self.feature_cache[symbol] = {}
        
        if date_key not in self.feature_cache[symbol]:
            self.feature_cache[symbol][date_key] = []
        
        self.feature_cache[symbol][date_key].append({
            "timestamp": timestamp.isoformat(),
            "features": features
        })
        
        # Update latest features
        self.latest_features[symbol] = {
            "timestamp": timestamp.isoformat(),
            "features": features
        }
        
        # Update statistics
        self._update_statistics(symbol, features)
        
        return True
    
    def store_batch(self, symbol: str, data: pd.DataFrame, 
                   feature_columns: List[str]) -> bool:
        """
        Store features from a DataFrame
        """
        for idx, row in data.iterrows():
            features = {col: row[col] for col in feature_columns}
            self.store_features(symbol, idx, features)
        
        # Save to disk
        self._save_to_disk(symbol, data, feature_columns)
        
        return True
    
    def get_features(self, symbol: str, start_date: datetime = None, 
                    end_date: datetime = None, limit: int = None) -> pd.DataFrame:
        """
        Get features for a symbol over a time range
        """
        if symbol not in self.feature_cache:
            return pd.DataFrame()
        
        records = []
        
        # Collect from cache
        for date_key, entries in self.feature_cache[symbol].items():
            for entry in entries:
                ts = datetime.fromisoformat(entry["timestamp"])
                
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue
                
                record = {"timestamp": ts}
                record.update(entry["features"])
                records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        if limit:
            df = df.tail(limit)
        
        return df
    
    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest features for a symbol
        """
        return self.latest_features.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all symbols with stored features
        """
        return list(self.feature_cache.keys())
    
    def normalize_features(self, features: Dict[str, Any], 
                          fit: bool = False) -> Dict[str, Any]:
        """
        Normalize/standardize features
        """
        normalized = {}
        
        for name, value in features.items():
            if name not in self.feature_definitions:
                normalized[name] = value
                continue
            
            ftype = self.feature_definitions[name]["type"]
            
            if ftype == "continuous":
                if fit:
                    # Update normalization parameters
                    if name not in self.normalization_params:
                        self.normalization_params[name] = {
                            "mean": value,
                            "std": 1.0,
                            "min": value,
                            "max": value,
                            "count": 1
                        }
                    else:
                        params = self.normalization_params[name]
                        params["count"] += 1
                        params["mean"] += (value - params["mean"]) / params["count"]
                        params["min"] = min(params["min"], value)
                        params["max"] = max(params["max"], value)
                
                # Apply normalization
                if name in self.normalization_params:
                    params = self.normalization_params[name]
                    
                    if self.normalization == "standard":
                        if params["std"] > 0:
                            normalized[name] = (value - params["mean"]) / params["std"]
                        else:
                            normalized[name] = 0
                    
                    elif self.normalization == "minmax":
                        if params["max"] > params["min"]:
                            normalized[name] = (value - params["min"]) / (params["max"] - params["min"])
                        else:
                            normalized[name] = 0.5
                    
                    elif self.normalization == "robust":
                        # Simplified robust scaling (would use percentiles in production)
                        normalized[name] = value
                else:
                    normalized[name] = value
            
            elif ftype == "categorical":
                # One-hot encoding would be applied here
                normalized[name] = value
            
            else:  # binary
                normalized[name] = float(value)
        
        return normalized
    
    def prepare_training_data(self, symbols: List[str], 
                             target_column: str,
                             feature_columns: List[str],
                             lookback_days: int = 60,
                             test_split: float = 0.2) -> Dict[str, Any]:
        """
        Prepare training data for ML models
        
        Returns:
        {
            "X_train": features for training,
            "y_train": targets for training,
            "X_test": features for testing,
            "y_test": targets for testing,
            "feature_names": list of feature names,
            "target_name": target column name
        }
        """
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            df = self.get_features(symbol, limit=lookback_days * 2)  # Get extra for lookback
            
            if df.empty or target_column not in df.columns:
                continue
            
            # Ensure all required feature columns exist
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                logging.warning(f"Missing features for {symbol}: {missing}")
                continue
            
            features = df[feature_columns].values
            targets = df[target_column].values
            
            all_features.append(features)
            all_targets.append(targets)
        
        if not all_features:
            return {}
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        # Remove any rows with NaN
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Train/test split
        split_idx = int(len(X) * (1 - test_split))
        
        # Shuffle
        indices = np.random.permutation(len(X))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        return {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
            "feature_names": feature_columns,
            "target_name": target_column,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx)
        }
    
    def get_feature_importance(self, feature_names: List[str], 
                              importance_scores: List[float]) -> Dict[str, float]:
        """
        Map importance scores to feature names
        """
        return dict(zip(feature_names, importance_scores))
    
    def _update_statistics(self, symbol: str, features: Dict[str, Any]):
        """
        Update running statistics for features
        """
        for name, value in features.items():
            if name not in self.feature_stats:
                self.feature_stats[name] = {
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "symbols": set()
                }
            
            stats = self.feature_stats[name]
            stats["count"] += 1
            stats["symbols"].add(symbol)
            
            # Update running mean and std (simplified)
            delta = value - stats["mean"]
            stats["mean"] += delta / stats["count"]
            delta2 = value - stats["mean"]
            stats["std"] = np.sqrt(((stats["count"] - 1) * stats["std"]**2 + delta * delta2) / stats["count"])
            
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
    
    def _save_to_disk(self, symbol: str, data: pd.DataFrame, 
                     feature_columns: List[str]):
        """
        Save features to disk
        """
        try:
            filename = os.path.join(self.data_dir, f"{symbol}_features.parquet")
            data[feature_columns].to_parquet(filename)
            logging.info(f"💾 Saved features for {symbol} to {filename}")
        except Exception as e:
            logging.error(f"Error saving features: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get feature store statistics
        """
        return {
            "features_registered": len(self.feature_definitions),
            "symbols_with_data": len(self.feature_cache),
            "normalization_method": self.normalization,
            "version": self.version,
            "feature_stats": {
                name: {
                    "count": stats["count"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "num_symbols": len(stats["symbols"])
                }
                for name, stats in self.feature_stats.items()
            }
        }