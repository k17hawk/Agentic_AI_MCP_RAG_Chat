"""
Config Updater - Automatically updates configuration based on learning
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml
import json
import os
from utils.logger import logger as  logging

class ConfigUpdater:
    """
    Config Updater - Automatically updates YAML configuration files based on learning
    
    Responsibilities:
    - Update trigger thresholds
    - Adjust analysis weights
    - Modify risk parameters
    - Version control for configs
    - Backup and restore
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Config paths
        self.config_dir = config.get("config_dir", "config")
        self.backup_dir = config.get("backup_dir", "config/backups")
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Config files
        self.config_files = {
            "triggers": os.path.join(self.config_dir, "triggers.yaml"),
            "analysis_weights": os.path.join(self.config_dir, "analysis_weights.yaml"),
            "risk_config": os.path.join(self.config_dir, "risk_config.yaml"),
            "logging_config": os.path.join(self.config_dir, "logging_config.yaml"),
            "database": os.path.join(self.config_dir, "database.yaml")
        }
        
        # Current configs cache
        self.current_configs = {}
        
        # Load current configs
        self._load_configs()
        
        # Update history
        self.update_history = []
        
        logging.info(f"✅ ConfigUpdater initialized")
    
    def _load_configs(self):
        """Load all config files"""
        for config_name, filepath in self.config_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                            self.current_configs[config_name] = yaml.safe_load(f)
                        else:
                            self.current_configs[config_name] = json.load(f)
                except Exception as e:
                    logging.error(f"Error loading {config_name}: {e}")
                    self.current_configs[config_name] = {}
            else:
                self.current_configs[config_name] = {}
    
    def _save_config(self, config_name: str, config_data: Dict):
        """Save a config file"""
        filepath = self.config_files.get(config_name)
        if not filepath:
            logging.error(f"Unknown config: {config_name}")
            return False
        
        try:
            # Create backup first
            self._create_backup(config_name)
            
            # Save new config
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            
            # Update cache
            self.current_configs[config_name] = config_data
            
            logging.info(f"✅ Saved config: {config_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving {config_name}: {e}")
            return False
    
    def _create_backup(self, config_name: str):
        """Create a backup of current config"""
        filepath = self.config_files.get(config_name)
        if not filepath or not os.path.exists(filepath):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_name}_{timestamp}.bak"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            import shutil
            shutil.copy2(filepath, backup_path)
            logging.info(f"💾 Created backup: {backup_filename}")
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
    
    def update_trigger_thresholds(self, signal_performance: Dict[str, Any]) -> bool:
        """
        Update trigger thresholds based on signal performance
        """
        triggers_config = self.current_configs.get("triggers", {})
        
        changes = []
        
        for signal_name, perf in signal_performance.items():
            if perf.get("appearances", 0) < 10:
                continue
            
            win_rate = perf.get("win_rate", 0.5)
            
            # Adjust threshold based on win rate
            if win_rate > 0.6:
                # Good signal - consider lowering threshold
                current_threshold = self._get_trigger_threshold(triggers_config, signal_name)
                if current_threshold:
                    new_threshold = current_threshold * 0.95  # Lower by 5%
                    self._set_trigger_threshold(triggers_config, signal_name, new_threshold)
                    changes.append(f"{signal_name}: {current_threshold:.2f} -> {new_threshold:.2f}")
            
            elif win_rate < 0.4:
                # Poor signal - consider raising threshold
                current_threshold = self._get_trigger_threshold(triggers_config, signal_name)
                if current_threshold:
                    new_threshold = current_threshold * 1.05  # Raise by 5%
                    self._set_trigger_threshold(triggers_config, signal_name, new_threshold)
                    changes.append(f"{signal_name}: {current_threshold:.2f} -> {new_threshold:.2f}")
        
        if changes:
            success = self._save_config("triggers", triggers_config)
            if success:
                self.update_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "trigger_thresholds",
                    "changes": changes
                })
            return success
        
        return True
    
    def update_analysis_weights(self, signal_performance: Dict[str, Any]) -> bool:
        """
        Update analysis weights based on signal performance
        """
        weights_config = self.current_configs.get("analysis_weights", {})
        
        changes = []
        
        # Get current regime weights
        regime_weights = weights_config.get("regime_weights", {})
        
        for signal_name, perf in signal_performance.items():
            if perf.get("appearances", 0) < 10:
                continue
            
            win_rate = perf.get("win_rate", 0.5)
            
            # Update weights in each regime
            for regime, weights in regime_weights.items():
                if signal_name in weights:
                    current_weight = weights[signal_name]
                    # Adjust weight based on win rate
                    adjustment = (win_rate - 0.5) * 0.1  # Max 5% adjustment
                    new_weight = max(0.05, min(0.5, current_weight + adjustment))
                    
                    if abs(new_weight - current_weight) > 0.01:
                        weights[signal_name] = new_weight
                        changes.append(f"{regime}.{signal_name}: {current_weight:.2f} -> {new_weight:.2f}")
        
        if changes:
            weights_config["regime_weights"] = regime_weights
            success = self._save_config("analysis_weights", weights_config)
            if success:
                self.update_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "analysis_weights",
                    "changes": changes
                })
            return success
        
        return True
    
    def update_risk_parameters(self, risk_performance: Dict[str, Any]) -> bool:
        """
        Update risk parameters based on performance
        """
        risk_config = self.current_configs.get("risk_config", {})
        
        changes = []
        
        # Update position sizing based on win rate
        if "position_sizing" in risk_config:
            win_rate = risk_performance.get("overall_win_rate", 0.5)
            
            if "max_position_pct" in risk_config["position_sizing"]:
                current = risk_config["position_sizing"]["max_position_pct"]
                # Increase position size slightly if win rate is good
                if win_rate > 0.55:
                    new = min(0.25, current * 1.05)
                else:
                    new = max(0.05, current * 0.95)
                
                if abs(new - current) > 0.001:
                    risk_config["position_sizing"]["max_position_pct"] = new
                    changes.append(f"max_position_pct: {current:.2f} -> {new:.2f}")
        
        # Update stop loss based on average loss
        if "stop_loss" in risk_config:
            avg_loss = risk_performance.get("avg_loss_pct", 2.0)
            current = risk_config["stop_loss"].get("default_pct", 2.0)
            
            # Adjust stop to be slightly larger than average loss
            new = max(1.0, min(5.0, avg_loss * 1.2))
            
            if abs(new - current) > 0.1:
                risk_config["stop_loss"]["default_pct"] = new
                changes.append(f"stop_loss_pct: {current:.1f} -> {new:.1f}")
        
        if changes:
            success = self._save_config("risk_config", risk_config)
            if success:
                self.update_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "risk_parameters",
                    "changes": changes
                })
            return success
        
        return True
    
    def _get_trigger_threshold(self, config: Dict, signal_name: str) -> Optional[float]:
        """Get threshold for a trigger"""
        # This would need to parse the actual trigger config structure
        # Simplified version
        return config.get("thresholds", {}).get(signal_name)
    
    def _set_trigger_threshold(self, config: Dict, signal_name: str, threshold: float):
        """Set threshold for a trigger"""
        if "thresholds" not in config:
            config["thresholds"] = {}
        config["thresholds"][signal_name] = threshold
    
    def get_config(self, config_name: str) -> Optional[Dict]:
        """Get current configuration"""
        return self.current_configs.get(config_name)
    
    def restore_backup(self, config_name: str, backup_file: str) -> bool:
        """Restore a configuration from backup"""
        backup_path = os.path.join(self.backup_dir, backup_file)
        
        if not os.path.exists(backup_path):
            logging.error(f"Backup not found: {backup_file}")
            return False
        
        try:
            # Load backup
            with open(backup_path, 'r') as f:
                if backup_path.endswith('.bak'):
                    # Assume same format as original
                    if self.config_files[config_name].endswith('.yaml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
            
            # Save as current
            success = self._save_config(config_name, config_data)
            
            if success:
                self.update_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "restore",
                    "config": config_name,
                    "backup": backup_file
                })
            
            return success
            
        except Exception as e:
            logging.error(f"Error restoring backup: {e}")
            return False
    
    def get_update_history(self, limit: int = 100) -> List[Dict]:
        """Get update history"""
        return self.update_history[-limit:]
    
    def suggest_updates(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest configuration updates based on performance
        """
        suggestions = []
        
        # Suggest trigger threshold adjustments
        for signal, perf in performance_data.get("signals", {}).items():
            if perf.get("appearances", 0) > 20:
                win_rate = perf.get("win_rate", 0.5)
                if win_rate > 0.65:
                    suggestions.append({
                        "type": "trigger",
                        "signal": signal,
                        "suggestion": "Lower threshold to catch more signals",
                        "current_threshold": perf.get("current_threshold", 0.5),
                        "suggested_threshold": perf.get("current_threshold", 0.5) * 0.9
                    })
                elif win_rate < 0.35:
                    suggestions.append({
                        "type": "trigger",
                        "signal": signal,
                        "suggestion": "Raise threshold to filter poor signals",
                        "current_threshold": perf.get("current_threshold", 0.5),
                        "suggested_threshold": perf.get("current_threshold", 0.5) * 1.1
                    })
        
        return suggestions