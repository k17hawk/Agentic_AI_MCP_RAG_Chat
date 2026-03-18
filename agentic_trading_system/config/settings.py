"""
Settings - Central configuration loader
Loads all YAML configuration files and provides unified access
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class Settings:
    """Central configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        # Load all YAML files
        self._load_all()
        
    def _load_all(self):
        """Load all YAML configuration files"""
        yaml_files = [
            "triggers.yaml",
            "analysis_weights.yaml", 
            "risk_config.yaml",
            "logging_config.yaml",
            "database.yaml"
        ]
        
        for filename in yaml_files:
            filepath = self.config_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Replace environment variables
                    content = self._replace_env_vars(content)
                    self.configs[filename.replace('.yaml', '')] = yaml.safe_load(content)
            else:
                print(f"Warning: Config file not found: {filepath}")
    
    def _replace_env_vars(self, content: str) -> str:
        """Replace ${VAR:-default} with environment variables"""
        pattern = r'\${([A-Za-z0-9_]+):-([^}]*)}'
        
        def replace(match):
            var_name, default = match.groups()
            return os.getenv(var_name, default)
        
        return re.sub(pattern, replace, content)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def get_triggers(self) -> Dict:
        """Get trigger configuration"""
        return self.configs.get('triggers', {})
    
    def get_analysis_weights(self) -> Dict:
        """Get analysis weights configuration"""
        return self.configs.get('analysis_weights', {})
    
    def get_risk_config(self) -> Dict:
        """Get risk configuration"""
        return self.configs.get('risk_config', {})
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self.configs.get('logging_config', {})
    
    def get_database_config(self) -> Dict:
        """Get database configuration"""
        return self.configs.get('database', {})
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable"""
        return os.getenv(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configurations"""
        return self.configs.copy()

# Global settings instance
settings = Settings()

# Example usage function
def get_config():
    """Get all configuration (for backward compatibility)"""
    return settings.get_all()