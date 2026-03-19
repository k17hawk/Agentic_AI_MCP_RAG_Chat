# config_loader.py
# Simple config loader for discovery-only mode
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class SimpleConfig:
    """Minimal config loader for discovery only"""
    
    def __init__(self, config_path="agentic_trading_system/config/discovery_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml()
        
    def _load_yaml(self):
        """Load and interpolate environment variables"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Replace ${VAR} with environment variables
        pattern = r'\${([^}:]+)(?::-[^}]*)?}'
        
        def replace(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        content = re.sub(pattern, replace, content)
        
        return yaml.safe_load(content)
    
    def get(self, key, default=None):
        """Get config value by dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value if value is not None else default
    
    def get_all(self):
        """Get all config"""
        return self.config

# Global instance
config = SimpleConfig()