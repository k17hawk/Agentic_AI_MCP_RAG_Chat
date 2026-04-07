import os
import yaml
import re
from pathlib import Path
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DiscoveryConfigLoader:
    """
    Loads discovery configuration from YAML files and environment variables.
    Automatically resolves correct config path based on project structure.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Optional explicit path to config file.
                        If not provided, resolves from project structure.
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Auto-resolve config path
            self.config_path = self._resolve_config_path()
        
        self.config = None
        self.config_version = None
        self._load_config()
    
    def _resolve_config_path(self) -> Path:
        """
        Resolve the discovery_config.yaml path based on project structure.
        
        Strategy:
        1. Look in current working directory
        2. Look in project root (parent of agentic_trading_system)
        3. Look in config/ subdirectory
        """
        possible_paths = []
        
        # Current working directory
        possible_paths.append(Path.cwd() / "config" / "discovery_config.yaml")
        possible_paths.append(Path.cwd() / "discovery_config.yaml")

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # agentic_trading_system -> project
        
        possible_paths.append(project_root / "config" / "discovery_config.yaml")
        possible_paths.append(project_root / "discovery_config.yaml")
        
        # Also check if we're in the Agentic_AI_MCP_RAG_Chat directory
        if "Agentic_AI_MCP_RAG_Chat" in str(project_root):
            # We're already in the project root
            possible_paths.append(project_root / "agentic_trading_system" / "config" / "discovery_config.yaml")
        
        # Try each path
        for path in possible_paths:
            if path.exists():
                print(f"✅ Found config at: {path}")
                return path
        
        # If not found, raise helpful error
        raise FileNotFoundError(
            f"Could not find discovery_config.yaml. Searched in:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def _load_config(self):
        """Load and interpolate environment variables."""
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
            
            # Replace ${VAR:-default} with environment variables
            pattern = r'\${([A-Za-z0-9_]+)(?:::-([^}]*))?}'
            
            def replace(match):
                var_name = match.group(1)
                default = match.group(2) or ''
                value = os.getenv(var_name, default)
                return value
            
            content = re.sub(pattern, replace, content)
            
            # Parse YAML
            self.config = yaml.safe_load(content)
            
            # Compute config version
            self.config_version = self._compute_config_version()
            
            print(f"✅ Loaded config version: {self.config_version}")
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            raise
    
    def _compute_config_version(self) -> str:
        """
        Compute a version hash from config content (excluding secrets).
        This allows tracking which config version was used for each run.
        """
        if not self.config:
            return "unknown"
        
        # Create a copy without sensitive fields
        config_copy = {}
        for key, value in self.config.items():
            # Skip sensitive fields
            if key in ["tavily_config", "news_config", "social_config", "macro_config"]:
                config_copy[key] = {}
                for k, v in value.items():
                    if k not in ["api_key", "news_api_key", "alpha_vantage_key", 
                                 "fmp_key", "twitter_bearer_token", "fred_api_key"]:
                        config_copy[key][k] = v
            else:
                config_copy[key] = value
        
        # Create hash
        config_str = yaml.dump(config_copy, sort_keys=True)
        return f"v{hashlib.md5(config_str.encode()).hexdigest()[:8]}"
    
    def get(self, key: str, default=None):
        """Get config value by dot notation."""
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
    
    def get_all(self) -> dict:
        """Get all config."""
        return self.config
    
    def get_version(self) -> str:
        """Get config version."""
        return self.config_version
    
    def get_metadata(self) -> dict:
        """Get config metadata."""
        return {
            "version": self.config_version,
            "loaded_at": datetime.now().isoformat(),
            "config_path": str(self.config_path)
        }

_loader = None


def get_discovery_config() -> dict:
    """Get discovery configuration."""
    global _loader
    if _loader is None:
        _loader = DiscoveryConfigLoader()
    return _loader.get_all()


def get_config_version() -> str:
    """Get configuration version."""
    global _loader
    if _loader is None:
        _loader = DiscoveryConfigLoader()
    return _loader.get_version()


def get_config_metadata() -> dict:
    """Get configuration metadata."""
    global _loader
    if _loader is None:
        _loader = DiscoveryConfigLoader()
    return _loader.get_metadata()


def reload_discovery_config(config_path: str = None):
    """Reload configuration."""
    global _loader
    _loader = DiscoveryConfigLoader(config_path)
    return _loader.get_all()

class SimpleConfig:
    """Minimal config loader for backward compatibility."""
    
    def __init__(self, config_path=None):
        self.loader = DiscoveryConfigLoader(config_path)
    
    def get(self, key, default=None):
        return self.loader.get(key, default)
    
    def get_all(self):
        return self.loader.get_all()
    
    def get_version(self):
        return self.loader.get_version()


# Global instance for backward compatibility
config = SimpleConfig()

if __name__ == "__main__":
    print("Testing Discovery Config Loader...\n")
    
    try:
        config = get_discovery_config()
        version = get_config_version()
        metadata = get_config_metadata()
        
        print(f"✅ Config loaded successfully")
        print(f"   Version: {version}")
        print(f"   Path: {metadata['config_path']}")
        print(f"   Loaded at: {metadata['loaded_at']}")
        
        # Print summary
        print(f"\n📋 Config Summary:")
        print(f"   Tavily enabled: {config.get('tavily_config', {}).get('enabled', False)}")
        print(f"   News enabled: {config.get('news_config', {}).get('enabled', False)}")
        print(f"   Social enabled: {config.get('social_config', {}).get('enabled', False)}")
        print(f"   SEC enabled: {config.get('sec_config', {}).get('enabled', False)}")
        print(f"   Options enabled: {config.get('options_config', {}).get('enabled', False)}")
        print(f"   Macro enabled: {config.get('macro_config', {}).get('enabled', False)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")