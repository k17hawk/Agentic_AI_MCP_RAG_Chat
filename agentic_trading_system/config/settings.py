"""
Settings - Central configuration loader
Loads all YAML configuration files and provides unified access
"""

import os
import yaml
import re
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Central configuration manager"""

    def __init__(self):
        # ✅ Correct path (relative to this file, NOT working directory)
        self.config_dir = Path(__file__).resolve().parent
        self.configs: Dict[str, Any] = {}

        # Debug (optional)
        # print(f"[DEBUG] Config dir: {self.config_dir}")

        self._load_all()

    def _load_all(self):
        """Load all YAML configuration files"""
        yaml_files = [
            "triggers.yaml",
            "analysis_weights.yaml",
            "risk_config.yaml",
            "logging_config.yaml",
            "database.yaml",
        ]

        for filename in yaml_files:
            filepath = self.config_dir / filename

            if filepath.exists():
                try:
                    with open(filepath, "r") as f:
                        content = f.read()

                        # Replace environment variables
                        content = self._replace_env_vars(content)

                        self.configs[filename.replace(".yaml", "")] = yaml.safe_load(content)

                except Exception as e:
                    print(f"❌ Error loading {filepath}: {e}")
            else:
                print(f"⚠️ Warning: Config file not found: {filepath}")

    def _replace_env_vars(self, content: str) -> str:
        """
        Replace ${VAR:-default} with environment variables
        Example:
            ${DB_HOST:-localhost}
        """
        pattern = r"\${([A-Za-z0-9_]+):-([^}]*)}"

        def replace(match):
            var_name, default = match.groups()
            return os.getenv(var_name, default)

        return re.sub(pattern, replace, content)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example:
            get("database.host")
        """
        keys = key.split(".")
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
        return self.configs.get("triggers", {})

    def get_analysis_weights(self) -> Dict:
        return self.configs.get("analysis_weights", {})

    def get_risk_config(self) -> Dict:
        return self.configs.get("risk_config", {})

    def get_logging_config(self) -> Dict:
        return self.configs.get("logging_config", {})

    def get_database_config(self) -> Dict:
        return self.configs.get("database", {})

    def get_env(self, key: str, default: Any = None) -> Any:
        return os.getenv(key, default)

    def get_all(self) -> Dict[str, Any]:
        return self.configs.copy()


# ✅ Lazy-loaded singleton (BEST PRACTICE)
_settings_instance: Settings = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# Optional backward compatibility
def get_config():
    return get_settings().get_all()