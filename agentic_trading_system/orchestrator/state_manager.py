import json
import os
from typing import Any, Dict
from logger import logging as logger

class StateManager:
    def __init__(self, storage_path="data/state/system_state.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self._cache = {}

    def update_state(self, key: str, value: Any):
        """Updates internal state and persists to disk"""
        self._cache[key] = value
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._cache, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"State persistence failed: {e}")

    def get_state(self, key: str, default=None):
        return self._cache.get(key, default)

    def load_from_disk(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self._cache = json.load(f)
                logger.info("Previous system state recovered from disk.")