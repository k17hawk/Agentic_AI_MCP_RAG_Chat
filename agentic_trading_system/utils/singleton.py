"""
Singleton - Singleton pattern implementation
"""
from typing import Any, Dict
import threading

class Singleton:
    """
    Singleton base class using metaclass
    
    Usage:
        class MyClass(metaclass=Singleton):
            pass
    """
    _instances: Dict[Any, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class ThreadSafeSingleton:
    """
    Thread-safe singleton decorator for classes
    """
    _instances: Dict = {}
    _lock: threading.Lock = threading.Lock()
    
    def __init__(self, cls):
        self.cls = cls
    
    def __call__(self, *args, **kwargs):
        if self.cls not in self._instances:
            with self._lock:
                if self.cls not in self._instances:
                    instance = self.cls(*args, **kwargs)
                    self._instances[self.cls] = instance
        return self._instances[self.cls]

def singleton(cls):
    """
    Simple singleton decorator
    """
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

class SingletonMeta(type):
    """
    Metaclass for creating singleton classes
    """
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]