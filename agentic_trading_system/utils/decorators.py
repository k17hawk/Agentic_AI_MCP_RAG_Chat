"""
Decorators - Common decorators for logging, retry, timing, etc.
"""
import asyncio
import functools
import time
from typing import Any, Callable, Optional, Type, Union
from datetime import datetime, timedelta
from utils.logger import logging
from utils.exceptions import RateLimitError

def retry(max_attempts: int = 3, delay: float = 1.0, 
          backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logging.error(f"Final retry failed for {func.__name__}: {e}")
                        raise
                    
                    logging.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logging.error(f"Final retry failed for {func.__name__}: {e}")
                        raise
                    
                    logging.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def timer(func: Callable) -> Callable:
    """
    Time execution of a function and log it
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end = time.time()
            logging.debug(f"{func.__name__} took {end - start:.3f}s")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end = time.time()
            logging.debug(f"{func.__name__} took {end - start:.3f}s")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def singleton(cls):
    """
    Singleton decorator for classes
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def log_execution(level: str = "INFO"):
    """
    Log function entry and exit
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logging.log(getattr(logging, level), f"Entering {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                logging.log(getattr(logging, level), f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logging.log(getattr(logging, level), f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logging.log(getattr(logging, level), f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def rate_limit(max_calls: int, period: int = 60):
    """
    Rate limit decorator
    
    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        calls = []
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            calls[:] = [c for c in calls if now - c < period]
            
            if len(calls) >= max_calls:
                wait_time = period - (now - calls[0])
                raise RateLimitError(
                    f"Rate limit exceeded. Max {max_calls} calls per {period}s. "
                    f"Wait {wait_time:.1f}s"
                )
            
            calls.append(now)
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if now - c < period]
            
            if len(calls) >= max_calls:
                wait_time = period - (now - calls[0])
                raise RateLimitError(
                    f"Rate limit exceeded. Max {max_calls} calls per {period}s. "
                    f"Wait {wait_time:.1f}s"
                )
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def deprecated(message: str = None):
    """
    Mark a function as deprecated
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning = f"Call to deprecated function {func.__name__}"
            if message:
                warning += f": {message}"
            logging.warning(warning)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def memoize(ttl: Optional[int] = None):
    """
    Cache function results
    
    Args:
        ttl: Time to live in seconds (None = forever)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        timestamps = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            # Check cache
            if key in cache:
                if ttl is None or now - timestamps[key] < ttl:
                    return cache[key]
            
            # Compute result
            result = await func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            if key in cache:
                if ttl is None or now - timestamps[key] < ttl:
                    return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator