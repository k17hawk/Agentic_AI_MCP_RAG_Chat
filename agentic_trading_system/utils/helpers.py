"""
Helpers - General utility functions
"""
from typing import Any, Dict, List, Optional, Union
import re
import json
from datetime import datetime
import psutil

def format_currency(value: float, currency: str = "USD") -> str:
    """Format currency value"""
    if currency == "USD":
        return f"${value:,.2f}"
    elif currency == "EUR":
        return f"€{value:,.2f}"
    elif currency == "GBP":
        return f"£{value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    return f"{value * 100:.{decimals}f}%"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with zero check"""
    if b == 0:
        return default
    return a / b

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if deep and isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def deep_get(obj: Dict, path: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation"""
    keys = path.split('.')
    value = obj
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        
        if value is None:
            return default
    
    return value

def deep_set(obj: Dict, path: str, value: Any) -> Dict:
    """Set nested dictionary value using dot notation"""
    keys = path.split('.')
    current = obj
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return obj

def flatten_list(nested_list: List) -> List:
    """Flatten a nested list"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def clean_dict(data: Dict, remove_none: bool = True, 
              remove_empty: bool = False) -> Dict:
    """Clean dictionary by removing None/empty values"""
    result = {}
    
    for key, value in data.items():
        if remove_none and value is None:
            continue
        
        if remove_empty and value in ("", [], {}, ()):
            continue
        
        if isinstance(value, dict):
            cleaned = clean_dict(value, remove_none, remove_empty)
            if cleaned or not remove_empty:
                result[key] = cleaned
        else:
            result[key] = value
    
    return result

def parse_bool(value: Union[str, bool, int]) -> bool:
    """Parse boolean from various formats"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 'y', 'on')
    return False

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug"""
    # Convert to lowercase
    text = text.lower()
    # Replace spaces with hyphens
    text = re.sub(r'\s+', '-', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9-]', '', text)
    # Remove multiple hyphens
    text = re.sub(r'-+', '-', text)
    # Strip hyphens from ends
    return text.strip('-')

def pretty_json(data: Any) -> str:
    """Convert data to pretty JSON string"""
    return json.dumps(data, indent=2, default=str)

def parse_json_safe(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except:
        return default

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }

def get_cpu_usage() -> float:
    """Get current CPU usage"""
    return psutil.cpu_percent(interval=1)

def generate_id(prefix: str = "") -> str:
    """Generate unique ID"""
    import uuid
    unique = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique}" if prefix else unique