"""
Validators - Input validation functions
"""
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import yaml
from utils.exceptions import ValidationError

# Ticker validation
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')
TICKER_BLACKLIST = {
    'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS',
    'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD',
    'NYSE', 'NASDAQ', 'SEC', 'CEO', 'CFO', 'EPS', 'YOY',
    'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD'
}

# Email validation
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Phone validation (simple US format)
PHONE_PATTERN = re.compile(r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$')

def validate_ticker(ticker: str, raise_error: bool = False) -> bool:
    """
    Validate stock ticker symbol
    
    Rules:
    - 1-5 uppercase letters
    - Not in blacklist
    """
    ticker = ticker.upper().strip()
    
    if not TICKER_PATTERN.match(ticker):
        if raise_error:
            raise ValidationError(f"Invalid ticker format: {ticker}")
        return False
    
    if ticker in TICKER_BLACKLIST:
        if raise_error:
            raise ValidationError(f"Ticker in blacklist: {ticker}")
        return False
    
    return True

def validate_email(email: str, raise_error: bool = False) -> bool:
    """Validate email address"""
    if not EMAIL_PATTERN.match(email.strip()):
        if raise_error:
            raise ValidationError(f"Invalid email format: {email}")
        return False
    return True

def validate_phone(phone: str, raise_error: bool = False) -> bool:
    """Validate phone number (simple US format)"""
    if not PHONE_PATTERN.match(phone.strip()):
        if raise_error:
            raise ValidationError(f"Invalid phone format: {phone}")
        return False
    return True

def validate_date(date_str: str, formats: List[str] = None, 
                  raise_error: bool = False) -> bool:
    """Validate date string"""
    if formats is None:
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
    
    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    
    if raise_error:
        raise ValidationError(f"Invalid date format: {date_str}")
    return False

def validate_price(price: float, min_price: float = 0, 
                   max_price: float = 1_000_000,
                   raise_error: bool = False) -> bool:
    """Validate price value"""
    if not isinstance(price, (int, float)):
        if raise_error:
            raise ValidationError(f"Price must be a number: {price}")
        return False
    
    if price < min_price or price > max_price:
        if raise_error:
            raise ValidationError(f"Price {price} outside range [{min_price}, {max_price}]")
        return False
    
    return True

def validate_quantity(quantity: int, min_qty: int = 1, 
                      max_qty: int = 1_000_000,
                      raise_error: bool = False) -> bool:
    """Validate quantity"""
    if not isinstance(quantity, int):
        if raise_error:
            raise ValidationError(f"Quantity must be integer: {quantity}")
        return False
    
    if quantity < min_qty or quantity > max_qty:
        if raise_error:
            raise ValidationError(f"Quantity {quantity} outside range [{min_qty}, {max_qty}]")
        return False
    
    return True

def validate_percentage(value: float, min_pct: float = 0, 
                        max_pct: float = 100,
                        raise_error: bool = False) -> bool:
    """Validate percentage value"""
    if not isinstance(value, (int, float)):
        if raise_error:
            raise ValidationError(f"Percentage must be a number: {value}")
        return False
    
    if value < min_pct or value > max_pct:
        if raise_error:
            raise ValidationError(f"Percentage {value} outside range [{min_pct}, {max_pct}]")
        return False
    
    return True

def validate_json(json_str: str, raise_error: bool = False) -> bool:
    """Validate JSON string"""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        if raise_error:
            raise ValidationError(f"Invalid JSON: {json_str[:100]}")
        return False

def validate_yaml(yaml_str: str, raise_error: bool = False) -> bool:
    """Validate YAML string"""
    try:
        yaml.safe_load(yaml_str)
        return True
    except yaml.YAMLError:
        if raise_error:
            raise ValidationError(f"Invalid YAML: {yaml_str[:100]}")
        return False

def validate_required_fields(data: Dict, required: List[str],
                           raise_error: bool = False) -> bool:
    """Validate required fields in dictionary"""
    missing = [field for field in required if field not in data]
    
    if missing:
        if raise_error:
            raise ValidationError(f"Missing required fields: {missing}")
        return False
    
    return True

def validate_range(value: float, min_val: float, max_val: float,
                  inclusive: bool = True, raise_error: bool = False) -> bool:
    """Validate value within range"""
    if inclusive:
        valid = min_val <= value <= max_val
    else:
        valid = min_val < value < max_val
    
    if not valid and raise_error:
        raise ValidationError(f"Value {value} outside range [{min_val}, {max_val}]")
    
    return valid

def validate_enum(value: Any, enum_class, raise_error: bool = False) -> bool:
    """Validate value against enum"""
    try:
        enum_class(value)
        return True
    except (ValueError, TypeError):
        if raise_error:
            valid_values = [e.value for e in enum_class]
            raise ValidationError(f"Invalid value {value}. Must be one of: {valid_values}")
        return False