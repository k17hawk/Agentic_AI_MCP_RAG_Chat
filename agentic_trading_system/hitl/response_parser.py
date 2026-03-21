"""
Response Parser - Parses human responses from various channels
"""
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
from agentic_trading_system.utils.logger import logger as  logging

class ResponseParser:
    """
    Response Parser - Parses natural language responses from humans
    
    Supports:
    - YES/NO approvals
    - Custom position sizes
    - Multiple symbols
    - Comments and feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Response patterns
        self.patterns = {
            "approve": [
                r'^\s*yes\s+([A-Z]+)\s*$',
                r'^\s*approve\s+([A-Z]+)\s*$',
                r'^\s*y\s+([A-Z]+)\s*$',
                r'^\s*ok\s+([A-Z]+)\s*$',
                r'^\s*sure\s+([A-Z]+)\s*$'
            ],
            "reject": [
                r'^\s*no\s+([A-Z]+)\s*$',
                r'^\s*reject\s+([A-Z]+)\s*$',
                r'^\s*n\s+([A-Z]+)\s*$',
                r'^\s*pass\s+([A-Z]+)\s*$',
                r'^\s*skip\s+([A-Z]+)\s*$'
            ],
            "modify": [
                r'^\s*modify\s+([A-Z]+)\s+(\d+)\s*$',
                r'^\s*change\s+([A-Z]+)\s+to\s+(\d+)\s*$',
                r'^\s*([A-Z]+)\s+(\d+)\s*shares?\s*$',
                r'^\s*buy\s+(\d+)\s+([A-Z]+)\s*$'
            ],
            "comment": [
                r'^\s*comment\s+(.+)$',
                r'^\s*note\s+(.+)$',
                r'^\s*reason:\s*(.+)$'
            ],
            "status": [
                r'^\s*status\s*$',
                r'^\s*what\'?s?\s+(?:the\s+)?status\s*$',
                r'^\s*how\'?s?\s+(?:the\s+)?portfolio\s*$'
            ],
            "help": [
                r'^\s*help\s*$',
                r'^\s*\?\s*$',
                r'^\s*commands?\s*$'
            ]
        }
        
        logging.info(f"✅ ResponseParser initialized")
    
    def parse(self, message: str, channel: str = "whatsapp") -> Dict[str, Any]:
        """
        Parse a human response
        """
        message = message.strip()
        
        result = {
            "raw": message,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
            "parsed": False,
            "type": "unknown",
            "symbols": [],
            "action": None,
            "quantity": None,
            "comment": None
        }
        
        # Try each pattern type
        for response_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.match(pattern, message, re.IGNORECASE)
                if match:
                    result["parsed"] = True
                    result["type"] = response_type
                    
                    if response_type == "approve":
                        result["action"] = "approve"
                        result["symbols"] = [match.group(1).upper()]
                        
                    elif response_type == "reject":
                        result["action"] = "reject"
                        result["symbols"] = [match.group(1).upper()]
                        
                    elif response_type == "modify":
                        if len(match.groups()) == 2:
                            # Try to determine order (symbol quantity or quantity symbol)
                            group1 = match.group(1)
                            group2 = match.group(2)
                            
                            if group1.isalpha() and group2.isdigit():
                                result["symbols"] = [group1.upper()]
                                result["quantity"] = int(group2)
                            elif group1.isdigit() and group2.isalpha():
                                result["symbols"] = [group2.upper()]
                                result["quantity"] = int(group1)
                            
                            result["action"] = "modify"
                        
                    elif response_type == "comment":
                        result["comment"] = match.group(1)
                        
                    elif response_type == "status":
                        result["action"] = "status"
                        
                    elif response_type == "help":
                        result["action"] = "help"
                    
                    break
        
        # Handle multiple symbols (e.g., "YES AAPL, TSLA")
        if not result["parsed"] and "," in message:
            parts = message.split(",")
            symbols = []
            action = None
            
            for part in parts:
                sub_result = self.parse(part.strip())
                if sub_result["parsed"]:
                    if not action:
                        action = sub_result["action"]
                    if sub_result["symbols"]:
                        symbols.extend(sub_result["symbols"])
            
            if symbols:
                result["parsed"] = True
                result["type"] = "multiple"
                result["action"] = action
                result["symbols"] = symbols
        
        return result
    
    def extract_approval(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Simple extraction for YES/NO responses
        Returns (action, symbol) where action is "approve" or "reject"
        """
        result = self.parse(message)
        if result["parsed"] and result["type"] in ["approve", "reject"]:
            return result["action"], result["symbols"][0] if result["symbols"] else None
        return None, None
    
    def is_affirmative(self, message: str) -> bool:
        """Check if message is affirmative"""
        affirmative = ["yes", "y", "ok", "sure", "approve", "correct", "right"]
        return message.lower().strip() in affirmative
    
    def is_negative(self, message: str) -> bool:
        """Check if message is negative"""
        negative = ["no", "n", "not", "reject", "skip", "pass"]
        return message.lower().strip() in negative
    
    def get_help_text(self) -> str:
        """Get help text for users"""
        return (
            "Available commands:\n"
            "• YES SYMBOL - Approve trade\n"
            "• NO SYMBOL - Reject trade\n"
            "• MODIFY SYMBOL QUANTITY - Modify position size\n"
            "• STATUS - Get system status\n"
            "• HELP - Show this message"
        )