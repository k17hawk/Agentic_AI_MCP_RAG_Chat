"""
Regex Extractor - Uses regex patterns to extract entities from text
"""
from typing import Dict, List, Optional, Any
import re
from datetime import datetime

from utils.logger import logger as logging

class RegexExtractor:
    """
    Extracts entities using regex patterns
    Faster than NLP for specific patterns like tickers, dates, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Compile regex patterns
        self.patterns = {
            "tickers": re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\b(?:\s+(?:stock|share|equity|price|trading))', re.IGNORECASE),
            "currencies": re.compile(r'[$€£¥](\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(?:dollars|euros|pounds|yen)\b', re.IGNORECASE),
            "percentages": re.compile(r'(\d+(?:\.\d+)?)\s*%|\b(?:up|down|increase|decrease)\s+(\d+(?:\.\d+)?)\s*percent', re.IGNORECASE),
            "dates": re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
            "companies": re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc|Corp|Corporation|Ltd|Limited|LLC|Co|Company|Group|Holdings|Technologies)\b', re.IGNORECASE),
            "people": re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\s+(?:CEO|CFO|COO|CTO|President|Chairman|Director|Founder)\b', re.IGNORECASE),
            "phone_numbers": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "urls": re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)'),
            "stock_exchanges": re.compile(r'\b(NYSE|NASDAQ|AMEX|TSX|LSE|TSE|HKEX|ASX|SSE)\b'),
            "market_indicators": re.compile(r'\b(S&P 500|Dow Jones|NASDAQ Composite|Russell 2000|VIX)\b', re.IGNORECASE),
            "financial_terms": re.compile(r'\b(earnings|revenue|profit|loss|EPS|P/E|dividend|yield|volatility|momentum|support|resistance)\b', re.IGNORECASE)
        }
        
        # Ticker blacklist (common false positives)
        self.ticker_blacklist = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS',
            'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD',
            'NYSE', 'NASDAQ', 'SEC', 'CEO', 'CFO', 'EPS', 'YOY',
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD',
            'INC', 'CORP', 'LTD', 'LLC', 'CO', 'GROUP'
        }
        
        logging.info(f"✅ RegexExtractor initialized with {len(self.patterns)} patterns")
    
    async def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using regex patterns
        """
        if not text:
            return self._empty_result()
        
        results = {
            "tickers": [],
            "companies": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "currencies": [],
            "industries": [],
            "percentages": [],
            "contact_info": [],
            "market_indicators": [],
            "stock_exchanges": []
        }
        
        # Apply each pattern
        for key, pattern in self.patterns.items():
            matches = pattern.findall(text)
            
            if key == "tickers":
                # Handle ticker pattern specially (has groups)
                tickers = []
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m and m not in self.ticker_blacklist and len(m) <= 5:
                                tickers.append(m.upper())
                    elif match and match not in self.ticker_blacklist:
                        tickers.append(match.upper())
                
                # Deduplicate
                results[key] = list(dict.fromkeys(tickers))
            
            elif key == "currencies":
                # Format currency values
                values = []
                for match in matches:
                    if isinstance(match, tuple):
                        values.append(match[0] or match[1])
                    else:
                        values.append(match)
                results[key] = list(dict.fromkeys(values))
            
            elif key in ["percentages", "dates", "phone_numbers", "emails", "urls"]:
                # Simple list of matches
                flat_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        flat_matches.extend([m for m in match if m])
                    else:
                        flat_matches.append(match)
                results[key] = list(dict.fromkeys(flat_matches))
            
            else:
                # For other patterns, just take the first group if tuple
                flat_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        # Take the first non-empty group
                        for m in match:
                            if m:
                                flat_matches.append(m)
                                break
                    else:
                        flat_matches.append(match)
                results[key] = list(dict.fromkeys(flat_matches))
        
        return results
    
    async def extract_tickers(self, text: str) -> List[str]:
        """
        Quickly extract only tickers (optimized)
        """
        tickers = []
        
        # Pattern for $TICKER
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        tickers.extend([t.upper() for t in dollar_tickers])
        
        # Pattern for standalone tickers with context
        context_pattern = re.compile(r'\b([A-Z]{1,5})\b(?:\s+(?:stock|share|equity|price|trading|shares))', re.IGNORECASE)
        context_tickers = context_pattern.findall(text)
        tickers.extend([t.upper() for t in context_tickers])
        
        # Filter blacklist
        filtered = [t for t in tickers if t not in self.ticker_blacklist]
        
        # Deduplicate
        return list(dict.fromkeys(filtered))
    
    async def extract_numbers(self, text: str) -> List[float]:
        """
        Extract all numbers from text
        """
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return [float(n) for n in numbers]
    
    def _empty_result(self) -> Dict[str, List[str]]:
        """Return empty result structure"""
        return {
            "tickers": [],
            "companies": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "currencies": [],
            "industries": [],
            "percentages": [],
            "contact_info": [],
            "market_indicators": [],
            "stock_exchanges": []
        }