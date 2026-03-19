"""
Regex Extractor - Uses regex patterns to extract entities from text
"""
from typing import Dict, List, Optional, Any
import re
from datetime import datetime

from agentic_trading_system.utils.logger import logger as logging

class RegexExtractor:
    """
    Extracts entities using regex patterns
    Faster than NLP for specific patterns like tickers, dates, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Ticker blacklist (common false positives)
        self.ticker_blacklist = {
            # Articles / conjunctions / pronouns
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'THESE', 'THOSE',
            'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD', 'SHOULD',
            'ALL', 'ANY', 'NOT', 'BUT', 'NOR', 'OR', 'YET', 'SO',
            'ITS', 'OUR', 'THEIR', 'HIS', 'HER', 'OWN',
            'BEEN', 'BEING', 'ALSO', 'MORE', 'THAN', 'JUST', 'ONLY',
            'THEY', 'THEM', 'THEN', 'WHEN', 'WHERE', 'WHAT', 'WHICH', 'WHO', 'WHY', 'HOW',
            # Finance / business abbreviations
            'NYSE', 'NASDAQ', 'AMEX', 'SEC', 'CEO', 'CFO', 'COO', 'CTO', 'CIO',
            'EPS', 'YOY', 'QOQ', 'TTM', 'ATH', 'ATL', 'GDP', 'CPI', 'PMI',
            'ESG', 'ROI', 'ROE', 'DCF', 'IPO', 'ETF', 'REIT', 'SPAC',
            # Currencies
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF', 'HKD', 'SGD',
            # Legal suffixes
            'INC', 'CORP', 'LTD', 'LLC', 'LP', 'PLC', 'CO', 'GROUP', 'HOLDINGS',
            # Market / trading words
            'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'PRICE', 'MARKET', 'MARKETS',
            'FUND', 'FUNDS', 'BOND', 'BONDS', 'TRADE', 'CHART', 'NEWS', 'DATA',
            'HIGH', 'LOW', 'OPEN', 'CLOSE', 'CALL', 'PUT', 'CALLS', 'PUTS',
            'BUY', 'SELL', 'HOLD', 'BULL', 'BEAR', 'LONG', 'SHORT',
            # Common web/UI words that appear in scraped Tavily content
            'IDEAS', 'QUOTE', 'QUOTE', 'VIEW', 'ALERT', 'ALERTS', 'LOGIN',
            'SIGN', 'TERMS', 'HELP', 'ABOUT', 'HOME', 'BACK', 'NEXT', 'PREV',
            'READ', 'MORE', 'LESS', 'SHOW', 'HIDE', 'FULL', 'LIVE', 'FREE',
            # Privacy / legal boilerplate common in scraped content
            'CCPA', 'GDPR', 'DMCA', 'EULA', 'TOS',
            # Common full company names (not tickers)
            'APPLE', 'GOOGLE', 'META', 'TESLA', 'AMAZON', 'MICROSOFT',
            'NVIDIA', 'DISNEY', 'NETFLIX', 'INTEL', 'CISCO', 'ORACLE',
            # Dates
            'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
            'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
        }
        
        # Compile regex patterns
        self.patterns = {
            # Require an explicit signal: $AAPL, (AAPL), "AAPL stock/shares/…", or "stock AAPL"
            # The old bare \b[A-Z]{1,5}\b group is removed — it grabs UI/legal words from
            # scraped content (IDEAS, QUOTE, CCPA, etc.)
            "tickers": re.compile(
                r'\$([A-Z]{1,5})\b'
                r'|\(([A-Z]{1,5})\)'
                r'|\b([A-Z]{1,5})\s+(?:stock|shares?|equity|price|trading)\b'
                r'|\b(?:stock|shares?|equity|price|trading)\s+([A-Z]{1,5})\b',
                re.IGNORECASE
            ),
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
        Quickly extract only tickers (optimized).
        Called by data_enricher.py.

        Uses self.patterns["tickers"] exclusively — the single source of truth
        for ticker patterns. Old inline patterns with inc/corp context words
        are removed; they produced false positives like THE, INC, STOCK, QUOTE.
        """
        if not text:
            return []

        tickers = []

        # Delegate entirely to the shared tickers pattern (already fixed/tightened)
        matches = self.patterns["tickers"].findall(text)
        for match in matches:
            if isinstance(match, tuple):
                for m in match:
                    if m and len(m) <= 5 and m.isalpha():
                        tickers.append(m.upper())
            elif match and len(match) <= 5 and match.isalpha():
                tickers.append(match.upper())

        # Filter through blacklist + validation
        filtered = [t for t in tickers if self._is_valid_ticker(t)]

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for ticker in filtered:
            if ticker not in seen:
                seen.add(ticker)
                unique.append(ticker)

        return unique[:10]
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Helper method to validate tickers
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.upper().strip()
        
        # Check length (1-5 characters)
        if len(ticker) < 1 or len(ticker) > 5:
            return False
        
        # Check if all characters are letters
        if not ticker.isalpha():
            return False
        
        # Check against blacklist
        if ticker in self.ticker_blacklist:
            return False
        
        # Avoid single letters unless they're common tickers
        if len(ticker) == 1:
            common_single_letter_tickers = {'A', 'C', 'F', 'G', 'H', 'J', 'M', 'R', 'T', 'V', 'Z'}
            return ticker in common_single_letter_tickers
        
        # Avoid common English words (even if not in blacklist)
        common_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'HAVE', 'WILL'}
        if ticker in common_words:
            return False
        
        return True
    
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