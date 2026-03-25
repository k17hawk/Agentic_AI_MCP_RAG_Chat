# =============================================================================
# discovery/entity_extractor/regex_extractor.py (UPDATED)
# =============================================================================
"""
Regex Extractor - Uses regex patterns to extract entities from text
"""

from typing import Dict, List, Any, Set
import re
from datetime import datetime

from agentic_trading_system.utils.logger import logger as logging

# Import from new config structure
from agentic_trading_system.constants import EntityExtraction,BLACKLIST, COMMON_WORDS
from agentic_trading_system.config.config__entity import RegexExtractorConfig


class RegexExtractor:
    """
    Extracts entities using regex patterns.
    Faster than NLP for specific patterns like tickers, dates, etc.
    """

    def __init__(self, config: RegexExtractorConfig):
        """
        Initialize regex extractor.
        
        Args:
            config: RegexExtractorConfig object
        """
        self.config = config

        # Ticker blacklist (common false positives)
        self.ticker_blacklist = self._build_blacklist(config.exclude_tickers)

        # Compile regex patterns
        self.patterns = self._compile_patterns()

        logging.info(f"✅ RegexExtractor initialized with {len(self.patterns)} patterns")

    def _build_blacklist(self, exclude_tickers: List[str]) -> Set[str]:
        """Build ticker blacklist."""
        blacklist = BLACKLIST
        # Add user-provided exclusions
        for ticker in exclude_tickers:
            blacklist.add(ticker.upper())

        return blacklist

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns."""
        return {
            # Ticker patterns - require explicit signals
            "tickers": re.compile(
                r'\$([A-Z]{1,5})\b'  # $AAPL
                r'|\(([A-Z]{1,5})\)'  # (AAPL)
                r'|\b([A-Z]{2,5})\s+(?:stock|shares?|equity|price|trading)\b'  # AAPL stock
                r'|\b(?:stock|shares?|equity|price|trading)\s+([A-Z]{2,5})\b',  # stock AAPL
                re.IGNORECASE
            ),
            # Currency pattern
            "currencies": re.compile(
                r'\$([0-9,]+(?:\.[0-9]+)?)'  # $ amounts
                r'|€([0-9,]+(?:\.[0-9]+)?)'  # Euro amounts
                r'|£([0-9,]+(?:\.[0-9]+)?)'  # Pound amounts
                r'|¥([0-9,]+(?:\.[0-9]+)?)'  # Yen amounts
                r'|\b(USD|EUR|GBP|JPY|CNY|AUD|CAD|CHF|HKD|SGD|NZD|KRW|INR|BRL|RUB|ZAR)\b',  # Currency codes
                re.IGNORECASE
            ),
            "percentages": re.compile(
                r'(\d+(?:\.\d+)?)\s*%|\b(?:up|down|increase|decrease)\s+(\d+(?:\.\d+)?)\s*percent',
                re.IGNORECASE
            ),
            "dates": re.compile(
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                re.IGNORECASE
            ),
            "companies": re.compile(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc|Corp|Corporation|Ltd|Limited|LLC|Co|Company|Group|Holdings|Technologies)\b',
                re.IGNORECASE
            ),
            "people": re.compile(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\s+(?:CEO|CFO|COO|CTO|President|Chairman|Director|Founder)\b',
                re.IGNORECASE
            ),
            "phone_numbers": re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            "emails": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "urls": re.compile(
                r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
            ),
            "stock_exchanges": re.compile(
                r'\b(NYSE|NASDAQ|AMEX|TSX|LSE|TSE|HKEX|ASX|SSE)\b'
            ),
            "market_indicators": re.compile(
                r'\b(S&P 500|Dow Jones|NASDAQ Composite|Russell 2000|VIX)\b',
                re.IGNORECASE
            ),
            "financial_terms": re.compile(
                r'\b(earnings|revenue|profit|loss|EPS|P/E|dividend|yield|volatility|momentum|support|resistance)\b',
                re.IGNORECASE
            )
        }

    async def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities
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
            "stock_exchanges": [],
            "financial_terms": []
        }

        # Apply each pattern
        for key, pattern in self.patterns.items():
            matches = pattern.findall(text)

            if key == "tickers":
                tickers = []
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m and m.upper() not in self.ticker_blacklist and len(m) <= 5 and m.isalpha():
                                tickers.append(m.upper())
                    elif match and match.upper() not in self.ticker_blacklist:
                        tickers.append(match.upper())

                results[key] = list(dict.fromkeys(tickers))

            elif key == "currencies":
                values = []
                for match in matches:
                    if isinstance(match, tuple):
                        if match[0]:  # $ amount
                            values.append("USD")
                        elif match[1]:  # € amount
                            values.append("EUR")
                        elif match[2]:  # £ amount
                            values.append("GBP")
                        elif match[3]:  # ¥ amount
                            values.append("JPY")
                        elif match[4]:  # Currency code
                            values.append(match[4].upper())
                results[key] = list(dict.fromkeys(values))

            elif key in ["percentages", "dates", "phone_numbers", "emails", "urls"]:
                flat_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        flat_matches.extend([m for m in match if m])
                    else:
                        flat_matches.append(match)
                results[key] = list(dict.fromkeys(flat_matches))

            else:
                flat_matches = []
                for match in matches:
                    if isinstance(match, tuple):
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
        
        Args:
            text: Input text
            
        Returns:
            List of ticker symbols
        """
        if not text:
            return []

        tickers = []
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

        return unique[:EntityExtraction.MAX_ENTITIES_PER_TYPE]

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol."""
        if not ticker or not isinstance(ticker, str):
            return False

        ticker = ticker.upper().strip()

        # Check length
        if len(ticker) < 1 or len(ticker) > 5:
            return False

        # Check if all characters are letters
        if not ticker.isalpha():
            return False

        # Check against blacklist
        if ticker in self.ticker_blacklist:
            return False

        # Single-letter validation
        if len(ticker) == 1:
            return ticker in EntityExtraction.VALID_SINGLE_LETTER_TICKERS

        # Avoid common English words
        common_words = COMMON_WORDS
        if ticker in common_words:
            return False

        return True

    async def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return [float(n) for n in numbers]

    def _empty_result(self) -> Dict[str, List[str]]:
        """Return empty result structure."""
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
            "stock_exchanges": [],
            "financial_terms": []
        }