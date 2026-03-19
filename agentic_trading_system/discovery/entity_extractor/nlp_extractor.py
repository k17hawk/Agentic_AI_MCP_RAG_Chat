"""
NLP Extractor - Uses NLP to extract entities from text
"""
from typing import Dict, List, Any
import re
import asyncio

# Safe spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception as e:
    spacy = None
    SPACY_AVAILABLE = False
    import logging as _logging
    _logging.warning(f"⚠️ spaCy not available: {e}")

from agentic_trading_system.utils.logger import logger as logging


class NLPExtractor:
    """
    Extracts entities using Natural Language Processing.
    """

    # ---------------------------------------------------------------------------
    # Words that must NEVER be treated as tickers.
    # Kept as a frozenset for O(1) lookup.
    # ---------------------------------------------------------------------------
    TICKER_FALSE_POSITIVES = frozenset({
        # Articles / determiners / conjunctions / prepositions
        'A', 'AN', 'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM', 'THAT',
        'THIS', 'THESE', 'THOSE', 'ON', 'IN', 'AT', 'TO', 'BY', 'OF', 'UP',
        'UPON', 'INTO', 'ONTO', 'WITHIN', 'OUT', 'OVER', 'UNDER', 'ABOVE',
        'BELOW', 'BETWEEN', 'AMONG', 'AS', 'IF', 'SO', 'YET', 'NOR',
        # Pronouns
        'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'ME', 'HIM', 'HER',
        'US', 'THEM', 'MY', 'YOUR', 'HIS', 'ITS', 'OUR', 'THEIR',
        'MINE', 'YOURS', 'HERS', 'OURS', 'THEIRS',
        # Common verbs / auxiliaries
        'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'AM',
        'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD',
        'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'CANNOT',
        'GET', 'GETS', 'GOT', 'GO', 'GOES', 'WENT', 'GONE',
        'SAY', 'SAYS', 'SAID', 'SEE', 'SAW', 'SEEN', 'KNOW', 'KNEW',
        'THINK', 'THOUGHT', 'TAKE', 'TOOK', 'TAKEN', 'COME', 'CAME',
        'GIVE', 'GAVE', 'GIVEN', 'MAKE', 'MADE', 'LET', 'SET', 'PUT',
        # Common adjectives / adverbs
        'ALL', 'ANY', 'EACH', 'EVERY', 'BOTH', 'NEITHER', 'NOT',
        'MORE', 'MOST', 'ALSO', 'JUST', 'ONLY', 'EVEN', 'THAN',
        'VERY', 'MUCH', 'SUCH', 'MANY', 'SOME',
        # Time / place
        'NOW', 'THEN', 'WHEN', 'WHERE', 'HERE', 'THERE',
        'TODAY', 'YESTERDAY', 'TOMORROW',
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
        'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
        # Business / legal suffixes — never stand-alone tickers
        'INC', 'CORP', 'LTD', 'LLC', 'LP', 'PLC', 'SA', 'AG', 'NV',
        'CO', 'GROUP', 'HOLDINGS', 'ACQUISITION', 'MERGER',
        # Finance abbreviations
        'ETF', 'ETFS', 'REIT', 'REITS', 'IPO', 'IPOS', 'SPAC', 'SPACS',
        'CEO', 'CFO', 'COO', 'CTO', 'CIO', 'CMO',
        'EPS', 'PE', 'PEG', 'ROE', 'ROA', 'ROI', 'EBITDA', 'DCF',
        'YOY', 'QOQ', 'MOM', 'YTD', 'ATH', 'ATL', 'AVG', 'TTM',
        'GDP', 'CPI', 'PMI', 'ESG',
        'BUY', 'SELL', 'HOLD', 'BULL', 'BEAR', 'LONG', 'SHORT',
        'CALL', 'PUT', 'CALLS', 'PUTS', 'OPTION', 'OPTIONS',
        'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'EQUITY', 'EQUITIES',
        'BOND', 'BONDS', 'FUND', 'FUNDS',
        # Price-action and analysis words adjacent to stock/price/trading
        'GAIN', 'GAINS', 'LOSS', 'RALLY', 'SURGE', 'DROP', 'FALL',
        'RISE', 'JUMP', 'SLIDE', 'SPIKE', 'DIP', 'PEAK', 'FLAT',
        'UP', 'DOWN', 'RANGE', 'LEVEL', 'MARK', 'BASE', 'ZONE',
        'MOVE', 'SETUP', 'PLAY', 'IDEA', 'PICK', 'RANK', 'SCAN',
        'ALERT', 'WATCH', 'BREAK', 'CROSS', 'STOP', 'LIMIT', 'CAP',
        'FLOOR', 'GRAPH', 'TREND', 'CHART', 'CHARTS', 'TRADE', 'TRADES',
        # Geopolitical / news words near trading/price in headlines
        'WAR', 'WARS', 'DEAL', 'RISK', 'FEAR', 'TALK', 'TALKS',
        'PLAN', 'BILL', 'ACT', 'LAW', 'RULE',
        # Time / recency words
        'WEEK', 'MONTH', 'YEAR', 'DAILY', 'LIVE', 'PRE', 'POST', 'LATE', 'EARLY',
        # UI / meta words common in scraped content
        'REAL', 'BEST', 'TOP', 'NEW', 'OLD', 'KEY', 'MAIN', 'INFO', 'SITE',
        'PAGE', 'LIST', 'FULL', 'PLUS', 'PRO', 'API', 'APP',
        # Exchanges / regulators
        'NYSE', 'NASDAQ', 'AMEX', 'NYSEARCA', 'NYSEAMERICAN',
        'LSE', 'LON', 'TSX', 'TSXV', 'CSE', 'HKEX', 'HKG', 'TSE',
        'SSE', 'SZSE', 'ASX', 'NZX', 'JSE', 'BSE', 'NSE', 'SGX',
        'SEC', 'FINRA', 'CFTC', 'FDIC', 'OCC', 'FED', 'ECB',
        # Currencies
        'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF',
        'HKD', 'SGD', 'NZD', 'KRW', 'INR', 'BRL', 'RUB', 'ZAR',
        # Full company names (tickers differ)
        'APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'META', 'TESLA',
        'NVIDIA', 'BERKSHIRE', 'JPMORGAN', 'JOHNSON', 'WALMART',
        'EXXON', 'CHEVRON', 'PFIZER', 'MERCK', 'COCA', 'PEPSI',
        'DISNEY', 'NETFLIX', 'INTEL', 'CISCO', 'ORACLE', 'SALESFORCE',
        'ADOBE', 'PAYPAL', 'UBER', 'LYFT', 'AIRBNB', 'STRIPE',
        # Words that appear adjacent to stock/price/trading in financial headlines
        'TOTAL', 'RETURN', 'VOLUME', 'VALUE', 'RATIO', 'SCORE', 'RATE',
        'VS', 'VERSUS', 'TODAY', 'YESTERDAY', 'TOMORROW', 'SINCE', 'AFTER', 'BEFORE', 'DURING', 'WHILE',
        'INDEX', 'YIELD', 'BETA', 'DELTA', 'GAMMA', 'THETA', 'SIGMA',
        # Common web / UI tokens from scraped content
        'IDEAS', 'IDEA', 'QUOTE', 'QUOTES', 'VIEW', 'VIEWS', 'ALERT',
        'ALERTS', 'LOGIN', 'SIGN', 'TERMS', 'HELP', 'HOME', 'BACK',
        'NEXT', 'PREV', 'READ', 'SHOW', 'HIDE', 'FULL', 'LIVE', 'FREE',
        # Privacy / legal boilerplate
        'CCPA', 'GDPR', 'DMCA', 'EULA', 'TOS',
        # False positives from logs
        'MASI',  # Not a common ticker
        'BANDS',  # Technical indicator, not a ticker
        'VTI',     # This is actually a valid ticker (VTI is Vanguard Total Stock Market ETF)
        'AI',      # This is actually a valid ticker (C3.ai Inc.)
        'DATA',    # This is actually a valid ticker (Tableau Software, but now acquired)
        'DATE',    # Not a ticker - common word
    })

    # Words that indicate a spaCy ORG span is actually a sentence fragment
    _FRAGMENT_WORDS = frozenset({
        'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'that',
        'this', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
        'by', 'what', 'which', 'who', 'when', 'where', 'how', 'why', 'if',
        'more', 'most', 'some', 'all', 'any', 'its', 'our', 'their', 'your',
        'see', 'find', 'get', 'think', 'know', 'analysts', 'latest', 'detailed',
        'low', 'high', 'new', 'old', 'level', 'support', 'resistance',
        # Financial news site names — these prefix company names in scraped titles
        'marketwatch', 'yahoo', 'barrons', "barron's", 'nasdaq', 'cnbc',
        'reuters', 'bloomberg', 'seeking', 'alpha', 'investing', 'motley',
        'fool', 'benzinga', 'zacks', 'morningstar', 'tradingview', 'finviz',
    })

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("spacy_model", "en_core_web_sm")

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.model_name)
                logging.info(f"✅ Loaded spaCy model: {self.model_name}")
            except Exception as e:
                logging.warning(f"⚠️ Could not load spaCy model: {e}")

        # Company name pattern: "Apple Inc.", "Tesla Corporation", etc.
        self.company_pattern = re.compile(
            r'\b([A-Z][A-Za-z0-9\.,&\'-]*(?:\s+[A-Z][A-Za-z0-9\.,&\'-]*)*)\s+'
            r'(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|L\.L\.C\.|'
            r'Co\.?|Company|Group|Holdings|Technologies|Tech\.?|'
            r'Pharma|Biosciences|Industries|International|Global)\b',
            re.IGNORECASE
        )

        # Ticker patterns: only high-confidence signals
        self._ticker_pattern = re.compile(
            r'\$([A-Z]{1,5})\b'                                              # $AAPL
            r'|\(([A-Z]{1,5})\)'                                              # (AAPL)
            r'|\b([A-Z]{2,5})\s+(?:stock|shares?|equity|price|trading)\b'    # AAPL stock
            r'|\b(?:stock|shares?|equity|price|trading)\s+([A-Z]{2,5})\b'    # stock AAPL
            r'|\b([A-Z]{1,5})\.([A-Z]{1,2})\b',                              # BRK.A / BF.B
            re.IGNORECASE
        )

        logging.info(f"✅ NLPExtractor initialized ({'spaCy + regex' if self.nlp else 'regex only'})")

    async def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        if not text or len(text) < 10:
            return self._empty_result()

        entities = self._empty_result()

        # --- spaCy pass ---
        if self.nlp:
            try:
                doc = self.nlp(text[:100000])
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        # Try ticker first, then company name.
                        if self._is_valid_ticker(ent.text):
                            entities["tickers"].append(ent.text.upper().strip())
                        elif self._is_valid_company_name(ent.text):
                            name = self._clean_entity(ent.text, "companies")
                            entities["companies"].append(name)
                            entities["organizations"].append(name)

                    elif ent.label_ == "PERSON":
                        if self._is_valid_person_name(ent.text):
                            entities["people"].append(ent.text.strip())

                    elif ent.label_ in ("GPE", "LOC"):
                        entities["locations"].append(ent.text.strip())

                    elif ent.label_ == "DATE":
                        entities["dates"].append(ent.text.strip())

                    elif ent.label_ == "MONEY":
                        entities["currencies"].append(ent.text.strip())

                    elif ent.label_ == "PRODUCT":
                        if self._is_valid_ticker(ent.text):
                            entities["tickers"].append(ent.text.upper().strip())

            except Exception as e:
                logging.warning(f"⚠️ spaCy extraction failed: {e}")

        # --- regex pass ---
        regex_entities = await self._extract_regex(text)
        for key in entities:
            entities[key].extend(regex_entities.get(key, []))

        # --- deduplicate + final validation ---
        for key in entities:
            seen: set = set()
            unique: List[str] = []
            for item in entities[key]:
                cleaned = self._clean_entity(item, key)
                if not cleaned or cleaned in seen:
                    continue
                # Re-validate after cleaning
                if key == "tickers" and not self._is_valid_ticker(cleaned):
                    continue
                if key in ("companies", "organizations") and not self._is_valid_company_name(cleaned):
                    continue
                seen.add(cleaned)
                unique.append(cleaned)
            entities[key] = unique[:20]

        return entities

    async def _extract_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        entities = self._empty_result()

        # Company names with legal suffix
        for match in self.company_pattern.finditer(text):
            company_name = match.group(1).strip()
            if self._is_valid_company_name(company_name):
                cleaned = self._clean_entity(company_name, "companies")
                entities["companies"].append(cleaned)
                entities["organizations"].append(cleaned)

        # Tickers — only high-signal patterns
        for match in self._ticker_pattern.finditer(text):
            for group in match.groups():
                if group and len(group) <= 5 and group.isalpha():
                    ticker = group.upper()
                    if self._is_valid_ticker(ticker):
                        entities["tickers"].append(ticker)

        return entities

    def _is_valid_ticker(self, text: str) -> bool:
        """Return True only if text looks like a real stock ticker."""
        if not text or not isinstance(text, str):
            return False

        ticker = text.upper().strip()
        # Handle dotted class shares: BRK.A, BF.B
        base = ticker.split('.')[0]

        # Length: 1–5 letters, all alpha
        if not (1 <= len(base) <= 5):
            return False
        if not base.isalpha():
            return False

        # Hard blacklist
        if ticker in self.TICKER_FALSE_POSITIVES:
            return False
        if base in self.TICKER_FALSE_POSITIVES:
            return False

        # Single-letter tickers: only a handful are real
        if len(base) == 1:
            return base in {'A', 'C', 'F', 'G', 'H', 'J', 'M', 'R', 'T', 'V', 'Z'}

        return True

    def _is_valid_company_name(self, text: str) -> bool:
        """Return True only if text looks like a real company name."""
        if not text or not isinstance(text, str):
            return False

        cleaned = text.strip()

        # Length bounds
        if len(cleaned) < 3 or len(cleaned) > 60:
            return False

        # Must start with an uppercase letter
        if not cleaned[0].isupper():
            return False

        # No newlines / tabs (scraped content artefacts)
        if '\n' in cleaned or '\t' in cleaned:
            return False

        # Valid character set
        if not re.match(r'^[A-Za-z0-9\s\.,&\'\-]+$', cleaned):
            return False

        # Too many words → almost certainly a fragment
        words = cleaned.split()
        if len(words) > 5:
            return False

        # Any function/fragment word present → reject
        word_set = {w.lower().rstrip('.,') for w in words}
        if word_set & self._FRAGMENT_WORDS:
            return False

        # Dangling preposition / article at end
        last = words[-1].lower().rstrip('.,')
        if last in {'the', 'and', 'for', 'with', 'of', 'in', 'a', 'an', 'or', 'at'}:
            return False

        # Single all-caps word (e.g. "APPLE") is a ticker candidate, not a company name.
        if len(words) == 1 and cleaned.isupper() and len(cleaned) > 2:
            return False
        
        # Filter out common false positives
        false_company_names = {'institutional', 'maxim', 'analyst', 'analysts', 'research'}
        if any(word.lower() in false_company_names for word in words):
            return False

        return True

    def _is_valid_person_name(self, text: str) -> bool:
        """Return True if text looks like a person's name."""
        if not text or not isinstance(text, str):
            return False
        cleaned = text.strip()
        if len(cleaned) < 3 or len(cleaned) > 50:
            return False
        if not re.search(r'[A-Za-z]', cleaned):
            return False
        # All-uppercase strings > 3 chars are almost certainly not person names
        if cleaned.isupper() and len(cleaned) > 3:
            return False
        return True

    def _clean_entity(self, text: str, entity_type: str) -> str:
        """Normalise whitespace and apply type-specific formatting."""
        if not text:
            return ""
        cleaned = ' '.join(text.split())

        if entity_type == "tickers":
            cleaned = cleaned.upper()
            cleaned = re.sub(r'^(?:STOCK|SHARE|PRICE|TICKER)\s+', '', cleaned)

        elif entity_type in ("companies", "organizations"):
            words = cleaned.split()
            result = []
            for word in words:
                if word.isupper() and len(word) > 3 and word not in ('INC', 'CORP', 'LTD', 'LLC', 'PLC'):
                    result.append(word.capitalize())
                else:
                    result.append(word)
            cleaned = ' '.join(result)

        return cleaned.strip()

    def _empty_result(self) -> Dict[str, List[str]]:
        return {
            "tickers": [],
            "companies": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "currencies": [],
            "industries": [],
        }