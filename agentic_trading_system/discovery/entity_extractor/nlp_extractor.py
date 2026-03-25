# =============================================================================
# discovery/entity_extractor/nlp_extractor.py (UPDATED)
# =============================================================================
"""
NLP Extractor - Uses NLP to extract entities from text
"""

from typing import Dict, List, Any, Set
import re
import asyncio

# Import from new config structure
from  agentic_trading_system.constants import EntityType, EntityExtraction

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

    # Words that must NEVER be treated as tickers
    TICKER_FALSE_POSITIVES: Set[str] = {
        'A', 'AN', 'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM', 'THAT',
        'THIS', 'THESE', 'THOSE', 'ON', 'IN', 'AT', 'TO', 'BY', 'OF', 'UP',
        'UPON', 'INTO', 'ONTO', 'WITHIN', 'OUT', 'OVER', 'UNDER', 'ABOVE',
        'BELOW', 'BETWEEN', 'AMONG', 'AS', 'IF', 'SO', 'YET', 'NOR',
        'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'ME', 'HIM', 'HER',
        'US', 'THEM', 'MY', 'YOUR', 'HIS', 'ITS', 'OUR', 'THEIR',
        'MINE', 'YOURS', 'HERS', 'OURS', 'THEIRS',
        'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'AM',
        'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD',
        'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'CANNOT',
        'INC', 'CORP', 'LTD', 'LLC', 'LP', 'PLC', 'SA', 'AG', 'NV',
        'CO', 'GROUP', 'HOLDINGS', 'ETF', 'IPO', 'CEO', 'CFO', 'COO',
        'CTO', 'CIO', 'EPS', 'PE', 'ROE', 'ROA', 'EBITDA',
        'NYSE', 'NASDAQ', 'AMEX', 'SEC', 'FED', 'ECB',
        'GDP', 'CPI', 'PMI', 'USD', 'EUR', 'GBP', 'JPY', 'CNY',
        'BUY', 'SELL', 'HOLD', 'BULL', 'BEAR', 'LONG', 'SHORT',
        'CALL', 'PUT', 'STOCK', 'SHARE', 'PRICE', 'VOLUME',
    }

    # Words that indicate a spaCy ORG span is actually a sentence fragment
    _FRAGMENT_WORDS: Set[str] = {
        'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'that',
        'this', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
        'by', 'what', 'which', 'who', 'when', 'where', 'how', 'why', 'if',
        'more', 'most', 'some', 'all', 'any', 'its', 'our', 'their', 'your',
        'see', 'find', 'get', 'think', 'know', 'analysts', 'latest', 'detailed',
        'low', 'high', 'new', 'old', 'level', 'support', 'resistance',
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NLP extractor.
        
        Args:
            config: NLP extractor configuration dictionary
        """
        self.config = config
        self.model_name = config.get("spacy_model", "en_core_web_sm")
        self.entity_types = config.get("entity_types", ["PERSON", "ORG", "GPE", "MONEY", "DATE", "PERCENT"])

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.model_name)
                logging.info(f"✅ Loaded spaCy model: {self.model_name}")
            except Exception as e:
                logging.warning(f"⚠️ Could not load spaCy model: {e}")

        # Company name pattern
        self.company_pattern = re.compile(
            r'\b([A-Z][A-Za-z0-9\.,&\'-]*(?:\s+[A-Z][A-Za-z0-9\.,&\'-]*)*)\s+'
            r'(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|L\.L\.C\.|'
            r'Co\.?|Company|Group|Holdings|Technologies|Tech\.?|'
            r'Pharma|Biosciences|Industries|International|Global)\b',
            re.IGNORECASE
        )

        # Ticker patterns
        self._ticker_pattern = re.compile(
            r'\$([A-Z]{1,5})\b'
            r'|\(([A-Z]{1,5})\)'
            r'|\b([A-Z]{2,5})\s+(?:stock|shares?|equity|price|trading)\b'
            r'|\b(?:stock|shares?|equity|price|trading)\s+([A-Z]{2,5})\b'
            r'|\b([A-Z]{1,5})\.([A-Z]{1,2})\b',
            re.IGNORECASE
        )

        logging.info(f"✅ NLPExtractor initialized ({'spaCy + regex' if self.nlp else 'regex only'})")

    async def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        if not text or len(text) < 10:
            return self._empty_result()

        entities = self._empty_result()

        # spaCy pass
        if self.nlp:
            try:
                doc = self.nlp(text[:EntityExtraction.MAX_TEXT_LENGTH])
                for ent in doc.ents:
                    if ent.label_ == "ORG":
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

            except Exception as e:
                logging.warning(f"⚠️ spaCy extraction failed: {e}")

        # Regex pass
        regex_entities = await self._extract_regex(text)
        for key in entities:
            entities[key].extend(regex_entities.get(key, []))

        # Deduplicate and validate
        for key in entities:
            seen: set = set()
            unique: List[str] = []
            for item in entities[key]:
                cleaned = self._clean_entity(item, key)
                if not cleaned or cleaned in seen:
                    continue
                if key == "tickers" and not self._is_valid_ticker(cleaned):
                    continue
                if key in ("companies", "organizations") and not self._is_valid_company_name(cleaned):
                    continue
                seen.add(cleaned)
                unique.append(cleaned)
            entities[key] = unique[:EntityExtraction.MAX_ENTITIES_PER_TYPE]

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

        # Tickers
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
        base = ticker.split('.')[0]

        if not (1 <= len(base) <= 5):
            return False
        if not base.isalpha():
            return False

        if ticker in self.TICKER_FALSE_POSITIVES:
            return False
        if base in self.TICKER_FALSE_POSITIVES:
            return False

        if len(base) == 1:
            return base in EntityExtraction.VALID_SINGLE_LETTER_TICKERS

        return True

    def _is_valid_company_name(self, text: str) -> bool:
        """Return True only if text looks like a real company name."""
        if not text or not isinstance(text, str):
            return False

        cleaned = text.strip()

        if len(cleaned) < 3 or len(cleaned) > 60:
            return False

        if not cleaned[0].isupper():
            return False

        if '\n' in cleaned or '\t' in cleaned:
            return False

        if not re.match(r'^[A-Za-z0-9\s\.,&\'\-]+$', cleaned):
            return False

        words = cleaned.split()
        if len(words) > 5:
            return False

        word_set = {w.lower().rstrip('.,') for w in words}
        if word_set & self._FRAGMENT_WORDS:
            return False

        last = words[-1].lower().rstrip('.,')
        if last in {'the', 'and', 'for', 'with', 'of', 'in', 'a', 'an', 'or', 'at'}:
            return False

        if len(words) == 1 and cleaned.isupper() and len(cleaned) > 2:
            return False

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
        }