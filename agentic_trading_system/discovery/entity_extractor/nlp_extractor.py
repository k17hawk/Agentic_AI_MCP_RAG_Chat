"""
NLP Extractor - Uses NLP to extract entities from text
"""
from typing import Dict, List, Optional, Any
import re
import asyncio
from collections import Counter

from utils.logger import logger as  logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("⚠️ spaCy not available - NLP extractor will use fallback methods")

class NLPExtractor:
    """
    Extracts entities using Natural Language Processing
    Uses spaCy for advanced entity recognition with fallback methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("spacy_model", "en_core_web_sm")
        
        # Load spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.model_name)
                logging.info(f"✅ Loaded spaCy model: {self.model_name}")
            except:
                logging.warning(f"⚠️ Could not load spaCy model {self.model_name}, using fallback")
        
        # Company name patterns
        self.company_indicators = [
            "inc", "corp", "corporation", "ltd", "limited", "llc",
            "co", "company", "group", "holdings", "technologies",
            "solutions", "systems", "software", "bank", "insurance"
        ]
        
        # Ticker patterns (uppercase letters, usually 1-5)
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Stock exchange indicators
        self.exchange_indicators = [
            "nyse", "nasdaq", "amex", "tsx", "lse", "tse", "hkex"
        ]
        
        logging.info(f"✅ NLPExtractor initialized")
    
    async def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using NLP
        """
        if not text or len(text) < 10:
            return self._empty_result()
        
        entities = {
            "tickers": [],
            "companies": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "currencies": [],
            "industries": []
        }
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text[:100000])  # Limit text length
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Check if it might be a ticker
                    if self._is_likely_ticker(ent.text):
                        entities["tickers"].append(ent.text)
                    else:
                        entities["organizations"].append(ent.text)
                        
                        # Check if it's a company
                        if self._is_likely_company(ent.text):
                            entities["companies"].append(ent.text)
                
                elif ent.label_ == "PERSON":
                    entities["people"].append(ent.text)
                
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                
                elif ent.label_ == "MONEY":
                    entities["currencies"].append(ent.text)
        
        # Always run regex extraction as backup
        regex_entities = await self._extract_regex(text)
        
        # Merge results
        for key in entities:
            entities[key].extend(regex_entities.get(key, []))
        
        # Clean and deduplicate
        for key in entities:
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for item in entities[key]:
                item_clean = item.strip()
                if item_clean and item_clean not in seen:
                    seen.add(item_clean)
                    unique.append(item_clean)
            entities[key] = unique
        
        return entities
    
    async def _extract_regex(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using regex patterns
        """
        entities = {
            "tickers": [],
            "companies": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "currencies": [],
            "industries": []
        }
        
        # Extract potential tickers
        for match in self.ticker_pattern.finditer(text):
            ticker = match.group()
            if self._is_likely_ticker(ticker):
                entities["tickers"].append(ticker)
        
        # Extract company names (Title Case followed by indicator)
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    # Check for "Company Name Inc" pattern
                    potential = f"{word} {words[i+1]}"
                    if any(ind in potential.lower() for ind in self.company_indicators):
                        entities["companies"].append(potential)
        
        # Extract currency symbols
        currency_pattern = re.compile(r'[$€£¥]')
        for match in currency_pattern.finditer(text):
            entities["currencies"].append(match.group())
        
        return entities
    
    def _is_likely_ticker(self, text: str) -> bool:
        """
        Check if text is likely a stock ticker
        """
        # Must be all uppercase letters
        if not text.isupper():
            return False
        
        # Length check
        if len(text) < 1 or len(text) > 5:
            return False
        
        # Common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS',
            'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD',
            'NYSE', 'NASDAQ', 'SEC', 'CEO', 'CFO', 'EPS', 'YOY',
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD'
        }
        
        if text in false_positives:
            return False
        
        # Check if followed by stock-related words
        # This would need context - handled by caller
        
        return True
    
    def _is_likely_company(self, text: str) -> bool:
        """
        Check if text is likely a company name
        """
        text_lower = text.lower()
        
        # Check for company indicators
        for indicator in self.company_indicators:
            if indicator in text_lower:
                return True
        
        # Check for common company name patterns
        if len(text.split()) <= 3 and text[0].isupper():
            return True
        
        return False
    
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
            "industries": []
        }