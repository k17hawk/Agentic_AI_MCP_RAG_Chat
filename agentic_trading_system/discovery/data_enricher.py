"""
Data Enricher - Enriches discovered data with additional context
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import yfinance as yf
import asyncio

from utils.logger import logger as logging
from discovery.entity_extractor.regex_extractor import RegexExtractor

class DataEnricher:
    """
    Enriches discovered data with additional context
    
    Enrichments:
    - Company information (sector, industry, market cap)
    - Current stock price and performance
    - Related news and social sentiment
    - Historical context
    - Relevance scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Cache for company info
        self.company_cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 60) * 60
        
        # Entity extractor for ticker detection
        self.extractor = RegexExtractor(config)
        
        logging.info(f"✅ DataEnricher initialized")
    
    async def enrich(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single data item with additional context
        """
        enriched = item.copy()
        
        # Extract tickers from content
        content = f"{item.get('title', '')} {item.get('content', '')}"
        tickers = await self.extractor.extract_tickers(content)
        
        if tickers:
            enriched["detected_tickers"] = tickers
            
            # Get company info for the first ticker
            primary_ticker = tickers[0]
            company_info = await self._get_company_info(primary_ticker)
            if company_info:
                enriched["company_info"] = company_info
                
                # Calculate relevance score
                enriched["relevance_score"] = self._calculate_relevance(item, company_info)
        
        # Add timestamp if not present
        if "timestamp" not in enriched:
            enriched["timestamp"] = datetime.now().isoformat()
        
        # Add content length and word count
        if "content" in enriched:
            enriched["content_length"] = len(enriched["content"])
            enriched["word_count"] = len(enriched["content"].split())
        
        # Add source authority score
        enriched["source_authority"] = self._get_source_authority(enriched.get("source", ""))
        
        # Add content type classification
        enriched["content_type"] = self._classify_content(enriched)
        
        return enriched
    
    async def enrich_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple items in parallel
        """
        tasks = [self.enrich(item) for item in items]
        return await asyncio.gather(*tasks)
    
    async def _get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company information from yfinance
        """
        # Check cache
        if ticker in self.company_cache:
            cached_time, info = self.company_cache[ticker]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return info
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                company_info = {
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": info.get("country", "Unknown"),
                    "website": info.get("website", ""),
                    "market_cap": info.get("marketCap"),
                    "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "day_change": info.get("regularMarketChangePercent"),
                    "year_high": info.get("fiftyTwoWeekHigh"),
                    "year_low": info.get("fiftyTwoWeekLow"),
                    "volume": info.get("volume"),
                    "avg_volume": info.get("averageVolume"),
                    "pe_ratio": info.get("trailingPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "updated_at": datetime.now().isoformat()
                }
                
                # Cache
                self.company_cache[ticker] = (datetime.now().timestamp(), company_info)
                
                return company_info
        except Exception as e:
            logging.debug(f"Error getting company info for {ticker}: {e}")
        
        return None
    
    def _calculate_relevance(self, item: Dict[str, Any], company_info: Dict) -> float:
        """
        Calculate relevance score for this item
        """
        score = 0.5  # Base score
        
        title = item.get("title", "").lower()
        content = item.get("content", "")[:500].lower()
        
        # Check for company name in title
        company_name = company_info.get("name", "").lower()
        if company_name and (company_name in title or company_name in content):
            score += 0.2
        
        # Check for sector/industry terms
        sector = company_info.get("sector", "").lower()
        industry = company_info.get("industry", "").lower()
        
        if sector and (sector in title or sector in content):
            score += 0.15
        if industry and (industry in title or industry in content):
            score += 0.1
        
        # Check for financial terms
        financial_terms = ["earnings", "revenue", "profit", "loss", "guidance", "forecast"]
        for term in financial_terms:
            if term in title or term in content:
                score += 0.05
        
        # Recency boost (if available)
        published = item.get("published_at")
        if published:
            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                if age_hours < 24:
                    score += 0.1
                elif age_hours < 72:
                    score += 0.05
            except:
                pass
        
        return float(min(1.0, score))
    
    def _get_source_authority(self, source: str) -> float:
        """
        Get authority score for a source
        """
        source_lower = source.lower()
        
        # High authority sources
        high_authority = {
            "reuters", "bloomberg", "wsj", "wall street journal",
            "financial times", "ft.com", "nytimes", "economist"
        }
        
        # Medium authority sources
        medium_authority = {
            "cnbc", "yahoo", "seeking alpha", "marketwatch",
            "forbes", "business insider", "investopedia"
        }
        
        # Low authority sources
        low_authority = {
            "twitter", "reddit", "stocktwits", "facebook",
            "linkedin", "medium", "wordpress"
        }
        
        for src in high_authority:
            if src in source_lower:
                return 1.0
        
        for src in medium_authority:
            if src in source_lower:
                return 0.7
        
        for src in low_authority:
            if src in source_lower:
                return 0.3
        
        return 0.5  # Default
    
    def _classify_content(self, item: Dict[str, Any]) -> str:
        """
        Classify content type
        """
        title = item.get("title", "").lower()
        content = item.get("content", "")[:200].lower()
        combined = title + " " + content
        
        # News categories
        if any(word in combined for word in ["earnings", "revenue", "profit", "loss"]):
            return "earnings"
        elif any(word in combined for word in ["merger", "acquisition", "buyout", "takeover"]):
            return "ma"
        elif any(word in combined for word in ["upgrade", "downgrade", "rating", "target"]):
            return "analyst_rating"
        elif any(word in combined for word in ["ipo", "offering", "listing"]):
            return "ipo"
        elif any(word in combined for word in ["dividend", "buyback", "split"]):
            return "corporate_action"
        elif any(word in combined for word in ["sec", "investigation", "lawsuit", "regulatory"]):
            return "regulatory"
        elif any(word in combined for word in ["ceo", "cfo", "executive", "management"]):
            return "management"
        elif any(word in combined for word in ["price", "target", "forecast", "prediction"]):
            return "price_target"
        else:
            return "general"