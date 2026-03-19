"""
SEC Filings Client - Fetches and parses SEC filings
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import xml.etree.ElementTree as ET
import re

from agentic_trading_system.utils.logger import logger as  logging
from agentic_trading_system.utils.decorators import retry
import asyncio
class SECFilingsClient:
    """
    Fetches and parses SEC filings (10-K, 10-Q, 8-K, Form 4, etc.)
    Uses SEC EDGAR API
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_agent = config.get("user_agent", "TradingBot/1.0 (your-email@example.com)")
        self.base_url = config.get("base_url", "https://www.sec.gov")
        
        # Rate limiting - SEC requires 10 requests per second max
        self.rate_limit = config.get("rate_limit", 5)
        self.request_timestamps = []
        
        # Filing types to fetch
        self.filing_types = config.get("filing_types", [
            "10-K", "10-Q", "8-K", "4", "13F-HR"
        ])
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_hours", 24) * 3600
        
        logging.info(f"✅ SECFilingsClient initialized")
    
    @retry(max_attempts=3, delay=2.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for SEC filings by company ticker
        """
        options = options or {}
        logging.info(f"📄 SEC filings search for: '{query}'")
        
        # Check cache
        cache_key = f"sec_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return cached_result
        
        await self._rate_limit()
        
        try:
            # First get CIK number for the ticker
            cik = await self._get_cik(query)
            if not cik:
                return {"items": [], "metadata": {"error": "CIK not found"}}
            
            # Get recent filings
            filings = await self._get_filings(cik, options)
            
            # Process filings
            processed_filings = []
            for filing in filings[:options.get("max_results", 20)]:
                processed = await self._process_filing(filing, options)
                if processed:
                    processed_filings.append(processed)
            
            result = {
                "items": processed_filings,
                "metadata": {
                    "cik": cik,
                    "ticker": query.upper(),
                    "total_found": len(filings),
                    "processed_count": len(processed_filings)
                }
            }
            
            self.cache[cache_key] = (datetime.now().timestamp(), result)
            
            logging.info(f"✅ SEC found {len(processed_filings)} filings")
            return result
            
        except Exception as e:
            logging.error(f"❌ SEC search error: {e}")
            return {"items": [], "metadata": {"error": str(e)}}
    
    async def _get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker
        """
        url = f"{self.base_url}/files/company_tickers.json"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": self.user_agent}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        ticker = ticker.upper()
                        for item in data.values():
                            if item.get("ticker") == ticker:
                                cik = str(item.get("cik_str")).zfill(10)
                                return cik
        except Exception as e:
            logging.debug(f"Error getting CIK: {e}")
        
        return None
    
    async def _get_filings(self, cik: str, options: Dict) -> List[Dict]:
        """
        Get recent filings for a CIK
        """
        url = f"{self.base_url}/cgi-bin/browse-edgar"
        
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": options.get("filing_type", ""),
            "dateb": options.get("end_date", ""),
            "owner": "exclude",
            "count": options.get("limit", 100),
            "output": "atom"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers={"User-Agent": self.user_agent}) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_filings_atom(text)
        except Exception as e:
            logging.debug(f"Error getting filings: {e}")
        
        return []
    
    def _parse_filings_atom(self, xml_text: str) -> List[Dict]:
        """
        Parse filings from Atom feed
        """
        filings = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                filing = {
                    "filing_type": self._extract_text(entry, ".//{http://www.w3.org/2005/Atom}category", "term"),
                    "company_name": self._extract_text(entry, ".//{http://www.w3.org/2005/Atom}title"),
                    "filing_date": self._extract_text(entry, ".//{http://www.w3.org/2005/Atom}updated"),
                    "filing_href": self._extract_text(entry, ".//{http://www.w3.org/2005/Atom}link", "href"),
                    "filing_id": self._extract_id_from_href(self._extract_text(entry, ".//{http://www.w3.org/2005/Atom}link", "href"))
                }
                
                if filing["filing_type"]:
                    filings.append(filing)
        
        except Exception as e:
            logging.debug(f"Error parsing Atom feed: {e}")
        
        return filings
    
    def _extract_text(self, element: ET.Element, path: str, attrib: str = None) -> str:
        """Extract text or attribute from XML element"""
        found = element.find(path)
        if found is not None:
            if attrib:
                return found.get(attrib, "")
            return found.text or ""
        return ""
    
    def _extract_id_from_href(self, href: str) -> str:
        """Extract filing ID from href"""
        match = re.search(r'/(\d+)-(\d+)-(\d+)', href)
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}"
        return ""
    
    async def _process_filing(self, filing: Dict, options: Dict) -> Optional[Dict]:
        """
        Process a filing and extract key information
        """
        filing_type = filing["filing_type"]
        
        # Determine filing importance
        importance = self._get_filing_importance(filing_type)
        
        # Extract key data based on filing type
        key_data = {}
        
        if filing_type in ["10-K", "10-Q"]:
            # Annual/Quarterly report
            key_data = {
                "period": "annual" if filing_type == "10-K" else "quarterly",
                "is_annual": filing_type == "10-K",
                "is_quarterly": filing_type == "10-Q"
            }
        elif filing_type == "8-K":
            # Current report (significant events)
            key_data = {
                "is_current_report": True,
                "event_type": self._guess_event_type(filing.get("filing_href", ""))
            }
        elif filing_type == "4":
            # Insider trading
            key_data = {
                "is_insider_trading": True
            }
        elif filing_type in ["13F-HR", "13F"]:
            # Institutional holdings
            key_data = {
                "is_institutional_holdings": True
            }
        
        return {
            **filing,
            "importance": importance,
            "key_data": key_data,
            "source": "sec"
        }
    
    def _get_filing_importance(self, filing_type: str) -> str:
        """
        Determine importance of filing type
        """
        importance_map = {
            "10-K": "high",
            "8-K": "high",
            "4": "high",
            "10-Q": "medium",
            "13F-HR": "medium",
            "DEF 14A": "medium",
            "S-1": "high",
            "S-4": "medium"
        }
        
        return importance_map.get(filing_type, "low")
    
    def _guess_event_type(self, href: str) -> str:
        """
        Guess event type from 8-K filing
        """
        if "item1" in href.lower():
            return "business_changes"
        elif "item2" in href.lower():
            "financial_info"
        elif "item3" in href.lower():
            "bankruptcy"
        elif "item4" in href.lower():
            "accounting_changes"
        elif "item5" in href.lower():
            "corporate_governance"
        elif "item7" in href.lower():
            "financial_statements"
        elif "item8" in href.lower():
            "other_events"
        else:
            return "general"
    
    async def _rate_limit(self):
        """Rate limiting for SEC API"""
        now = datetime.now().timestamp()
        
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < 1]  # Per second
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 1 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)