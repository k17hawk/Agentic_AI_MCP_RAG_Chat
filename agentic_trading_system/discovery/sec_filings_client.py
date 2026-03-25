# =============================================================================
# discovery/sec_filings_client.py (UPDATED)
# =============================================================================
"""
SEC Filings Client - Fetches and parses SEC filings
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import xml.etree.ElementTree as ET
import re
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

# Import from new config structure
from agentic_trading_system.constants import Source, FilingType, RateLimit, CacheTTL
from agentic_trading_system.config.config__entity import SECFilingsConfig


class SECFilingsClient:
    """
    Fetches and parses SEC filings (10-K, 10-Q, 8-K, Form 4, etc.)
    Uses SEC EDGAR API
    """

    def __init__(self, config: SECFilingsConfig):
        """
        Initialize SEC filings client.
        
        Args:
            config: SECFilingsConfig object
        """
        self.config = config
        self.user_agent = config.user_agent
        self.base_url = config.base_url

        # Rate limiting - SEC requires 10 requests per second max
        self.rate_limit = config.rate_limit
        self.request_timestamps: List[float] = []

        # Filing types to fetch
        self.filing_types = config.filing_types

        # Cache
        self.cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60

        # Cache for CIK mappings
        self.cik_cache: Dict[str, Optional[str]] = {}
        self.known_ciks = config.known_ciks

        logging.info(f"✅ SECFilingsClient initialized")

    @retry(max_attempts=3, delay=2.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for SEC filings by company ticker.
        
        Args:
            query: Company ticker or name
            options: Search options
            
        Returns:
            Dict with 'items' and 'metadata' keys
        """
        options = options or {}

        # Clean query - remove numbers and extra words
        clean_query = re.sub(r'[^A-Za-z]', '', query).upper()
        if not clean_query:
            clean_query = query.upper()

        logging.info(f"📄 SEC filings search for: '{clean_query}' (original: '{query}')")

        # Check cache
        cache_key = f"sec_{clean_query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached SEC results for '{clean_query}'")
                return cached_result

        await self._rate_limit()

        try:
            # First get CIK number for the ticker
            cik = await self._get_cik(clean_query)
            if not cik:
                logging.warning(f"⚠️ CIK not found for ticker: {clean_query}")
                return {
                    "items": [],
                    "metadata": {
                        "error": f"CIK not found for {clean_query}",
                        "ticker": clean_query
                    }
                }

            logging.info(f"✅ Found CIK {cik} for {clean_query}")

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
                    "ticker": clean_query,
                    "total_found": len(filings),
                    "processed_count": len(processed_filings)
                }
            }

            self.cache[cache_key] = (datetime.now().timestamp(), result)

            logging.info(f"✅ SEC found {len(processed_filings)} filings for {clean_query}")
            return result

        except Exception as e:
            logging.error(f"❌ SEC search error: {e}")
            return {"items": [], "metadata": {"error": str(e)}}

    async def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        ticker = ticker.upper()

        # Check cache
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]

        # Check known CIKs from config
        if ticker in self.known_ciks:
            cik = self.known_ciks[ticker]
            self.cik_cache[ticker] = cik
            return cik

        # Try from company_tickers.json
        cik = await self._get_cik_from_tickers_json(ticker)
        if not cik:
            cik = await self._get_cik_from_search(ticker)

        # Cache the result (even if None)
        self.cik_cache[ticker] = cik
        return cik

    async def _get_cik_from_tickers_json(self, ticker: str) -> Optional[str]:
        """Get CIK from company_tickers.json."""
        url = f"{self.base_url}/files/company_tickers.json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        ticker = ticker.upper()
                        for item in data.values():
                            if item.get("ticker") == ticker:
                                cik = str(item.get("cik_str")).zfill(10)
                                return cik
        except asyncio.TimeoutError:
            logging.warning(f"⏰ SEC tickers.json timeout")
        except Exception as e:
            logging.debug(f"Error getting CIK from tickers.json: {e}")

        return None

    async def _get_cik_from_search(self, ticker: str) -> Optional[str]:
        """Get CIK by searching EDGAR."""
        url = f"{self.base_url}/cgi-bin/browse-edgar"

        params = {
            "action": "getcompany",
            "CIK": ticker,
            "output": "atom"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    headers={"User-Agent": self.user_agent},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Parse CIK from response
                        match = re.search(r'CIK=(\d{10})', text)
                        if match:
                            return match.group(1)
        except asyncio.TimeoutError:
            logging.warning(f"⏰ SEC search timeout for {ticker}")
        except Exception as e:
            logging.debug(f"Error getting CIK from search: {e}")

        return None

    async def _get_filings(self, cik: str, options: Dict) -> List[Dict]:
        """Get recent filings for a CIK."""
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
                await self._rate_limit()

                async with session.get(
                    url, params=params,
                    headers={"User-Agent": self.user_agent},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_filings_atom(text)
                    else:
                        logging.debug(f"SEC filings request failed: {response.status}")
        except asyncio.TimeoutError:
            logging.warning(f"⏰ SEC filings timeout for CIK {cik}")
        except Exception as e:
            logging.debug(f"Error getting filings: {e}")

        return []

    def _parse_filings_atom(self, xml_text: str) -> List[Dict]:
        """Parse filings from Atom feed."""
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

        except ET.ParseError as e:
            logging.debug(f"Error parsing Atom feed: {e}")
        except Exception as e:
            logging.debug(f"Error parsing filings: {e}")

        return filings

    def _extract_text(self, element: ET.Element, path: str, attrib: str = None) -> str:
        """Extract text or attribute from XML element."""
        found = element.find(path)
        if found is not None:
            if attrib:
                return found.get(attrib, "")
            return found.text or ""
        return ""

    def _extract_id_from_href(self, href: str) -> str:
        """Extract filing ID from href."""
        match = re.search(r'/(\d+)-(\d+)-(\d+)', href)
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}"
        return ""

    async def _process_filing(self, filing: Dict, options: Dict) -> Optional[Dict]:
        """Process a filing and extract key information."""
        filing_type = filing.get("filing_type", "")

        # Determine filing importance
        importance = FilingType.IMPORTANCE_MAP.get(filing_type, FilingType.IMPORTANCE_LOW)

        # Extract key data based on filing type
        key_data = {}

        if filing_type in [FilingType.FORM_10K, FilingType.FORM_10Q]:
            key_data = {
                "period": "annual" if filing_type == FilingType.FORM_10K else "quarterly",
                "is_annual": filing_type == FilingType.FORM_10K,
                "is_quarterly": filing_type == FilingType.FORM_10Q
            }
        elif filing_type == FilingType.FORM_8K:
            key_data = {
                "is_current_report": True,
                "event_type": self._guess_event_type(filing.get("filing_href", ""))
            }
        elif filing_type == FilingType.FORM_4:
            key_data = {
                "is_insider_trading": True
            }
        elif filing_type in [FilingType.FORM_13F_HR, "13F"]:
            key_data = {
                "is_institutional_holdings": True
            }

        return {
            **filing,
            "importance": importance,
            "key_data": key_data,
            "source": Source.SEC,
            "title": f"{filing.get('company_name', '')} - {filing_type} Filing",
            "content": f"SEC {filing_type} filing filed on {filing.get('filing_date', '')}",
            "url": filing.get("filing_href", ""),
            "published_at": filing.get("filing_date", ""),
            "type": "filing"
        }

    def _guess_event_type(self, href: str) -> str:
        """Guess event type from 8-K filing."""
        href_lower = href.lower()
        if "item1" in href_lower:
            return "business_changes"
        elif "item2" in href_lower:
            return "financial_info"
        elif "item3" in href_lower:
            return "bankruptcy"
        elif "item4" in href_lower:
            return "accounting_changes"
        elif "item5" in href_lower:
            return "corporate_governance"
        elif "item7" in href_lower:
            return "financial_statements"
        elif "item8" in href_lower:
            return "other_events"
        else:
            return "general"

    async def _rate_limit(self):
        """Rate limiting for SEC API (per second)."""
        now = datetime.now().timestamp()

        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if now - ts < 1]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 1 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        self.cik_cache.clear()
        logging.info("🧹 SEC filings cache cleared")