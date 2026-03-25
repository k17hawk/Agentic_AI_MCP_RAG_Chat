# =============================================================================
# discovery/macro_data_client.py (UPDATED)
# =============================================================================
"""
Macro Data Client - Fetches macroeconomic indicators
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import numpy as np
import re

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

# Import from new config structure
from agentic_trading_system.constants import Source, RateLimit, CacheTTL
from agentic_trading_system.config.config__entity import MacroDataConfig


class MacroDataClient:
    """
    Fetches macroeconomic indicators from FRED and other sources.
    """

    def __init__(self, config: MacroDataConfig):
        """
        Initialize macro data client.
        
        Args:
            config: MacroDataConfig object
        """
        self.config = config

        # API keys
        self.fred_api_key = config.fred_api_key

        # Rate limiting
        self.rate_limit = config.rate_limit
        self.request_timestamps: List[float] = []

        # Cache
        self.cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60

        # Load indicators from config
        self.indicators = config.indicators

        logging.info(f"✅ MacroDataClient initialized with {len(self.indicators)} indicators")

    @retry(max_attempts=3, delay=2.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for macroeconomic data.
        
        Args:
            query: Macroeconomic indicator name (e.g., "GDP", "inflation")
            options: Search options
            
        Returns:
            Dict with 'items' and 'metadata' keys
        """
        options = options or {}

        logging.info(f"📈 Macro data search for: '{query}'")

        # Check cache
        cache_key = f"macro_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached macro results for '{query}'")
                return cached_result

        await self._rate_limit()

        # Find matching indicator
        indicator_key, indicator = self._find_indicator(query)

        if not indicator:
            logging.warning(f"⚠️ No indicator found for query: '{query}', trying direct FRED search")
            return await self._direct_fred_search(query, options)

        # Log the FRED series ID
        fred_series = indicator.get("fred")
        logging.info(f"📡 Fetching FRED series: {fred_series} for indicator: {indicator.get('name')}")

        items = []
        sources_used = []

        # Try FRED
        if self.fred_api_key and indicator.get("fred"):
            fred_data = await self._fetch_fred(indicator["fred"], options)
            if fred_data:
                item = self._format_item(fred_data, indicator_key, indicator, Source.FRED)
                items.append(item)
                sources_used.append(Source.FRED)
            else:
                logging.warning(f"⚠️ No data returned from FRED for series {indicator['fred']}")

        result = {
            "items": items,
            "metadata": {
                "query": query,
                "indicator": indicator.get("name", query),
                "description": indicator.get("description", f"{indicator.get('name', query)} data from FRED"),
                "sources_used": sources_used,
                "total_found": len(items)
            }
        }

        # Cache result
        self.cache[cache_key] = (datetime.now().timestamp(), result)

        logging.info(f"✅ Macro data: found {len(items)} items")
        return result

    async def _direct_fred_search(self, query: str, options: Dict) -> Dict[str, Any]:
        """Direct FRED search when no indicator mapping found."""
        if not self.fred_api_key:
            return {
                "items": [],
                "metadata": {
                    "message": f"No macroeconomic indicator found for '{query}' and FRED API key not configured"
                }
            }

        # Try to search FRED series
        url = "https://api.stlouisfed.org/fred/series/search"

        params = {
            "search_text": query,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "limit": 5
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        series_list = data.get("seriess", [])

                        if series_list:
                            series = series_list[0]
                            series_id = series.get("id")

                            fred_data = await self._fetch_fred(series_id, options)
                            if fred_data:
                                items = [self._format_item(
                                    fred_data,
                                    series_id,
                                    {
                                        "name": series.get("title", query),
                                        "unit": series.get("units", ""),
                                        "frequency": series.get("frequency", ""),
                                        "description": series.get("notes", "")
                                    },
                                    Source.FRED
                                )]

                                return {
                                    "items": items,
                                    "metadata": {
                                        "query": query,
                                        "indicator": series.get("title", query),
                                        "description": series.get("notes", ""),
                                        "sources_used": [Source.FRED]
                                    }
                                }
        except asyncio.TimeoutError:
            logging.warning(f"⏰ FRED search timeout for '{query}'")
        except Exception as e:
            logging.warning(f"⚠️ FRED search error: {e}")

        return {
            "items": [],
            "metadata": {
                "message": f"No data found for '{query}'. Try: gdp, inflation, unemployment, fed_funds"
            }
        }

    def _find_indicator(self, query: str) -> tuple:
        """Find matching indicator."""
        query_lower = query.lower().strip()

        # Direct match on key
        for key in self.indicators:
            if key.lower() == query_lower:
                return key, self.indicators[key]

        # Match on name
        for key, indicator in self.indicators.items():
            if indicator.get("name", "").lower() == query_lower:
                return key, indicator

        # Partial match
        for key, indicator in self.indicators.items():
            if (query_lower in key.lower() or
                query_lower in indicator.get("name", "").lower()):
                return key, indicator

        # Common aliases
        aliases = {
            "gdp": ["gdp", "gross domestic product", "economic growth"],
            "inflation": ["inflation", "cpi", "consumer price index", "prices"],
            "unemployment": ["unemployment", "jobless", "jobs", "labor"],
            "fed_funds": ["fed funds", "interest rates", "federal funds", "rates"]
        }

        for indicator_key, alias_list in aliases.items():
            if query_lower in alias_list:
                return indicator_key, self.indicators.get(indicator_key)

        return None, None

    async def _fetch_fred(self, series_id: str, options: Dict) -> Optional[Dict]:
        """Fetch data from FRED."""
        if not self.fred_api_key:
            logging.warning("⚠️ FRED API key not configured")
            return None

        url = "https://api.stlouisfed.org/fred/series/observations"

        # Default to last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)

        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "limit": 100,
            "sort_order": "desc"
        }

        try:
            async with aiohttp.ClientSession() as session:
                await self._rate_limit()

                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        observations = []
                        for obs in data.get("observations", []):
                            if obs.get("value") != ".":
                                observations.append({
                                    "date": obs.get("date"),
                                    "value": float(obs.get("value"))
                                })

                        if not observations:
                            logging.warning(f"⚠️ No observations for FRED series {series_id}")
                            return None

                        # Get series info
                        info = await self._get_fred_info(series_id)

                        # Calculate statistics
                        values = [o["value"] for o in observations]

                        return {
                            "series_id": series_id,
                            "title": info.get("title", series_id),
                            "units": info.get("units", ""),
                            "frequency": info.get("frequency", ""),
                            "observations": observations[:20],
                            "latest_value": observations[0]["value"] if observations else None,
                            "latest_date": observations[0]["date"] if observations else None,
                            "statistics": {
                                "mean": float(np.mean(values)) if values else None,
                                "median": float(np.median(values)) if values else None,
                                "min": float(min(values)) if values else None,
                                "max": float(max(values)) if values else None,
                                "trend": self._calculate_trend(values) if values else None
                            }
                        }
                    else:
                        logging.warning(f"⚠️ FRED API error: {response.status}")
                        return None

        except asyncio.TimeoutError:
            logging.warning(f"⏰ FRED timeout for series {series_id}")
        except Exception as e:
            logging.warning(f"⚠️ FRED error: {e}")

        return None

    async def _get_fred_info(self, series_id: str) -> Dict:
        """Get FRED series information."""
        url = "https://api.stlouisfed.org/fred/series"

        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        series = data.get("seriess", [{}])[0]
                        return {
                            "title": series.get("title"),
                            "units": series.get("units"),
                            "frequency": series.get("frequency"),
                            "notes": series.get("notes", "")
                        }
        except Exception:
            pass

        return {}

    def _format_item(self, data: Dict, indicator_key: str, indicator: Dict, source: str) -> Dict:
        """Format as a display item."""
        title = indicator.get("name", data.get("title", f"Macro Data: {indicator_key}"))

        content_lines = []

        if data.get("latest_value") is not None:
            unit = indicator.get("unit", data.get("units", ""))
            content_lines.append(f"Latest: {data['latest_value']:.2f} {unit} as of {data['latest_date']}")

        if indicator.get("description"):
            content_lines.append(f"\n{indicator['description']}")

        if data.get("statistics"):
            stats = data["statistics"]
            content_lines.append(f"\nStatistics (last 5 years):")
            content_lines.append(f"  Mean: {stats['mean']:.2f}")
            content_lines.append(f"  Median: {stats['median']:.2f}")
            content_lines.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")

            if stats.get("trend"):
                trend = stats["trend"]
                content_lines.append(f"  Recent Trend: {trend.get('direction', 'unknown')} ({trend.get('strength', 0):.1f}%)")

        if data.get("observations"):
            freq = indicator.get("frequency", data.get("frequency", ""))
            content_lines.append(f"\nRecent {freq} Values:")
            for obs in data["observations"][:5]:
                content_lines.append(f"  {obs['date']}: {obs['value']:.2f}")

        return {
            "title": f"{title} - {source.upper()}",
            "content": "\n".join(content_lines),
            "source": source,
            "indicator": title,
            "latest_value": data.get("latest_value"),
            "latest_date": data.get("latest_date"),
            "unit": indicator.get("unit", data.get("units", "")),
            "frequency": indicator.get("frequency", data.get("frequency", "")),
            "description": indicator.get("description", ""),
            "statistics": data.get("statistics", {}),
            "type": "macro_data",
            "published_at": data.get("latest_date"),
            "relevance_score": 0.9
        }

    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics."""
        if len(values) < 4:
            return {"direction": "unknown", "strength": 0}

        recent = values[:3]
        older = values[3:6]

        if older:
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)

            change = recent_avg - older_avg
            change_pct = (change / abs(older_avg)) * 100 if older_avg != 0 else 0

            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            strength = min(100, abs(change_pct))

            return {
                "direction": direction,
                "strength": float(strength),
                "change": float(change),
                "change_percent": float(change_pct)
            }

        return {"direction": "unknown", "strength": 0}

    async def test_fred_connection(self) -> bool:
        """Test FRED API connection."""
        if not self.fred_api_key:
            logging.error("❌ No FRED API key")
            return False

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "GDP",
            "api_key": self.fred_api_key,
            "file_type": "json",
            "limit": 1
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logging.info(f"✅ FRED API test successful: {data.get('count', 0)} observations")
                        return True
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ FRED API test failed: {response.status} - {error_text}")
                        return False
        except Exception as e:
            logging.error(f"❌ FRED API test exception: {e}")
            return False

    async def _rate_limit(self):
        """Rate limiting."""
        now = datetime.now().timestamp()

        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if now - ts < RateLimit.WINDOW_SECONDS]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = RateLimit.WINDOW_SECONDS - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        logging.info("🧹 Macro data cache cleared")