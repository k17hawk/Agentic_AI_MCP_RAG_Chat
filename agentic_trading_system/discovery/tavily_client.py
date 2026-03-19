"""
Tavily Client - Web search integration
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import asyncio

from agentic_trading_system.utils.logger import logger as  logging
from agentic_trading_system.utils.decorators import retry

class TavilyClient:
    """
    Tavily API client for web search
    Provides high-quality search results optimized for AI applications
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.tavily.com")
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 10)  # requests per minute
        self.request_timestamps = []
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 30) * 60
        
        logging.info(f"✅ TavilyClient initialized")
    
    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Perform web search using Tavily
        """
        options = options or {}
        logging.info(f"🔍 Tavily searching for: '{query}'")
        
        # Check cache
        cache_key = f"tavily_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached Tavily results for '{query}'")
                return cached_result
        
        await self._rate_limit()
        
        try:
            # Prepare request
            url = f"{self.base_url}/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": options.get("search_depth", "basic"),
                "include_answer": options.get("include_answer", True),
                "include_raw_content": options.get("include_raw_content", False),
                "max_results": options.get("max_results", 10),
                "include_domains": options.get("include_domains", []),
                "exclude_domains": options.get("exclude_domains", [])
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format results
                        items = []
                        for result in data.get("results", []):
                            items.append({
                                "title": result.get("title"),
                                "url": result.get("url"),
                                "content": result.get("content"),
                                "score": result.get("score"),
                                "published_at": result.get("published_date"),
                                "source": "tavily",
                                "type": "web_search"
                            })
                        
                        result = {
                            "items": items,
                            "metadata": {
                                "query": data.get("query"),
                                "answer": data.get("answer"),
                                "response_time": data.get("response_time"),
                                "result_count": len(items)
                            }
                        }
                        
                        # Cache result
                        self.cache[cache_key] = (datetime.now().timestamp(), result)
                        
                        logging.info(f"✅ Tavily found {len(items)} results")
                        return result
                    
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ Tavily API error {response.status}: {error_text}")
                        return {"items": [], "metadata": {"error": error_text}}
        
        except Exception as e:
            logging.error(f"❌ Tavily search error: {e}")
            return {"items": [], "metadata": {"error": str(e)}}
    
    async def extract(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract content from specific URLs
        """
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/extract"
            
            payload = {
                "api_key": self.api_key,
                "urls": urls
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "results": data.get("results", []),
                            "failed_results": data.get("failed_results", [])
                        }
                    else:
                        return {"results": [], "failed_results": urls}
        
        except Exception as e:
            logging.error(f"❌ Tavily extract error: {e}")
            return {"results": [], "failed_results": urls}
    
    async def _rate_limit(self):
        """Rate limiting to avoid API throttling"""
        now = datetime.now().timestamp()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < 60]
        
        # Check if we've exceeded rate limit
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logging.warning(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)