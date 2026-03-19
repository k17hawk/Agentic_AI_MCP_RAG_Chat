"""
Search Aggregator - Coordinates all data sources for discovery
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agentic_trading_system.logger.logger import logger as  logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import all discovery clients
from agentic_trading_system.discovery.tavily_client import TavilyClient
from agentic_trading_system.discovery.news_api_client import NewsAPIClient
from agentic_trading_system.discovery.social_media_client import SocialMediaClient
from agentic_trading_system.discovery.sec_filings_client import SECFilingsClient
from agentic_trading_system.discovery.options_flow_client import OptionsFlowClient
from agentic_trading_system.discovery.macro_data_client import MacroDataClient
from agentic_trading_system.discovery.entity_extractor.nlp_extractor import NLPExtractor
from agentic_trading_system.discovery.entity_extractor.regex_extractor import RegexExtractor
from agentic_trading_system.discovery.data_enricher import DataEnricher

class SearchAggregator(BaseAgent):
    """
    Coordinates all data sources for discovery
    
    Responsibilities:
    - Aggregate data from multiple sources
    - Deduplicate and merge results
    - Rank by relevance and recency
    - Extract entities (tickers, companies)
    - Enrich data with additional context
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Aggregates data from multiple discovery sources",
            config=config
        )
        
        # Initialize clients
        self.tavily = TavilyClient(config.get("tavily_config", {}))
        self.news = NewsAPIClient(config.get("news_config", {}))
        self.social = SocialMediaClient(config.get("social_config", {}))
        self.sec = SECFilingsClient(config.get("sec_config", {}))
        self.options = OptionsFlowClient(config.get("options_config", {}))
        self.macro = MacroDataClient(config.get("macro_config", {}))
        
        # Initialize extractors
        self.nlp_extractor = NLPExtractor(config.get("nlp_config", {}))
        self.regex_extractor = RegexExtractor(config.get("regex_config", {}))
        
        # Initialize enricher
        self.enricher = DataEnricher(config.get("enricher_config", {}))
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 5))
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(minutes=config.get("cache_ttl_minutes", 15))
        
        # Source weights for ranking
        self.source_weights = config.get("source_weights", {
            "tavily": 0.25,
            "news": 0.20,
            "social": 0.15,
            "sec": 0.15,
            "options": 0.15,
            "macro": 0.10
        })
        
        # Map source names to their client search methods
        self._source_clients = {
            "tavily": self.tavily,
            "news": self.news,
            "social": self.social,
            "sec": self.sec,
            "options": self.options,
            "macro": self.macro,
        }
        
        logging.info(f"✅ SearchAggregator initialized with {len(self.source_weights)} sources")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process discovery requests
        """
        if message.message_type == "discovery_request":
            query = message.content.get("query")
            options = message.content.get("options", {})
            
            results = await self.discover(query, options)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="discovery_result",
                content=results
            )
        
        elif message.message_type == "extract_entities":
            text = message.content.get("text")
            
            entities = await self.extract_entities(text)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="entities_result",
                content={"entities": entities}
            )
        
        return None
    
    async def discover(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Discover information from all sources
        """
        options = options or {}
        logging.info(f"🔍 Discovery request for: '{query}'")
        
        # Check cache
        cache_key = f"discovery_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached discovery results for '{query}'")
                return cached_result
        
        # Determine sources to query
        sources_to_query = self._get_sources_to_query(options)

        # FIX: Build tasks and source names together so indices never get out of sync
        task_pairs = [
            (source_name, self._source_clients[source_name].search(query, options))
            for source_name in sources_to_query
            if source_name in self._source_clients
        ]
        source_names = [name for name, _ in task_pairs]
        tasks = [task for _, task in task_pairs]

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_items = []
        source_stats = {}
        
        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logging.warning(f"⚠️ Error from {source_name}: {result}")
                source_stats[source_name] = {"status": "error", "count": 0}
            else:
                items = result.get("items", [])
                all_items.extend(items)
                source_stats[source_name] = {
                    "status": "success",
                    "count": len(items),
                    "metadata": result.get("metadata", {})
                }
        
        # Deduplicate items
        unique_items = self._deduplicate_items(all_items)
        
        # Rank items by relevance
        ranked_items = self._rank_items(unique_items, query)
        
        # Extract entities from all items
        all_text = " ".join([item.get("title", "") + " " + item.get("content", "") 
                            for item in ranked_items[:20]])
        entities = await self.extract_entities(all_text)
        
        # Enrich top items
        enriched_items = []
        for item in ranked_items[:10]:  # Enrich top 10
            enriched = await self.enricher.enrich(item)
            enriched_items.append(enriched)
        
        result = {
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": len(all_items),
            "unique_items": len(unique_items),
            "items": enriched_items,
            "entities": entities,
            "source_stats": source_stats,
            "sources_queried": source_names
        }
        
        # Cache result
        self.cache[cache_key] = (datetime.now(), result)
        
        logging.info(f"✅ Discovery complete: {len(unique_items)} unique items from {len(source_names)} sources")
        
        return result
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using both NLP and regex.
        Results from both extractors are merged here, then re-validated
        so that anything slipping through one extractor is caught by the other's
        validation logic before reaching the caller.
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

        # Regex extractor — fast, high-precision patterns
        regex_entities = await self.regex_extractor.extract(text)
        for key in regex_entities:
            if key in entities:
                entities[key].extend(regex_entities[key])

        # NLP extractor — spaCy-based, catches names regex misses
        nlp_entities = await self.nlp_extractor.extract(text)
        for key in nlp_entities:
            if key in entities:
                entities[key].extend(nlp_entities[key])

        # Deduplicate with per-type validation as the final safety net.
        # This catches anything that slipped through either extractor individually.
        ticker_fp = self.nlp_extractor.TICKER_FALSE_POSITIVES
        fragment_words = self.nlp_extractor._FRAGMENT_WORDS

        def _valid_ticker(t: str) -> bool:
            t = t.upper().strip()
            base = t.split('.')[0]
            if not (1 <= len(base) <= 5) or not base.isalpha():
                return False
            if t in ticker_fp or base in ticker_fp:
                return False
            if len(base) == 1:
                return base in {'A', 'C', 'F', 'G', 'H', 'J', 'M', 'R', 'T', 'V', 'Z'}
            return True

        def _valid_company(name: str) -> bool:
            s = name.strip()
            if len(s) < 3 or len(s) > 60:
                return False
            if not s[0].isupper():
                return False
            words = s.split()
            if len(words) > 5:
                return False
            word_set = {w.lower().rstrip('.,') for w in words}
            if word_set & fragment_words:
                return False
            return True

        for key in entities:
            seen: set = set()
            unique = []
            for item in entities[key]:
                item = item.strip()
                if not item or item in seen:
                    continue
                if key == "tickers" and not _valid_ticker(item):
                    continue
                if key in ("companies", "organizations") and not _valid_company(item):
                    continue
                seen.add(item)
                unique.append(item)
            entities[key] = unique[:10]

        return entities
    
    def _get_sources_to_query(self, options: Dict) -> List[str]:
        """
        Determine which sources to query based on options
        """
        all_sources = ["tavily", "news", "social", "sec", "options", "macro"]
        
        # If specific sources requested
        if "sources" in options:
            requested = options["sources"]
            return [s for s in all_sources if s in requested]
        
        # If search type specified
        search_type = options.get("search_type", "general")
        if search_type == "news":
            return ["news", "tavily"]
        elif search_type == "social":
            return ["social", "tavily"]
        elif search_type == "fundamental":
            return ["sec", "news", "tavily"]
        elif search_type == "technical":
            return ["options", "news", "tavily"]
        elif search_type == "macro":
            return ["macro", "news", "tavily"]
        
        # Default: use all sources
        return all_sources
    
    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate items based on title and content similarity
        """
        unique_items = []
        seen_titles = set()
        seen_urls = set()
        
        for item in items:
            title = item.get("title", "").lower().strip()
            url = item.get("url", "").lower().strip()
            
            # Check for duplicates
            if title and title in seen_titles:
                continue
            if url and url in seen_urls:
                continue
            
            # Add to seen sets
            if title:
                seen_titles.add(title)
            if url:
                seen_urls.add(url)
            
            unique_items.append(item)
        
        return unique_items
    
    def _rank_items(self, items: List[Dict], query: str) -> List[Dict]:
        """
        Rank items by relevance, recency, and source authority
        """
        query_terms = set(query.lower().split())

        # FIX: Use timezone-aware UTC now for correct comparison with API timestamps
        now_utc = datetime.now(timezone.utc)

        for item in items:
            score = 0
            
            # Relevance score (term matching)
            title = item.get("title", "").lower()
            content = item.get("content", "")[:500].lower()
            
            # Title matches (highest weight)
            title_matches = sum(1 for term in query_terms if term in title)
            score += title_matches * 10
            
            # Content matches
            content_matches = sum(1 for term in query_terms if term in content)
            score += content_matches * 3
            
            # Source authority
            source = item.get("source", "").lower()
            authority_scores = {
                "reuters": 10, "bloomberg": 10, "wsj": 10,
                "cnbc": 8, "yahoo": 6, "seeking_alpha": 7,
                "twitter": 3, "reddit": 2
            }
            for src, auth_score in authority_scores.items():
                if src in source:
                    score += auth_score
                    break
            
            # Recency bonus
            published = item.get("published_at")
            if published:
                try:
                    pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    # FIX: Ensure pub_date is timezone-aware before comparing
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    age_hours = (now_utc - pub_date).total_seconds() / 3600
                    if age_hours < 24:
                        score += 5
                    elif age_hours < 72:
                        score += 2
                except Exception:
                    pass
            
            item["_rank_score"] = score
        
        # Sort by rank score
        items.sort(key=lambda x: x.get("_rank_score", 0), reverse=True)
        
        return items