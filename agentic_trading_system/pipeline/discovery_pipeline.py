import sys
import os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Now imports will work
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.constants import Source, SearchType, ScoringWeights, EntityExtraction
from agentic_trading_system.config.config__entity import DiscoveryConfig
from agentic_trading_system.config.artifact_enity import (
    DiscoveryArtifact,
    SourceResult,
    SearchResultItem,
    EnrichedItemArtifact,
    EntityExtractionArtifact
)

# Import all clients
from agentic_trading_system.discovery.tavily_client import TavilyClient
from agentic_trading_system.discovery.news_api_client import NewsAPIClient
from agentic_trading_system.discovery.social_media_client import SocialMediaClient
from agentic_trading_system.discovery.sec_filings_client import SECFilingsClient
from agentic_trading_system.discovery.options_flow_client import OptionsFlowClient
from agentic_trading_system.discovery.macro_data_client import MacroDataClient
from agentic_trading_system.discovery.entity_extractor.nlp_extractor import NLPExtractor
from agentic_trading_system.discovery.entity_extractor.regex_extractor import RegexExtractor
from agentic_trading_system.discovery.data_enricher import DataEnricher

from agentic_trading_system.config.loader import get_discovery_config as get_config_object
    
    

class DiscoveryPipeline:
    """
    Orchestrates the discovery pipeline.
    Runs all components in parallel and aggregates results.
    """
    
    def __init__(self, config: DiscoveryConfig):
        """
        Initialize the discovery pipeline.
        
        Args:
            config: Discovery configuration
        """
        self.config = config
        
        # Initialize clients (lazy loading)
        self._clients: Dict[str, Any] = {}
        self._client_errors: Dict[str, str] = {}
        
        # Initialize extractors
        self.nlp_extractor = None
        self.regex_extractor = None
        self.enricher = None
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=config.cache_ttl_minutes)
        
        # Track initialization status
        self._initialized = False
        
        logging.info(f"✅ DiscoveryPipeline initialized")
    
    def _ensure_initialized(self):
        """Ensure all components are initialized before use."""
        if self._initialized:
            return
        
        # Initialize clients
        self._init_clients()
        
        # Initialize extractors
        self._init_extractors()
        
        # Initialize enricher
        self._init_enricher()
        
        self._initialized = True
    
    def _init_clients(self) -> None:
        """Initialize all data source clients."""
        logging.info("Initializing data source clients...")
        
        # Tavily
        if self.config.tavily.enabled and self.config.tavily.api_key:
            try:
                self._clients[Source.TAVILY] = TavilyClient(self.config.tavily)
                logging.info(f"  ✅ {Source.TAVILY} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.TAVILY}: {e}")
                self._client_errors[Source.TAVILY] = str(e)
        elif self.config.tavily.enabled:
            logging.warning(f"  ⚠️ {Source.TAVILY} disabled: API key missing")
        
        # News API
        if self.config.news.enabled:
            try:
                self._clients[Source.NEWS] = NewsAPIClient(self.config.news)
                logging.info(f"  ✅ {Source.NEWS} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.NEWS}: {e}")
                self._client_errors[Source.NEWS] = str(e)
        
        # Social Media
        if self.config.social.enabled:
            try:
                self._clients[Source.SOCIAL] = SocialMediaClient(self.config.social)
                logging.info(f"  ✅ {Source.SOCIAL} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.SOCIAL}: {e}")
                self._client_errors[Source.SOCIAL] = str(e)
        
        # SEC Filings
        if self.config.sec.enabled:
            try:
                self._clients[Source.SEC] = SECFilingsClient(self.config.sec)
                logging.info(f"  ✅ {Source.SEC} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.SEC}: {e}")
                self._client_errors[Source.SEC] = str(e)
        
        # Options Flow
        if self.config.options.enabled and self.config.options.fmp_key:
            try:
                self._clients[Source.OPTIONS] = OptionsFlowClient(self.config.options)
                logging.info(f"  ✅ {Source.OPTIONS} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.OPTIONS}: {e}")
                self._client_errors[Source.OPTIONS] = str(e)
        elif self.config.options.enabled:
            logging.warning(f"  ⚠️ {Source.OPTIONS} disabled: FMP API key missing")
        
        # Macro Data
        if self.config.macro.enabled and self.config.macro.fred_api_key:
            try:
                self._clients[Source.MACRO] = MacroDataClient(self.config.macro)
                logging.info(f"  ✅ {Source.MACRO} client initialized")
            except Exception as e:
                logging.error(f"  ❌ Failed to initialize {Source.MACRO}: {e}")
                self._client_errors[Source.MACRO] = str(e)
        elif self.config.macro.enabled:
            logging.warning(f"  ⚠️ {Source.MACRO} disabled: FRED API key missing")
        
        logging.info(f"Initialized {len(self._clients)}/{len(self._get_available_sources())} sources")
    
    def _init_extractors(self) -> None:
        """Initialize entity extractors."""
        logging.info("Initializing entity extractors...")
        
        # NLP extractor config
        nlp_config = {
            "spacy_model": self.config.nlp.spacy_model,
            "entity_types": self.config.nlp.entity_types
        }
        
        # Regex extractor config
        regex_config = {
            "ticker_pattern": self.config.regex.ticker_pattern,
            "exclude_tickers": self.config.regex.exclude_tickers
        }
        
        try:
            self.nlp_extractor = NLPExtractor(nlp_config)
            logging.info(f"  ✅ NLP extractor initialized")
        except Exception as e:
            logging.warning(f"  ⚠️ NLP extractor failed: {e}")
            self.nlp_extractor = None
        
        try:
            self.regex_extractor = RegexExtractor(regex_config)
            logging.info(f"  ✅ Regex extractor initialized")
        except Exception as e:
            logging.warning(f"  ⚠️ Regex extractor failed: {e}")
            self.regex_extractor = None
    
    def _init_enricher(self) -> None:
        """Initialize data enricher."""
        logging.info("Initializing data enricher...")
        
        try:
            self.enricher = DataEnricher(self.config.enricher)
            # Set extractor reference if available
            if self.regex_extractor:
                self.enricher.set_extractor(self.regex_extractor)
            logging.info(f"  ✅ Data enricher initialized")
        except Exception as e:
            logging.error(f"  ❌ Failed to initialize enricher: {e}")
            self.enricher = None
    
    def _get_available_sources(self) -> List[str]:
        """Get list of all available sources (regardless of initialization status)."""
        sources = []
        if self.config.tavily.enabled:
            sources.append(Source.TAVILY)
        if self.config.news.enabled:
            sources.append(Source.NEWS)
        if self.config.social.enabled:
            sources.append(Source.SOCIAL)
        if self.config.sec.enabled:
            sources.append(Source.SEC)
        if self.config.options.enabled:
            sources.append(Source.OPTIONS)
        if self.config.macro.enabled:
            sources.append(Source.MACRO)
        return sources
    
    async def run(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None
    ) -> DiscoveryArtifact:
        """
        Run the complete discovery pipeline.
        
        Args:
            query: Search query
            options: Additional options (search_type, max_results, etc.)
            
        Returns:
            DiscoveryArtifact with all results
        """
        # Ensure all components are initialized
        self._ensure_initialized()
        
        start_time = time.time()
        options = options or {}
        
        logging.info(f"🚀 Starting discovery pipeline for: '{query}'")
        
        # Check cache
        cache_key = f"discovery_{query}_{hash(str(options))}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached discovery results for '{query}'")
                return cached_result
        
        # Determine which sources to query - RUN ALL AVAILABLE SOURCES
        sources_to_query = self._get_sources_to_query(options)
        
        # Query all sources in parallel
        source_results = await self._query_sources_parallel(query, sources_to_query, options)
        
        # Collect all items
        all_items = []
        for result in source_results.values():
            all_items.extend(result.items)
        
        # Deduplicate items
        unique_items = self._deduplicate_items(all_items)
        
        # Rank items by relevance
        ranked_items = self._rank_items(unique_items, query)
        
        # Extract entities from top items
        entities = await self._extract_entities(ranked_items[:20])
        
        # Enrich top items
        enriched_items = await self._enrich_items(ranked_items[:10])
        
        # Build final artifact
        response_time_ms = (time.time() - start_time) * 1000
        
        artifact = DiscoveryArtifact(
            query=query,
            total_items=len(all_items),
            unique_items=len(unique_items),
            source_results=source_results,
            items=enriched_items,
            entities=entities,
            response_time_ms=response_time_ms,
            sources_queried=list(source_results.keys()),
            sources_succeeded=[s for s, r in source_results.items() if r.status == "success"],
            sources_failed=[s for s, r in source_results.items() if r.status == "error"],
            options_used=options,
            timestamp=datetime.now(),
            config_version=self.config.config_version,
            metadata={
                "search_type": options.get("search_type", SearchType.GENERAL),
                "max_results": options.get("max_results", 50)
            }
        )
        
        # Cache result
        self._cache[cache_key] = (datetime.now(), artifact)
        
        logging.info(
            f"✅ Discovery pipeline complete: {artifact.unique_items} unique items "
            f"from {len(artifact.sources_succeeded)} sources in {artifact.response_time_ms:.0f}ms"
        )
        
        return artifact
    
    async def _query_sources_parallel(
        self,
        query: str,
        sources: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, SourceResult]:
        """
        Query multiple sources in parallel.
        
        Args:
            query: Search query
            sources: List of source names to query
            options: Search options
            
        Returns:
            Dictionary mapping source name to SourceResult
        """
        async def query_one(source_name: str) -> tuple:
            """Query a single source and return result."""
            start_time = time.time()
            client = self._clients.get(source_name)
            
            if not client:
                return source_name, SourceResult(
                    source=source_name,
                    status="error",
                    error=f"Client not available: {self._client_errors.get(source_name, 'Unknown error')}"
                )
            
            try:
                # Call the client's search method
                result = await client.search(query, options)
                
                # Convert to SearchResultItems
                items = []
                for item in result.get("items", []):
                    # Parse published_at
                    published_at = None
                    pub_str = item.get("published_at")
                    if pub_str:
                        try:
                            published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        except:
                            pass
                    
                    items.append(SearchResultItem(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        url=item.get("url", ""),
                        published_at=published_at,
                        score=item.get("score", 0.0),
                        source=source_name,
                        content_type=item.get("type", "general"),
                        sentiment=item.get("sentiment"),
                        sentiment_score=item.get("sentiment_score"),
                        detected_tickers=item.get("detected_tickers", []),
                        detected_companies=item.get("detected_companies", []),
                        metadata=item.get("metadata", {})
                    ))
                
                response_time_ms = (time.time() - start_time) * 1000
                
                logging.info(f"  ✅ {source_name}: {len(items)} items in {response_time_ms:.0f}ms")
                
                return source_name, SourceResult(
                    source=source_name,
                    items=items,
                    status="success",
                    response_time_ms=response_time_ms,
                    metadata=result.get("metadata", {})
                )
                
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                logging.error(f"  ❌ {source_name} error: {e}")
                return source_name, SourceResult(
                    source=source_name,
                    status="error",
                    error=str(e),
                    response_time_ms=response_time_ms
                )
        
        # Filter to only initialized sources
        available_sources = [s for s in sources if s in self._clients]
        
        if not available_sources:
            logging.warning("No available sources to query")
            return {}
        
        # Run all queries in parallel
        tasks = [query_one(source) for source in available_sources]
        results = await asyncio.gather(*tasks)
        
        # Build result dictionary
        source_results = {}
        for source_name, result in results:
            source_results[source_name] = result
        
        return source_results
    
    def _get_sources_to_query(self, options: Dict[str, Any]) -> List[str]:
        """
        Determine which sources to query.
        By default, queries ALL available sources.
        
        Args:
            options: Search options
            
        Returns:
            List of source names
        """
        # If specific sources requested, use those
        if "sources" in options:
            requested = options["sources"]
            return [s for s in requested if s in self._clients]
        
        # If search type specified, use that mapping
        search_type = options.get("search_type", SearchType.GENERAL)
        
        search_type_mapping = {
            SearchType.GENERAL: list(self._clients.keys()),  # ALL SOURCES
            SearchType.NEWS: [s for s in self._clients.keys() if s in [Source.NEWS, Source.TAVILY]],
            SearchType.SOCIAL: [s for s in self._clients.keys() if s in [Source.SOCIAL, Source.TAVILY]],
            SearchType.FUNDAMENTAL: [s for s in self._clients.keys() if s in [Source.SEC, Source.NEWS, Source.TAVILY]],
            SearchType.TECHNICAL: [s for s in self._clients.keys() if s in [Source.OPTIONS, Source.NEWS, Source.TAVILY]],
            SearchType.MACRO: [s for s in self._clients.keys() if s in [Source.MACRO, Source.NEWS, Source.TAVILY]]
        }
        
        # Default to ALL sources if mapping doesn't exist
        return search_type_mapping.get(search_type, list(self._clients.keys()))
    
    def _deduplicate_items(self, items: List[SearchResultItem]) -> List[SearchResultItem]:
        """
        Remove duplicate items based on title and URL.
        
        Args:
            items: List of search result items
            
        Returns:
            Deduplicated list
        """
        unique_items = []
        seen_titles = set()
        seen_urls = set()
        
        for item in items:
            title = item.title.lower().strip() if item.title else ""
            url = item.url.lower().strip() if item.url else ""
            
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
    
    def _rank_items(
        self,
        items: List[SearchResultItem],
        query: str
    ) -> List[SearchResultItem]:
        """
        Rank items by relevance, recency, and source authority.
        
        Args:
            items: List of search result items
            query: Original query
            
        Returns:
            Ranked list
        """
        query_terms = set(query.lower().split())
        now_utc = datetime.now(timezone.utc)
        
        for item in items:
            score = 0.0
            
            # Relevance score (term matching)
            title = item.title.lower() if item.title else ""
            content = item.content.lower()[:500] if item.content else ""
            
            # Title matches (highest weight)
            title_matches = sum(1 for term in query_terms if term in title)
            score += title_matches * ScoringWeights.TITLE_MATCH_MULTIPLIER
            
            # Content matches
            content_matches = sum(1 for term in query_terms if term in content)
            score += content_matches * ScoringWeights.CONTENT_MATCH_MULTIPLIER
            
            # Source authority bonus
            authority_bonus = self._get_authority_bonus(item.source)
            score += authority_bonus
            
            # Recency bonus
            if item.published_at:
                try:
                    pub_date = item.published_at
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    age_hours = (now_utc - pub_date).total_seconds() / 3600
                    if age_hours < 24:
                        score += ScoringWeights.RECENT_24H_BONUS
                    elif age_hours < 72:
                        score += ScoringWeights.RECENT_72H_BONUS
                except Exception:
                    pass
            
            # Calculate relevance score (normalized to 0-1)
            max_possible_score = 100.0
            item.relevance_score = min(
                EntityExtraction.MAX_RELEVANCE,
                max(EntityExtraction.MIN_RELEVANCE, score / max_possible_score)
            ) if score > 0 else EntityExtraction.DEFAULT_RELEVANCE
            item.score = score
        
        # Sort by score
        items.sort(key=lambda x: x.score, reverse=True)
        
        return items
    
    def _get_authority_bonus(self, source: str) -> float:
        """
        Get authority bonus for a source.
        
        Args:
            source: Source name
            
        Returns:
            Authority bonus score
        """
        source_lower = source.lower()
        
        for high in EntityExtraction.HIGH_AUTHORITY_SOURCES:
            if high in source_lower:
                return ScoringWeights.HIGH_AUTHORITY_BONUS
        
        for medium in EntityExtraction.MEDIUM_AUTHORITY_SOURCES:
            if medium in source_lower:
                return ScoringWeights.MEDIUM_AUTHORITY_BONUS
        
        for low in EntityExtraction.LOW_AUTHORITY_SOURCES:
            if low in source_lower:
                return ScoringWeights.LOW_AUTHORITY_BONUS
        
        return 5.0  # Default bonus
    
    async def _extract_entities(
        self,
        items: List[SearchResultItem]
    ) -> EntityExtractionArtifact:
        """
        Extract entities from items using both NLP and regex.
        
        Args:
            items: List of search result items
            
        Returns:
            EntityExtractionArtifact with extracted entities
        """
        # Combine all text
        all_text = " ".join([
            f"{item.title or ''} {item.content or ''}"
            for item in items
            if (item.title or item.content)
        ])
        
        if not all_text:
            return EntityExtractionArtifact()
        
        # Run both extractors in parallel if available
        tasks = []
        if self.regex_extractor:
            tasks.append(self.regex_extractor.extract(all_text))
        else:
            tasks.append(asyncio.sleep(0, result={}))
        
        if self.nlp_extractor:
            tasks.append(self.nlp_extractor.extract(all_text))
        else:
            tasks.append(asyncio.sleep(0, result={}))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        regex_result = results[0] if not isinstance(results[0], Exception) else {}
        nlp_result = results[1] if not isinstance(results[1], Exception) else {}
        
        # Merge results
        artifact = EntityExtractionArtifact(
            tickers=list(set(regex_result.get("tickers", []) + nlp_result.get("tickers", []))),
            companies=list(set(regex_result.get("companies", []) + nlp_result.get("companies", []))),
            people=list(set(regex_result.get("people", []) + nlp_result.get("people", []))),
            organizations=list(set(regex_result.get("organizations", []) + nlp_result.get("organizations", []))),
            locations=list(set(regex_result.get("locations", []) + nlp_result.get("locations", []))),
            dates=list(set(regex_result.get("dates", []) + nlp_result.get("dates", []))),
            currencies=list(set(regex_result.get("currencies", []) + nlp_result.get("currencies", []))),
            industries=list(set(regex_result.get("industries", []) + nlp_result.get("industries", []))),
            percentages=regex_result.get("percentages", []),
            market_indicators=regex_result.get("market_indicators", []),
            stock_exchanges=regex_result.get("stock_exchanges", []),
            financial_terms=regex_result.get("financial_terms", []),
            text_length=len(all_text),
            extraction_methods=[
                m for m in ["regex", "nlp"] 
                if (m == "regex" and self.regex_extractor) or (m == "nlp" and self.nlp_extractor)
            ],
            timestamp=datetime.now()
        )
        
        return artifact
    
    async def _enrich_items(
        self,
        items: List[SearchResultItem]
    ) -> List[EnrichedItemArtifact]:
        """
        Enrich items with additional context.
        
        Args:
            items: List of search result items
            
        Returns:
            List of enriched items
        """
        if not self.enricher or not items:
            # Return basic items if enricher not available
            return [
                EnrichedItemArtifact(
                    title=item.title,
                    content=item.content,
                    url=item.url,
                    published_at=item.published_at,
                    score=item.score,
                    relevance_score=item.relevance_score,
                    source=item.source,
                    content_type=item.content_type,
                    detected_tickers=item.detected_tickers,
                    detected_companies=item.detected_companies,
                    sentiment=item.sentiment,
                    sentiment_score=item.sentiment_score,
                    metadata=item.metadata
                )
                for item in items
            ]
        
        # Convert to dict for enricher
        item_dicts = []
        for item in items:
            item_dict = {
                "title": item.title,
                "content": item.content,
                "url": item.url,
                "published_at": item.published_at.isoformat() if item.published_at else None,
                "score": item.score,
                "source": item.source,
                "type": item.content_type,
                "sentiment": item.sentiment,
                "sentiment_score": item.sentiment_score,
                "detected_tickers": item.detected_tickers,
                "detected_companies": item.detected_companies,
                "metadata": item.metadata
            }
            item_dicts.append(item_dict)
        
        # Enrich in parallel
        try:
            enriched_dicts = await self.enricher.enrich_batch(item_dicts)
        except Exception as e:
            logging.error(f"Enrichment failed: {e}")
            enriched_dicts = item_dicts
        
        # Convert back to artifacts
        enriched_items = []
        for item_dict in enriched_dicts:
            # Parse published_at
            published_at = None
            pub_str = item_dict.get("published_at")
            if pub_str and isinstance(pub_str, str):
                try:
                    published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except:
                    pass
            elif isinstance(pub_str, datetime):
                published_at = pub_str
            
            enriched_items.append(EnrichedItemArtifact(
                title=item_dict.get("title", ""),
                content=item_dict.get("content", ""),
                url=item_dict.get("url", ""),
                published_at=published_at,
                score=item_dict.get("score", 0.0),
                relevance_score=item_dict.get("relevance_score", 0.5),
                authority_score=item_dict.get("source_authority", 0.5),
                source=item_dict.get("source", ""),
                content_type=item_dict.get("content_type", "general"),
                detected_tickers=item_dict.get("detected_tickers", []),
                detected_companies=item_dict.get("detected_companies", []),
                sentiment=item_dict.get("sentiment"),
                sentiment_score=item_dict.get("sentiment_score"),
                company_info=item_dict.get("company_info"),
                content_length=item_dict.get("content_length", len(item_dict.get("content", ""))),
                word_count=item_dict.get("word_count", len(item_dict.get("content", "").split())),
                metadata=item_dict.get("metadata", {})
            ))
        
        return enriched_items
    
    async def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        if self.enricher and hasattr(self.enricher, 'clear_cache'):
            await self.enricher.clear_cache()
        
        # Clear client caches if they have the method
        for client in self._clients.values():
            if hasattr(client, 'clear_cache'):
                try:
                    await client.clear_cache()
                except Exception as e:
                    logging.debug(f"Error clearing {client} cache: {e}")
        
        logging.info("🧹 Discovery pipeline cache cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "initialized": self._initialized,
            "sources_initialized": list(self._clients.keys()),
            "sources_failed": list(self._client_errors.keys()),
            "source_errors": self._client_errors,
            "cache_size": len(self._cache),
            "nlp_extractor": self.nlp_extractor is not None,
            "regex_extractor": self.regex_extractor is not None,
            "enricher": self.enricher is not None
        }


async def _run_single_ticker(ticker: str, output_dir: Path) -> DiscoveryArtifact:
    """Run discovery for a single ticker."""
    print(f"\n🔍 Running discovery for: {ticker}")
    

    # This returns a DiscoveryConfig object directly
    config = get_config_object()
    
    # Create pipeline
    pipeline = DiscoveryPipeline(config)
    
    # Run discovery
    start_time = time.time()
    
    artifact = await pipeline.run(
        query=ticker,
        options={
            "search_type": "general",
            "max_results": 50
        }
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    output_path = artifact.save(output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" RESULTS - {ticker}")
    print(f"{'='*70}")
    
    print(f"\n✅ Complete in {elapsed:.2f}s")
    print(f"📈 Run ID: {artifact.run_id}")
    print(f"🔧 Config Version: {artifact.config_version}")
    print(f"📄 Unique items: {artifact.unique_items} from {len(artifact.sources_succeeded)} sources")
    
    if artifact.entities.tickers:
        print(f"\n🏷️  Tickers detected: {', '.join(artifact.entities.tickers[:10])}")
    
    if artifact.entities.companies:
        print(f"🏢 Companies detected: {', '.join(artifact.entities.companies[:5])}")
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return artifact


async def _run_multiple_tickers(tickers: List[str], output_dir: Path) -> List[DiscoveryArtifact]:
    """Run discovery for multiple tickers."""
    print(f"\n{'='*70}")
    print(f" RUNNING DISCOVERY FOR {len(tickers)} TICKERS")
    print(f"{'='*70}")
    
    artifacts = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        print("-" * 50)
        
        try:
            artifact = await _run_single_ticker(ticker, output_dir)
            artifacts.append(artifact)
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
        
        # Small delay between tickers to avoid rate limiting
        if i < len(tickers):
            await asyncio.sleep(2)
    
    return artifacts


def main():
    """
    Simple main function - runs with no input arguments.
    
    To change which tickers to run, modify the TICKERS list below.
    - If single ticker: runs just that one
    - If multiple tickers: runs all in sequence
    
    Default: ['AAPL', 'MSFT', 'TSLA'] - runs all three
    """
    
    # ============================================================
    # CONFIGURE YOUR TICKERS HERE
    # ============================================================
    # Single ticker example:
    # TICKERS = ['AAPL']
    
    # Multiple tickers example:
    TICKERS = ['AAPL', 'MSFT', 'TSLA']
    

    print("\n" + "=" * 70)
    print(" DISCOVERY PIPELINE")
    print("=" * 70)
    print(f"\n📋 Tickers to process: {', '.join(TICKERS)}")
    
    # Setup output directory
    output_dir = Path("discovery_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run discovery
    if len(TICKERS) == 1:
        # Single ticker - run directly
        artifact = asyncio.run(_run_single_ticker(TICKERS[0], output_dir))
        
        # Print final summary
        print(f"\n{'='*70}")
        print(" COMPLETE")
        print(f"{'='*70}")
        print(f"\n✅ Successfully processed: {TICKERS[0]}")
        print(f"📁 Output directory: {output_dir}")
        
    else:
        # Multiple tickers - run all
        artifacts = asyncio.run(_run_multiple_tickers(TICKERS, output_dir))
        
        # Print final summary
        print(f"\n{'='*70}")
        print(" COMPLETE - SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n✅ Successfully processed: {len(artifacts)}/{len(TICKERS)} tickers")
        
        if artifacts:
            print(f"\n📊 Summary Table:")
            print(f"{'Ticker':<10} {'Items':<8} {'Sources':<8} {'Tickers Found':<15} {'Time':<8}")
            print("-" * 60)
            
            for artifact in artifacts:
                ticker = artifact.query
                items = artifact.unique_items
                sources = len(artifact.sources_succeeded)
                tickers_found = len(artifact.entities.tickers)
                time_ms = artifact.response_time_ms
                print(f"{ticker:<10} {items:<8} {sources:<8} {tickers_found:<15} {time_ms/1000:<8.1f}s")
        
        print(f"\n📁 All results saved to: {output_dir}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    """
    Direct execution - just run: python discovery_pipeline.py
    
    To change tickers, modify the TICKERS list in main() above.
    """
    main()