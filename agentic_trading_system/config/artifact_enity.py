# =============================================================================
# discovery/artifact/artifact_entity.py
# =============================================================================
"""
Artifact entities for the discovery package.
Typed output classes for each component and final pipeline output.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


# =============================================================================
# Base Artifacts
# =============================================================================
@dataclass
class BaseArtifact:
    """Base artifact with common fields."""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResultItem(BaseArtifact):
    """Individual search result item."""
    title: str = ""
    content: str = ""
    url: str = ""
    published_at: Optional[datetime] = None
    score: float = 0.0
    relevance_score: float = 0.0
    authority_score: float = 0.0
    content_type: str = ""
    detected_tickers: List[str] = field(default_factory=list)
    detected_companies: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "score": self.score,
            "relevance_score": self.relevance_score,
            "authority_score": self.authority_score,
            "content_type": self.content_type,
            "detected_tickers": self.detected_tickers,
            "detected_companies": self.detected_companies,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SourceResult(BaseArtifact):
    """Result from a single data source."""
    source: str
    items: List[SearchResultItem] = field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None
    count: int = 0
    response_time_ms: float = 0.0
    
    def __post_init__(self):
        self.count = len(self.items)
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "status": self.status,
            "error": self.error,
            "count": self.count,
            "response_time_ms": self.response_time_ms,
            "items": [item.to_dict() for item in self.items],
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# Entity Extraction Artifact
# =============================================================================
@dataclass
class EntityExtractionArtifact(BaseArtifact):
    """Result from entity extraction."""
    tickers: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    percentages: List[str] = field(default_factory=list)
    market_indicators: List[str] = field(default_factory=list)
    stock_exchanges: List[str] = field(default_factory=list)
    financial_terms: List[str] = field(default_factory=list)
    
    # Metadata about extraction
    text_length: int = 0
    extraction_methods: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tickers": self.tickers,
            "companies": self.companies,
            "people": self.people,
            "organizations": self.organizations,
            "locations": self.locations,
            "dates": self.dates,
            "currencies": self.currencies,
            "industries": self.industries,
            "percentages": self.percentages,
            "market_indicators": self.market_indicators,
            "stock_exchanges": self.stock_exchanges,
            "financial_terms": self.financial_terms,
            "text_length": self.text_length,
            "extraction_methods": self.extraction_methods,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def merge(self, other: "EntityExtractionArtifact") -> "EntityExtractionArtifact":
        """Merge with another extraction result."""
        return EntityExtractionArtifact(
            tickers=list(set(self.tickers + other.tickers)),
            companies=list(set(self.companies + other.companies)),
            people=list(set(self.people + other.people)),
            organizations=list(set(self.organizations + other.organizations)),
            locations=list(set(self.locations + other.locations)),
            dates=list(set(self.dates + other.dates)),
            currencies=list(set(self.currencies + other.currencies)),
            industries=list(set(self.industries + other.industries)),
            percentages=list(set(self.percentages + other.percentages)),
            market_indicators=list(set(self.market_indicators + other.market_indicators)),
            stock_exchanges=list(set(self.stock_exchanges + other.stock_exchanges)),
            financial_terms=list(set(self.financial_terms + other.financial_terms)),
            text_length=self.text_length + other.text_length,
            extraction_methods=list(set(self.extraction_methods + other.extraction_methods)),
            timestamp=datetime.now(),
            metadata={**self.metadata, **other.metadata}
        )


# =============================================================================
# Enriched Item Artifact
# =============================================================================
@dataclass
class EnrichedItemArtifact(SearchResultItem):
    """Enriched search result with additional context."""
    company_info: Optional[Dict[str, Any]] = None
    price_data: Optional[Dict[str, Any]] = None
    content_length: int = 0
    word_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result.update({
            "company_info": self.company_info,
            "price_data": self.price_data,
            "content_length": self.content_length,
            "word_count": self.word_count
        })
        return result


# =============================================================================
# Main Discovery Artifact
# =============================================================================
@dataclass
class DiscoveryArtifact(BaseArtifact):
    """Complete discovery pipeline output."""
    query: str = ""
    total_items: int = 0
    unique_items: int = 0
    
    # Source results
    source_results: Dict[str, SourceResult] = field(default_factory=dict)
    
    # Enriched items
    items: List[EnrichedItemArtifact] = field(default_factory=list)
    
    # Entity extraction
    entities: EntityExtractionArtifact = field(default_factory=EntityExtractionArtifact)
    
    # Aggregated metrics
    response_time_ms: float = 0.0
    sources_queried: List[str] = field(default_factory=list)
    sources_succeeded: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)
    
    # Options used
    options_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_items": self.total_items,
            "unique_items": self.unique_items,
            "response_time_ms": self.response_time_ms,
            "sources_queried": self.sources_queried,
            "sources_succeeded": self.sources_succeeded,
            "sources_failed": self.sources_failed,
            "source_results": {
                name: result.to_dict() 
                for name, result in self.source_results.items()
            },
            "items": [item.to_dict() for item in self.items],
            "entities": self.entities.to_dict(),
            "options_used": self.options_used,
            "metadata": self.metadata
        }
    
    def get_all_tickers(self) -> List[str]:
        """Get all unique tickers from all sources."""
        tickers = set()
        for item in self.items:
            tickers.update(item.detected_tickers)
        tickers.update(self.entities.tickers)
        return sorted(tickers)
    
    def get_items_by_type(self, content_type: str) -> List[EnrichedItemArtifact]:
        """Filter items by content type."""
        return [item for item in self.items if item.content_type == content_type]
    
    def get_items_by_source(self, source: str) -> List[EnrichedItemArtifact]:
        """Filter items by source."""
        return [item for item in self.items if item.source == source]