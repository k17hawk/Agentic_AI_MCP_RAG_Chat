from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path


@dataclass
class BaseArtifact:
    """Base artifact with common fields."""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


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
    run_id: str = ""
    query: str = ""
    total_items: int = 0
    unique_items: int = 0
    config_version: str = ""
    
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
    
    def __post_init__(self):
        """Generate run_id if not provided."""
        if not self.run_id:
            self.run_id = self.timestamp.strftime("%Y%m%d_%H%M%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_items": self.total_items,
            "unique_items": self.unique_items,
            "config_version": self.config_version,
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
    
    def save(self, output_dir: Path) -> Path:
        """
        Save artifact to disk with timestamp-based directory structure.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Path to saved artifact directory
        """
        # Create run directory
        run_dir = output_dir / f"{self.run_id}_{self.query}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main artifact
        artifact_file = run_dir / "artifact.json"
        with open(artifact_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        # Save metadata separately
        metadata = {
            "run_id": self.run_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "config_version": self.config_version,
            "sources_succeeded": self.sources_succeeded,
            "sources_failed": self.sources_failed,
            "total_items": self.unique_items,
            "response_time_ms": self.response_time_ms
        }
        
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save entities separately for easy access
        entities_file = run_dir / "entities.json"
        with open(entities_file, "w") as f:
            json.dump(self.entities.to_dict(), f, indent=2)
        
        # Save per-source raw data
        sources_dir = run_dir / "sources"
        sources_dir.mkdir(exist_ok=True)
        
        for source_name, source_result in self.source_results.items():
            source_file = sources_dir / f"{source_name}.json"
            with open(source_file, "w") as f:
                json.dump(source_result.to_dict(), f, indent=2)
        
        return run_dir
    
    @classmethod
    def load(cls, run_dir: Path) -> "DiscoveryArtifact":
        """
        Load artifact from disk.
        
        Args:
            run_dir: Directory containing saved artifact
            
        Returns:
            Loaded DiscoveryArtifact
        """
        artifact_file = run_dir / "artifact.json"
        
        if not artifact_file.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_file}")
        
        with open(artifact_file, "r") as f:
            data = json.load(f)
        
        artifact = cls(
            run_id=data.get("run_id", ""),
            query=data.get("query", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            total_items=data.get("total_items", 0),
            unique_items=data.get("unique_items", 0),
            config_version=data.get("config_version", ""),
            response_time_ms=data.get("response_time_ms", 0),
            sources_queried=data.get("sources_queried", []),
            sources_succeeded=data.get("sources_succeeded", []),
            sources_failed=data.get("sources_failed", [])
        )
        
        return artifact
    
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_items": self.unique_items,
            "sources_succeeded": len(self.sources_succeeded),
            "sources_failed": len(self.sources_failed),
            "response_time_ms": self.response_time_ms,
            "unique_tickers": len(self.get_all_tickers()),
            "unique_companies": len(set(self.entities.companies)),
            "content_types": list(set(item.content_type for item in self.items))
        }