# run_discovery.py - WORKING SCRIPT
#!/usr/bin/env python3
"""
Complete working script to run the discovery pipeline
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import discovery modules
from agentic_trading_system.pipeline.discovery_pipeline import DiscoveryPipeline, SearchType, DiscoveryConfig
from agentic_trading_system.config.loader import get_discovery_config


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result_summary(result, title: str = "Results"):
    """Print a summary of discovery results."""
    print_section(title)
    
    print(f"\n📊 Query: {result.query}")
    print(f"📈 Found {result.unique_items} unique items (total: {result.total_items})")
    print(f"⏱️  Response time: {result.response_time_ms:.0f}ms")
    
    # Sources
    print(f"\n📡 Sources:")
    print(f"   Success: {', '.join(result.sources_succeeded) or 'None'}")
    if result.sources_failed:
        print(f"   Failed: {', '.join(result.sources_failed)}")
    
    # Entities
    if result.entities.tickers:
        print(f"\n📊 Tickers found: {', '.join(result.entities.tickers[:10])}")
    if result.entities.companies:
        print(f"🏢 Companies found: {', '.join(result.entities.companies[:5])}")
    if result.entities.people:
        print(f"👤 People found: {', '.join(result.entities.people[:5])}")
    
    # Top results
    if result.items:
        print(f"\n📰 Top Results:")
        for i, item in enumerate(result.items[:5], 1):
            print(f"\n   {i}. {item.title[:80]}")
            print(f"      Source: {item.source} | Relevance: {item.relevance_score:.2f}")
            if item.detected_tickers:
                print(f"      Tickers: {', '.join(item.detected_tickers)}")
            if item.content:
                content_preview = item.content[:150].replace('\n', ' ')
                print(f"      Preview: {content_preview}...")


async def run_basic_search():
    """Run a basic search."""
    print_section("BASIC SEARCH")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    query = input("\n🔍 Enter search query (default: 'NVDA stock news'): ").strip()
    if not query:
        query = "NVDA stock news"
    
    print(f"\n📡 Searching: {query}")
    
    try:
        result = await pipeline.run(
            query=query,
            options={
                "search_type": SearchType.GENERAL,
                "max_results": 15
            }
        )
        
        print_result_summary(result, "SEARCH RESULTS")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_news_search():
    """Run a news-specific search."""
    print_section("NEWS SEARCH")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    ticker = input("\n📊 Enter stock ticker (default: 'AAPL'): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
    
    print(f"\n📡 Searching news for: {ticker}")
    
    try:
        result = await pipeline.run(
            query=ticker,
            options={
                "search_type": SearchType.NEWS,
                "max_results": 15
            }
        )
        
        print_result_summary(result, f"NEWS RESULTS FOR {ticker}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_social_search():
    """Run a social media search."""
    print_section("SOCIAL MEDIA SEARCH")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    ticker = input("\n📊 Enter stock ticker (default: 'TSLA'): ").strip().upper()
    if not ticker:
        ticker = "TSLA"
    
    print(f"\n📡 Searching social media for: {ticker}")
    
    try:
        result = await pipeline.run(
            query=ticker,
            options={
                "search_type": SearchType.SOCIAL,
                "max_results": 15
            }
        )
        
        print_result_summary(result, f"SOCIAL MEDIA RESULTS FOR {ticker}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_macro_search():
    """Run a macroeconomic search."""
    print_section("MACROECONOMIC SEARCH")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    print("\n📈 Available macro indicators: GDP, inflation, unemployment, fed_funds")
    indicator = input("Enter indicator (default: 'GDP'): ").strip().lower()
    if not indicator:
        indicator = "GDP"
    
    print(f"\n📡 Searching macro data for: {indicator}")
    
    try:
        result = await pipeline.run(
            query=indicator,
            options={
                "search_type": SearchType.MACRO,
                "max_results": 10
            }
        )
        
        print_result_summary(result, f"MACRO DATA: {indicator.upper()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def run_comprehensive_search():
    """Run a comprehensive search across all sources."""
    print_section("COMPREHENSIVE SEARCH")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    query = input("\n🔍 Enter search query (default: 'AI stocks 2024'): ").strip()
    if not query:
        query = "AI stocks 2024"
    
    print(f"\n📡 Comprehensive search for: {query}")
    print("   (This may take a moment as it queries all sources in parallel)")
    
    try:
        result = await pipeline.run(
            query=query,
            options={
                "search_type": SearchType.GENERAL,
                "max_results": 20
            }
        )
        
        print_result_summary(result, "COMPREHENSIVE SEARCH RESULTS")
        
        # Additional statistics
        print(f"\n📊 Additional Stats:")
        print(f"   Items by type:")
        content_types = {}
        for item in result.items:
            ct = item.content_type
            content_types[ct] = content_types.get(ct, 0) + 1
        for ct, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      {ct}: {count}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def save_results_to_file():
    """Run search and save results to file."""
    print_section("SAVE RESULTS TO FILE")
    
    config = get_discovery_config()
    pipeline = DiscoveryPipeline(config)
    
    query = input("\n🔍 Enter search query (default: 'Microsoft outlook'): ").strip()
    if not query:
        query = "Microsoft outlook"
    
    print(f"\n📡 Searching: {query}")
    
    try:
        result = await pipeline.run(
            query=query,
            options={
                "search_type": SearchType.GENERAL,
                "max_results": 20
            }
        )
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"discovery_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {filename}")
        print(f"   File size: {os.path.getsize(filename):,} bytes")
        
        print_result_summary(result, "SEARCH SUMMARY")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_configuration():
    """Test configuration and API keys."""
    print_section("CONFIGURATION TEST")
    
    try:
        config = get_discovery_config()
        
        print("\n📋 Loaded Configuration:")
        print(f"   Tavily enabled: {config.tavily.enabled}")
        print(f"   Tavily API key: {'✓ Set' if config.tavily.api_key else '✗ Missing'}")
        print(f"   News API key: {'✓ Set' if config.news.news_api_key else '✗ Missing'}")
        print(f"   Alpha Vantage key: {'✓ Set' if config.news.alpha_vantage_key else '✗ Missing'}")
        print(f"   FMP key: {'✓ Set' if config.news.fmp_key else '✗ Missing'}")
        print(f"   FRED key: {'✓ Set' if config.macro.fred_api_key else '✗ Missing'}")
        print(f"   Max workers: {config.max_workers}")
        print(f"   Cache TTL: {config.cache_ttl_minutes} minutes")
        
        # Test pipeline initialization
        print("\n🔧 Testing pipeline initialization...")
        pipeline = DiscoveryPipeline(config)
        status = pipeline.get_status()
        
        print(f"\n📡 Pipeline Status:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Sources initialized: {status['sources_initialized'] or 'None'}")
        print(f"   NLP extractor: {'✓' if status['nlp_extractor'] else '✗'}")
        print(f"   Regex extractor: {'✓' if status['regex_extractor'] else '✗'}")
        print(f"   Enricher: {'✓' if status['enricher'] else '✗'}")
        
        if status['sources_failed']:
            print(f"\n⚠️ Failed sources: {status['sources_failed']}")
            for source, error in status['source_errors'].items():
                print(f"   {source}: {error[:100]}")
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main menu."""
    print("\n" + "=" * 70)
    print(" 🚀 DISCOVERY PIPELINE - Market Intelligence Gathering")
    print("=" * 70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    menu = {
        "1": ("Basic Search", run_basic_search),
        "2": ("News Search", run_news_search),
        "3": ("Social Media Search", run_social_search),
        "4": ("Macroeconomic Search", run_macro_search),
        "5": ("Comprehensive Search", run_comprehensive_search),
        "6": ("Save Results to File", save_results_to_file),
        "7": ("Test Configuration", test_configuration),
    }
    
    print("\n📋 Available Options:")
    for key, (name, _) in menu.items():
        print(f"   {key}. {name}")
    print("   q. Quit")
    
    while True:
        choice = input("\n👉 Select option (1-7/q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Goodbye!")
            break
        
        if choice in menu:
            print("\n" + "-" * 70)
            try:
                await menu[choice][1]()
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
            
            input("\n\nPress Enter to continue...")
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()