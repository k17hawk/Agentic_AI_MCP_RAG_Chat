#!/usr/bin/env python3
"""
Run Discovery module standalone
This runs ONLY the discovery module with minimal dependencies
"""
import asyncio
import sys
from pathlib import Path
from pprint import pprint
import argparse
import types
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
    
    # Debug: Show which keys are loaded (without revealing full keys)
    print("\n🔑 API Keys Status:")
    tavily_key = os.getenv('TAVILY_API_KEY')
    print(f"   TAVILY_API_KEY: {'✅ ' + tavily_key[:5] + '...' if tavily_key else '❌ Missing'}")
    
    news_key = os.getenv('NEWS_API_KEY')
    print(f"   NEWS_API_KEY: {'✅ ' + news_key[:5] + '...' if news_key else '❌ Missing'}")
    
    alpha_key = os.getenv('ALPHA_VANTAGE_KEY')
    print(f"   ALPHA_VANTAGE_KEY: {'✅ ' + alpha_key[:5] + '...' if alpha_key else '❌ Missing'}")
    
    fmp_key = os.getenv('FMP_KEY')
    print(f"   FMP_KEY: {'✅ ' + fmp_key[:5] + '...' if fmp_key else '❌ Missing'}")
    
    twitter_key = os.getenv('TWITTER_BEARER_TOKEN')
    print(f"   TWITTER_BEARER_TOKEN: {'✅ ' + twitter_key[:5] + '...' if twitter_key else '❌ Missing'}")
    
    fred_key = os.getenv('FRED_API_KEY')
    print(f"   FRED_API_KEY: {'✅ ' + fred_key[:5] + '...' if fred_key else '❌ Missing'}")
else:
    print("⚠️  No .env file found. Please create one with your API keys.")

# Add discovery folder to path
sys.path.insert(0, str(Path(__file__).parent))

# Simple logger for standalone mode
class SimpleLogger:
    def info(self, msg): print(f"📘 {msg}")
    def error(self, msg): print(f"❌ {msg}")
    def warning(self, msg): print(f"⚠️ {msg}")
    def debug(self, msg): print(f"🔍 {msg}")

# Override the logger module under every import path used across the codebase.
# discovery clients:    from agentic_trading_system.utils.logger import logger
# search_aggregator:    from agentic_trading_system.logger.logger import logger
# (the bare "utils.logger" key was the original bug — it never matched either path)
_simple_logger = SimpleLogger()

for _module_path in (
    "agentic_trading_system.utils.logger",
    "agentic_trading_system.logger.logger",
    "utils.logger",           # keep for safety
):
    _mod = types.ModuleType(_module_path)
    _mod.logger = _simple_logger
    sys.modules[_module_path] = _mod

# Import config - use the correct import path
try:
    from agentic_trading_system.config.config_loader_discovery import config
    print(f"✅ Config loaded from: {config.config_path}")
    
    # 🚨 ADD THIS DEBUG CODE
    print("\n🔍 DEBUG: Social config content:")
    social_config = config.get('social_config', {})
    print(f"   Raw social config: {social_config}")
    
    if 'reddit' in social_config:
        print(f"   Reddit limit type: {type(social_config['reddit'].get('limit'))}")
        print(f"   Reddit limit value: {social_config['reddit'].get('limit')}")
    else:
        print("   ⚠️ No 'reddit' key in social_config!")
        
        # Temporary fix: Add it manually
        print("   🔧 Adding reddit config manually...")
        # This modifies the config in memory
        config._config['social_config']['reddit'] = {
            'limit': 100,
            'time_filter': 'week',
            'user_agent': 'TradingBot/1.0 (market sentiment aggregator)',
            'subreddits': ['wallstreetbets', 'stocks', 'investing'],
            'comment_limit': 10,
            'sort': 'hot'
        }
except ImportError as e:
    print(f"❌ Failed to import config: {e}")
    print("Attempting alternative import...")
    try:
        from agentic_trading_system.config.config_loader_discovery import config
        print(f"✅ Config loaded from alternative path: {config.config_path}")
    except ImportError as e2:
        print(f"❌ Also failed: {e2}")
        sys.exit(1)

from agentic_trading_system.discovery.search_aggregator import SearchAggregator


class DiscoveryRunner:
    """Standalone Discovery module runner"""
    
    def __init__(self):
        # Build config for discovery
        self.discovery_config = {
            "tavily_config":   config.get("tavily_config", {}),
            "news_config":     config.get("news_config", {}),
            "social_config":   config.get("social_config", {}),
            "sec_config":      config.get("sec_config", {}),
            "options_config":  config.get("options_config", {}),
            "macro_config":    config.get("macro_config", {}),
            "enricher_config": config.get("enricher_config", {}),
            "nlp_config":      config.get("nlp_config", {}),
            "regex_config":    config.get("regex_config", {}),
            "cache_ttl_minutes": config.get("cache_ttl_minutes", 15),
            "max_workers":     config.get("max_workers", 5),
            "source_weights":  config.get("source_weights", {})
        }
        
        # Create discovery instance
        self.discovery = SearchAggregator(
            name="StandaloneDiscovery",
            config=self.discovery_config
        )
        
        print("\n" + "=" * 60)
        print("🚀 DISCOVERY MODULE - STANDALONE MODE")
        print("=" * 60)
        print(f"📊 Sources configured: {list(self.discovery_config['source_weights'].keys())}")
        
        # Show API key status from config
        print("\n🔑 API Keys Status (from config):")
        tavily_key = self.discovery_config['tavily_config'].get('api_key')
        print(f"   Tavily: {'✅' if tavily_key else '❌'}")

        news_key = self.discovery_config['news_config'].get('news_api_key')
        print(f"   NewsAPI: {'✅' if news_key else '❌'}")

        alpha_key = self.discovery_config['news_config'].get('alpha_vantage_key')
        print(f"   Alpha Vantage: {'✅' if alpha_key else '❌'}")

        fmp_key = self.discovery_config['news_config'].get('fmp_key')
        print(f"   FMP: {'✅' if fmp_key else '❌'}")

        twitter_key = self.discovery_config['social_config'].get('twitter_bearer_token')
        print(f"   Twitter: {'✅' if twitter_key else '❌'}")

        fred_key = self.discovery_config['macro_config'].get('fred_api_key')
        print(f"   FRED: {'✅' if fred_key else '❌'}")
        
        print("=" * 60)
    
    async def search(self, query, sources=None, max_results=10):
        """Perform a search"""
        print(f"\n🔍 Searching: '{query}'")

        options = {"max_results": max_results}

        if sources:
            options["sources"] = sources

        result = await self.discovery.discover(query, options)
        self._print_results(result)
        return result
    
    async def extract(self, text):
        """Extract entities from text"""
        print(f"\n🔍 Extracting entities from: '{text[:100]}...'")
        
        entities = await self.discovery.extract_entities(text)
        
        print("\n🏷️  Found entities:")
        for k, v in entities.items():
            if v:
                print(f"   • {k}: {v}")
        
        return entities
    
    def _print_results(self, result):
        """Pretty print results"""
        print(f"\n📊 Results Summary:")
        print(f"   • Total items   : {result.get('total_items', 0)}")
        print(f"   • Unique items  : {result.get('unique_items', 0)}")
        print(f"   • Sources queried: {result.get('sources_queried', [])}")
        
        # Source stats
        print(f"\n📈 Source Statistics:")
        for source, stats in result.get("source_stats", {}).items():
            status = "✅" if stats.get("status") == "success" else "❌"
            count = stats.get('count', 0)
            print(f"   {status} {source}: {count} items")
            
            # Show error if any
            if stats.get("metadata", {}).get("error"):
                print(f"      Error: {stats['metadata']['error']}")
        
        # Top items
        items = result.get("items", [])
        # In discovery_loader.py, in _print_results method, after the top items section:
        if items:
            print(f"\n📄 Top {min(5, len(items))} items:")
            for i, item in enumerate(items[:5]):
                print(f"\n   [{i+1}] {item.get('title', 'No title')}")
                print(f"       Source    : {item.get('source', 'unknown')}")
                print(f"       Relevance : {item.get('relevance_score', 0):.2f}")
                
                # Add this to see the actual content
                print(f"\n       Content Preview:")
                content = item.get('content', '')
                if content:
                    # Print first 500 chars
                    for line in content.split('\n')[:10]:
                        print(f"           {line}")
                else:
                    print("           No content")
                
                if "company_info" in item:
                    ci = item["company_info"]
                    print(f"       Company   : {ci.get('name', 'N/A')}")
                    print(f"       Sector    : {ci.get('sector', 'N/A')}")
        else:
            print("\n   ⚠️  No items returned")
        
        # Entities
        entities = result.get("entities", {})
        if any(entities.values()):
            print(f"\n🏷️  Extracted Entities:")
            for k, v in entities.items():
                if v:
                    print(f"   • {k}: {v[:5]}")
    
    async def interactive(self):
        """Run interactive mode"""
        print("\n💬 Interactive mode - Type 'quit' to exit")
        print("   Commands: search <query>, extract <text>, stats, config")
        
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if cmd.lower() in ["quit", "exit"]:
                    break
                
                elif cmd.lower() == "stats":
                    stats = getattr(self.discovery, "get_stats", None)
                    print(f"\n📊 Stats: {stats() if stats else 'not available'}")
                
                elif cmd.lower() == "config":
                    print("\n📋 Current config:")
                    pprint(self.discovery_config)
                
                elif cmd.startswith("search "):
                    query = cmd[7:]
                    await self.search(query)
                
                elif cmd.startswith("extract "):
                    text = cmd[8:]
                    await self.extract(text)
                
                else:
                    print("Unknown command. Try: search AAPL, extract 'Apple Inc.', stats, config")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def batch(self, queries_file):
        """Run batch mode from file"""
        with open(queries_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"\n📋 Running {len(queries)} queries from {queries_file}")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query}")
            await self.search(query)
            await asyncio.sleep(2)  # Rate limiting between queries


async def main():
    parser = argparse.ArgumentParser(description="Run Discovery module standalone")
    parser.add_argument("--search",      type=str,          help="Search query")
    parser.add_argument("--extract",     type=str,          help="Text to extract entities from")
    parser.add_argument("--sources",     type=str, nargs="+", help="Sources to use (tavily, news, social, sec, options, macro)")
    parser.add_argument("--max",         type=int, default=10, help="Max results")
    parser.add_argument("--batch",       type=str,          help="File with queries to run in batch")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    runner = DiscoveryRunner()
    
    if args.interactive:
        await runner.interactive()
    elif args.batch:
        await runner.batch(args.batch)
    elif args.search:
        await runner.search(args.search, args.sources, args.max)
    elif args.extract:
        await runner.extract(args.extract)
    else:
        # Default: interactive mode
        await runner.interactive()


if __name__ == "__main__":
    asyncio.run(main())