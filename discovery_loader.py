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

# Add discovery folder to path
sys.path.insert(0, str(Path(__file__).parent))

# Simple logger for standalone mode
class SimpleLogger:
    def info(self, msg): print(f"📘 {msg}")
    def error(self, msg): print(f"❌ {msg}")
    def warning(self, msg): print(f"⚠️ {msg}")
    def debug(self, msg): print(f"🔍 {msg}")

# FIX: Override the logger as a proper module with a logger attribute
# so that `from utils.logger import logger as logging` works correctly
fake_logger_module = types.ModuleType("utils.logger")
fake_logger_module.logger = SimpleLogger()
sys.modules["utils.logger"] = fake_logger_module

from agentic_trading_system.config.config_loader_discovery  import config
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
        print(f"📋 Config loaded from: {config.config_path}")
        print(f"📊 Sources configured: {list(self.discovery_config['source_weights'].keys())}")
        print("=" * 60)
    
    async def search(self, query, sources=None, max_results=10):
        """Perform a search"""
        print(f"\n🔍 Searching: '{query}'")

        options = {"max_results": max_results}

        # FIX: Only add sources key if actually provided — passing None causes
        # TypeError in _get_sources_to_query when it tries to iterate over it
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
            print(f"   {status} {source}: {stats.get('count', 0)} items")
        
        # Top items
        items = result.get("items", [])
        if items:
            print(f"\n📄 Top {min(5, len(items))} items:")
            for i, item in enumerate(items[:5]):
                print(f"\n   [{i+1}] {item.get('title', 'No title')}")
                print(f"       Source    : {item.get('source', 'unknown')}")
                print(f"       Relevance : {item.get('relevance_score', 0):.2f}")
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