#!/usr/bin/env python3
"""
Debug macro data client
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
import json

# Load environment
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.config.config_loader_discovery import config
from agentic_trading_system.discovery.macro_data_client import MacroDataClient

async def test_fred_direct():
    """Test FRED API directly"""
    fred_key = os.getenv('FRED_API_KEY')
    print(f"FRED API Key: {fred_key[:5]}..." if fred_key else "No FRED key")
    
    if not fred_key:
        print("❌ No FRED API key found")
        return
    
    # Test GDP series
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GDP",
        "api_key": fred_key,
        "file_type": "json",
        "limit": 5,
        "sort_order": "desc"
    }
    
    print(f"\n📡 Testing FRED API directly...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    observations = data.get("observations", [])
                    print(f"Found {len(observations)} observations")
                    for obs in observations[:3]:
                        if obs.get("value") != ".":
                            print(f"  {obs.get('date')}: {obs.get('value')}")
                    return True
                else:
                    text = await response.text()
                    print(f"Error: {text[:200]}")
                    return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

async def test_macro_client():
    """Test the macro client"""
    print("\n" + "="*60)
    print("Testing MacroDataClient")
    print("="*60)
    
    # Get config
    macro_config = config.get("macro_config", {})
    print(f"Macro config: {macro_config}")
    
    # Create client
    client = MacroDataClient(macro_config)
    
    # Test different queries
    queries = ["gdp", "GDP", "inflation", "unemployment", "fed_funds"]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Searching for: '{query}'")
        print(f"{'='*60}")
        
        try:
            result = await client.search(query)
            items = result.get("items", [])
            sources = result.get("sources_used", [])
            
            print(f"Sources used: {sources}")
            print(f"Items found: {len(items)}")
            
            for i, item in enumerate(items):
                print(f"\n--- Item {i+1} ---")
                print(f"Title: {item.get('title', 'No title')}")
                print(f"Source: {item.get('source', 'unknown')}")
                print(f"Latest: {item.get('latest_value')} ({item.get('latest_date')})")
                print(f"Content: {item.get('content', '')[:200]}...")
                
        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(1)  # Rate limiting

async def main():
    # First test FRED directly
    fred_working = await test_fred_direct()
    
    if not fred_working:
        print("\n⚠️ FRED API not working. Check your API key.")
        print("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    # Then test the client
    await test_macro_client()

if __name__ == "__main__":
    asyncio.run(main())