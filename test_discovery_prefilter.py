#!/usr/bin/env python3
"""
run_prefilter_from_discovery.py

Loads the latest discovery artifact, extracts tickers,
runs the prefilter quality gates, and saves results.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Set
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.prefilter.quality_gates import QualityGates

# Prefilter configuration (adjust as needed)
PREFILTER_CONFIG = {
    "exchange_config": {
        "allowed_exchanges": ["NYSE", "NASDAQ", "AMEX", "BATS", "IEX"]
    },
    "price_config": {
        "min_price": 1.0,
        "max_price": 10000.0,
        "max_daily_change": 100.0
    },
    "volume_config": {
        "min_volume": 100000,
        "min_avg_volume": 50000,
        "min_dollar_volume": 1_000_000
    },
    "market_cap_config": {
        "min_market_cap": 50_000_000,
        "max_market_cap": float('inf')
    },
    "data_quality_config": {
        "min_history_days": 60,
        "min_data_points": 50
    },
    "queue_config": {
        "max_size": 1000,
        "item_ttl_seconds": 3600
    },
    "cache_ttl_minutes": 5
}

def extract_tickers_from_artifact(run_dir: Path) -> Set[str]:
    """
    Extract unique tickers from a discovery run directory.
    Reads entities.json and the run name (which contains the query ticker).
    """
    tickers = set()
    
    # 1. Query ticker from directory name (e.g., "20260402_180511_AAPL")
    run_name = run_dir.name
    if "_" in run_name:
        query_ticker = run_name.split("_")[-1]
        tickers.add(query_ticker)
    
    # 2. Load entities.json for detected tickers
    entities_file = run_dir / "entities.json"
    if entities_file.exists():
        with open(entities_file, 'r') as f:
            entities = json.load(f)
        detected = entities.get("tickers", [])
        tickers.update(detected)
    
    # 3. Optionally, also scan artifact.json for items' detected_tickers
    artifact_file = run_dir / "artifact.json"
    if artifact_file.exists():
        with open(artifact_file, 'r') as f:
            artifact_data = json.load(f)
        for item in artifact_data.get("items", []):
            tickers.update(item.get("detected_tickers", []))
    
    # Filter out common false positives (like "TIME", "QUOTE", etc.)
    false_positives = {"TIME", "QUOTE", "ADD", "VS", "PER", "WHEN", "LAST", "TAG", "ONE", "HIGH", "FALLS", "TESLA"}
    tickers = {t for t in tickers if t not in false_positives and len(t) <= 5 and t.isalpha()}
    
    return tickers

async def run_prefilter(tickers: List[str], output_dir: Path) -> dict:
    """Run prefilter on a list of tickers and save results."""
    quality_gates = QualityGates(name="Prefilter", config=PREFILTER_CONFIG)
    result = await quality_gates.filter_tickers(tickers, source="discovery")
    
    print(f"\n✅ Prefilter complete")
    print(f"   Processed: {result['total_processed']}")
    print(f"   Passed:    {result['passed_count']}")
    print(f"   Rejected:  {result['rejected_count']}")
    
    if result['rejected']:
        print("\n❌ Rejections:")
        for r in result['rejected']:
            print(f"   {r['ticker']}: {', '.join(r['reasons'])}")
    
    if result['passed']:
        print(f"\n✅ Passed {len(result['passed'])} tickers (queued for analysis):")
        for p in result['passed']:
            # Ensure we print the ticker
            ticker = p.get('ticker', p.get('symbol', 'unknown'))
            print(f"   {ticker}")
    
    # Save prefilter results to a JSON file
    prefilter_output = {
        "timestamp": datetime.now().isoformat(),
        "source": "discovery",
        "tickers_processed": tickers,
        "passed": [
            {
                "ticker": p.get('ticker', p.get('symbol')),
                "info": p.get('info', {}),
                "checks_passed": p.get('checks_passed', {})
            }
            for p in result['passed']
        ],
        "rejected": [
            {
                "ticker": r['ticker'],
                "reasons": r['reasons']
            }
            for r in result['rejected']
        ],
        "stats": result.get('stats', {})
    }
    
    output_file = output_dir / "prefilter_results.json"
    with open(output_file, 'w') as f:
        json.dump(prefilter_output, f, indent=2, default=str)
    print(f"\n💾 Prefilter results saved to: {output_file}")
    
    # Also save the queue state (passed items) for analysis
    queue_items = await quality_gates.passed_queue.get_all()
    queue_file = output_dir / "passed_queue.json"
    with open(queue_file, 'w') as f:
        json.dump(queue_items, f, indent=2, default=str)
    print(f"💾 Passed queue saved to: {queue_file}")
    
    return result

def main():
    output_base = Path("discovery_outputs")
    if not output_base.exists():
        print("❌ No discovery_outputs directory found. Run discovery first.")
        return
    
    # Find the latest run directory (most recent by name, which includes timestamp)
    run_dirs = sorted(output_base.glob("2026*_*"), reverse=True)
    if not run_dirs:
        print("❌ No discovery run directories found.")
        return
    
    latest_run = run_dirs[0]
    print(f"📁 Using latest discovery run: {latest_run.name}")
    
    # Create a prefilter output directory inside the run folder
    prefilter_out_dir = latest_run / "prefilter"
    prefilter_out_dir.mkdir(exist_ok=True)
    
    tickers = extract_tickers_from_artifact(latest_run)
    if not tickers:
        print("❌ No tickers extracted from artifact.")
        return
    
    print(f"🔍 Extracted tickers: {sorted(tickers)}")
    asyncio.run(run_prefilter(list(tickers), prefilter_out_dir))

if __name__ == "__main__":
    main()