#!/usr/bin/env python3
"""
run_analysis_on_passed.py - Direct analysis without message passing
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.analysis.technical.technical_analyzer import TechnicalAnalyzer
from agentic_trading_system.analysis.fundamental.fundamental_analyzer import FundamentalAnalyzer
from agentic_trading_system.analysis.sentiment.sentiment_analyzer import SentimentAnalyzer
from agentic_trading_system.analysis.regime_detector import RegimeDetector
from agentic_trading_system.analysis.weighted_score_engine import WeightedScoreEngine

async def analyze_ticker(ticker: str):
    """Run all analyses directly and combine scores."""
    print(f"\n📊 Analyzing {ticker}...")
    
    # Initialize analyzers (with empty configs for now)
    tech = TechnicalAnalyzer("TechnicalAnalyzer", {})
    fund = FundamentalAnalyzer("FundamentalAnalyzer", {})
    sent = SentimentAnalyzer("SentimentAnalyzer", {})
    regime = RegimeDetector("RegimeDetector", {})
    scorer = WeightedScoreEngine("WeightedScoreEngine", {})
    
    # Run each analysis
    tech_score, tech_details = await tech.analyze(ticker)
    print(f"   Technical score: {tech_score:.2f}")
    
    fund_score, fund_details = await fund.analyze(ticker)
    print(f"   Fundamental score: {fund_score:.2f}")
    
    sent_score, sent_details = await sent.analyze(ticker)
    print(f"   Sentiment score: {sent_score:.2f}")
    
    regime_result = await regime.detect_regime()  # market regime, not ticker-specific
    market_regime = regime_result.get("regime", "unknown")
    print(f"   Market regime: {market_regime}")
    
    # Combine scores using the score engine
    scores = {
        "technical": {"score": tech_score, "confidence": 0.7},
        "fundamental": {"score": fund_score, "confidence": 0.6},
        "sentiment": {"score": sent_score, "confidence": 0.6},
        "timeframe": {"score": 0.5, "confidence": 0.5},  # placeholder
        "risk": {"score": 0.7, "confidence": 0.8}        # placeholder
    }
    
    combined = await scorer.combine_scores(
        analysis_id=f"analysis_{ticker}",
        scores=scores,
        regime=market_regime
    )
    
    return {
        "ticker": ticker,
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "sentiment_score": sent_score,
        "market_regime": market_regime,
        "final_score": combined["final_score"],
        "recommendation": combined["recommendation"]["action"],
        "confidence": combined["confidence"],
        "details": combined
    }

async def main():
    # Find latest discovery run and its passed queue
    discovery_base = Path("discovery_outputs")
    run_dirs = sorted(discovery_base.glob("2026*_*"), reverse=True)
    if not run_dirs:
        print("No discovery runs found")
        return
    
    latest_run = run_dirs[0]
    queue_file = latest_run / "prefilter" / "passed_queue.json"
    if not queue_file.exists():
        print(f"No passed queue found at {queue_file}")
        return
    
    with open(queue_file) as f:
        passed_items = json.load(f)
    
    tickers = [item["ticker"] for item in passed_items]
    print(f"🔍 Analyzing {len(tickers)} tickers: {tickers}")
    
    results = []
    for ticker in tickers:
        result = await analyze_ticker(ticker)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['ticker']:6} | Score: {r['final_score']:.2f} | {r['recommendation']:10} | Conf: {r['confidence']:.2f}")
    
    # Save results
    output_dir = latest_run / "analysis"
    output_dir.mkdir(exist_ok=True)
    for r in results:
        out_file = output_dir / f"{r['ticker']}_analysis.json"
        with open(out_file, "w") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())