"""
Rejected Logger - Logs rejected stocks for learning and analysis
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import pandas as pd
import asyncio
from agentic_trading_system.utils.logger import logger as logging

class RejectedLogger:
    """
    Logs rejected stocks and reasons for rejection
    
    Used for:
    - Analyzing rejection patterns
    - Improving filter thresholds
    - Debugging
    - Reporting
    - Machine learning feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage
        self.log_file = config.get("log_file", "data/rejected_stocks.json")
        self.csv_file = config.get("csv_file", "data/rejected_stocks.csv")
        self.max_log_entries = config.get("max_log_entries", 10000)
        self.keep_days = config.get("keep_days", 30)  # Keep logs for 30 days
        
        # In-memory cache
        self.recent_rejections = []
        self.rejection_counts = defaultdict(int)
        self.rejection_reasons = defaultdict(lambda: defaultdict(int))
        self.rejection_by_source = defaultdict(int)
        self.rejection_by_exchange = defaultdict(int)
        
        # Load existing log if available
        self._load_log()
        
        # Start cleanup task
        self._start_cleanup()
        
        logging.info(f"✅ RejectedLogger initialized with max {self.max_log_entries} entries")
    
    async def log(self, ticker: str, reasons: List[str], source: str, 
                  info: Optional[Dict] = None):
        """
        Log a rejected stock
        """
        entry = {
            "ticker": ticker,
            "reasons": reasons,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "exchange": info.get("exchange") if info else None,
            "price": info.get("current_price") if info else None,
            "volume": info.get("volume") if info else None,
            "market_cap": info.get("market_cap") if info else None,
            "sector": info.get("sector") if info else None,
            "primary_reason": reasons[0] if reasons else "unknown",
            "reason_count": len(reasons)
        }
        
        # Add to recent list
        self.recent_rejections.append(entry)
        
        # Update counts
        self.rejection_counts[ticker] += 1
        self.rejection_by_source[source] += 1
        
        if info and info.get("exchange"):
            self.rejection_by_exchange[info["exchange"]] += 1
        
        for reason in reasons:
            self.rejection_reasons[reason][ticker] += 1
        
        # Trim if needed
        if len(self.recent_rejections) > self.max_log_entries:
            # Remove oldest entries
            self.recent_rejections = self.recent_rejections[-self.max_log_entries:]
        
        # Save to file periodically (every 10 entries)
        if len(self.recent_rejections) % 10 == 0:
            self._save_log()
            self._save_csv()
        
        logging.debug(f"📝 Logged rejection for {ticker}: {reasons}")
    
    async def log_batch(self, rejections: List[Dict[str, Any]]):
        """
        Log multiple rejections at once
        """
        for rejection in rejections:
            await self.log(
                ticker=rejection["ticker"],
                reasons=rejection["reasons"],
                source=rejection.get("source", "unknown"),
                info=rejection.get("info")
            )
    
    async def get_recent(self, limit: int = 100, source: Optional[str] = None) -> List[Dict]:
        """
        Get recent rejections, optionally filtered by source
        """
        if source:
            filtered = [r for r in self.recent_rejections if r["source"] == source]
            return filtered[-limit:]
        return self.recent_rejections[-limit:]
    
    async def get_by_ticker(self, ticker: str, limit: int = 100) -> List[Dict]:
        """
        Get rejections for a specific ticker
        """
        return [r for r in self.recent_rejections if r["ticker"] == ticker][-limit:]
    
    async def get_by_reason(self, reason: str, limit: int = 100) -> List[Dict]:
        """
        Get rejections by reason
        """
        return [r for r in self.recent_rejections if reason in r["reasons"]][-limit:]
    
    async def get_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Get rejections within a date range
        """
        return [
            r for r in self.recent_rejections
            if start_date <= datetime.fromisoformat(r["timestamp"]) <= end_date
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rejection statistics
        """
        # Count reasons
        reason_counts = defaultdict(int)
        for rejection in self.recent_rejections:
            for reason in rejection["reasons"]:
                reason_counts[reason] += 1
        
        # Get top rejected tickers
        top_tickers = sorted(
            self.rejection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Get top reasons
        top_reasons = sorted(
            reason_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Get top sources
        top_sources = sorted(
            self.rejection_by_source.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate time-based stats
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        rejections_last_hour = sum(
            1 for r in self.recent_rejections
            if datetime.fromisoformat(r["timestamp"]) > last_hour
        )
        
        rejections_last_24h = sum(
            1 for r in self.recent_rejections
            if datetime.fromisoformat(r["timestamp"]) > last_24h
        )
        
        rejections_last_7d = sum(
            1 for r in self.recent_rejections
            if datetime.fromisoformat(r["timestamp"]) > last_7d
        )
        
        # Calculate average rejection rate
        avg_daily = rejections_last_7d / 7 if rejections_last_7d > 0 else 0
        
        # Calculate most common primary reasons
        primary_reasons = defaultdict(int)
        for rejection in self.recent_rejections:
            primary_reasons[rejection["primary_reason"]] += 1
        
        top_primary = sorted(
            primary_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_rejections": len(self.recent_rejections),
            "unique_tickers": len(self.rejection_counts),
            "unique_reasons": len(reason_counts),
            "unique_sources": len(self.rejection_by_source),
            "unique_exchanges": len(self.rejection_by_exchange),
            "rejections_last_hour": rejections_last_hour,
            "rejections_last_24h": rejections_last_24h,
            "rejections_last_7d": rejections_last_7d,
            "avg_daily_rejections": round(avg_daily, 1),
            "top_rejected_tickers": [
                {"ticker": t, "count": c, "percentage": round(c/len(self.recent_rejections)*100, 1) if self.recent_rejections else 0}
                for t, c in top_tickers
            ],
            "top_rejection_reasons": [
                {"reason": r, "count": c, "percentage": round(c/len(self.recent_rejections)*100, 1) if self.recent_rejections else 0}
                for r, c in top_reasons
            ],
            "top_primary_reasons": [
                {"reason": r, "count": c, "percentage": round(c/len(self.recent_rejections)*100, 1) if self.recent_rejections else 0}
                for r, c in top_primary
            ],
            "top_sources": [
                {"source": s, "count": c, "percentage": round(c/len(self.recent_rejections)*100, 1) if self.recent_rejections else 0}
                for s, c in top_sources
            ],
            "rejection_rate_per_hour": round(rejections_last_24h / 24, 1) if rejections_last_24h > 0 else 0,
            "exchanges": dict(self.rejection_by_exchange)
        }
    
    async def get_rejection_analysis(self) -> Dict[str, Any]:
        """
        Get detailed rejection analysis for ML feedback
        """
        if not self.recent_rejections:
            return {}
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.recent_rejections)
        
        analysis = {
            "correlations": {},
            "patterns": {},
            "threshold_suggestions": {}
        }
        
        # Analyze price-based rejections
        price_rejections = df[df['reasons'].apply(lambda x: any('price' in r.lower() for r in x))]
        if not price_rejections.empty:
            analysis["patterns"]["price"] = {
                "avg_price": float(price_rejections['price'].mean()),
                "min_price": float(price_rejections['price'].min()),
                "max_price": float(price_rejections['price'].max()),
                "count": len(price_rejections)
            }
            
            # Suggest price threshold adjustments
            if len(price_rejections) > 10:
                p25 = price_rejections['price'].quantile(0.25)
                p75 = price_rejections['price'].quantile(0.75)
                analysis["threshold_suggestions"]["price"] = {
                    "current_min": self.config.get("min_price", 1.0),
                    "suggested_min": round(max(0.1, p25 * 0.8), 2),
                    "current_max": self.config.get("max_price", 10000),
                    "suggested_max": round(p75 * 1.2, 2)
                }
        
        # Analyze volume-based rejections
        volume_rejections = df[df['reasons'].apply(lambda x: any('volume' in r.lower() for r in x))]
        if not volume_rejections.empty:
            analysis["patterns"]["volume"] = {
                "avg_volume": float(volume_rejections['volume'].mean()),
                "min_volume": float(volume_rejections['volume'].min()),
                "max_volume": float(volume_rejections['volume'].max()),
                "count": len(volume_rejections)
            }
        
        # Analyze market cap rejections
        mcap_rejections = df[df['reasons'].apply(lambda x: any('market cap' in r.lower() for r in x))]
        if not mcap_rejections.empty:
            analysis["patterns"]["market_cap"] = {
                "avg_mcap": float(mcap_rejections['market_cap'].mean()),
                "min_mcap": float(mcap_rejections['market_cap'].min()),
                "max_mcap": float(mcap_rejections['market_cap'].max()),
                "count": len(mcap_rejections)
            }
        
        # Analyze by sector
        sector_stats = df.groupby('sector').size().to_dict()
        analysis["patterns"]["by_sector"] = {k: int(v) for k, v in sector_stats.items() if k}
        
        # Analyze by exchange
        exchange_stats = df.groupby('exchange').size().to_dict()
        analysis["patterns"]["by_exchange"] = {k: int(v) for k, v in exchange_stats.items() if k}
        
        # Analyze rejection reason combinations
        from collections import Counter
        reason_combinations = Counter()
        for r in df['reasons']:
            reason_combinations[tuple(sorted(r))] += 1
        
        top_combinations = reason_combinations.most_common(10)
        analysis["patterns"]["common_combinations"] = [
            {"reasons": list(combo), "count": count}
            for combo, count in top_combinations
        ]
        
        return analysis
    
    def size(self) -> int:
        """Get number of logged rejections"""
        return len(self.recent_rejections)
    
    async def clear(self):
        """Clear all logs"""
        self.recent_rejections = []
        self.rejection_counts = defaultdict(int)
        self.rejection_reasons = defaultdict(lambda: defaultdict(int))
        self.rejection_by_source = defaultdict(int)
        self.rejection_by_exchange = defaultdict(int)
        self._save_log()
        self._save_csv()
        logging.info("🧹 Cleared rejection logs")
    
    async def cleanup_old(self):
        """Remove entries older than keep_days"""
        if not self.keep_days:
            return
        
        cutoff = datetime.now() - timedelta(days=self.keep_days)
        self.recent_rejections = [
            r for r in self.recent_rejections
            if datetime.fromisoformat(r["timestamp"]) > cutoff
        ]
        
        # Rebuild counts
        self.rejection_counts = defaultdict(int)
        self.rejection_reasons = defaultdict(lambda: defaultdict(int))
        self.rejection_by_source = defaultdict(int)
        self.rejection_by_exchange = defaultdict(int)
        
        for r in self.recent_rejections:
            ticker = r["ticker"]
            self.rejection_counts[ticker] += 1
            self.rejection_by_source[r["source"]] += 1
            if r["exchange"]:
                self.rejection_by_exchange[r["exchange"]] += 1
            for reason in r["reasons"]:
                self.rejection_reasons[reason][ticker] += 1
        
        logging.info(f"🧹 Cleaned up logs older than {self.keep_days} days")
    
    def _start_cleanup(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(86400)  # Run daily
                await self.cleanup_old()
        
        asyncio.create_task(cleanup_loop())
    
    def _save_log(self):
        """Save log to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump({
                    "rejections": self.recent_rejections,
                    "counts": dict(self.rejection_counts),
                    "by_source": dict(self.rejection_by_source),
                    "by_exchange": dict(self.rejection_by_exchange),
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving rejection log: {e}")
    
    def _save_csv(self):
        """Save log to CSV file for easy analysis"""
        try:
            if not self.recent_rejections:
                return
            
            df = pd.DataFrame(self.recent_rejections)
            
            # Flatten reasons for CSV
            df['reasons_str'] = df['reasons'].apply(lambda x: '|'.join(x))
            df = df.drop('reasons', axis=1)
            
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            df.to_csv(self.csv_file, index=False)
            
        except Exception as e:
            logging.error(f"Error saving rejection CSV: {e}")
    
    def _load_log(self):
        """Load log from JSON file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.recent_rejections = data.get("rejections", [])
                    
                    # Rebuild counts
                    self.rejection_counts = defaultdict(int)
                    self.rejection_reasons = defaultdict(lambda: defaultdict(int))
                    self.rejection_by_source = defaultdict(int)
                    self.rejection_by_exchange = defaultdict(int)
                    
                    for rejection in self.recent_rejections:
                        ticker = rejection["ticker"]
                        self.rejection_counts[ticker] += 1
                        self.rejection_by_source[rejection["source"]] += 1
                        if rejection.get("exchange"):
                            self.rejection_by_exchange[rejection["exchange"]] += 1
                        for reason in rejection["reasons"]:
                            self.rejection_reasons[reason][ticker] += 1
                    
                    logging.info(f"📂 Loaded {len(self.recent_rejections)} rejection records")
        except Exception as e:
            logging.error(f"Error loading rejection log: {e}")