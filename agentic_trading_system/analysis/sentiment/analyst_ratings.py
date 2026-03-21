"""
Analyst Ratings - Analyzes and aggregates analyst recommendations
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from agentic_trading_system.utils.logger import logger as  logging
import numpy as np
class AnalystRatings:
    """
    Fetches and analyzes analyst ratings and price targets
    Sources: Yahoo Finance, Financial Modeling Prep, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fmp_api_key = config.get("fmp_api_key")
        
        # Rating mapping
        self.rating_scores = {
            "strong_buy": 1.0,
            "buy": 0.8,
            "outperform": 0.75,
            "hold": 0.5,
            "underperform": 0.3,
            "sell": 0.2,
            "strong_sell": 0.1
        }
        
        # Number of analysts required for confidence
        self.min_analysts = config.get("min_analysts", 3)
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(hours=6)
        
        logging.info(f"📊 AnalystRatings initialized")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment from analyst ratings
        """
        logging.info(f"📊 Fetching analyst ratings for {symbol}")
        
        # Check cache
        cache_key = f"analyst_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Fetch analyst data
            ratings = await self._fetch_analyst_ratings(symbol)
            
            if not ratings:
                return {
                    "score": 0.5,
                    "confidence": 0.3,
                    "details": {"error": "No analyst data found"},
                    "source": "analyst"
                }
            
            # Calculate metrics
            avg_score = self._calculate_average_score(ratings)
            consensus = self._get_consensus(ratings)
            price_target = await self._get_price_target(symbol)
            revisions = await self._get_rating_revisions(symbol)
            
            # Calculate confidence
            confidence = self._calculate_confidence(ratings, revisions)
            
            result = {
                "score": float(avg_score),
                "confidence": float(confidence),
                "source": "analyst",
                "details": {
                    "analyst_count": len(ratings),
                    "consensus": consensus,
                    "price_target": price_target,
                    "rating_breakdown": self._get_rating_breakdown(ratings),
                    "revisions": revisions,
                    "top_firms": self._get_top_firms(ratings)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyst ratings for {symbol}: {e}")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "source": "analyst",
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_analyst_ratings(self, symbol: str) -> List[Dict]:
        """Fetch analyst ratings from API"""
        if not self.fmp_api_key:
            return self._get_mock_ratings(symbol)
        
        url = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{symbol}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"apikey": self.fmp_api_key}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [{
                            "firm": item.get("analystFirm", "Unknown"),
                            "rating": item.get("rating", "hold").lower(),
                            "previous_rating": item.get("previousRating", "").lower(),
                            "date": item.get("date", ""),
                            "action": item.get("action", "")
                        } for item in data[:20]]  # Last 20 ratings
        except Exception as e:
            logging.debug(f"Error fetching analyst ratings: {e}")
        
        return self._get_mock_ratings(symbol)
    
    def _get_mock_ratings(self, symbol: str) -> List[Dict]:
        """Get mock ratings for testing"""
        # Mock data for popular stocks
        mock_data = {
            "AAPL": [
                {"firm": "Morgan Stanley", "rating": "buy", "previous_rating": "hold", "date": "2024-02-15"},
                {"firm": "Goldman Sachs", "rating": "strong_buy", "previous_rating": "buy", "date": "2024-02-10"},
                {"firm": "JP Morgan", "rating": "buy", "previous_rating": "buy", "date": "2024-02-05"},
                {"firm": "Bank of America", "rating": "hold", "previous_rating": "hold", "date": "2024-01-30"},
            ],
            "TSLA": [
                {"firm": "Morgan Stanley", "rating": "hold", "previous_rating": "sell", "date": "2024-02-14"},
                {"firm": "Goldman Sachs", "rating": "sell", "previous_rating": "sell", "date": "2024-02-08"},
            ]
        }
        
        return mock_data.get(symbol, [
            {"firm": "Analyst 1", "rating": "hold", "previous_rating": "hold", "date": "2024-02-01"}
        ])
    
    def _calculate_average_score(self, ratings: List[Dict]) -> float:
        """Calculate average rating score"""
        if not ratings:
            return 0.5
        
        scores = []
        for r in ratings:
            rating = r.get("rating", "").lower()
            score = self.rating_scores.get(rating, 0.5)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _get_consensus(self, ratings: List[Dict]) -> str:
        """Get consensus rating"""
        if not ratings:
            return "neutral"
        
        scores = [self.rating_scores.get(r.get("rating", "").lower(), 0.5) for r in ratings]
        avg_score = np.mean(scores)
        
        if avg_score >= 0.8:
            return "strong_buy"
        elif avg_score >= 0.65:
            return "buy"
        elif avg_score >= 0.45:
            return "hold"
        elif avg_score >= 0.25:
            return "sell"
        else:
            return "strong_sell"
    
    def _get_rating_breakdown(self, ratings: List[Dict]) -> Dict[str, int]:
        """Get breakdown of ratings by category"""
        breakdown = {category: 0 for category in self.rating_scores.keys()}
        
        for r in ratings:
            rating = r.get("rating", "").lower()
            if rating in breakdown:
                breakdown[rating] += 1
        
        return breakdown
    
    async def _get_price_target(self, symbol: str) -> Dict[str, Any]:
        """Get price target data"""
        # Mock data for now
        return {
            "high": 200.0,
            "low": 150.0,
            "mean": 175.0,
            "median": 172.5,
            "analyst_count": 15
        }
    
    async def _get_rating_revisions(self, symbol: str) -> Dict[str, Any]:
        """Get rating revision data"""
        upgrades = 0
        downgrades = 0
        initiations = 0
        
        # Mock data
        if symbol == "AAPL":
            upgrades = 2
            downgrades = 0
            initiations = 1
        elif symbol == "TSLA":
            upgrades = 0
            downgrades = 1
            initiations = 0
        
        return {
            "upgrades": upgrades,
            "downgrades": downgrades,
            "initiations": initiations,
            "net_revisions": upgrades - downgrades
        }
    
    def _get_top_firms(self, ratings: List[Dict]) -> List[Dict]:
        """Get top investment firms"""
        firms = {}
        for r in ratings:
            firm = r.get("firm", "Unknown")
            if firm not in firms:
                firms[firm] = {
                    "name": firm,
                    "rating": r.get("rating", ""),
                    "date": r.get("date", "")
                }
        
        return list(firms.values())[:5]  # Top 5 firms
    
    def _calculate_confidence(self, ratings: List[Dict], revisions: Dict) -> float:
        """Calculate confidence in analyst ratings"""
        if not ratings:
            return 0.3
        
        # Number of analysts factor
        analyst_count = len(ratings)
        count_factor = min(1.0, analyst_count / self.min_analysts)
        
        # Recency factor (more recent = higher confidence)
        recent_count = sum(1 for r in ratings[:3] if r.get("date"))
        recency_factor = recent_count / 3
        
        # Revision factor (upgrades increase confidence)
        net_revisions = revisions.get("net_revisions", 0)
        revision_factor = 0.5 + (net_revisions * 0.1)
        revision_factor = min(1.0, max(0.3, revision_factor))
        
        confidence = (
            count_factor * 0.4 +
            recency_factor * 0.3 +
            revision_factor * 0.3
        )
        
        return float(min(1.0, max(0.3, confidence)))