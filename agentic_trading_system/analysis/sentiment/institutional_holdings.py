"""
Institutional Holdings - Analyzes institutional investor activity
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from utils.logger import logger as  logging

class InstitutionalHoldings:
    """
    Analyzes institutional holdings (13F filings)
    Tracks what big money is doing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fmp_api_key = config.get("fmp_api_key")
        
        # Lookback period (quarters)
        self.lookback_quarters = config.get("lookback_quarters", 4)
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(days=1)  # Update daily
        
        logging.info(f"📊 InstitutionalHoldings initialized")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment from institutional holdings
        """
        logging.info(f"📊 Fetching institutional holdings for {symbol}")
        
        # Check cache
        cache_key = f"institutional_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Fetch institutional data
            holdings = await self._fetch_institutional_holdings(symbol)
            
            if not holdings:
                return {
                    "score": 0.5,
                    "confidence": 0.3,
                    "details": {"error": "No institutional data found"},
                    "source": "institutional"
                }
            
            # Calculate metrics
            analysis = self._analyze_holdings(holdings)
            
            # Calculate sentiment score
            score = self._calculate_score(analysis)
            
            # Calculate confidence
            confidence = self._calculate_confidence(holdings, analysis)
            
            result = {
                "score": float(score),
                "confidence": float(confidence),
                "source": "institutional",
                "details": {
                    "institution_count": len(holdings),
                    "total_holders": analysis.get("total_holders", 0),
                    **analysis
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in institutional holdings for {symbol}: {e}")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "source": "institutional",
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_institutional_holdings(self, symbol: str) -> List[Dict]:
        """Fetch institutional holdings from API"""
        if not self.fmp_api_key:
            return self._get_mock_holdings(symbol)
        
        url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{symbol}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"apikey": self.fmp_api_key}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return [{
                            "holder": item.get("holder", "Unknown"),
                            "shares": item.get("shares", 0),
                            "date_reported": item.get("dateReported", ""),
                            "change": item.get("change", 0),
                            "change_percent": item.get("changePercent", 0),
                            "weight_percent": item.get("weightPercent", 0)
                        } for item in data[:50]]  # Top 50 holders
        except Exception as e:
            logging.debug(f"Error fetching institutional holdings: {e}")
        
        return self._get_mock_holdings(symbol)
    
    def _get_mock_holdings(self, symbol: str) -> List[Dict]:
        """Get mock institutional holdings for testing"""
        mock_data = {
            "AAPL": [
                {
                    "holder": "Vanguard Group Inc",
                    "shares": 1000000000,
                    "date_reported": "2023-12-31",
                    "change": 5000000,
                    "change_percent": 0.5,
                    "weight_percent": 7.5
                },
                {
                    "holder": "BlackRock Inc",
                    "shares": 950000000,
                    "date_reported": "2023-12-31",
                    "change": 3000000,
                    "change_percent": 0.3,
                    "weight_percent": 7.2
                },
                {
                    "holder": "Berkshire Hathaway",
                    "shares": 900000000,
                    "date_reported": "2023-12-31",
                    "change": -1000000,
                    "change_percent": -0.1,
                    "weight_percent": 6.8
                }
            ],
            "TSLA": [
                {
                    "holder": "Vanguard Group Inc",
                    "shares": 200000000,
                    "date_reported": "2023-12-31",
                    "change": -5000000,
                    "change_percent": -2.5,
                    "weight_percent": 3.2
                }
            ]
        }
        
        return mock_data.get(symbol, [])
    
    def _analyze_holdings(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Analyze institutional holdings data"""
        total_shares = sum(h.get("shares", 0) for h in holdings)
        total_change = sum(h.get("change", 0) for h in holdings)
        
        # Calculate net flows
        increasing = [h for h in holdings if h.get("change", 0) > 0]
        decreasing = [h for h in holdings if h.get("change", 0) < 0]
        
        # Calculate institutional concentration
        top_10_shares = sum(h.get("shares", 0) for h in holdings[:10])
        top_10_concentration = top_10_shares / total_shares if total_shares > 0 else 0
        
        # Identify notable institutions
        notable_holders = []
        notable_names = ["Berkshire Hathaway", "Bridgewater", "Renaissance", "Citadel"]
        
        for h in holdings[:5]:  # Check top 5
            holder = h.get("holder", "")
            if any(name in holder for name in notable_names):
                notable_holders.append({
                    "name": holder,
                    "action": "increasing" if h.get("change", 0) > 0 else "decreasing",
                    "change": h.get("change_percent", 0)
                })
        
        return {
            "total_shares": int(total_shares),
            "total_change": int(total_change),
            "net_flow": int(total_change),
            "increasing_count": len(increasing),
            "decreasing_count": len(decreasing),
            "increasing_ratio": len(increasing) / len(holdings) if holdings else 0.5,
            "top_10_concentration": float(top_10_concentration),
            "notable_holders": notable_holders
        }
    
    def _calculate_score(self, analysis: Dict) -> float:
        """Calculate sentiment score from institutional activity"""
        score = 0.5  # Start neutral
        
        # Net flow direction
        net_flow = analysis.get("net_flow", 0)
        if net_flow > 0:
            score += min(0.2, net_flow / 10000000)  # Scale by 10M shares
        elif net_flow < 0:
            score -= min(0.2, abs(net_flow) / 10000000)
        
        # Increasing/decreasing ratio
        inc_ratio = analysis.get("increasing_ratio", 0.5)
        score += (inc_ratio - 0.5) * 0.3
        
        # Notable holders
        notable = analysis.get("notable_holders", [])
        for n in notable:
            if n.get("action") == "increasing":
                score += 0.05
            else:
                score -= 0.05
        
        return float(min(1.0, max(0.0, score)))
    
    def _calculate_confidence(self, holdings: List[Dict], analysis: Dict) -> float:
        """Calculate confidence in institutional sentiment"""
        if not holdings:
            return 0.3
        
        # Number of holders factor
        holder_count = len(holdings)
        count_factor = min(1.0, holder_count / 20)
        
        # Total shares factor (more shares = more confidence)
        total_shares = analysis.get("total_shares", 0)
        shares_factor = min(1.0, total_shares / 100000000)  # Scale by 100M shares
        
        # Concentration factor (high concentration = higher confidence)
        concentration = analysis.get("top_10_concentration", 0)
        concentration_factor = min(1.0, concentration * 1.5)
        
        confidence = (
            count_factor * 0.4 +
            shares_factor * 0.3 +
            concentration_factor * 0.3
        )
        
        return float(min(1.0, max(0.3, confidence)))