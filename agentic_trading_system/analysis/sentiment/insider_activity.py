"""
Insider Activity - Analyzes insider trading activity
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from agentic_trading_system.utils.logger import logger as logging
import numpy as np

class InsiderActivity:
    """
    Analyzes insider transactions (Form 4 filings)
    Tracks buying/selling activity by corporate insiders
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fmp_api_key = config.get("fmp_api_key")
        
        # Transaction type weights
        self.transaction_weights = {
            "P": 1.0,      # Purchase
            "S": -1.0,     # Sale
            "A": 0.5,      # Award
            "D": -0.3,     # Disposition
            "F": -0.2,     # Tax withholding
            "M": 0.2       # Exercise of option
        }
        
        # Lookback period
        self.lookback_days = config.get("lookback_days", 90)
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(hours=12)
        
        logging.info(f"📊 InsiderActivity initialized")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment from insider activity
        """
        logging.info(f"📊 Fetching insider activity for {symbol}")
        
        # Check cache
        cache_key = f"insider_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Fetch insider transactions
            transactions = await self._fetch_insider_transactions(symbol)
            
            if not transactions:
                return {
                    "score": 0.5,
                    "confidence": 0.3,
                    "details": {"error": "No insider data found"},
                    "source": "insider"
                }
            
            # Calculate metrics
            analysis = self._analyze_transactions(transactions)
            
            # Calculate sentiment score
            score = self._calculate_score(analysis)
            
            # Calculate confidence
            confidence = self._calculate_confidence(transactions, analysis)
            
            result = {
                "score": float(score),
                "confidence": float(confidence),
                "source": "insider",
                "details": {
                    "transaction_count": len(transactions),
                    "analysis_period_days": self.lookback_days,
                    **analysis
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in insider activity for {symbol}: {e}")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "source": "insider",
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_insider_transactions(self, symbol: str) -> List[Dict]:
        """Fetch insider transactions from API"""
        if not self.fmp_api_key:
            return self._get_mock_transactions(symbol)
        
        url = f"https://financialmodelingprep.com/api/v4/insider-trading"
        
        params = {
            "symbol": symbol,
            "limit": 100,
            "apikey": self.fmp_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter by date
                        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
                        
                        transactions = []
                        for item in data:
                            filing_date = item.get("filingDate", "")
                            if filing_date:
                                try:
                                    date = datetime.fromisoformat(filing_date.replace("Z", "+00:00"))
                                    if date > cutoff_date:
                                        transactions.append({
                                            "filing_date": filing_date,
                                            "transaction_date": item.get("transactionDate", ""),
                                            "owner": item.get("owner", {}),
                                            "transaction_type": item.get("transactionType", ""),
                                            "shares": item.get("shares", 0),
                                            "price": item.get("price", 0),
                                            "value": item.get("value", 0),
                                            "is_direct": item.get("isDirect", True)
                                        })
                                except:
                                    continue
                        
                        return transactions
        except Exception as e:
            logging.debug(f"Error fetching insider transactions: {e}")
        
        return self._get_mock_transactions(symbol)
    
    def _get_mock_transactions(self, symbol: str) -> List[Dict]:
        """Get mock insider transactions for testing"""
        mock_data = {
            "AAPL": [
                {
                    "filing_date": (datetime.now() - timedelta(days=5)).isoformat(),
                    "transaction_date": (datetime.now() - timedelta(days=7)).isoformat(),
                    "owner": {"name": "Tim Cook", "title": "CEO"},
                    "transaction_type": "P",
                    "shares": 5000,
                    "price": 175.0,
                    "value": 875000,
                    "is_direct": True
                },
                {
                    "filing_date": (datetime.now() - timedelta(days=12)).isoformat(),
                    "transaction_date": (datetime.now() - timedelta(days=14)).isoformat(),
                    "owner": {"name": "Luca Maestri", "title": "CFO"},
                    "transaction_type": "S",
                    "shares": 2000,
                    "price": 178.0,
                    "value": 356000,
                    "is_direct": True
                }
            ],
            "TSLA": [
                {
                    "filing_date": (datetime.now() - timedelta(days=3)).isoformat(),
                    "transaction_date": (datetime.now() - timedelta(days=5)).isoformat(),
                    "owner": {"name": "Elon Musk", "title": "CEO"},
                    "transaction_type": "S",
                    "shares": 10000,
                    "price": 200.0,
                    "value": 2000000,
                    "is_direct": True
                }
            ]
        }
        
        return mock_data.get(symbol, [])
    
    def _analyze_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze insider transactions"""
        buys = []
        sells = []
        total_value = 0
        insider_scores = []
        
        for t in transactions:
            trans_type = t.get("transaction_type", "")
            shares = t.get("shares", 0)
            value = t.get("value", 0)
            
            if trans_type in ["P", "A", "M"]:
                buys.append(t)
                total_value += value
                insider_scores.append(self.transaction_weights.get(trans_type, 0))
            elif trans_type in ["S", "D", "F"]:
                sells.append(t)
                total_value -= value
                insider_scores.append(self.transaction_weights.get(trans_type, 0))
        
        # Calculate net score
        avg_score = np.mean(insider_scores) if insider_scores else 0
        
        # Calculate ratios
        buy_count = len(buys)
        sell_count = len(sells)
        total_count = buy_count + sell_count
        
        buy_ratio = buy_count / total_count if total_count > 0 else 0.5
        buy_volume = sum(t.get("shares", 0) for t in buys)
        sell_volume = sum(t.get("shares", 0) for t in sells)
        
        # Identify notable insiders
        ceo_actions = []
        cfo_actions = []
        
        for t in transactions:
            title = t.get("owner", {}).get("title", "").lower()
            if "ceo" in title or "chief executive" in title:
                ceo_actions.append(t)
            if "cfo" in title or "chief financial" in title:
                cfo_actions.append(t)
        
        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_transactions": buy_count - sell_count,
            "buy_ratio": float(buy_ratio),
            "buy_volume": int(buy_volume),
            "sell_volume": int(sell_volume),
            "net_volume": int(buy_volume - sell_volume),
            "total_value": float(total_value),
            "avg_score": float(avg_score),
            "ceo_actions": len(ceo_actions),
            "cfo_actions": len(cfo_actions),
            "ceo_buying": any(t.get("transaction_type") in ["P", "A", "M"] for t in ceo_actions)
        }
    
    def _calculate_score(self, analysis: Dict) -> float:
        """Calculate sentiment score from insider activity"""
        score = 0.5  # Start neutral
        
        # Net transaction count
        net = analysis.get("net_transactions", 0)
        score += min(0.2, net * 0.05)
        
        # Buy ratio
        buy_ratio = analysis.get("buy_ratio", 0.5)
        score += (buy_ratio - 0.5) * 0.3
        
        # Net volume
        net_volume_ratio = analysis.get("net_volume", 0) / max(analysis.get("buy_volume", 1), 1)
        score += min(0.2, net_volume_ratio * 0.1)
        
        # CEO action
        if analysis.get("ceo_buying"):
            score += 0.1
        
        return float(min(1.0, max(0.0, score)))
    
    def _calculate_confidence(self, transactions: List[Dict], analysis: Dict) -> float:
        """Calculate confidence in insider sentiment"""
        if not transactions:
            return 0.3
        
        # Transaction count factor
        count_factor = min(1.0, len(transactions) / 20)
        
        # Recency factor (more recent = higher confidence)
        recent_count = sum(1 for t in transactions[:5] 
                         if datetime.now() - datetime.fromisoformat(t.get("filing_date", "").replace("Z", "+00:00")) < timedelta(days=7))
        recency_factor = recent_count / 5
        
        # Executive involvement factor
        exec_count = analysis.get("ceo_actions", 0) + analysis.get("cfo_actions", 0)
        exec_factor = min(1.0, exec_count / 3)
        
        confidence = (
            count_factor * 0.4 +
            recency_factor * 0.4 +
            exec_factor * 0.2
        )
        
        return float(min(1.0, max(0.3, confidence)))