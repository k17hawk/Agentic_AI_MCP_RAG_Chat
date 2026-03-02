"""
Query Engine - Unified query interface across all memory tiers
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import re
from utils.logger import logger as logging

class QueryEngine:
    """
    Query Engine - Unified query interface across all memory tiers
    
    Features:
    - Natural language-like queries
    - Multi-tier searches
    - Aggregation functions
    - Time-based queries
    - Filtering and sorting
    """
    
    def __init__(self, config: Dict[str, Any], memory_orchestrator):
        self.config = config
        self.memory = memory_orchestrator
        
        # Query patterns
        self.patterns = {
            "get_trades": re.compile(r"get\s+trades?\s*(?:for\s+(\w+))?\s*(?:from\s+([\d-]+))?\s*(?:to\s+([\d-]+))?", re.IGNORECASE),
            "get_signals": re.compile(r"get\s+signals?\s*(?:for\s+(\w+))?\s*(?:from\s+([\d-]+))?\s*(?:to\s+([\d-]+))?", re.IGNORECASE),
            "get_performance": re.compile(r"get\s+performance\s*(?:for\s+(\d+)\s*days?)?", re.IGNORECASE),
            "get_positions": re.compile(r"get\s+positions?\s*(?:for\s+(\w+))?", re.IGNORECASE),
            "get_pnl": re.compile(r"get\s+pnl\s*(?:for\s+(\d+)\s*days?)?", re.IGNORECASE),
            "get_win_rate": re.compile(r"get\s+win\s*rate\s*(?:for\s+(\d+)\s*days?)?", re.IGNORECASE),
            "get_best_trades": re.compile(r"get\s+best\s+trades?\s*(\d+)?", re.IGNORECASE),
            "get_worst_trades": re.compile(r"get\s+worst\s+trades?\s*(\d+)?", re.IGNORECASE),
            "get_signal_accuracy": re.compile(r"get\s+signal\s+accuracy\s*(?:for\s+(\w+))?", re.IGNORECASE),
            "get_model_weights": re.compile(r"get\s+model\s+weights\s*(?:for\s+(\w+))?", re.IGNORECASE),
            "search": re.compile(r"search\s+(.+?)\s*(?:in\s+(\w+))?\s*(?:from\s+([\d-]+))?\s*(?:to\s+([\d-]+))?", re.IGNORECASE)
        }
        
        logging.info(f"✅ QueryEngine initialized")
    
    async def query(self, query_string: str, params: Dict = None) -> Dict[str, Any]:
        """
        Execute a query (natural language or structured)
        """
        params = params or {}
        result = {
            "query": query_string,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "data": None,
            "error": None
        }
        
        # Try to parse natural language
        parsed = self._parse_query(query_string)
        
        if parsed["type"]:
            # Execute parsed query
            try:
                data = await self._execute_parsed(parsed, params)
                result["success"] = True
                result["data"] = data
                result["parsed"] = parsed
            except Exception as e:
                result["error"] = str(e)
                logging.error(f"Query execution error: {e}")
        else:
            # Fall back to structured query
            try:
                data = await self._execute_structured(query_string, params)
                result["success"] = True
                result["data"] = data
            except Exception as e:
                result["error"] = str(e)
                logging.error(f"Structured query error: {e}")
        
        return result
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query
        """
        result = {
            "type": None,
            "target": None,
            "symbol": None,
            "start_date": None,
            "end_date": None,
            "limit": None,
            "filters": {}
        }
        
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(query)
            if match:
                result["type"] = pattern_name
                
                if pattern_name == "get_trades":
                    result["symbol"] = match.group(1)
                    result["start_date"] = match.group(2)
                    result["end_date"] = match.group(3)
                
                elif pattern_name == "get_signals":
                    result["symbol"] = match.group(1)
                    result["start_date"] = match.group(2)
                    result["end_date"] = match.group(3)
                
                elif pattern_name == "get_performance":
                    days = match.group(1)
                    result["days"] = int(days) if days else 30
                
                elif pattern_name == "get_positions":
                    result["symbol"] = match.group(1)
                
                elif pattern_name == "get_pnl":
                    days = match.group(1)
                    result["days"] = int(days) if days else 30
                
                elif pattern_name == "get_win_rate":
                    days = match.group(1)
                    result["days"] = int(days) if days else 30
                
                elif pattern_name == "get_best_trades":
                    limit = match.group(1)
                    result["limit"] = int(limit) if limit else 10
                
                elif pattern_name == "get_worst_trades":
                    limit = match.group(1)
                    result["limit"] = int(limit) if limit else 10
                
                elif pattern_name == "get_signal_accuracy":
                    result["signal_type"] = match.group(1)
                
                elif pattern_name == "get_model_weights":
                    result["model_name"] = match.group(1)
                
                elif pattern_name == "search":
                    result["search_term"] = match.group(1)
                    result["target"] = match.group(2)
                    result["start_date"] = match.group(3)
                    result["end_date"] = match.group(4)
                
                break
        
        return result
    
    async def _execute_parsed(self, parsed: Dict, params: Dict) -> Any:
        """
        Execute parsed query
        """
        query_type = parsed["type"]
        
        if query_type == "get_trades":
            return await self.get_trades(
                symbol=parsed["symbol"],
                start_date=parsed["start_date"],
                end_date=parsed["end_date"],
                **params
            )
        
        elif query_type == "get_signals":
            return await self.get_signals(
                symbol=parsed["symbol"],
                start_date=parsed["start_date"],
                end_date=parsed["end_date"],
                **params
            )
        
        elif query_type == "get_performance":
            return await self.get_performance(
                days=parsed.get("days", 30),
                **params
            )
        
        elif query_type == "get_positions":
            return await self.get_positions(
                symbol=parsed["symbol"],
                **params
            )
        
        elif query_type == "get_pnl":
            return await self.get_pnl(
                days=parsed.get("days", 30),
                **params
            )
        
        elif query_type == "get_win_rate":
            return await self.get_win_rate(
                days=parsed.get("days", 30),
                **params
            )
        
        elif query_type == "get_best_trades":
            return await self.get_best_trades(
                limit=parsed.get("limit", 10),
                **params
            )
        
        elif query_type == "get_worst_trades":
            return await self.get_worst_trades(
                limit=parsed.get("limit", 10),
                **params
            )
        
        elif query_type == "get_signal_accuracy":
            return await self.get_signal_accuracy(
                signal_type=parsed["signal_type"],
                **params
            )
        
        elif query_type == "get_model_weights":
            return await self.get_model_weights(
                model_name=parsed["model_name"],
                **params
            )
        
        elif query_type == "search":
            return await self.search(
                term=parsed["search_term"],
                target=parsed["target"],
                start_date=parsed["start_date"],
                end_date=parsed["end_date"],
                **params
            )
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    async def _execute_structured(self, query: str, params: Dict) -> Any:
        """
        Execute structured query (JSON-like)
        """
        # Handle dot notation queries
        if query.startswith("trades."):
            parts = query.split(".")
            if len(parts) == 2:
                return await self.get_trades(symbol=parts[1])
            elif len(parts) == 3 and parts[2] == "recent":
                return await self.get_recent_trades(limit=params.get("limit", 10))
        
        elif query.startswith("signals."):
            parts = query.split(".")
            if len(parts) == 2:
                return await self.get_signals(symbol=parts[1])
            elif len(parts) == 3 and parts[2] == "active":
                return await self.get_active_signals()
        
        elif query == "performance.daily":
            return await self.get_performance(days=1)
        
        elif query == "performance.weekly":
            return await self.get_performance(days=7)
        
        elif query == "performance.monthly":
            return await self.get_performance(days=30)
        
        elif query == "positions.all":
            return await self.get_positions()
        
        elif query == "pnl.today":
            return await self.get_pnl(days=1)
        
        elif query == "pnl.week":
            return await self.get_pnl(days=7)
        
        elif query == "pnl.month":
            return await self.get_pnl(days=30)
        
        elif query == "stats.win_rate":
            return await self.get_win_rate()
        
        elif query == "stats.signal_accuracy":
            return await self.get_signal_accuracy()
        
        elif query == "models.active":
            return await self.get_active_models()
        
        # Pass through to memory orchestrator
        return await self.memory.query(query, params)
    
    # Query implementations
    
    async def get_trades(self, symbol: str = None, start_date: str = None,
                        end_date: str = None, limit: int = 100,
                        sort_by: str = "entry_time", sort_order: str = "desc") -> List[Dict]:
        """Get trades with filters"""
        trades = []
        
        if symbol:
            trades = self.memory.trade_repo.get_by_symbol(symbol, limit)
        elif start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            trades = self.memory.trade_repo.get_by_date_range(start, end)
        else:
            trades = self.memory.trade_repo.get_recent(limit)
        
        # Convert to dict
        result = [t.dict() for t in trades]
        
        # Sort
        reverse = sort_order.lower() == "desc"
        result.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)
        
        return result[:limit]
    
    async def get_signals(self, symbol: str = None, signal_type: str = None,
                         start_date: str = None, end_date: str = None,
                         limit: int = 100) -> List[Dict]:
        """Get signals with filters"""
        signals = []
        
        if symbol:
            signals = self.memory.signal_repo.get_by_symbol(symbol, limit)
        elif signal_type:
            signals = self.memory.signal_repo.get_by_type(signal_type, limit)
        elif start_date and end_date:
            # This would need date range implementation
            pass
        else:
            signals = self.memory.signal_repo.get_recent(limit)
        
        return [s.dict() for s in signals]
    
    async def get_active_signals(self) -> List[Dict]:
        """Get active (non-expired) signals"""
        signals = self.memory.signal_repo.get_active_signals()
        return [s.dict() for s in signals]
    
    async def get_performance(self, days: int = 30, metrics: List[str] = None) -> Dict:
        """Get performance metrics"""
        performance = await self.memory.get_performance(days)
        
        if metrics:
            # Filter to requested metrics
            filtered = {}
            for metric in metrics:
                if metric in performance:
                    filtered[metric] = performance[metric]
            return filtered
        
        return performance
    
    async def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get current positions"""
        if symbol:
            position = await self.memory.session.get_position(symbol)
            return [position] if position else []
        else:
            positions = await self.memory.session.get_all_positions()
            return list(positions.values())
    
    async def get_pnl(self, days: int = 30) -> Dict[str, Any]:
        """Get P&L summary"""
        performance = await self.memory.get_performance(days)
        
        return {
            "period_days": days,
            "net_pnl": performance.get("current", {}).get("net_pnl", 0),
            "gross_profit": performance.get("current", {}).get("gross_profit", 0),
            "gross_loss": performance.get("current", {}).get("gross_loss", 0),
            "profit_factor": performance.get("current", {}).get("profit_factor", 0),
            "best_trade": await self._get_best_trade(days),
            "worst_trade": await self._get_worst_trade(days)
        }
    
    async def get_win_rate(self, days: int = 30) -> Dict[str, Any]:
        """Get win rate statistics"""
        performance = await self.memory.get_performance(days)
        current = performance.get("current", {})
        
        return {
            "period_days": days,
            "win_rate": current.get("win_rate", 0),
            "wins": current.get("winning_trades", 0),
            "losses": current.get("losing_trades", 0),
            "total_trades": current.get("total_trades", 0)
        }
    
    async def get_best_trades(self, limit: int = 10, metric: str = "pnl") -> List[Dict]:
        """Get best performing trades"""
        trades = self.memory.trade_repo.get_recent(1000)  # Get more to find best
        
        # Filter trades with P&L
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        
        if metric == "pnl":
            trades_with_pnl.sort(key=lambda x: x.pnl or 0, reverse=True)
        elif metric == "return":
            trades_with_pnl.sort(key=lambda x: x.pnl_percent or 0, reverse=True)
        
        return [t.dict() for t in trades_with_pnl[:limit]]
    
    async def get_worst_trades(self, limit: int = 10, metric: str = "pnl") -> List[Dict]:
        """Get worst performing trades"""
        trades = self.memory.trade_repo.get_recent(1000)
        
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        
        if metric == "pnl":
            trades_with_pnl.sort(key=lambda x: x.pnl or 0)
        elif metric == "return":
            trades_with_pnl.sort(key=lambda x: x.pnl_percent or 0)
        
        return [t.dict() for t in trades_with_pnl[:limit]]
    
    async def get_signal_accuracy(self, signal_type: str = None) -> Dict[str, Any]:
        """Get signal accuracy statistics"""
        if signal_type:
            accuracy = self.memory.signal_repo.get_accuracy_by_type()
            return {signal_type: accuracy.get(signal_type, 0)}
        else:
            return self.memory.signal_repo.get_accuracy_by_type()
    
    async def get_model_weights(self, model_name: str = None) -> Dict[str, Any]:
        """Get model weights"""
        if model_name:
            weights = self.memory.weights_repo.get_active(model_name)
            return weights.dict() if weights else {}
        else:
            # Get all active models
            # This would need a method to list models
            return {}
    
    async def get_active_models(self) -> List[str]:
        """Get list of active models"""
        # This would need implementation in weights repo
        return []
    
    async def search(self, term: str, target: str = None, 
                    start_date: str = None, end_date: str = None,
                    limit: int = 50) -> List[Dict]:
        """Search across memory tiers"""
        results = []
        term_lower = term.lower()
        
        # Search trades
        if not target or target == "trades":
            trades = self.memory.trade_repo.get_recent(1000)
            for trade in trades:
                if (term_lower in trade.symbol.lower() or 
                    term_lower in trade.strategy.lower() or
                    (trade.notes and term_lower in trade.notes.lower())):
                    results.append({"type": "trade", "data": trade.dict()})
        
        # Search signals
        if not target or target == "signals":
            signals = self.memory.signal_repo.get_recent(1000)
            for signal in signals:
                if (term_lower in signal.symbol.lower() or
                    term_lower in signal.signal_type.lower() or
                    term_lower in signal.source.lower()):
                    results.append({"type": "signal", "data": signal.dict()})
        
        # Filter by date if specified
        if start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            results = [
                r for r in results
                if start <= datetime.fromisoformat(r["data"].get("entry_time", r["data"].get("generated_at", "2000-01-01"))) <= end
            ]
        
        return results[:limit]
    
    async def _get_best_trade(self, days: int) -> Optional[Dict]:
        """Get best trade in period"""
        end = datetime.now()
        start = end - timedelta(days=days)
        trades = self.memory.trade_repo.get_by_date_range(start, end)
        
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        if not trades_with_pnl:
            return None
        
        best = max(trades_with_pnl, key=lambda x: x.pnl or 0)
        return best.dict()
    
    async def _get_worst_trade(self, days: int) -> Optional[Dict]:
        """Get worst trade in period"""
        end = datetime.now()
        start = end - timedelta(days=days)
        trades = self.memory.trade_repo.get_by_date_range(start, end)
        
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        if not trades_with_pnl:
            return None
        
        worst = min(trades_with_pnl, key=lambda x: x.pnl or 0)
        return worst.dict()
    
    def explain(self, query_result: Dict) -> str:
        """Generate human-readable explanation of query result"""
        if not query_result["success"]:
            return f"Query failed: {query_result.get('error', 'Unknown error')}"
        
        data = query_result["data"]
        query = query_result["query"]
        
        if isinstance(data, list):
            if len(data) == 0:
                return f"No results found for '{query}'"
            elif len(data) == 1:
                return f"Found 1 result for '{query}'"
            else:
                return f"Found {len(data)} results for '{query}'"
        
        elif isinstance(data, dict):
            # Try to generate meaningful summary
            if "net_pnl" in data:
                return f"P&L: ${data['net_pnl']:,.2f} over {data.get('period_days', '?')} days"
            elif "win_rate" in data:
                return f"Win rate: {data['win_rate']*100:.1f}% ({data.get('wins', 0)}W/{data.get('losses', 0)}L)"
            elif "total_trades" in data:
                return f"Total trades: {data['total_trades']}"
            else:
                return f"Found data for '{query}'"
        
        return f"Query executed successfully"