"""
Memory Orchestrator - Main orchestrator for all memory tiers
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from utils.logger import logger as logging
from agents.base_agent import BaseAgent, AgentMessage

# Import all memory components
from memory.models import Trade, Signal, PerformanceMetrics, ModelWeights
from memory.repositories.trade_repository import TradeRepository
from memory.repositories.signal_repository import SignalRepository
from memory.repositories.performance_repository import PerformanceRepository
from memory.repositories.model_weights_repository import ModelWeightsRepository
from memory.short_term.redis_client import RedisClient
from memory.short_term.session_cache import SessionCache

class MemoryOrchestrator(BaseAgent):
    """
    Memory Orchestrator - Main orchestrator for all memory tiers
    
    Responsibilities:
    - Coordinate between memory tiers
    - Provide unified query interface
    - Handle data retention policies
    - Manage backup and archiving
    - Monitor memory usage
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Tiered memory management system",
            config=config
        )
        
        # Initialize repositories (medium-term)
        self.trade_repo = TradeRepository(config.get("trade_repo_config", {}))
        self.signal_repo = SignalRepository(config.get("signal_repo_config", {}))
        self.performance_repo = PerformanceRepository(config.get("performance_repo_config", {}))
        self.weights_repo = ModelWeightsRepository(config.get("weights_repo_config", {}))
        
        # Initialize short-term memory
        self.redis = RedisClient(config.get("redis_config", {}))
        self.session = SessionCache(config.get("session_config", {}), self.redis)
        
        # Long-term storage (S3, etc.) - placeholder
        self.long_term_enabled = config.get("long_term_enabled", False)
        self.archive_days = config.get("archive_days", 90)
        
        # Statistics
        self.stats = {
            "short_term_ops": 0,
            "medium_term_ops": 0,
            "long_term_ops": 0,
            "errors": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
        
        logging.info(f"✅ MemoryOrchestrator initialized")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired()
        
        asyncio.create_task(cleanup_loop())
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process memory-related requests
        """
        msg_type = message.message_type
        
        # Short-term memory operations
        if msg_type == "cache_set":
            result = await self.cache_set(
                message.content.get("key"),
                message.content.get("value"),
                message.content.get("ttl")
            )
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="cache_set_result",
                content={"success": result}
            )
        
        elif msg_type == "cache_get":
            value = await self.cache_get(message.content.get("key"))
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="cache_get_result",
                content={"value": value}
            )
        
        # Trade operations
        elif msg_type == "save_trade":
            trade_data = message.content
            trade = Trade(**trade_data)
            trade_id = await self.save_trade(trade)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="trade_saved",
                content={"trade_id": trade_id}
            )
        
        elif msg_type == "get_trade":
            trade = await self.get_trade(message.content.get("trade_id"))
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="trade_data",
                content=trade.dict() if trade else None
            )
        
        elif msg_type == "get_recent_trades":
            trades = await self.get_recent_trades(message.content.get("limit", 100))
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="recent_trades",
                content={"trades": [t.dict() for t in trades]}
            )
        
        # Signal operations
        elif msg_type == "save_signal":
            signal_data = message.content
            signal = Signal(**signal_data)
            signal_id = await self.save_signal(signal)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="signal_saved",
                content={"signal_id": signal_id}
            )
        
        elif msg_type == "update_signal_outcome":
            updated = await self.update_signal_outcome(
                message.content.get("signal_id"),
                message.content.get("led_to_trade"),
                message.content.get("trade_id")
            )
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="signal_updated",
                content={"success": updated}
            )
        
        # Performance operations
        elif msg_type == "get_performance":
            days = message.content.get("days", 30)
            performance = await self.get_performance(days)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="performance_data",
                content=performance
            )
        
        # Model weights operations
        elif msg_type == "save_weights":
            weights_data = message.content
            weights = ModelWeights(**weights_data)
            weights_id = await self.save_weights(weights)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="weights_saved",
                content={"weights_id": weights_id}
            )
        
        elif msg_type == "get_active_weights":
            weights = await self.get_active_weights(message.content.get("model_name"))
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="active_weights",
                content=weights.dict() if weights else None
            )
        
        # Session operations
        elif msg_type == "get_session_stats":
            stats = await self.session.get_stats()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="session_stats",
                content=stats
            )
        
        # Query operations
        elif msg_type == "query":
            query = message.content.get("query")
            params = message.content.get("params", {})
            results = await self.query(query, params)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="query_results",
                content={"results": results}
            )
        
        return None
    
    # Short-term memory methods
    async def cache_set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in short-term cache"""
        self.stats["short_term_ops"] += 1
        return await self.session.set("global", key, value, ttl)
    
    async def cache_get(self, key: str) -> Any:
        """Get value from short-term cache"""
        self.stats["short_term_ops"] += 1
        return await self.session.get("global", key)
    
    # Trade methods
    async def save_trade(self, trade: Trade) -> str:
        """Save a trade to medium-term storage"""
        self.stats["medium_term_ops"] += 1
        trade_id = self.trade_repo.save(trade)
        
        # Also cache in short-term for quick access
        await self.session.set("trades", trade_id, trade.dict(), ttl=86400)
        
        return trade_id
    
    async def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID"""
        self.stats["medium_term_ops"] += 1
        
        # Try short-term first
        cached = await self.session.get("trades", trade_id)
        if cached:
            return Trade(**cached)
        
        # Fall back to repository
        return self.trade_repo.get(trade_id)
    
    async def get_recent_trades(self, limit: int = 100) -> List[Trade]:
        """Get recent trades"""
        self.stats["medium_term_ops"] += 1
        return self.trade_repo.get_recent(limit)
    
    async def update_trade_outcome(self, trade_id: str, pnl: float, exit_time: datetime) -> bool:
        """Update trade outcome"""
        self.stats["medium_term_ops"] += 1
        return self.trade_repo.update_outcome(trade_id, pnl, exit_time)
    
    # Signal methods
    async def save_signal(self, signal: Signal) -> str:
        """Save a signal"""
        self.stats["medium_term_ops"] += 1
        signal_id = self.signal_repo.save(signal)
        
        # Cache in short-term
        await self.session.add_signal(signal_id, signal.dict())
        
        return signal_id
    
    async def update_signal_outcome(self, signal_id: str, led_to_trade: bool, trade_id: str = None) -> bool:
        """Update signal outcome"""
        self.stats["medium_term_ops"] += 1
        return self.signal_repo.update_outcome(signal_id, led_to_trade, trade_id)
    
    async def get_best_signals(self, min_samples: int = 10) -> List[Dict]:
        """Get best performing signal types"""
        self.stats["medium_term_ops"] += 1
        return self.signal_repo.get_best_performing_signals(min_samples)
    
    # Performance methods
    async def get_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics"""
        self.stats["medium_term_ops"] += 1
        
        # Get trades from period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        trades = self.trade_repo.get_by_date_range(start_date, end_date)
        
        # Calculate metrics
        metrics = self.performance_repo.calculate_performance(trades, start_date, end_date)
        
        # Add historical comparison
        previous_trades = self.trade_repo.get_by_date_range(
            start_date - timedelta(days=days),
            start_date
        )
        previous_metrics = self.performance_repo.calculate_performance(
            previous_trades,
            start_date - timedelta(days=days),
            start_date
        )
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "current": metrics.dict(),
            "previous": previous_metrics.dict() if previous_trades else None,
            "improvement": {
                "pnl": metrics.net_pnl - (previous_metrics.net_pnl if previous_metrics else 0),
                "win_rate": metrics.win_rate - (previous_metrics.win_rate if previous_metrics else 0)
            }
        }
    
    # Model weights methods
    async def save_weights(self, weights: ModelWeights) -> str:
        """Save model weights"""
        self.stats["medium_term_ops"] += 1
        return self.weights_repo.save(weights)
    
    async def get_active_weights(self, model_name: str) -> Optional[ModelWeights]:
        """Get active weights for a model"""
        self.stats["medium_term_ops"] += 1
        return self.weights_repo.get_active(model_name)
    
    async def set_active_weights(self, model_name: str, weights_id: str) -> bool:
        """Set active weights for a model"""
        self.stats["medium_term_ops"] += 1
        return self.weights_repo.set_active(model_name, weights_id)
    
    # Query interface
    async def query(self, query: str, params: Dict = None) -> List[Dict]:
        """
        Unified query interface across memory tiers
        Supports simple key-value queries
        """
        self.stats["medium_term_ops"] += 1
        results = []
        
        # Parse query (simplified)
        if query.startswith("trades:"):
            symbol = query.replace("trades:", "")
            trades = self.trade_repo.get_by_symbol(symbol)
            results = [t.dict() for t in trades]
        
        elif query == "recent_trades":
            trades = self.trade_repo.get_recent(params.get("limit", 100))
            results = [t.dict() for t in trades]
        
        elif query.startswith("signals:"):
            signal_type = query.replace("signals:", "")
            signals = self.signal_repo.get_by_type(signal_type)
            results = [s.dict() for s in signals]
        
        elif query == "active_signals":
            signals = self.signal_repo.get_active_signals()
            results = [s.dict() for s in signals]
        
        elif query == "performance":
            days = params.get("days", 30)
            results = [await self.get_performance(days)]
        
        elif query == "model_versions":
            model_name = params.get("model_name")
            if model_name:
                versions = self.weights_repo.get_all_versions(model_name)
                results = [v.dict() for v in versions]
        
        return results
    
    # Cleanup and maintenance
    async def _cleanup_expired(self):
        """Clean up expired data"""
        # Clean expired signals
        expired_count = self.signal_repo.cleanup_expired()
        
        # Archive old trades
        if self.long_term_enabled:
            cutoff = datetime.now() - timedelta(days=self.archive_days)
            # Archive logic here
        
        logging.info(f"🧹 Cleanup complete: {expired_count} signals expired")
    
    async def archive_old_data(self, days: int = 90):
        """Archive data older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get old trades
        old_trades = self.trade_repo.get_by_date_range(
            datetime(2000, 1, 1),  # From beginning
            cutoff
        )
        
        # Archive logic would go here
        # For now, just log
        logging.info(f"📦 Would archive {len(old_trades)} trades older than {days} days")
        
        # Delete from active storage (optional)
        for trade in old_trades:
            self.trade_repo.delete(trade.trade_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory orchestrator statistics"""
        return {
            **self.stats,
            "short_term": {
                "session_id": self.session.session_id,
                "session_duration": (datetime.now() - self.session.session_start).total_seconds(),
                "session_stats": self.session.stats
            },
            "medium_term": {
                "trades": self.trade_repo.count(),
                "signals": self.signal_repo.count(),
                "model_weights": self.weights_repo.count()
            },
            "long_term_enabled": self.long_term_enabled,
            "archive_days": self.archive_days
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        base_health = await super().health_check()
        
        # Check Redis connection
        redis_healthy = self.redis.is_connected if hasattr(self.redis, 'is_connected') else True
        
        base_health.update({
            "redis_connected": redis_healthy,
            "session_active": True,
            "total_trades": self.trade_repo.count(),
            "total_signals": self.signal_repo.count()
        })
        
        return base_health