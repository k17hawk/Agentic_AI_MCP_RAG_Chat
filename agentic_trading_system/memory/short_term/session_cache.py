"""
Session Cache - Short-term cache for current trading session
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from collections import defaultdict
from utils.logger import logger as logging
from memory.short_term.redis_client import RedisClient

class SessionCache:
    """
    Session Cache - Short-term cache for current trading session
    
    Stores:
    - Current positions
    - Today's signals
    - Recent prices
    - Session statistics
    - Pending approvals
    """
    
    def __init__(self, config: Dict[str, Any], redis_client: RedisClient = None):
        self.config = config
        
        # Redis client (optional - falls back to memory)
        self.redis = redis_client
        
        # Session namespace
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prefix = f"session:{self.session_id}"
        
        # In-memory fallback
        self.memory_cache = defaultdict(dict)
        
        # Session start time
        self.session_start = datetime.now()
        
        # Statistics
        self.stats = {
            "signals_today": 0,
            "trades_today": 0,
            "approvals_pending": 0,
            "approvals_completed": 0
        }
        
        logging.info(f"✅ SessionCache initialized (session: {self.session_id})")
    
    def _key(self, *parts) -> str:
        """Build Redis key"""
        return f"{self.prefix}:{':'.join(str(p) for p in parts)}"
    
    async def set(self, namespace: str, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in the session cache
        """
        full_key = self._key(namespace, key)
        
        if self.redis:
            return await self.redis.set(full_key, value, ttl)
        else:
            self.memory_cache[namespace][key] = {
                "value": value,
                "expires": datetime.now() + timedelta(seconds=ttl or 3600) if ttl else None
            }
            return True
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a value from the session cache
        """
        full_key = self._key(namespace, key)
        
        if self.redis:
            return await self.redis.get(full_key)
        else:
            if namespace in self.memory_cache and key in self.memory_cache[namespace]:
                entry = self.memory_cache[namespace][key]
                if entry.get("expires") and datetime.now() > entry["expires"]:
                    del self.memory_cache[namespace][key]
                    return None
                return entry["value"]
            return None
    
    async def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a value from the session cache
        """
        full_key = self._key(namespace, key)
        
        if self.redis:
            return await self.redis.delete(full_key)
        else:
            if namespace in self.memory_cache and key in self.memory_cache[namespace]:
                del self.memory_cache[namespace][key]
                return True
            return False
    
    async def get_all(self, namespace: str) -> Dict[str, Any]:
        """
        Get all values in a namespace
        """
        if self.redis:
            # Would need to use SCAN in production
            return {}
        else:
            if namespace in self.memory_cache:
                # Filter expired
                current = {}
                for key, entry in self.memory_cache[namespace].items():
                    if entry.get("expires") and datetime.now() > entry["expires"]:
                        continue
                    current[key] = entry["value"]
                return current
            return {}
    
    # Position management
    async def set_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Store current position"""
        return await self.set("positions", symbol, position)
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position"""
        return await self.get("positions", symbol)
    
    async def get_all_positions(self) -> Dict[str, Any]:
        """Get all positions"""
        return await self.get_all("positions")
    
    async def remove_position(self, symbol: str) -> bool:
        """Remove a position"""
        return await self.delete("positions", symbol)
    
    # Signal management
    async def add_signal(self, signal_id: str, signal: Dict[str, Any]) -> bool:
        """Store a signal"""
        result = await self.set("signals", signal_id, signal, ttl=86400)  # 24 hours
        if result:
            self.stats["signals_today"] += 1
        return result
    
    async def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a signal"""
        return await self.get("signals", signal_id)
    
    async def get_today_signals(self) -> List[Dict[str, Any]]:
        """Get all signals from today"""
        signals = await self.get_all("signals")
        return list(signals.values())
    
    # Price cache
    async def set_price(self, symbol: str, price: float, ttl: int = 60) -> bool:
        """Cache current price (short TTL)"""
        return await self.set("prices", symbol, {"price": price, "timestamp": datetime.now().isoformat()}, ttl)
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price"""
        data = await self.get("prices", symbol)
        return data.get("price") if data else None
    
    async def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices"""
        prices = await self.get_all("prices")
        return {k: v["price"] for k, v in prices.items()}
    
    # Pending approvals
    async def add_pending_approval(self, approval_id: str, data: Dict[str, Any]) -> bool:
        """Add a pending approval"""
        result = await self.set("pending", approval_id, data, ttl=300)  # 5 minutes
        if result:
            self.stats["approvals_pending"] += 1
        return result
    
    async def get_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Get a pending approval"""
        return await self.get("pending", approval_id)
    
    async def remove_pending_approval(self, approval_id: str) -> bool:
        """Remove a pending approval"""
        result = await self.delete("pending", approval_id)
        if result:
            self.stats["approvals_pending"] = max(0, self.stats["approvals_pending"] - 1)
            self.stats["approvals_completed"] += 1
        return result
    
    async def get_all_pending(self) -> Dict[str, Any]:
        """Get all pending approvals"""
        return await self.get_all("pending")
    
    # Session statistics
    async def increment_stat(self, stat_name: str, amount: int = 1) -> int:
        """Increment a session statistic"""
        current = self.stats.get(stat_name, 0)
        self.stats[stat_name] = current + amount
        await self.set("stats", stat_name, self.stats[stat_name])
        return self.stats[stat_name]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        # Add duration
        duration = (datetime.now() - self.session_start).total_seconds()
        stats = self.stats.copy()
        stats["session_duration_seconds"] = duration
        stats["session_id"] = self.session_id
        stats["session_start"] = self.session_start.isoformat()
        return stats
    
    # Cleanup
    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all values in a namespace"""
        if self.redis:
            # Would need to use SCAN and DEL in production
            return False
        else:
            if namespace in self.memory_cache:
                self.memory_cache[namespace].clear()
                return True
            return False
    
    async def clear_all(self) -> bool:
        """Clear entire session cache"""
        if self.redis:
            # Would need to use SCAN and DEL
            return False
        else:
            self.memory_cache.clear()
            self.stats = {
                "signals_today": 0,
                "trades_today": 0,
                "approvals_pending": 0,
                "approvals_completed": 0
            }
            return True