"""
Redis Client - Fast in-memory data store
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import pickle
import asyncio
from utils.logger import logger as  logging
import redis
REDIS_AVAILABLE = True
class RedisClient:
    """
    Redis Client for short-term memory storage
    
    Features:
    - Key-value storage with TTL
    - Pub/sub for real-time updates
    - List, set, hash operations
    - Automatic serialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Connection parameters
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.default_ttl = config.get("default_ttl_seconds", 3600)  # 1 hour
        
        # Connection pool
        self.client = None
        self.pubsub = None
        self.is_connected = False
        
        # Mock storage for development
        self.mock_storage = {}
        self.mock_ttl = {}
        self.mock_pubsub = {}
        
        # Connect
        self._connect()
        
        logging.info(f"✅ RedisClient initialized ({'connected' if self.is_connected else 'mock mode'})")
    
    def _connect(self):
        """Connect to Redis"""
        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False
                )
                self.is_connected = True
            except Exception as e:
                logging.error(f"Redis connection error: {e}")
                self.is_connected = False
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a key-value pair with optional TTL
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Serialize value
        try:
            serialized = pickle.dumps(value)
        except:
            serialized = str(value).encode()
        
        if self.is_connected and self.client:
            try:
                await self.client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logging.error(f"Redis set error: {e}")
                return False
        else:
            # Mock mode
            self.mock_storage[key] = value
            self.mock_ttl[key] = datetime.now() + timedelta(seconds=ttl)
            return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value for a key
        """
        if self.is_connected and self.client:
            try:
                data = await self.client.get(key)
                if data:
                    try:
                        return pickle.loads(data)
                    except:
                        return data.decode()
                return None
            except Exception as e:
                logging.error(f"Redis get error: {e}")
                return None
        else:
            # Mock mode with TTL check
            if key in self.mock_storage:
                if key in self.mock_ttl and datetime.now() > self.mock_ttl[key]:
                    del self.mock_storage[key]
                    del self.mock_ttl[key]
                    return None
                return self.mock_storage[key]
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key
        """
        if self.is_connected and self.client:
            try:
                result = await self.client.delete(key)
                return result > 0
            except Exception as e:
                logging.error(f"Redis delete error: {e}")
                return False
        else:
            if key in self.mock_storage:
                del self.mock_storage[key]
                if key in self.mock_ttl:
                    del self.mock_ttl[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists
        """
        if self.is_connected and self.client:
            try:
                result = await self.client.exists(key)
                return result > 0
            except Exception as e:
                logging.error(f"Redis exists error: {e}")
                return False
        else:
            if key in self.mock_storage:
                if key in self.mock_ttl and datetime.now() > self.mock_ttl[key]:
                    del self.mock_storage[key]
                    del self.mock_ttl[key]
                    return False
                return True
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiry on existing key
        """
        if self.is_connected and self.client:
            try:
                result = await self.client.expire(key, ttl)
                return result
            except Exception as e:
                logging.error(f"Redis expire error: {e}")
                return False
        else:
            if key in self.mock_storage:
                self.mock_ttl[key] = datetime.now() + timedelta(seconds=ttl)
                return True
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key
        """
        if self.is_connected and self.client:
            try:
                return await self.client.ttl(key)
            except Exception as e:
                logging.error(f"Redis ttl error: {e}")
                return -2
        else:
            if key in self.mock_ttl:
                remaining = (self.mock_ttl[key] - datetime.now()).total_seconds()
                return max(-1, int(remaining))
            return -2
    
    async def incr(self, key: str) -> int:
        """
        Increment a counter
        """
        if self.is_connected and self.client:
            try:
                return await self.client.incr(key)
            except Exception as e:
                logging.error(f"Redis incr error: {e}")
                return 0
        else:
            current = await self.get(key) or 0
            new_value = int(current) + 1
            await self.set(key, new_value)
            return new_value
    
    async def sadd(self, key: str, *values) -> int:
        """
        Add to set
        """
        if self.is_connected and self.client:
            try:
                return await self.client.sadd(key, *values)
            except Exception as e:
                logging.error(f"Redis sadd error: {e}")
                return 0
        else:
            current = await self.get(key) or set()
            if not isinstance(current, set):
                current = set()
            current.update(values)
            await self.set(key, current)
            return len(values)
    
    async def smembers(self, key: str) -> set:
        """
        Get all members of a set
        """
        if self.is_connected and self.client:
            try:
                members = await self.client.smembers(key)
                return {m.decode() if isinstance(m, bytes) else m for m in members}
            except Exception as e:
                logging.error(f"Redis smembers error: {e}")
                return set()
        else:
            result = await self.get(key)
            if isinstance(result, set):
                return result
            return set()
    
    async def lpush(self, key: str, *values) -> int:
        """
        Push to list (left)
        """
        if self.is_connected and self.client:
            try:
                return await self.client.lpush(key, *values)
            except Exception as e:
                logging.error(f"Redis lpush error: {e}")
                return 0
        else:
            current = await self.get(key) or []
            if not isinstance(current, list):
                current = []
            current = list(values) + current
            await self.set(key, current)
            return len(values)
    
    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """
        Get range from list
        """
        if self.is_connected and self.client:
            try:
                data = await self.client.lrange(key, start, end)
                return [d.decode() if isinstance(d, bytes) else d for d in data]
            except Exception as e:
                logging.error(f"Redis lrange error: {e}")
                return []
        else:
            current = await self.get(key) or []
            if not isinstance(current, list):
                return []
            return current[start:end] if end >= 0 else current[start:]
    
    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel
        """
        if self.is_connected and self.client:
            try:
                serialized = json.dumps(message, default=str)
                return await self.client.publish(channel, serialized)
            except Exception as e:
                logging.error(f"Redis publish error: {e}")
                return 0
        else:
            # Mock pub/sub
            if channel not in self.mock_pubsub:
                self.mock_pubsub[channel] = []
            self.mock_pubsub[channel].append({
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            return 1
    
    async def subscribe(self, channel: str, callback) -> bool:
        """
        Subscribe to a channel (requires separate connection)
        """
        if self.is_connected and self.client:
            try:
                self.pubsub = self.client.pubsub()
                await self.pubsub.subscribe(channel)
                
                # Start listener task
                asyncio.create_task(self._listen(channel, callback))
                return True
            except Exception as e:
                logging.error(f"Redis subscribe error: {e}")
                return False
        else:
            # Mock subscription
            logging.info(f"📡 Mock subscribed to {channel}")
            return True
    
    async def _listen(self, channel: str, callback):
        """Listen for messages on channel"""
        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = json.loads(message['data'])
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"Redis listen error: {e}")
                break
    
    async def flushall(self) -> bool:
        """
        Flush all data (use with caution)
        """
        if self.is_connected and self.client:
            try:
                await self.client.flushall()
                return True
            except Exception as e:
                logging.error(f"Redis flushall error: {e}")
                return False
        else:
            self.mock_storage.clear()
            self.mock_ttl.clear()
            self.mock_pubsub.clear()
            return True
    
    async def close(self):
        """Close connection"""
        if self.is_connected and self.client:
            await self.client.close()
            if self.pubsub:
                await self.pubsub.close()