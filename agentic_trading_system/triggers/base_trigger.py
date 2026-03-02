"""
Abstract Base Trigger - Foundation for all trigger agents
All triggers MUST inherit from this class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import asyncio
import uuid
import hashlib
from pydantic import BaseModel, Field, validator
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats
import json

class TriggerPriority(Enum):
    """Priority levels for triggers"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TriggerStatus(Enum):
    """Status of trigger execution"""
    IDLE = "idle"
    RUNNING = "running"
    COOLDOWN = "cooldown"
    ERROR = "error"
    DISABLED = "disabled"

class TriggerConfig(BaseModel):
    """Base configuration for all triggers"""
    name: str
    enabled: bool = True
    priority: TriggerPriority = TriggerPriority.MEDIUM
    cooldown_seconds: int = 300  # Min time between triggers for same symbol
    max_signals_per_run: int = 10
    require_confirmation: bool = False
    min_confidence: float = 0.6
    lookback_days: int = 60  # YOUR 60-DAY REQUIREMENT!
    execution_mode: str = "realtime"  # realtime, batch, scheduled
    schedule: Optional[str] = None  # Cron expression if scheduled
    
    # Statistical parameters
    z_score_threshold: float = 2.0
    p_value_threshold: float = 0.05
    min_sample_size: int = 30
    
    # Rate limiting
    max_calls_per_minute: int = 60
    max_calls_per_hour: int = 1000
    
    class Config:
        use_enum_values = True

class TriggerEvent(BaseModel):
    """
    Standard trigger event model - ALL triggers produce these
    This is what gets sent to the fusion agent
    """
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    symbol: str
    source_trigger: str
    trigger_version: str = "1.0.0"
    event_type: str
    confidence: float = 0.5
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    processed_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistical significance (YOUR 60-DAY REQUIREMENT!)
    z_score: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    volatility: Optional[float] = None
    
    # Multi-timeframe (for fusion)
    timeframes_detected: List[str] = Field(default_factory=list)
    primary_timeframe: str = "1d"
    
    # Market context
    market_regime: Optional[str] = None
    sector: Optional[str] = None
    
    # For correlation
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Expiry
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        return max(0.0, min(1.0, v))
    
    def is_statistically_significant(self) -> bool:
        """Check if event meets statistical significance"""
        if self.p_value is None or self.z_score is None:
            return False
        return self.p_value < 0.05 and abs(self.z_score) > 2.0
    
    def should_fuse(self) -> bool:
        """Determine if this event should be considered for fusion"""
        return self.confidence >= 0.6 or self.is_statistically_significant()
    
    def dict_for_fusion(self) -> Dict[str, Any]:
        """Convert to format expected by fusion agent"""
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "source": self.source_trigger,
            "confidence": self.confidence,
            "stat_sig": self.is_statistically_significant(),
            "z_score": self.z_score,
            "timeframes": self.timeframes_detected,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }

class BaseTrigger(ABC):
    """
    ABSTRACT BASE TRIGGER - ALL triggers MUST inherit from this
    
    Provides:
    - Common configuration
    - Cooldown management
    - Rate limiting
    - Statistical utilities
    - Event creation
    - Logging
    """
    
    def __init__(self, 
                 name: str,
                 config: Dict[str, Any],
                 memory_agent=None,
                 message_bus=None):
        """
        Initialize base trigger with common functionality
        """
        self.name = name
        self.config = TriggerConfig(**config)
        self.memory = memory_agent  # For storing historical data
        self.message_bus = message_bus  # For sending events
        
        # State management
        self.status = TriggerStatus.IDLE
        self.last_run_time = None
        self.last_trigger_times: Dict[str, datetime] = {}  # Per-symbol cooldown
        self.error_count = 0
        self.total_signals_generated = 0
        
        # Rate limiting
        self.call_timestamps: List[datetime] = []
        
        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        logger.info(f"✅ Initialized {self.name} (Priority: {self.config.priority.value})")
    
    @abstractmethod
    async def scan(self) -> List[TriggerEvent]:
        """
        MAIN METHOD TO IMPLEMENT - Scan for triggers
        Returns list of trigger events
        """
        pass
    
    @abstractmethod
    async def validate(self, event: TriggerEvent) -> bool:
        """
        Validate a potential trigger event
        Returns True if event should be emitted
        """
        pass
    
    async def execute(self) -> List[TriggerEvent]:
        """
        Execute the trigger scan with all safety checks
        This is called by the orchestrator
        """
        # Check if enabled
        if not self.config.enabled:
            logger.debug(f"{self.name} is disabled")
            return []
        
        # Check cooldown
        if self.status == TriggerStatus.COOLDOWN:
            logger.debug(f"{self.name} in cooldown")
            return []
        
        # Rate limiting
        if not self._check_rate_limit():
            logger.warning(f"{self.name} rate limit exceeded")
            return []
        
        try:
            self.status = TriggerStatus.RUNNING
            self.last_run_time = datetime.utcnow()
            
            # Execute the scan (implemented by child class)
            events = await self.scan()
            
            # Validate each event
            valid_events = []
            for event in events:
                if await self.validate(event):
                    # Enrich with common data
                    event = await self._enrich_event(event)
                    valid_events.append(event)
                    
                    # Update cooldown
                    self.last_trigger_times[event.symbol] = datetime.utcnow()
                    
                    logger.info(f"🔔 {self.name} triggered for {event.symbol} "
                              f"(conf: {event.confidence:.2f})")
                else:
                    logger.debug(f"{self.name} event for {event.symbol} failed validation")
            
            # Update stats
            self.total_signals_generated += len(valid_events)
            self.error_count = 0
            
            # Store in memory if available
            if self.memory and valid_events:
                await self._store_events(valid_events)
            
            return valid_events
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ {self.name} execution error: {e}")
            
            if self.error_count > 5:
                self.status = TriggerStatus.ERROR
                logger.critical(f"{self.name} disabled due to repeated errors")
            
            return []
        
        finally:
            self.status = TriggerStatus.IDLE
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Remove old timestamps
        self.call_timestamps = [
            ts for ts in self.call_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        # Check minute limit
        if len(self.call_timestamps) >= self.config.max_calls_per_minute:
            return False
        
        # Check hour limit (simplified)
        hour_ago = now - timedelta(hours=1)
        hour_calls = sum(1 for ts in self.call_timestamps if ts > hour_ago)
        if hour_calls >= self.config.max_calls_per_hour:
            return False
        
        self.call_timestamps.append(now)
        return True
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown"""
        if symbol not in self.last_trigger_times:
            return True
        
        last_trigger = self.last_trigger_times[symbol]
        cooldown_passed = (datetime.utcnow() - last_trigger).seconds > self.config.cooldown_seconds
        
        return cooldown_passed
    
    async def _enrich_event(self, event: TriggerEvent) -> TriggerEvent:
        """Add common enrichment to all events"""
        # Add source trigger
        event.source_trigger = self.name
        
        # Add market context if available
        if self.memory:
            try:
                # Get market regime from memory
                regime = await self.memory.get("current_market_regime")
                if regime:
                    event.market_regime = regime
            except:
                pass
        
        return event
    
    async def _store_events(self, events: List[TriggerEvent]):
        """Store events in memory for future learning"""
        if not self.memory:
            return
        
        for event in events:
            await self.memory.store(
                key=f"trigger:{self.name}:{event.event_id}",
                value=event.dict(),
                tier="short"
            )
    
    def get_statistical_significance(self, 
                                    values: List[float], 
                                    current_value: float) -> Dict[str, float]:
        """
        Calculate statistical significance for a value
        THIS IS YOUR 60-DAY REQUIREMENT IMPLEMENTATION!
        """
        if len(values) < self.config.min_sample_size:
            return {
                "z_score": 0,
                "p_value": 1.0,
                "significant": False,
                "sample_size": len(values)
            }
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return {
                "z_score": 0,
                "p_value": 1.0,
                "significant": False,
                "sample_size": len(values)
            }
        
        # Z-score
        z_score = (current_value - mean) / std
        
        # Approximate p-value (simplified)
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": p_value < self.config.p_value_threshold and abs(z_score) > self.config.z_score_threshold,
            "sample_size": len(values),
            "mean": float(mean),
            "std": float(std)
        }
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            if datetime.utcnow() < self._cache_ttl.get(key, datetime.min):
                return self._cache[key]
        return None
    
    def set_cache(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache with TTL"""
        self._cache[key] = value
        self._cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    
    def get_status(self) -> Dict[str, Any]:
        """Get trigger status for monitoring"""
        return {
            "name": self.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "priority": self.config.priority.value,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "total_signals": self.total_signals_generated,
            "error_count": self.error_count,
            "cache_size": len(self._cache)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring system"""
        return {
            "name": self.name,
            "healthy": self.error_count < 5,
            "status": self.status.value,
            "last_signal": max(self.last_trigger_times.values()) if self.last_trigger_times else None,
            "memory_usage": len(self._cache) * 1024  # Approximate
        }