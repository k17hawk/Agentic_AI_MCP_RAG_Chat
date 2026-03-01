"""
Trigger Fusion Agent - Combines multiple trigger signals
Correlates events by symbol, time, and statistical significance
"""
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import asyncio
import uuid
from collections import defaultdict
from logger import logging as logger
from typing import Any
import numpy as np
from pydantic import BaseModel,Field
from triggers.base_trigger import TriggerEvent, TriggerPriority

class FusionConfig:
    """Configuration for fusion engine"""
    CORRELATION_WINDOW_SECONDS = 300  # 5 minutes
    MIN_CONFIDENCE_FOR_FUSION = 0.6
    REQUIRED_SOURCES_FOR_HIGH_CONFIDENCE = 2
    TIME_DECAY_FACTOR = 0.1  # How fast confidence decays over time
    
class FusedSignal(BaseModel):
    """
    A signal created by fusing multiple trigger events
    """
    fusion_id: str = Field(default_factory=lambda: f"fus_{uuid.uuid4().hex[:12]}")
    symbol: str
    source_events: List[str]  # Event IDs
    source_triggers: List[str]  # Trigger names
    confidence: float
    market_regime: Optional[str]
    primary_timeframe: str
    timeframes_covered: List[str]
    statistically_significant: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))
    
    # Analysis
    consensus_level: float  # How much sources agree (0-1)
    average_z_score: float
    max_confidence: float
    min_confidence: float
    
    def should_execute(self) -> bool:
        """Determine if this fused signal should trigger execution"""
        return (
            self.confidence >= 0.75 and
            len(self.source_triggers) >= 2 and
            self.consensus_level >= 0.6
        )

class TriggerFusion:
    """
    Fuses multiple trigger events to create higher confidence signals
    
    How it works:
    1. Collects events from all triggers within a time window
    2. Groups by symbol and correlation_id
    3. Calculates combined confidence based on:
       - Number of sources
       - Individual confidences
       - Statistical significance
       - Time proximity
    4. Emits fused signals for execution engine
    """
    
    def __init__(self, message_bus=None, memory_agent=None):
        self.message_bus = message_bus
        self.memory = memory_agent
        
        # Event storage
        self.pending_events: Dict[str, List[TriggerEvent]] = defaultdict(list)
        self.processed_events: Set[str] = set()
        self.fused_signals: Dict[str, FusedSignal] = {}
        
        # Configuration
        self.window_seconds = FusionConfig.CORRELATION_WINDOW_SECONDS
        self.min_confidence = FusionConfig.MIN_CONFIDENCE_FOR_FUSION
        
        # Control
        self.is_running = False
        self._processor_task = None
        
        logger.info("✅ TriggerFusion initialized")
    
    async def start(self):
        """Start the fusion processor"""
        self.is_running = True
        self._processor_task = asyncio.create_task(self._fusion_loop())
        logger.info("🚀 Fusion processor started")
    
    async def stop(self):
        """Stop the fusion processor"""
        self.is_running = False
        if self._processor_task:
            self._processor_task.cancel()
        logger.info("🛑 Fusion processor stopped")
    
    async def add_event(self, event: TriggerEvent):
        """Add a new trigger event to the fusion engine"""
        if not event.should_fuse():
            logger.debug(f"Event {event.event_id} does not meet fusion criteria")
            return
        
        # Store event
        self.pending_events[event.symbol].append(event)
        
        # Clean old events
        await self._clean_old_events()
        
        logger.debug(f"Added event {event.event_id} for {event.symbol} "
                    f"(pending: {len(self.pending_events[event.symbol])})")
    
    async def _fusion_loop(self):
        """Main fusion processing loop"""
        while self.is_running:
            try:
                # Process each symbol
                for symbol in list(self.pending_events.keys()):
                    events = self.pending_events[symbol]
                    
                    if len(events) >= 2:
                        # Try to fuse events
                        fused = await self._fuse_events(symbol, events)
                        
                        if fused:
                            # Emit fused signal
                            await self._emit_fused_signal(fused)
                            
                            # Mark events as processed
                            for event in events:
                                self.processed_events.add(event.event_id)
                            
                            # Clear pending
                            del self.pending_events[symbol]
                    
                    # Check for timeouts
                    elif events:
                        oldest = min(e.timestamp for e in events)
                        if datetime.utcnow() - oldest > timedelta(seconds=self.window_seconds):
                            # Window expired - emit best single event if good enough
                            best_event = max(events, key=lambda e: e.confidence)
                            if best_event.confidence >= 0.8:  # High confidence single
                                await self._emit_single_event(best_event)
                            
                            # Clear
                            del self.pending_events[symbol]
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fusion loop error: {e}")
                await asyncio.sleep(5)
    
    async def _fuse_events(self, symbol: str, events: List[TriggerEvent]) -> Optional[FusedSignal]:
        """
        Fuse multiple events for the same symbol
        """
        if len(events) < 2:
            return None
        
        # Check if events are within time window
        timestamps = [e.timestamp for e in events]
        time_span = max(timestamps) - min(timestamps)
        if time_span.seconds > self.window_seconds:
            return None
        
        # Calculate fusion metrics
        confidences = [e.confidence for e in events]
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        # Calculate consensus (how much they agree)
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0
        consensus = 1 - confidence_std  # Lower std = higher consensus
        
        # Time decay (more recent events weighted higher)
        now = datetime.utcnow()
        time_weights = [
            np.exp(-FusionConfig.TIME_DECAY_FACTOR * (now - e.timestamp).seconds / 60)
            for e in events
        ]
        weighted_confidence = np.average(confidences, weights=time_weights)
        
        # Count statistically significant events
        sig_count = sum(1 for e in events if e.is_statistically_significant())
        
        # Final confidence calculation
        final_confidence = (
            weighted_confidence * 0.4 +
            (len(events) / 5) * 0.2 +  # More sources = better
            (sig_count / len(events)) * 0.2 +  # Statistical significance
            consensus * 0.2  # Agreement level
        )
        
        final_confidence = min(1.0, final_confidence)
        
        # Only fuse if confidence is high enough
        if final_confidence < self.min_confidence:
            return None
        
        # Collect timeframes
        timeframes = set()
        for e in events:
            timeframes.update(e.timeframes_detected)
        
        # Create fused signal
        fused = FusedSignal(
            symbol=symbol,
            source_events=[e.event_id for e in events],
            source_triggers=list(set(e.source_trigger for e in events)),
            confidence=float(final_confidence),
            market_regime=events[0].market_regime,  # Use first event's regime
            primary_timeframe=events[0].primary_timeframe,
            timeframes_covered=list(timeframes),
            statistically_significant=sig_count > 0,
            consensus_level=float(consensus),
            average_z_score=float(np.mean([e.z_score for e in events if e.z_score])),
            max_confidence=float(max_confidence),
            min_confidence=float(min_confidence)
        )
        
        logger.info(f"🎯 FUSED: {symbol} from {len(events)} sources "
                   f"→ confidence: {final_confidence:.2f}")
        
        return fused
    
    async def _emit_fused_signal(self, fused: FusedSignal):
        """Emit a fused signal to the execution engine"""
        # Store
        self.fused_signals[fused.fusion_id] = fused
        
        # Send to message bus
        if self.message_bus:
            await self.message_bus.publish(
                topic="fused_signals",
                message=fused.dict()
            )
            
            # Send to execution engine
            await self.message_bus.send_to_agent(
                agent_name="ExecutionEngine",
                message={
                    "type": "fused_signal",
                    "signal": fused.dict()
                }
            )
        
        # Store in memory
        if self.memory:
            await self.memory.store(
                key=f"fused_signal:{fused.fusion_id}",
                value=fused.dict(),
                tier="short"
            )
        
        # Log
        if fused.should_execute():
            logger.info(f"🚀 EXECUTION READY: {fused.symbol} - "
                       f"Confidence: {fused.confidence:.2f}")
    
    async def _emit_single_event(self, event: TriggerEvent):
        """Emit a single high-confidence event"""
        if self.message_bus:
            await self.message_bus.publish(
                topic="high_confidence_singles",
                message=event.dict_for_fusion()
            )
    
    async def _clean_old_events(self):
        """Remove old events from pending storage"""
        now = datetime.utcnow()
        
        for symbol in list(self.pending_events.keys()):
            events = self.pending_events[symbol]
            
            # Keep only events within window
            fresh_events = [
                e for e in events
                if (now - e.timestamp).seconds < self.window_seconds
            ]
            
            if fresh_events:
                self.pending_events[symbol] = fresh_events
            else:
                del self.pending_events[symbol]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        return {
            "pending_symbols": len(self.pending_events),
            "pending_events": sum(len(e) for e in self.pending_events.values()),
            "fused_signals": len(self.fused_signals),
            "processed_events": len(self.processed_events)
        }