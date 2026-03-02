"""
Trigger Orchestrator - Manages all trigger agents
Coordinates execution, priority, and resource allocation
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from enum import Enum

class TriggerPriority(Enum):
    """Priority levels for triggers"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TriggerEvent:
    """
    Simple trigger event class for internal use
    """
    def __init__(self, 
                 symbol: str,
                 source_trigger: str,
                 event_type: str,
                 confidence: float = 0.5,
                 data: dict = None):
        self.event_id = f"evt_{datetime.utcnow().timestamp()}"
        self.symbol = symbol
        self.source_trigger = source_trigger
        self.event_type = event_type
        self.confidence = confidence
        self.data = data or {}
        self.timestamp = datetime.utcnow()
    
    def dict_for_fusion(self) -> dict:
        """Convert to dict for fusion"""
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "source": self.source_trigger,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

class BaseTrigger:
    """
    Simplified base trigger for testing
    """
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.priority = TriggerPriority[config.get("priority", "MEDIUM")]
        
    async def execute(self) -> List[TriggerEvent]:
        """Execute trigger - to be overridden"""
        return []
    
    async def health_check(self) -> dict:
        """Health check"""
        return {
            "name": self.name,
            "healthy": True,
            "status": "running"
        }

class TriggerOrchestrator:
    """
    Orchestrates all trigger agents in the system
    
    Responsibilities:
    - Start/stop all triggers
    - Manage execution order by priority
    - Handle resource allocation
    - Collect and route trigger events
    - Monitor trigger health
    """
    
    def __init__(self, memory_agent=None, message_bus=None):
        self.triggers: Dict[str, BaseTrigger] = {}
        self.memory = memory_agent
        self.message_bus = message_bus
        
        # Priority queues
        self.priority_queues = {
            TriggerPriority.CRITICAL: asyncio.Queue(),
            TriggerPriority.HIGH: asyncio.Queue(),
            TriggerPriority.MEDIUM: asyncio.Queue(),
            TriggerPriority.LOW: asyncio.Queue()
        }
        
        # Control flags
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_trigger": {},
            "errors": 0,
            "start_time": None
        }
        
        logger.info("✅ TriggerOrchestrator initialized")
    
    def register_trigger(self, trigger: BaseTrigger):
        """Register a trigger with the orchestrator"""
        self.triggers[trigger.name] = trigger
        self.stats["events_by_trigger"][trigger.name] = 0
        logger.info(f"📝 Registered trigger: {trigger.name} "
                   f"(Priority: {trigger.priority.value})")
    
    def register_triggers(self, triggers: List[BaseTrigger]):
        """Register multiple triggers at once"""
        for trigger in triggers:
            self.register_trigger(trigger)
    
    async def start(self):
        """Start all triggers and begin processing"""
        logger.info("🚀 Starting Trigger Orchestrator")
        self.is_running = True
        self.stats["start_time"] = datetime.utcnow()
        
        # Group triggers by priority
        triggers_by_priority = {
            TriggerPriority.CRITICAL: [],
            TriggerPriority.HIGH: [],
            TriggerPriority.MEDIUM: [],
            TriggerPriority.LOW: []
        }
        
        for trigger in self.triggers.values():
            if trigger.enabled:
                triggers_by_priority[trigger.priority].append(trigger)
        
        # Start priority queue processors
        processor_tasks = []
        for priority, queue in self.priority_queues.items():
            for i in range(self._get_workers_for_priority(priority)):
                task = asyncio.create_task(
                    self._process_priority_queue(priority, queue)
                )
                processor_tasks.append(task)
                logger.debug(f"Started processor {i+1} for {priority.name} queue")
        
        # Start trigger execution loops
        trigger_tasks = []
        for priority, triggers_list in triggers_by_priority.items():
            for trigger in triggers_list:
                task = asyncio.create_task(
                    self._run_trigger_loop(trigger)
                )
                trigger_tasks.append(task)
                logger.info(f"Started trigger: {trigger.name}")
        
        # Start health monitor
        health_task = asyncio.create_task(self._health_monitor())
        
        logger.info(f"✅ Started {len(trigger_tasks)} triggers and {len(processor_tasks)} processors")
        
        # Wait for shutdown
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("Orchestrator received shutdown signal")
        
        # Cleanup
        logger.info("🛑 Shutting down Trigger Orchestrator")
        for task in trigger_tasks + processor_tasks + [health_task]:
            task.cancel()
    
    async def _run_trigger_loop(self, trigger: BaseTrigger):
        """Run a trigger's execution loop"""
        while self.is_running:
            try:
                # Execute trigger
                events = await trigger.execute()
                
                # Queue events with appropriate priority
                for event in events:
                    await self.priority_queues[trigger.priority].put(event)
                    self.stats["total_events"] += 1
                    self.stats["events_by_trigger"][trigger.name] = \
                        self.stats["events_by_trigger"].get(trigger.name, 0) + 1
                
                # Dynamic sleep based on priority
                if trigger.priority == TriggerPriority.CRITICAL:
                    await asyncio.sleep(1)  # Check every second
                elif trigger.priority == TriggerPriority.HIGH:
                    await asyncio.sleep(5)  # Check every 5 seconds
                elif trigger.priority == TriggerPriority.MEDIUM:
                    await asyncio.sleep(15)  # Check every 15 seconds
                else:  # LOW
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
            except asyncio.CancelledError:
                logger.info(f"Trigger {trigger.name} stopped")
                break
            except Exception as e:
                logger.error(f"Error in trigger {trigger.name}: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(5)  # Back off on errors
    
    async def _process_priority_queue(self, 
                                      priority: TriggerPriority, 
                                      queue: asyncio.Queue):
        """Process events from a priority queue"""
        while self.is_running:
            try:
                # Get event from queue (with timeout)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process event
                await self._process_event(event)
                
                # Log based on priority
                if priority in [TriggerPriority.CRITICAL, TriggerPriority.HIGH]:
                    logger.debug(f"⚡ Processing {priority.name} event: "
                               f"{event.symbol} from {event.source_trigger}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing priority queue {priority}: {e}")
    
    async def _process_event(self, event: TriggerEvent):
        """Process a single trigger event"""
        try:
            # Send to message bus if available
            if self.message_bus:
                if hasattr(self.message_bus, 'publish'):
                    await self.message_bus.publish(
                        topic="trigger_events",
                        message=event.dict_for_fusion()
                    )
            
            # Store in memory if available
            if self.memory and hasattr(self.memory, 'store'):
                await self.memory.store(
                    key=f"trigger_event:{event.event_id}",
                    value=event.__dict__,
                    tier="short"
                )
            
            # Send to fusion agent (simulated)
            if self.message_bus and hasattr(self.message_bus, 'send_to_agent'):
                await self.message_bus.send_to_agent(
                    agent_name="FusionAgent",
                    message={
                        "type": "trigger_event",
                        "event": event.dict_for_fusion()
                    }
                )
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error processing event {getattr(event, 'event_id', 'unknown')}: {e}")
    
    def _get_workers_for_priority(self, priority: TriggerPriority) -> int:
        """Determine number of workers for a priority level"""
        if priority == TriggerPriority.CRITICAL:
            return 2
        elif priority == TriggerPriority.HIGH:
            return 2
        elif priority == TriggerPriority.MEDIUM:
            return 1
        else:  # LOW
            return 1
    
    async def _health_monitor(self):
        """Monitor health of all triggers"""
        while self.is_running:
            await asyncio.sleep(60)  # Check every minute
            
            unhealthy_triggers = []
            for name, trigger in self.triggers.items():
                try:
                    health = await trigger.health_check()
                    if not health.get("healthy", True):
                        unhealthy_triggers.append(name)
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")
                    unhealthy_triggers.append(name)
            
            if unhealthy_triggers:
                logger.warning(f"Unhealthy triggers: {unhealthy_triggers}")
            
            # Log stats
            queue_sizes = {p.name: q.qsize() for p, q in self.priority_queues.items()}
            logger.info(f"📊 Stats: {self.stats['total_events']} events, "
                       f"{self.stats['errors']} errors, "
                       f"Queues: {queue_sizes}")
    
    async def stop(self):
        """Stop the orchestrator gracefully"""
        logger.info("🛑 Stopping Trigger Orchestrator")
        self.is_running = False
        self._shutdown_event.set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "is_running": self.is_running,
            "active_triggers": len([t for t in self.triggers.values() if t.enabled]),
            "total_triggers": len(self.triggers),
            "stats": self.stats,
            "queue_sizes": {
                p.name: q.qsize() 
                for p, q in self.priority_queues.items()
            },
            "uptime": str(datetime.utcnow() - self.stats["start_time"]) if self.stats["start_time"] else None
        }

# Example test trigger for verification
class TestTrigger(BaseTrigger):
    """Simple test trigger"""
    def __init__(self, name: str, priority: str = "MEDIUM"):
        super().__init__(name, {
            "enabled": True,
            "priority": priority
        })
        self.count = 0
    
    async def execute(self) -> List[TriggerEvent]:
        self.count += 1
        if self.count % 10 == 0:  # Every 10th execution
            return [
                TriggerEvent(
                    symbol="TEST",
                    source_trigger=self.name,
                    event_type="TEST_EVENT",
                    confidence=0.8,
                    data={"count": self.count}
                )
            ]
        return []
    
    async def health_check(self) -> dict:
        return {
            "name": self.name,
            "healthy": True,
            "status": "running",
            "executions": self.count
        }

# Quick test function
async def test_orchestrator():
    """Test the orchestrator"""
    logger.info("Testing TriggerOrchestrator...")
    
    # Create orchestrator
    orchestrator = TriggerOrchestrator()
    
    # Create test triggers
    triggers = [
        TestTrigger("CriticalTrigger", "CRITICAL"),
        TestTrigger("HighTrigger", "HIGH"),
        TestTrigger("MediumTrigger", "MEDIUM"),
        TestTrigger("LowTrigger", "LOW")
    ]
    
    # Register triggers
    orchestrator.register_triggers(triggers)
    
    # Start orchestrator
    start_task = asyncio.create_task(orchestrator.start())
    
    # Let it run for 10 seconds
    await asyncio.sleep(10)
    
    # Check status
    status = orchestrator.get_status()
    logger.info(f"Orchestrator status: {status}")
    
    # Stop
    await orchestrator.stop()
    start_task.cancel()
    
    logger.info("Test complete!")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_orchestrator())