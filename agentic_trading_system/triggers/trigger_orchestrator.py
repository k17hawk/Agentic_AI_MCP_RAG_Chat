"""
Trigger Orchestrator - Manages all trigger agents
"""
from typing import Dict, List, Optional, Type,Any
from datetime import datetime, timedelta
import asyncio
from loguru import logger
import signal
from concurrent.futures import ThreadPoolExecutor
import psutil

from agentic_trading_system.triggers.base_trigger import BaseTrigger, TriggerEvent, TriggerPriority

class TriggerOrchestrator:
    """
    Orchestrates all trigger agents in the system
    """
    
    def __init__(self, 
                 memory_agent=None,
                 message_bus=None,
                 max_concurrent_triggers: int = 5):
        
        self.triggers: Dict[str, BaseTrigger] = {}
        self.memory = memory_agent
        self.message_bus = message_bus
        self.max_concurrent = max_concurrent_triggers
        
        # Execution queue
        self.event_queue: asyncio.Queue = asyncio.Queue()
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
        
        # Thread pool for CPU-intensive triggers
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("✅ TriggerOrchestrator initialized")
    
    def register_trigger(self, trigger: BaseTrigger):
        """Register a trigger with the orchestrator"""
        self.triggers[trigger.name] = trigger
        self.stats["events_by_trigger"][trigger.name] = 0
        
        # FIXED: Handle priority display correctly
        priority_value = trigger.priority if isinstance(trigger.priority, int) else trigger.priority.value
        logger.info(f"📝 Registered trigger: {trigger.name} "
                   f"(Priority: {priority_value})")
    
    def register_triggers(self, triggers: List[BaseTrigger]):
        for trigger in triggers:
            self.register_trigger(trigger)
    
    async def start(self):
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
            # Get priority as enum for grouping
            if hasattr(trigger.config, 'priority'):
                if isinstance(trigger.config.priority, int):
                    # Map int to enum
                    priority_map = {1: TriggerPriority.LOW, 2: TriggerPriority.MEDIUM, 
                                   3: TriggerPriority.HIGH, 4: TriggerPriority.CRITICAL}
                    priority = priority_map.get(trigger.config.priority, TriggerPriority.MEDIUM)
                else:
                    priority = trigger.config.priority
            else:
                priority = TriggerPriority.MEDIUM
            
            if trigger.config.enabled:
                triggers_by_priority[priority].append(trigger)
        
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
                    self._run_trigger_loop(trigger, priority)
                )
                trigger_tasks.append(task)
                logger.info(f"Started trigger: {trigger.name}")
        
        # Start health monitor
        health_task = asyncio.create_task(self._health_monitor())
        
        logger.info(f"✅ Started {len(trigger_tasks)} triggers and {len(processor_tasks)} processors")
        
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("Orchestrator received shutdown signal")
        
        # Cleanup
        logger.info("🛑 Shutting down Trigger Orchestrator")
        for task in trigger_tasks + processor_tasks + [health_task]:
            task.cancel()
    
    async def _run_trigger_loop(self, trigger: BaseTrigger, priority: TriggerPriority):
        while self.is_running:
            try:
                events = await trigger.execute()
                
                for event in events:
                    await self.priority_queues[priority].put(event)
                    self.stats["total_events"] += 1
                    self.stats["events_by_trigger"][trigger.name] = \
                        self.stats["events_by_trigger"].get(trigger.name, 0) + 1
                
                if trigger.config.execution_mode == "realtime":
                    await asyncio.sleep(1)
                elif trigger.config.execution_mode == "scheduled":
                    await asyncio.sleep(60)
                else:
                    await asyncio.sleep(300)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trigger {trigger.name}: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(5)
    
    async def _process_priority_queue(self, 
                                      priority: TriggerPriority, 
                                      queue: asyncio.Queue):
        while self.is_running:
            try:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                await self._process_event(event)
                
                if priority in [TriggerPriority.CRITICAL, TriggerPriority.HIGH]:
                    logger.debug(f"⚡ Processing {priority.name} event: "
                               f"{event.symbol} from {event.source_trigger}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing priority queue {priority}: {e}")
    
    async def _process_event(self, event: TriggerEvent):
        try:
            if self.message_bus and hasattr(self.message_bus, 'publish'):
                await self.message_bus.publish(
                    topic="trigger_events",
                    message=event.dict_for_fusion()
                )
            
            if self.memory and hasattr(self.memory, 'store'):
                await self.memory.store(
                    key=f"trigger_event:{event.event_id}",
                    value=event.__dict__,
                    tier="short"
                )
            
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
        if priority == TriggerPriority.CRITICAL:
            return 2
        elif priority == TriggerPriority.HIGH:
            return 2
        elif priority == TriggerPriority.MEDIUM:
            return 1
        else:
            return 1
    
    async def _health_monitor(self):
        while self.is_running:
            await asyncio.sleep(60)
            
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
            
            queue_sizes = {p.name: q.qsize() for p, q in self.priority_queues.items()}
            logger.info(f"📊 Stats: {self.stats['total_events']} events, "
                       f"{self.stats['errors']} errors, "
                       f"Queues: {queue_sizes}")
    
    async def stop(self):
        logger.info("🛑 Stopping Trigger Orchestrator")
        self.is_running = False
        self._shutdown_event.set()
        self.executor.shutdown(wait=True)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "active_triggers": len([t for t in self.triggers.values() if t.config.enabled]),
            "total_triggers": len(self.triggers),
            "stats": self.stats,
            "queue_sizes": {
                p.name: q.qsize() 
                for p, q in self.priority_queues.items()
            },
            "uptime": str(datetime.utcnow() - self.stats["start_time"]) if self.stats["start_time"] else None
        }