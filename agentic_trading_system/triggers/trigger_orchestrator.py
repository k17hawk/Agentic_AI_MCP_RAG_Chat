"""
Trigger Orchestrator - Manages all trigger agents
"""
from typing import Dict, List, Optional, Type, Any
from datetime import datetime, timedelta
import asyncio
from loguru import logger
import signal
from concurrent.futures import ThreadPoolExecutor
import psutil
from pathlib import Path
from agentic_trading_system.triggers.base_trigger import BaseTrigger, TriggerEvent, TriggerPriority


class TriggerOrchestrator:
    """
    Orchestrates all trigger agents in the system
    """

    def __init__(self,
                 config_path: str = None,
                 memory_agent=None,
                 message_bus=None,
                 max_concurrent_triggers: int = 5):

        self.triggers: Dict[str, BaseTrigger] = {}
        self.memory = memory_agent
        self.message_bus = message_bus
        self.max_concurrent = max_concurrent_triggers

        # NOTE: asyncio.Queue and asyncio.Event MUST be created inside the
        # running event loop (i.e. inside start()), NOT here in __init__.
        # Creating them here (before asyncio.run() starts) binds them to no
        # loop in Python 3.10+ and causes silent failures.
        self.event_queue: Optional[asyncio.Queue] = None
        self.priority_queues: Optional[Dict] = None

        # Control flags
        self.is_running = False
        self._shutdown_event: Optional[asyncio.Event] = None

        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_trigger": {},
            "errors": 0,
            "start_time": None
        }

        # Thread pool for CPU-intensive triggers
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._load_and_register_triggers(config_path)

        logger.info("✅ TriggerOrchestrator initialized")

    def _load_and_register_triggers(self, config_path: str = None):
        """Load triggers from YAML and register them"""
        import yaml
        from pathlib import Path

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "triggers.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Config file not found: {config_path}")
            return

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            triggers_config = config.get('triggers', [])

            for trigger_config in triggers_config:
                if not trigger_config.get('enabled', True):
                    logger.info(f"⏭️ Skipping disabled trigger: {trigger_config['name']}")
                    continue

                # Register the trigger
                trigger_instance = self._create_trigger_instance(trigger_config)

                if trigger_instance:
                    self.register_trigger(trigger_instance)
                    logger.info(f"✅ Registered trigger: {trigger_config['name']} ({trigger_config['type']})")

            logger.info(f"📊 Registered {len(self.triggers)} triggers from config")

        except Exception as e:
            logger.error(f"❌ Failed to load triggers: {e}")

    def _create_trigger_instance(self, trigger_config: dict):
        trigger_type = trigger_config['type']
        trigger_name = trigger_config['name']
        priority = trigger_config.get('priority', 'MEDIUM')
        config = trigger_config.get('config', {})

        try:
            if trigger_type == 'price_alert':
                from agentic_trading_system.triggers.price_alert_trigger import PriceAlertTrigger
                return PriceAlertTrigger(name=trigger_name, config=config, priority=priority)
            elif trigger_type == 'volume_spike':
                from agentic_trading_system.triggers.volume_spike_trigger import VolumeSpikeTrigger
                return VolumeSpikeTrigger(name=trigger_name, config=config, priority=priority)
            elif trigger_type == 'news_alert':
                from agentic_trading_system.triggers.news_alert_trigger import NewsAlertTrigger
                return NewsAlertTrigger(name=trigger_name, config=config, priority=priority)
            elif trigger_type == 'pattern_recognition':
                from agentic_trading_system.triggers.pattern_recognition_trigger import PatternRecognitionTrigger
                return PatternRecognitionTrigger(name=trigger_name, config=config, priority=priority)
            elif trigger_type == 'social_sentiment':
                from agentic_trading_system.triggers.social_sentiment_trigger import SocialSentimentTrigger
                return SocialSentimentTrigger(name=trigger_name, config=config, priority=priority)
            elif trigger_type == 'scheduled':
                from agentic_trading_system.triggers.scheduled_trigger import ScheduledTrigger
                return ScheduledTrigger(name=trigger_name, config=config, priority=priority)
            else:
                logger.warning(f"⚠️ Unknown trigger type: {trigger_type} for {trigger_name}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to create trigger {trigger_name}: {e}")
            return None

    def register_trigger(self, trigger: BaseTrigger):
        """Register a trigger with the orchestrator"""
        self.triggers[trigger.name] = trigger
        self.stats["events_by_trigger"][trigger.name] = 0

        # Get priority value safely
        if hasattr(trigger, 'priority') and trigger.priority is not None:
            priority_value = trigger.priority if isinstance(trigger.priority, int) else trigger.priority.value
        elif hasattr(trigger.config, 'priority'):
            priority_value = trigger.config.priority if isinstance(trigger.config.priority, int) else trigger.config.priority.value
        else:
            priority_value = 2  

        logger.info(f"📝 Registered trigger: {trigger.name} (Priority: {priority_value})")

    def register_triggers(self, triggers: List[BaseTrigger]):
        for trigger in triggers:
            self.register_trigger(trigger)

    async def start(self):
        logger.info("🚀 Starting Trigger Orchestrator")
        self.is_running = True
        self.stats["start_time"] = datetime.utcnow()

        # Create asyncio primitives
        self.event_queue = asyncio.Queue()
        self.priority_queues = {
            TriggerPriority.CRITICAL: asyncio.Queue(),
            TriggerPriority.HIGH: asyncio.Queue(),
            TriggerPriority.MEDIUM: asyncio.Queue(),
            TriggerPriority.LOW: asyncio.Queue()
        }
        
        self._shutdown_event = asyncio.Event()

        # Group triggers by priority
        triggers_by_priority = {
            TriggerPriority.CRITICAL: [],
            TriggerPriority.HIGH: [],
            TriggerPriority.MEDIUM: [],
            TriggerPriority.LOW: []
        }

        for trigger in self.triggers.values():
            # Get priority value - handle both int and enum
            priority_value = 2  # Default MEDIUM
            
            if hasattr(trigger.config, 'priority'):
                if isinstance(trigger.config.priority, int):
                    priority_value = trigger.config.priority
                elif hasattr(trigger.config.priority, 'value'):
                    priority_value = trigger.config.priority.value
                elif isinstance(trigger.config.priority, str):
                    # Handle string priority
                    priority_str_map = {
                        'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4
                    }
                    priority_value = priority_str_map.get(trigger.config.priority.upper(), 2)
            
            # Map integer priority to enum
            priority_map = {
                1: TriggerPriority.LOW,
                2: TriggerPriority.MEDIUM,
                3: TriggerPriority.HIGH,
                4: TriggerPriority.CRITICAL
            }
            priority_enum = priority_map.get(priority_value, TriggerPriority.MEDIUM)
            
            if trigger.config.enabled:
                triggers_by_priority[priority_enum].append(trigger)
                logger.debug(f"Added {trigger.name} to {priority_enum.name} priority")

        # Start priority queue processors
        processor_tasks = []
        for priority, queue in self.priority_queues.items():
            for i in range(self._get_workers_for_priority(priority)):
                task = asyncio.create_task(
                    self._process_priority_queue(priority, queue)
                )
                processor_tasks.append(task)
                logger.debug(f"Started processor {i + 1} for {priority.name} queue")

        # Start trigger execution loops
        trigger_tasks = []
        for priority, triggers_list in triggers_by_priority.items():
            for trigger in triggers_list:
                task = asyncio.create_task(
                    self._run_trigger_loop(trigger, priority)
                )
                trigger_tasks.append(task)
                logger.info(f"Started trigger: {trigger.name} (Priority: {priority.name})")

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
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run_trigger_loop(self, trigger: BaseTrigger, priority: TriggerPriority):
        """Run a single trigger's execution loop."""
        while self.is_running:
            try:
                events = await trigger.execute()
                
                # Debug log for number of events
                logger.debug(f"Trigger {trigger.name} generated {len(events)} events")
                
                for event in events:
                    # Validate confidence (optional, but good practice)
                    if event.confidence < 0.3:
                        logger.debug(f"Skipping {event.event_id} for {event.symbol} - low confidence {event.confidence:.2f}")
                        continue
                    
                    # Place event into the correct priority queue
                    if priority in self.priority_queues:
                        await self.priority_queues[priority].put(event)
                        self.stats["total_events"] += 1
                        self.stats["events_by_trigger"][trigger.name] = \
                            self.stats["events_by_trigger"].get(trigger.name, 0) + 1
                        logger.info(f"📊 {trigger.name} → {event.event_type} for {event.symbol} (conf: {event.confidence:.2f})")
                    else:
                        logger.error(f"Priority queue {priority} not found for event {event.event_id}")
                
                # Sleep according to execution mode
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
            logger.info(f"🔄 Processing event: {event.event_id} for {event.symbol} from {event.source_trigger}")
            
            if self.message_bus and hasattr(self.message_bus, 'publish'):
                await self.message_bus.publish(
                    topic="trigger_events",
                    message=event.dict_for_fusion()
                )
                logger.debug(f"📡 Published event to message bus")

            if self.memory and hasattr(self.memory, 'store'):
                await self.memory.store(
                    key=f"trigger_event:{event.event_id}",
                    value=event.__dict__,
                    tier="short"
                )
                logger.debug(f"💾 Stored event in memory")

            if self.message_bus and hasattr(self.message_bus, 'send_to_agent'):
                await self.message_bus.send_to_agent(
                    agent_name="FusionAgent",
                    message={
                        "type": "trigger_event",
                        "event": event.dict_for_fusion()
                    }
                )
                logger.debug(f"🔀 Sent event to FusionAgent")

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
        if self._shutdown_event is not None:
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
            } if self.priority_queues else {},
            "uptime": str(datetime.utcnow() - self.stats["start_time"]) if self.stats["start_time"] else None
        }

    def _load_triggers(self):
        """Load triggers from configuration (backward compatibility)"""
        config_path = Path(__file__).parent.parent / "config" / "triggers.yaml"
        logger.info(f"Looking for config at: {config_path.absolute()}")

        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return []

