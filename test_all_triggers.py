import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from triggers.trigger_orchestrator import TriggerOrchestrator, BaseTrigger, TriggerEvent
from triggers.base_trigger import TriggerPriority


# ==============================
# LOGGER CONFIGURATION
# ==============================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{message}</cyan>"
)


# ==============================
# MOCK DEPENDENCIES
# ==============================

class SimpleMemory:
    async def get(self, key):
        return None

    async def store(self, key, value, tier):
        pass


class SimpleMessageBus:
    async def publish(self, topic, message):
        logger.info(f"📨 [{topic}] {message.get('type', 'unknown')}")

    async def send_to_agent(self, agent_name, message):
        logger.info(f"📨 → {agent_name}")


# ==============================
# TEST TRIGGERS
# ==============================

class TestTrigger(BaseTrigger):
    """Simple test trigger for orchestrator testing"""

    def __init__(self, name: str, priority: TriggerPriority, memory=None, message_bus=None):
        config = {
            "name": name,
            "enabled": True,
            "priority": priority.value
        }
        super().__init__(name, config, memory, message_bus)
        self.priority = priority
        self.count = 0

    async def scan(self):
        self.count += 1
        if self.count % 3 == 0:  # Generate event every 3 scans
            return [
                TriggerEvent(
                    symbol=f"TEST{self.count}",
                    source_trigger=self.name,
                    event_type=f"TEST_EVENT_{self.priority.name}",
                    confidence=0.5 + (self.priority.value * 0.1)
                )
            ]
        return []

    async def validate(self, event):
        return True

    async def health_check(self):
        return {
            "name": self.name,
            "healthy": True,
            "status": "running",
            "executions": self.count
        }


# ==============================
# TEST FUNCTIONS
# ==============================

async def test_orchestrator():
    """Test the trigger orchestrator"""
    logger.info("🚀 Starting Trigger Orchestrator Test")
    logger.info("=" * 60)

    memory = SimpleMemory()
    message_bus = SimpleMessageBus()

    try:
        # Create orchestrator
        orchestrator = TriggerOrchestrator(
            memory_agent=memory,
            message_bus=message_bus
        )
        logger.info("✅ Created TriggerOrchestrator")

        # Create test triggers with different priorities
        triggers = [
            TestTrigger("Critical-1", TriggerPriority.CRITICAL, memory, message_bus),
            TestTrigger("Critical-2", TriggerPriority.CRITICAL, memory, message_bus),
            TestTrigger("High-1", TriggerPriority.HIGH, memory, message_bus),
            TestTrigger("Medium-1", TriggerPriority.MEDIUM, memory, message_bus),
            TestTrigger("Low-1", TriggerPriority.LOW, memory, message_bus),
        ]

        # Register triggers
        orchestrator.register_triggers(triggers)
        logger.info(f"📝 Registered {len(triggers)} triggers")

        # Start orchestrator
        logger.info("\n🚀 Starting orchestrator for 10 seconds...")
        start_task = asyncio.create_task(orchestrator.start())

        # Monitor status every 2 seconds
        for i in range(5):
            await asyncio.sleep(2)
            status = orchestrator.get_status()
            logger.info(f"\n📊 Status at {i*2}s:")
            logger.info(f"   Total events: {status['stats']['total_events']}")
            logger.info(f"   Queue sizes: {status['queue_sizes']}")
            logger.info(f"   Active triggers: {status['active_triggers']}")

        # Stop orchestrator
        logger.info("\n🛑 Stopping orchestrator...")
        await orchestrator.stop()
        start_task.cancel()

        # Final stats
        final_status = orchestrator.get_status()
        logger.info(f"\n📊 Final Stats:")
        logger.info(f"   Total events generated: {final_status['stats']['total_events']}")
        logger.info(f"   Errors: {final_status['stats']['errors']}")
        logger.info(f"   Uptime: {final_status['uptime']}")

    except Exception as e:
        logger.exception(f"❌ Error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Trigger Orchestrator Test Complete!")


# ==============================
# ENTRY POINT
# ==============================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Trigger Orchestrator - Test Suite                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    asyncio.run(test_orchestrator())


if __name__ == "__main__":
    main()