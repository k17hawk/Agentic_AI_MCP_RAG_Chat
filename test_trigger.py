# test_trigger_verbose.py
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Configure verbose logging
logger.remove()
logger.add(sys.stdout, level="DEBUG")

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator

async def test():
    print("\n🔍 Testing Trigger Orchestrator with Verbose Logging")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = TriggerOrchestrator()
    
    # Check loaded triggers
    print(f"\n📊 Loaded {len(orchestrator.triggers)} triggers:")
    for name, trigger in orchestrator.triggers.items():
        priority = trigger.config.priority.value if hasattr(trigger.config.priority, 'value') else trigger.config.priority
        print(f"  ✅ {name}: {type(trigger).__name__} (Priority: {priority})")
    
    # Start orchestrator
    print("\n🚀 Starting orchestrator...")
    start_task = asyncio.create_task(orchestrator.start())
    
    # Let it run for 10 seconds to see events
    await asyncio.sleep(10)
    
    # Get status
    status = orchestrator.get_status()
    print(f"\n📊 Final Status:")
    print(f"   Running: {status['is_running']}")
    print(f"   Total Events: {status['stats']['total_events']}")
    print(f"   Events by trigger: {status['stats']['events_by_trigger']}")
    print(f"   Errors: {status['stats']['errors']}")
    print(f"   Queue sizes: {status['queue_sizes']}")
    
    # Stop
    await orchestrator.stop()
    print("\n✅ Test complete")

if __name__ == "__main__":
    asyncio.run(test())