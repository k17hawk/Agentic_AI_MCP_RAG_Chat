# test_trigger_fixed.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator

async def test():
    print("🔍 Testing Trigger Orchestrator Fix")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = TriggerOrchestrator()
    
    # Check loaded triggers
    print(f"\n📊 Loaded {len(orchestrator.triggers)} triggers:")
    for name, trigger in orchestrator.triggers.items():
        print(f"  ✅ {name}: {type(trigger).__name__}")
    
    # Start orchestrator
    print("\n🚀 Starting orchestrator...")
    start_task = asyncio.create_task(orchestrator.start())
    
    # Let it run for a few seconds
    await asyncio.sleep(5)
    
    # Get status
    status = orchestrator.get_status()
    print(f"\n📊 Status: {status}")
    
    # Stop
    await orchestrator.stop()
    print("\n✅ Test complete")

if __name__ == "__main__":
    asyncio.run(test())