#!/usr/bin/env python3
"""Fixed test script for trigger system"""

import sys
import yaml
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.triggers.trigger_orchestrator import TriggerOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trigger_loading():
    """Test trigger loading and starting"""
    
    print("🔍 Testing Trigger Orchestrator...")
    
    # Initialize orchestrator
    orchestrator = TriggerOrchestrator()
    
    # Check what attributes exist
    print(f"\n📋 Orchestrator attributes: {[attr for attr in dir(orchestrator) if not attr.startswith('_')]}")
    
    # Check if triggers were loaded
    if hasattr(orchestrator, 'triggers'):
        print(f"\n📊 Active triggers: {len(orchestrator.triggers)}")
        if len(orchestrator.triggers) > 0:
            for trigger in orchestrator.triggers:
                print(f"  ✅ {trigger}")
        else:
            print("  ❌ No triggers found!")
    else:
        print("  ❌ orchestrator.triggers attribute not found!")
    
    # Check for trigger dictionary or list
    if hasattr(orchestrator, 'trigger_dict'):
        print(f"\n📊 Trigger dict: {len(orchestrator.trigger_dict)} items")
    
    # Try to manually load config
    config_path = Path("agentic_trading_system/config/triggers.yaml")
    if config_path.exists():
        print(f"\n📁 Config file found: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        triggers = config.get('triggers', [])
        enabled = [t for t in triggers if t.get('enabled', True)]
        print(f"📊 YAML has {len(triggers)} total, {len(enabled)} enabled triggers")
        
        # Print first few triggers
        for i, trigger in enumerate(enabled[:5]):
            print(f"  {i+1}. {trigger['name']} ({trigger['type']}) - enabled: {trigger.get('enabled', True)}")
    
    return orchestrator

def inspect_orchestrator_methods():
    """Inspect what methods are available"""
    print("\n🔍 Inspecting TriggerOrchestrator class...")
    
    # Get the actual class definition
    import inspect
    from agentic_trading_system.triggers import trigger_orchestrator
    
    # Get source file path
    source_file = inspect.getfile(trigger_orchestrator)
    print(f"📁 Source file: {source_file}")
    
    # Get methods
    methods = [m for m in dir(TriggerOrchestrator) if not m.startswith('_')]
    print(f"📋 Available methods: {methods}")

if __name__ == "__main__":
    # First inspect the class
    inspect_orchestrator_methods()
    
    # Then test loading
    orch = test_trigger_loading()
    
    # Try to start if method exists
    if hasattr(orch, 'start'):
        print("\n🚀 Attempting to start orchestrator...")
        try:
            orch.start()
            print("✅ Start method executed")
        except Exception as e:
            print(f"❌ Error starting: {e}")