#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Change to the project root so relative imports work
project_root = Path(__file__).parent / "agentic_trading_system"
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from orchestrator.main import main

if __name__ == "__main__":
    main()