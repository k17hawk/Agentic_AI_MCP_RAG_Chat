#!/usr/bin/env python3
"""
Launcher script to run trading system from parent directory
"""

import sys
import os
from pathlib import Path
from agentic_trading_system.orchestrator.main import main


if __name__ == "__main__":  
    main()