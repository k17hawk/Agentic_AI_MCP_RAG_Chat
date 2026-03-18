#!/usr/bin/env python3
"""
Initialize all data directories for the trading system
Run this first before starting the system
"""
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {path}")

def main():
    """Create all data directories"""
    print("=" * 60)
    print("🚀 Initializing Trading System Data Directories")
    print("=" * 60)
    
    # Base data directories
    data_dirs = [
        "data/raw",
        "data/raw/market_data",
        "data/raw/news",
        "data/raw/fundamentals",
        "data/raw/options",
        "data/processed",
        "data/processed/features",
        "data/processed/training_data",
        "data/models",
        "data/models/technical",
        "data/models/fundamental",
        "data/models/sentiment",
        "data/models/ensemble",
        "data/reports",
        "data/reports/daily",
        "data/reports/weekly",
        "data/reports/monthly",
        "data/reports/confirmations",
        "data/charts",
        "data/charts/technical",
        "data/charts/performance",
        "data/charts/portfolio",
        "data/logs",
        "data/logs/agents",
        "data/logs/triggers",
        "data/logs/analysis",
        "data/logs/execution",
        "data/logs/trading",
        "data/logs/audit",
        "data/cache",
        "data/backups",
        "data/exports",
        "data/state",
        "data/temp"
    ]
    
    for dir_path in data_dirs:
        create_directory(dir_path)
    
    # Create .gitkeep files to preserve empty directories in git
    gitkeep_dirs = [
        "data/raw/market_data",
        "data/raw/news",
        "data/raw/fundamentals",
        "data/processed/features",
        "data/models/technical",
        "data/reports/daily",
        "data/charts/technical",
        "data/logs/agents"
    ]
    
    for dir_path in gitkeep_dirs:
        gitkeep_path = Path(dir_path) / ".gitkeep"
        gitkeep_path.touch(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("✅ All data directories initialized successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()