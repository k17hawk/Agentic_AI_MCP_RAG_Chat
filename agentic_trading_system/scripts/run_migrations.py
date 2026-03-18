#!/usr/bin/env python3
"""
Run database migrations using Alembic
"""
import os
import sys
from pathlib import Path
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.settings import settings
except ImportError:
    print("⚠️  Config not found, using defaults")
    settings = None

def setup_alembic():
    """Initialize Alembic if not already set up"""
    if not Path("alembic").exists():
        print("📦 Initializing Alembic...")
        subprocess.run(["alembic", "init", "alembic"], check=True)
        
        # Update alembic.ini with database URL
        if settings:
            db_config = settings.get_database_config()
            pg_config = db_config.get('postgresql', {})
            db_url = f"postgresql://{pg_config.get('user')}:{pg_config.get('password')}@{pg_config.get('host')}:{pg_config.get('port')}/{pg_config.get('database')}"
            
            with open("alembic.ini", "r") as f:
                content = f.read()
            content = content.replace("sqlalchemy.url = driver://user:pass@localhost/dbname", f"sqlalchemy.url = {db_url}")
            with open("alembic.ini", "w") as f:
                f.write(content)

def run_migrations():
    """Run database migrations"""
    print("=" * 60)
    print("🚀 Running Database Migrations")
    print("=" * 60)
    
    # Check if alembic is installed
    try:
        import alembic
        print("✅ Alembic found")
    except ImportError:
        print("❌ Alembic not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "alembic"], check=True)
    
    # Setup alembic if needed
    setup_alembic()
    
    # Create initial migration if none exists
    if not Path("alembic/versions").exists() or not list(Path("alembic/versions").glob("*.py")):
        print("📦 Creating initial migration...")
        subprocess.run(["alembic", "revision", "--autogenerate", "-m", "Initial migration"], check=True)
    
    # Run migrations
    print("📦 Applying migrations...")
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Migrations applied successfully!")
        print(result.stdout)
    else:
        print("❌ Migration failed:")
        print(result.stderr)
        sys.exit(1)
    
    # Show current version
    print("\n📊 Current migration version:")
    subprocess.run(["alembic", "current"])

def create_migration(message: str):
    """Create a new migration"""
    print(f"📦 Creating migration: {message}")
    subprocess.run(["alembic", "revision", "--autogenerate", "-m", message], check=True)

def downgrade_migration(revision: str = "-1"):
    """Downgrade to a previous revision"""
    print(f"📦 Downgrading to: {revision}")
    subprocess.run(["alembic", "downgrade", revision], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("--create", type=str, help="Create a new migration with message")
    parser.add_argument("--downgrade", type=str, help="Downgrade to revision (default: -1)", nargs="?", const="-1")
    
    args = parser.parse_args()
    
    if args.create:
        create_migration(args.create)
    elif args.downgrade:
        downgrade_migration(args.downgrade)
    else:
        run_migrations()