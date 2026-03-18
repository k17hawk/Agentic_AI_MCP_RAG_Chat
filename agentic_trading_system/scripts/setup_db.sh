#!/bin/bash
# scripts/setup_db.sh
# Initialize databases for the trading system

set -e  # Exit on error

echo "========================================="
echo "🚀 Setting up Trading System Databases"
echo "========================================="

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "⚠️  .env file not found, using defaults"
fi

# Default values
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_DB=${POSTGRES_DB:-trading}
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}

REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

echo ""
echo "📦 Setting up PostgreSQL..."

# Check if PostgreSQL is running
if pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER > /dev/null 2>&1; then
    echo "✅ PostgreSQL is running"
    
    # Create database if it doesn't exist
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -tc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" | grep -q 1 || \
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -c "CREATE DATABASE $POSTGRES_DB"
    
    echo "✅ Database '$POSTGRES_DB' is ready"
    
    # Run migrations
    echo ""
    echo "📦 Running database migrations..."
    python scripts/run_migrations.py
else
    echo "❌ PostgreSQL is not running. Please start it first:"
    echo "   docker run -d --name trading-postgres -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD -p $POSTGRES_PORT:5432 postgres:15"
    exit 1
fi

echo ""
echo "📦 Setting up Redis..."

# Check if Redis is running
if redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
    
    # Clear Redis cache (optional)
    read -p "Clear Redis cache? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        redis-cli -h $REDIS_HOST -p $REDIS_PORT flushall
        echo "✅ Redis cache cleared"
    fi
else
    echo "⚠️  Redis is not running. Starting with Docker..."
    docker run -d --name trading-redis -p $REDIS_PORT:6379 redis:7-alpine
    echo "✅ Redis started"
fi

echo ""
echo "========================================="
echo "✅ Database setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run 'python scripts/seed_data.py' to load sample data"
echo "  2. Run 'python scripts/init_directories.py' to create data folders"
echo "  3. Start the system with 'python -m orchestrator.main'"
echo ""