#!/bin/bash
# scripts/restore.sh
# Restore trading system from backup

set -e

echo "========================================="
echo "🔄 Restoring Trading System from Backup"
echo "========================================="

BACKUP_DIR=${BACKUP_DIR:-"data/backups"}

# List available backups
echo ""
echo "Available backups:"
echo "------------------"
ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null | awk '{print NR")", $9, "("$5")"}'

if [ $? -ne 0 ]; then
    echo "No backups found in $BACKUP_DIR"
    exit 1
fi

echo ""
read -p "Enter backup number to restore: " BACKUP_NUM

# Get backup file
BACKUP_FILE=$(ls "$BACKUP_DIR"/*.tar.gz | sed -n "${BACKUP_NUM}p")

if [ -z "$BACKUP_FILE" ]; then
    echo "❌ Invalid backup number"
    exit 1
fi

echo ""
echo "Selected backup: $BACKUP_FILE"
read -p "This will overwrite existing data. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo ""
echo "📦 Extracting backup to $TEMP_DIR..."
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# Find the extracted directory
EXTRACTED_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d | tail -1)

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# 1. Restore PostgreSQL
echo ""
echo "📦 Restoring PostgreSQL..."
if [ -f "$EXTRACTED_DIR/postgres/trading.dump" ]; then
    if command -v pg_restore &> /dev/null; then
        # Drop and recreate database
        PGPASSWORD=${POSTGRES_PASSWORD:-postgres} dropdb -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-postgres} --if-exists ${POSTGRES_DB:-trading}
        PGPASSWORD=${POSTGRES_PASSWORD:-postgres} createdb -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-postgres} ${POSTGRES_DB:-trading}
        
        # Restore from dump
        PGPASSWORD=${POSTGRES_PASSWORD:-postgres} pg_restore -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} "$EXTRACTED_DIR/postgres/trading.dump"
        echo "   ✅ PostgreSQL restore complete"
    else
        echo "   ⚠️  pg_restore not found, skipping PostgreSQL restore"
    fi
fi

# 2. Restore Redis
echo ""
echo "📦 Restoring Redis..."
if [ -f "$EXTRACTED_DIR/redis/dump.rdb" ]; then
    if command -v redis-cli &> /dev/null; then
        # Stop Redis, replace dump file, restart
        sudo systemctl stop redis 2>/dev/null || echo "   ⚠️  Could not stop Redis"
        sudo cp "$EXTRACTED_DIR/redis/dump.rdb" /var/lib/redis/dump.rdb
        sudo chown redis:redis /var/lib/redis/dump.rdb
        sudo systemctl start redis 2>/dev/null || echo "   ⚠️  Could not start Redis"
        echo "   ✅ Redis restore complete"
    else
        echo "   ⚠️  redis-cli not found, skipping Redis restore"
    fi
fi

# 3. Restore configuration
echo ""
echo "📦 Restoring configuration..."
if [ -d "$EXTRACTED_DIR/config" ]; then
    cp -rf "$EXTRACTED_DIR/config" ./
    echo "   ✅ Config restore complete"
fi

# 4. Restore models
echo ""
echo "📦 Restoring ML models..."
if [ -d "$EXTRACTED_DIR/models" ]; then
    mkdir -p data
    cp -rf "$EXTRACTED_DIR/models" data/
    echo "   ✅ Models restore complete"
fi

# 5. Restore reports
echo ""
echo "📦 Restoring reports..."
if [ -d "$EXTRACTED_DIR/reports" ]; then
    mkdir -p data
    cp -rf "$EXTRACTED_DIR/reports" data/
    echo "   ✅ Reports restore complete"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "========================================="
echo "✅ Restore complete!"
echo "========================================="