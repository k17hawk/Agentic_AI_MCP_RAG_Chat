set -e

echo "========================================="
echo "💾 Backing up Trading System Data"
echo "========================================="

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Configuration
BACKUP_DIR=${BACKUP_DIR:-"data/backups"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="trading_backup_$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_PATH"
mkdir -p "$BACKUP_PATH"/{postgres,redis,config,models,reports}

echo ""
echo "📦 Creating backup at: $BACKUP_PATH"

# 1. Backup PostgreSQL
echo ""
echo "📦 Backing up PostgreSQL..."
if command -v pg_dump &> /dev/null; then
    PGPASSWORD=${POSTGRES_PASSWORD:-postgres} pg_dump -h ${POSTGRES_HOST:-localhost} -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} -F c -f "$BACKUP_PATH/postgres/trading.dump"
    echo "   ✅ PostgreSQL backup complete"
else
    echo "   ⚠️  pg_dump not found, skipping PostgreSQL backup"
fi

# 2. Backup Redis (if running)
echo ""
echo "📦 Backing up Redis..."
if command -v redis-cli &> /dev/null && redis-cli -h ${REDIS_HOST:-localhost} ping > /dev/null 2>&1; then
    redis-cli -h ${REDIS_HOST:-localhost} save
    cp /var/lib/redis/dump.rdb "$BACKUP_PATH/redis/" 2>/dev/null || echo "   ⚠️  Could not copy Redis dump"
    echo "   ✅ Redis backup complete"
else
    echo "   ⚠️  Redis not running or redis-cli not found"
fi

# 3. Backup configuration files
echo ""
echo "📦 Backing up configuration..."
if [ -d "config" ]; then
    cp -r config "$BACKUP_PATH/"
    echo "   ✅ Config backup complete"
fi

# 4. Backup trained models
echo ""
echo "📦 Backing up ML models..."
if [ -d "data/models" ]; then
    cp -r data/models "$BACKUP_PATH/"
    echo "   ✅ Models backup complete"
fi

# 5. Backup reports
echo ""
echo "📦 Backing up reports..."
if [ -d "data/reports" ]; then
    cp -r data/reports "$BACKUP_PATH/"
    echo "   ✅ Reports backup complete"
fi

# 6. Create archive
echo ""
echo "📦 Creating compressed archive..."
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"
cd - > /dev/null

echo ""
echo "========================================="
echo "✅ Backup complete: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
echo "========================================="

# 7. Cleanup old backups (keep last 30 days)
echo ""
echo "🧹 Cleaning up backups older than 30 days..."
find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +30 -delete

# Show backup size
BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_NAME.tar.gz" | cut -f1)
echo "   Backup size: $BACKUP_SIZE"
echo "   Free space: $(df -h . | awk 'NR==2 {print $4}')"