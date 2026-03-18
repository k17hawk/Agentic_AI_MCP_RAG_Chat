Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "🚀 Setting up Trading System Databases" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Load environment variables
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

# Default values
$POSTGRES_HOST = [Environment]::GetEnvironmentVariable("POSTGRES_HOST") ?? "localhost"
$POSTGRES_PORT = [Environment]::GetEnvironmentVariable("POSTGRES_PORT") ?? "5432"
$POSTGRES_DB = [Environment]::GetEnvironmentVariable("POSTGRES_DB") ?? "trading"
$POSTGRES_USER = [Environment]::GetEnvironmentVariable("POSTGRES_USER") ?? "postgres"
$POSTGRES_PASSWORD = [Environment]::GetEnvironmentVariable("POSTGRES_PASSWORD") ?? "postgres"

$REDIS_HOST = [Environment]::GetEnvironmentVariable("REDIS_HOST") ?? "localhost"
$REDIS_PORT = [Environment]::GetEnvironmentVariable("REDIS_PORT") ?? "6379"

Write-Host ""
Write-Host "📦 Setting up PostgreSQL..." -ForegroundColor Yellow

# Check if PostgreSQL is running
$pgIsRunning = $false
try {
    $connectionString = "Host=$POSTGRES_HOST;Port=$POSTGRES_PORT;Username=$POSTGRES_USER;Password=$POSTGRES_PASSWORD;Database=postgres"
    $connection = New-Object Npgsql.NpgsqlConnection($connectionString)
    $connection.Open()
    $connection.Close()
    $pgIsRunning = $true
    Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ PostgreSQL is not running" -ForegroundColor Red
}

if ($pgIsRunning) {
    # Check if database exists
    $connectionString = "Host=$POSTGRES_HOST;Port=$POSTGRES_PORT;Username=$POSTGRES_USER;Password=$POSTGRES_PASSWORD;Database=postgres"
    $connection = New-Object Npgsql.NpgsqlConnection($connectionString)
    $connection.Open()
    $command = $connection.CreateCommand()
    $command.CommandText = "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'"
    $result = $command.ExecuteScalar()
    $connection.Close()
    
    if (-not $result) {
        $connection = New-Object Npgsql.NpgsqlConnection($connectionString)
        $connection.Open()
        $command = $connection.CreateCommand()
        $command.CommandText = "CREATE DATABASE $POSTGRES_DB"
        $command.ExecuteNonQuery()
        $connection.Close()
        Write-Host "✅ Created database '$POSTGRES_DB'" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "📦 Running database migrations..." -ForegroundColor Yellow
    python scripts/run_migrations.py
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ Database setup complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan