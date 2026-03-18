# Makefile for trading system
.PHONY: help setup test run docker-build docker-up docker-down deploy backup restore clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Initialize system (directories, databases, seed data)"
	@echo "  make test       - Run all tests"
	@echo "  make run        - Run the system locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up  - Start Docker containers"
	@echo "  make docker-down - Stop Docker containers"
	@echo "  make deploy     - Deploy to production"
	@echo "  make backup     - Create backup"
	@echo "  make restore    - Restore from backup"
	@echo "  make clean      - Clean temporary files"

setup:
	@echo "🔧 Setting up system..."
	@python scripts/init_directories.py
	@chmod +x scripts/setup_db.sh
	@./scripts/setup_db.sh
	@python scripts/seed_data.py
	@echo "✅ Setup complete"

test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v --cov=. --cov-report=html
	@echo "✅ Tests complete"

run:
	@echo "🚀 Starting system..."
	@python -m orchestrator.main

docker-build:
	@echo "📦 Building Docker image..."
	@docker build -t trading-system:latest -f docker/Dockerfile .

docker-up:
	@echo "🐳 Starting Docker containers..."
	@docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Containers started"

docker-down:
	@echo "🐳 Stopping Docker containers..."
	@docker-compose -f docker/docker-compose.yml down
	@echo "✅ Containers stopped"

deploy:
	@echo "🚀 Deploying system..."
	@./scripts/deploy.sh $(env)

backup:
	@echo "💾 Creating backup..."
	@./scripts/backup.sh

restore:
	@echo "🔄 Restoring from backup..."
	@./scripts/restore.sh

clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"