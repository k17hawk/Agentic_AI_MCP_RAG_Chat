#!/bin/bash
# scripts/rollback.sh
# Rollback to previous version

set -e

echo "========================================="
echo "↩️ Rolling back Trading System"
echo "========================================="

ENV=${1:-production}

if [ "$ENV" == "production" ]; then
    echo "Rolling back production..."
    kubectl rollout undo deployment/trading-system -n production
    kubectl rollout status deployment/trading-system -n production
    
elif [ "$ENV" == "staging" ]; then
    echo "Rolling back staging..."
    kubectl rollout undo deployment/trading-system -n staging
    kubectl rollout status deployment/trading-system -n staging
    
elif [ "$ENV" == "docker" ]; then
    echo "Rolling back Docker deployment..."
    docker-compose -f docker/docker-compose.yml down
    docker-compose -f docker/docker-compose.yml up -d
fi

echo ""
echo "========================================="
echo "✅ Rollback complete!"
echo "========================================="