#!/bin/bash
# scripts/deploy.sh
# Deploy the trading system to production

set -e

echo "========================================="
echo "🚀 Deploying Trading System"
echo "========================================="

# Configuration
ENV=${1:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}
IMAGE_NAME="trading-system"
TAG=$(git rev-parse --short HEAD)

echo "Environment: $ENV"
echo "Image: $DOCKER_REGISTRY/$IMAGE_NAME:$TAG"
echo ""

# 1. Run tests
echo "📦 Running tests..."
python -m pytest tests/ -v --cov=.

# 2. Build Docker image
echo ""
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME:$TAG -f docker/Dockerfile .
docker tag $IMAGE_NAME:$TAG $DOCKER_REGISTRY/$IMAGE_NAME:$TAG

# 3. Push to registry
echo ""
echo "📦 Pushing to registry..."
docker push $DOCKER_REGISTRY/$IMAGE_NAME:$TAG

# 4. Deploy based on environment
if [ "$ENV" == "production" ]; then
    echo ""
    echo "📦 Deploying to production..."
    
    # Update Kubernetes deployment
    kubectl set image deployment/trading-system trading-system=$DOCKER_REGISTRY/$IMAGE_NAME:$TAG -n production
    
    # Wait for rollout
    kubectl rollout status deployment/trading-system -n production
    
elif [ "$ENV" == "staging" ]; then
    echo ""
    echo "📦 Deploying to staging..."
    
    kubectl set image deployment/trading-system trading-system=$DOCKER_REGISTRY/$IMAGE_NAME:$TAG -n staging
    kubectl rollout status deployment/trading-system -n staging
    
elif [ "$ENV" == "docker" ]; then
    echo ""
    echo "📦 Deploying with Docker Compose..."
    
    # Update docker-compose.yml with new image
    sed -i "s|image:.*trading-system.*|image: $DOCKER_REGISTRY/$IMAGE_NAME:$TAG|g" docker/docker-compose.yml
    
    # Restart services
    docker-compose -f docker/docker-compose.yml down
    docker-compose -f docker/docker-compose.yml up -d
fi

# 5. Health check
echo ""
echo "📦 Running health check..."
sleep 10
if [ "$ENV" == "docker" ]; then
    curl -f http://localhost:8000/health || echo "⚠️  Health check failed"
else
    kubectl get pods -l app=trading-system
fi

echo ""
echo "========================================="
echo "✅ Deployment complete!"
echo "========================================="