#!/bin/bash

# =============================================================================
# DOCKER QUICK START - Federated Learning Weather Prediction
# =============================================================================
# Script ini mengotomasi build dan startup seluruh sistem FL dengan 15 klien
# =============================================================================

set -e  # Exit on error

PROJECT_PATH="/home/rna_13/FedLearning/flweatherpred"
cd "$PROJECT_PATH"

echo "🚀 FL Weather Prediction - Docker Quick Start"
echo "=============================================="
echo ""

# Step 1: Check prerequisites
echo "✅ Step 1: Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "   ✓ Docker version: $(docker --version)"
echo "   ✓ Docker Compose version: $(docker compose version | head -1)"
echo ""

# Step 2: Check system resources
echo "✅ Step 2: Checking system resources..."
AVAILABLE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
echo "   ✓ Available RAM: $AVAILABLE_RAM (need ≥12GB)"
echo ""

# Step 3: Verify data files
echo "✅ Step 3: Verifying data files..."
CITY_COUNT=$(ls data/processed/*.csv | wc -l)
if [ "$CITY_COUNT" -eq 15 ]; then
    echo "   ✓ Found 15 city CSV files"
else
    echo "❌ Expected 15 cities, found $CITY_COUNT. Aborting."
    exit 1
fi
echo ""

# Step 4: Create logs directory
echo "✅ Step 4: Setting up directories..."
mkdir -p logs
chmod 777 logs
echo "   ✓ Created logs/ directory"
echo ""

# Step 5: Build Docker images
echo "✅ Step 5: Building Docker images..."
echo "   (This may take 5-10 minutes on first build)"
echo ""

docker compose build --no-cache

echo ""
echo "   ✓ Build complete!"
echo ""

# Step 6: Start services
echo "✅ Step 6: Starting FL services..."
docker compose up -d

echo "   ⏳ Waiting for services to stabilize (10s)..."
sleep 10

echo ""
echo "✅ All services started!"
echo ""

# Step 7: Verify status
echo "✅ Step 7: Verifying service status..."
echo ""

RUNNING=$(docker compose ps | grep -c "Up" || true)
EXPECTED=16  # 1 server + 15 clients

if [ "$RUNNING" -eq "$EXPECTED" ]; then
    echo "   ✓ All $RUNNING services running"
else
    echo "   ⚠ Expected $EXPECTED services, found $RUNNING running"
    echo "   (Some may still be starting, check 'docker compose ps' after 30s)"
fi

echo ""
echo "📊 Service Status:"
docker compose ps --no-trunc | head -3
echo "   ..."
echo "   (Use 'docker compose ps' for full list)"

echo ""
echo "💾 Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | head -5
echo "   ..."
echo "   (Use 'docker stats' for live monitoring)"

echo ""
echo "📋 Next Steps:"
echo ""
echo "   1. View all logs (real-time):"
echo "      docker compose logs -f"
echo ""
echo "   2. View specific service logs:"
echo "      docker compose logs -f fl-server"
echo "      docker compose logs -f fl-client-0"
echo ""
echo "   3. Monitor memory usage:"
echo "      docker stats"
echo ""
echo "   4. Stop all services:"
echo "      docker compose down"
echo ""
echo "   5. Full documentation:"
echo "      cat DOCKER_SETUP.md"
echo ""
echo "=============================================="
echo "✅ Docker setup complete!"
echo "=============================================="
