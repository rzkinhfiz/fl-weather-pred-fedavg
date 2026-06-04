#!/bin/bash

# =============================================================================
# setup-k8s.sh - Kubernetes Deployment Quick Setup Script
# =============================================================================
# This script automates the entire Kubernetes deployment process.
#
# Usage:
#   bash setup-k8s.sh                  # Interactive mode
#   bash setup-k8s.sh minikube         # Use minikube
#   bash setup-k8s.sh kind             # Use kind
#   bash setup-k8s.sh gcp              # Deploy to GCP
# =============================================================================

set -e

CLUSTER_TYPE=${1:-"local"}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 FL Kubernetes Deployment Setup                             ║"
echo "║  Federated Learning with Flower Framework v1.8+               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify prerequisites
echo "✅ Step 1: Checking Prerequisites..."
echo ""

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi
echo "   ✓ kubectl $(kubectl version --client --short)"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ docker not found. Please install Docker first."
    exit 1
fi
echo "   ✓ docker $(docker --version)"

# Check cluster type
if [ "$CLUSTER_TYPE" = "minikube" ]; then
    if ! command -v minikube &> /dev/null; then
        echo "❌ minikube not found. Please install minikube."
        exit 1
    fi
    echo "   ✓ Using Minikube"
elif [ "$CLUSTER_TYPE" = "kind" ]; then
    if ! command -v kind &> /dev/null; then
        echo "❌ kind not found. Please install kind."
        exit 1
    fi
    echo "   ✓ Using Kind"
elif [ "$CLUSTER_TYPE" = "gcp" ]; then
    if ! command -v gcloud &> /dev/null; then
        echo "❌ gcloud not found. Please install Google Cloud SDK."
        exit 1
    fi
    echo "   ✓ Using GCP"
else
    echo "   ✓ Using current kubectl context"
fi

echo ""

# Step 2: Build Docker images
echo "✅ Step 2: Building Docker Images..."
echo ""

docker build -f Dockerfile.server -t fl-weather-server:latest .
echo "   ✓ Built fl-weather-server:latest"

docker build -f Dockerfile.client -t fl-weather-client:latest .
echo "   ✓ Built fl-weather-client:latest"

echo ""

# Step 3: Load/Push images
echo "✅ Step 3: Making Images Available..."
echo ""

case "$CLUSTER_TYPE" in
    minikube)
        eval $(minikube docker-env)
        echo "   ✓ Using Minikube's Docker daemon"
        ;;
    kind)
        kind load docker-image fl-weather-server:latest
        kind load docker-image fl-weather-client:latest
        echo "   ✓ Loaded images into Kind cluster"
        ;;
    gcp)
        PROJECT_ID=$(gcloud config get-value project)
        docker tag fl-weather-server:latest gcr.io/$PROJECT_ID/fl-weather-server:latest
        docker tag fl-weather-client:latest gcr.io/$PROJECT_ID/fl-weather-client:latest
        docker push gcr.io/$PROJECT_ID/fl-weather-server:latest
        docker push gcr.io/$PROJECT_ID/fl-weather-client:latest
        
        # Update YAML files
        sed -i "s|fl-weather-server:latest|gcr.io/$PROJECT_ID/fl-weather-server:latest|g" fl-server.yaml
        sed -i "s|fl-weather-client:latest|gcr.io/$PROJECT_ID/fl-weather-client:latest|g" fl-clients.yaml
        
        echo "   ✓ Pushed images to GCP Container Registry"
        ;;
    *)
        echo "   ⚠ Using local images (make sure they're available to cluster)"
        ;;
esac

echo ""

# Step 4: Deploy to Kubernetes
echo "✅ Step 4: Deploying to Kubernetes..."
echo ""

# Apply server
kubectl apply -f fl-server.yaml
echo "   ✓ Applied fl-server.yaml"

# Generate and apply clients
bash generate-fl-clients.sh | kubectl apply -f -
echo "   ✓ Applied all 15 FL clients"

echo ""

# Step 5: Wait for deployment
echo "✅ Step 5: Waiting for Deployment to be Ready..."
echo ""

echo "   ⏳ Waiting for server pod (max 5 minutes)..."
kubectl wait --for=condition=ready pod \
  -l app=flower,component=server \
  -n flower-fl \
  --timeout=300s

echo "   ✓ Server is ready"

echo ""

# Step 6: Display status
echo "✅ Step 6: Deployment Status"
echo ""

echo "   Server:"
kubectl get pods -n flower-fl -l component=server -o wide | tail -1

echo ""
echo "   Clients:"
CLIENT_COUNT=$(kubectl get pods -n flower-fl -l component=client --no-headers | wc -l)
READY_COUNT=$(kubectl get pods -n flower-fl -l component=client --no-headers | grep "Running" | wc -l)
echo "   Status: $READY_COUNT/$CLIENT_COUNT running"

echo ""

# Step 7: Display next steps
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  📊 DEPLOYMENT COMPLETE                                         ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  🎯 Next Steps:                                                 ║"
echo "║                                                                 ║"
echo "║  1. Check all pods are running:                                 ║"
echo "║     kubectl get pods -n flower-fl                              ║"
echo "║                                                                 ║"
echo "║  2. Monitor server logs:                                        ║"
echo "║     kubectl logs -f deployment/fl-server -n flower-fl          ║"
echo "║                                                                 ║"
echo "║  3. Monitor client logs:                                        ║"
echo "║     kubectl logs -f pod/fl-client-0-berlin -n flower-fl        ║"
echo "║                                                                 ║"
echo "║  4. Check resource usage:                                       ║"
echo "║     kubectl top pods -n flower-fl                              ║"
echo "║                                                                 ║"
echo "║  5. View full deployment guide:                                 ║"
echo "║     cat KUBERNETES_DEPLOYMENT_GUIDE.md                         ║"
echo "║                                                                 ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  📞 Useful Commands:                                            ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  List all resources:                                            ║"
echo "║    kubectl get all -n flower-fl                                ║"
echo "║                                                                 ║"
echo "║  Describe pod:                                                  ║"
echo "║    kubectl describe pod fl-client-0-berlin -n flower-fl        ║"
echo "║                                                                 ║"
echo "║  Port-forward (for local testing):                              ║"
echo "║    kubectl port-forward svc/fl-server 8080:8080 -n flower-fl   ║"
echo "║                                                                 ║"
echo "║  Cleanup:                                                       ║"
echo "║    kubectl delete namespace flower-fl                          ║"
echo "║                                                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Setup complete! Federated Learning system is live on Kubernetes."
echo ""
