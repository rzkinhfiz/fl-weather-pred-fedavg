# 🚀 Kubernetes Deployment Guide - Federated Learning (Flower v1.8+)

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Deployment Steps](#deployment-steps)
5. [Kubectl Commands](#kubectl-commands)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Scaling & Production](#scaling--production)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This guide covers deploying a complete Federated Learning system on Kubernetes with:
- ✅ **1 Central Server** (Flower FedAvg aggregator, GCP e2-micro compatible)
- ✅ **15 City Clients** (PyTorch LSTM training, CPU-only)
- ✅ **Memory Optimized** (10.5GB total for 15 clients)
- ✅ **Production Ready** (health checks, RBAC, resource limits)

---

## 📦 Prerequisites

### Software Requirements

```bash
# 1. Kubernetes cluster (1.18+)
kubectl version --client
# Expected: version v1.24+

# 2. kubectl CLI
which kubectl
# Expected: /usr/bin/kubectl or similar

# 3. Docker images available in registry
# Option A: Local images (kind/minikube)
# Option B: Cloud registry (GCP Container Registry, Docker Hub)

# 4. Sufficient cluster resources
# - CPU: 8+ cores total
# - Memory: 12GB+ available
# - Storage: 5GB+ free
```

### Cluster Setup Options

#### Option 1: Local Testing (Minikube)
```bash
minikube start --cpus 8 --memory 12288 --disk-size 20gb
eval $(minikube docker-env)  # Use minikube's Docker daemon
```

#### Option 2: Local Testing (Kind)
```bash
cat > kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 8080
        hostPort: 8080
        protocol: TCP
  - role: worker
  - role: worker
EOF

kind create cluster --config kind-config.yaml
```

#### Option 3: GCP Kubernetes Engine (Production)
```bash
# Create GKE cluster
gcloud container clusters create flower-fl \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-stackdriver-kubernetes

# Get credentials
gcloud container clusters get-credentials flower-fl --zone us-central1-a
```

#### Option 4: AWS EKS
```bash
eksctl create cluster --name flower-fl --region us-east-1 --nodes 3
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Namespace: flower-fl                             │ │
│  │                                                     │ │
│  │  ┌──────────────────────────────────────────────┐ │ │
│  │  │ Deployment: fl-server (1 replica)          │ │ │
│  │  │ ├─ Pod: fl-server-xxxxx                    │ │ │
│  │  │ │  ├─ CPU: 100m req, 200m limit            │ │ │
│  │  │ │  ├─ Memory: 128Mi req, 256Mi limit       │ │ │
│  │  │ │  └─ Port: 8080 (gRPC)                    │ │ │
│  │  │ └─ Service: fl-server (LoadBalancer)       │ │ │
│  │  │                                             │ │ │
│  │  ├──────────────────────────────────────────────┤ │ │
│  │  │ Pods: fl-client-0-berlin (through ...)    │ │ │
│  │  │ ├─ fl-client-0-berlin      (CPU/Memory)   │ │ │
│  │  │ ├─ fl-client-1-cairo       (CPU/Memory)   │ │ │
│  │  │ ├─ ...                                     │ │ │
│  │  │ └─ fl-client-14-toronto    (CPU/Memory)   │ │ │
│  │  │                                             │ │ │
│  │  │ Resource Summary:                          │ │ │
│  │  │ - Total CPU: ~8 cores (0.5 × 15 + 0.2)    │ │ │
│  │  │ - Total Memory: ~10.5GB (0.75 × 15 + 0.25)│ │ │
│  │  └──────────────────────────────────────────────┘ │ │
│  │                                                     │ │
│  │  ConfigMaps, ServiceAccounts, RBAC               │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Deployment Steps

### Step 1: Prepare Docker Images

```bash
# Navigate to project
cd /home/rna_13/FedLearning/flweatherpred

# Build server image
docker build -f Dockerfile.server -t fl-weather-server:latest .

# Build client image
docker build -f Dockerfile.client -t fl-weather-client:latest .

# Verify images
docker images | grep fl-weather
```

### Step 2: Push Images to Registry

#### Option A: Local Development (Minikube/Kind)
```bash
# Minikube: images are accessible directly
# Kind: load images into cluster
kind load docker-image fl-weather-server:latest
kind load docker-image fl-weather-client:latest

# Verify
docker exec kind-worker docker images | grep fl-weather
```

#### Option B: Cloud Registry (GCP)
```bash
# Tag images for GCP
docker tag fl-weather-server:latest gcr.io/YOUR_PROJECT_ID/fl-weather-server:latest
docker tag fl-weather-client:latest gcr.io/YOUR_PROJECT_ID/fl-weather-client:latest

# Push to registry
docker push gcr.io/YOUR_PROJECT_ID/fl-weather-server:latest
docker push gcr.io/YOUR_PROJECT_ID/fl-weather-client:latest

# Update image references in YAML files
sed -i 's|fl-weather-server:latest|gcr.io/YOUR_PROJECT_ID/fl-weather-server:latest|g' fl-server.yaml
sed -i 's|fl-weather-client:latest|gcr.io/YOUR_PROJECT_ID/fl-weather-client:latest|g' fl-clients.yaml
```

#### Option C: Docker Hub
```bash
# Tag and push
docker tag fl-weather-server:latest DOCKER_USERNAME/fl-weather-server:latest
docker push DOCKER_USERNAME/fl-weather-server:latest

docker tag fl-weather-client:latest DOCKER_USERNAME/fl-weather-client:latest
docker push DOCKER_USERNAME/fl-weather-client:latest

# Update references in YAML
sed -i 's|fl-weather-server:latest|DOCKER_USERNAME/fl-weather-server:latest|g' fl-server.yaml
sed -i 's|fl-weather-client:latest|DOCKER_USERNAME/fl-weather-client:latest|g' fl-clients.yaml
```

### Step 3: Deploy Server

```bash
# Apply server deployment
kubectl apply -f fl-server.yaml

# Verify deployment
kubectl get deployment -n flower-fl
kubectl get pods -n flower-fl -l component=server

# Wait for server to be ready
kubectl wait --for=condition=ready pod \
  -l app=flower,component=server \
  -n flower-fl \
  --timeout=300s

# Get server service
kubectl get svc -n flower-fl fl-server

# Get external IP (for clients to connect)
EXTERNAL_IP=$(kubectl get svc -n flower-fl fl-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Server external IP: $EXTERNAL_IP"
```

### Step 4: Deploy Clients

#### Option A: Auto-Generate All 15 Clients
```bash
# Generate all 15 client manifests
bash generate-fl-clients.sh > fl-clients-generated.yaml

# Apply to cluster
kubectl apply -f fl-clients-generated.yaml

# Verify all clients are running
kubectl get pods -n flower-fl -l component=client
```

#### Option B: Manual (Using Template)
```bash
# Apply the provided fl-clients.yaml (contains 3 + template for 12)
kubectl apply -f fl-clients.yaml

# Then manually copy/paste template 12 times for remaining clients
# (Edit fl-clients.yaml to add clients 3-14)
```

### Step 5: Verify Deployment

```bash
# List all resources
kubectl get all -n flower-fl

# Verify server is ready
kubectl get pods -n flower-fl -l component=server -o wide

# Verify clients are ready
kubectl get pods -n flower-fl -l component=client -o wide

# Expected output:
# NAME                                READY   STATUS    RESTARTS
# fl-server-xxxxx                    1/1     Running   0
# fl-client-0-berlin                 1/1     Running   0
# fl-client-1-cairo                  1/1     Running   0
# ... (15 clients total)
```

---

## 💻 Kubectl Commands

### Status & Information

```bash
# Get all pods
kubectl get pods -n flower-fl

# Get pods with more details
kubectl get pods -n flower-fl -o wide

# Describe specific pod
kubectl describe pod -n flower-fl fl-server-xxxxx

# Get resource usage
kubectl top pods -n flower-fl  # Requires metrics-server

# Get events
kubectl get events -n flower-fl --sort-by='.lastTimestamp'
```

### Logs & Debugging

```bash
# View server logs (real-time)
kubectl logs -f deployment/fl-server -n flower-fl

# View client logs (real-time)
kubectl logs -f pod/fl-client-0-berlin -n flower-fl

# View all client logs
kubectl logs -f -l component=client -n flower-fl

# View logs with timestamps
kubectl logs -f --timestamps=true deployment/fl-server -n flower-fl

# View previous logs (if pod restarted)
kubectl logs --previous pod/fl-client-0-berlin -n flower-fl

# Tail last 100 lines
kubectl logs --tail=100 pod/fl-server-xxxxx -n flower-fl
```

### Scaling & Management

```bash
# Scale server replicas (usually keep at 1)
kubectl scale deployment/fl-server --replicas=1 -n flower-fl

# Restart deployment
kubectl rollout restart deployment/fl-server -n flower-fl

# Check rollout status
kubectl rollout status deployment/fl-server -n flower-fl

# Rollback to previous version
kubectl rollout undo deployment/fl-server -n flower-fl
```

### Port Forwarding (for local testing)

```bash
# Forward server port
kubectl port-forward -n flower-fl svc/fl-server 8080:8080

# Forward specific pod
kubectl port-forward -n flower-fl pod/fl-client-0-berlin 8000:8000

# Forward from different port
kubectl port-forward -n flower-fl svc/fl-server 9000:8080
```

### Exec into Pod

```bash
# Open shell in server pod
kubectl exec -it deployment/fl-server -n flower-fl -- /bin/bash

# Run command in client pod
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- python -c "import torch; print(torch.__version__)"

# Check data mount
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- ls -la /app/data/processed/
```

### Cleanup

```bash
# Delete specific pod (will restart due to RestartPolicy)
kubectl delete pod fl-client-0-berlin -n flower-fl

# Delete all clients
kubectl delete pods -l component=client -n flower-fl

# Delete entire deployment
kubectl delete -f fl-server.yaml
kubectl delete -f fl-clients.yaml

# Delete namespace (removes everything)
kubectl delete namespace flower-fl
```

---

## 📊 Monitoring & Debugging

### Monitor Resource Usage

```bash
# Real-time resource usage
kubectl top pods -n flower-fl --sort-by=memory

# Watch in real-time
watch kubectl top pods -n flower-fl

# Expected output:
# NAME                        CPU(m)   MEMORY(Mi)
# fl-server-xxxxx            50m      180Mi
# fl-client-0-berlin         400m     700Mi
# fl-client-1-cairo          410m     705Mi
# ... total ~10.5GB memory
```

### Check Pod Status

```bash
# Detailed pod info
kubectl get pods -n flower-fl -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,CPU_REQ:.spec.containers[0].resources.requests.cpu,MEM_REQ:.spec.containers[0].resources.requests.memory

# Check init container status
kubectl describe pod -n flower-fl fl-client-0-berlin | grep -A 10 "Init Containers"

# Check events for pod
kubectl describe pod -n flower-fl fl-client-0-berlin | grep -A 10 "Events:"
```

### Verify Networking

```bash
# Test DNS resolution from pod
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- \
  nslookup fl-server.flower-fl.svc.cluster.local

# Test connectivity to server
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- \
  nc -zv fl-server.flower-fl.svc.cluster.local 8080

# Get service endpoint
kubectl get endpoints -n flower-fl fl-server
```

### View Training Progress

```bash
# Monitor server aggregation (watch logs)
kubectl logs -f deployment/fl-server -n flower-fl | grep "Round\|Aggregating"

# Monitor client training
kubectl logs -f pod/fl-client-0-berlin -n flower-fl | grep "Training\|Epoch"

# Tail all client logs in real-time
kubectl logs -f -l component=client -n flower-fl --all-containers=true
```

---

## 🔧 Scaling & Production

### Add More Clients (Beyond 15)

If you need more than 15 cities:

```bash
# Option 1: Modify generate script
# Edit generate-fl-clients.sh, add more cities to CITIES array

# Option 2: Use Kustomization for parameterized deployment
cat > kustomization.yaml << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - fl-server.yaml

vars:
  - name: CLIENT_COUNT
    value: "20"  # Change to deploy 20 clients

patchesStrategicMerge:
  - clients-patch.yaml
EOF
```

### High Availability Server

```bash
# Edit fl-server.yaml, change replicas to 2+
# WARNING: This requires special Flower setup for multi-server federation
# For now, keep replicas: 1 (standard setup)
```

### Resource Quotas (for multi-tenant clusters)

```bash
# Set namespace quotas
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: flower-fl-quota
  namespace: flower-fl
spec:
  hard:
    requests.cpu: "8"
    requests.memory: "12Gi"
    limits.cpu: "12"
    limits.memory: "16Gi"
    pods: "20"
EOF
```

### Network Policies (Security)

```bash
# Restrict traffic to only server <-> clients
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: flower-fl-network-policy
  namespace: flower-fl
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: flower-fl
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: flower-fl
EOF
```

---

## 🐛 Troubleshooting

### Pod Stuck in "Pending"

```bash
# Check why pod is pending
kubectl describe pod -n flower-fl fl-client-0-berlin

# Likely causes:
# 1. Insufficient resources
#    Fix: Scale down other pods or add nodes
#    kubectl describe nodes | grep -A 5 "Allocated resources"

# 2. Image not found
#    Fix: Push images to accessible registry
#    docker push <image>

# 3. PullBackOff - registry auth issue
#    Fix: Create image pull secret
#    kubectl create secret docker-registry regcred \
#      --docker-server=<registry> \
#      --docker-username=<user> \
#      --docker-password=<pass>
```

### Pod OOMKilled

```bash
# Check for OOM events
kubectl describe pod -n flower-fl fl-client-0-berlin | grep OOMKilled

# Solutions:
# 1. Increase memory limit
#    kubectl set resources pod fl-client-0-berlin \
#      --limits=memory=1Gi -n flower-fl

# 2. Reduce resource requests (hint for scheduler)
#    Edit YAML, reduce requests

# 3. Monitor actual usage
#    kubectl top pod -n flower-fl fl-client-0-berlin
```

### Init Container Hanging

```bash
# Check init container logs
kubectl logs -n flower-fl fl-client-0-berlin -c wait-for-server

# If server not reachable, ensure server is running
kubectl get pods -n flower-fl -l component=server

# Test DNS from client
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- \
  nslookup fl-server.flower-fl.svc.cluster.local

# If DNS fails, check coredns
kubectl get pods -n kube-system | grep coredns
```

### Clients Can't Connect to Server

```bash
# 1. Verify server is running
kubectl get pods -n flower-fl -l component=server

# 2. Verify service is created
kubectl get svc -n flower-fl fl-server

# 3. Test from client pod
kubectl exec -it pod/fl-client-0-berlin -n flower-fl -- \
  python -c "import socket; socket.create_connection(('fl-server.flower-fl.svc.cluster.local', 8080))"

# 4. Check server logs for gRPC errors
kubectl logs deployment/fl-server -n flower-fl | grep -i "grpc\|error"

# 5. Port-forward and test locally
kubectl port-forward -n flower-fl svc/fl-server 8080:8080
# In another terminal:
python -c "import socket; socket.create_connection(('localhost', 8080))"
```

### Training Not Starting

```bash
# 1. Check server logs for "Round"
kubectl logs -f deployment/fl-server -n flower-fl | grep "Round"

# 2. Check if enough clients connected
kubectl logs deployment/fl-server -n flower-fl | grep -i "clients\|available"

# 3. Verify MIN_FIT_CLIENTS in ConfigMap
kubectl get configmap -n flower-fl fl-server-config -o yaml

# 4. Check client logs for connection attempts
kubectl logs pod/fl-client-0-berlin -n flower-fl | grep -i "connect\|error"
```

### High Memory Usage

```bash
# Identify high-memory pods
kubectl top pods -n flower-fl --sort-by=memory

# Reduce client memory limit (edit YAML)
# Change: memory: 750Mi → memory: 600Mi
# Reapply: kubectl apply -f fl-clients.yaml

# Check actual vs requested
kubectl get pods -n flower-fl -o custom-columns=NAME:.metadata.name,MEMORY_REQUEST:.spec.containers[0].resources.requests.memory,MEMORY_LIMIT:.spec.containers[0].resources.limits.memory
```

### Pod CrashLooping

```bash
# Check restart count
kubectl get pods -n flower-fl -o wide

# View crash logs
kubectl logs -n flower-fl fl-client-0-berlin --previous

# Common causes:
# 1. Image not found → Check image name/tag
# 2. Python error → Check application logs
# 3. Out of memory → Increase limits

# Fix and restart
kubectl delete pod fl-client-0-berlin -n flower-fl
# Pod will automatically restart
```

---

## 📚 Additional Resources

- **Kubernetes Docs**: https://kubernetes.io/docs/
- **Kubectl Cheat Sheet**: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- **Flower Framework**: https://flower.ai/docs/framework/
- **GKE Documentation**: https://cloud.google.com/kubernetes-engine/docs
- **Minikube**: https://minikube.sigs.k8s.io/

---

## 🎯 Quick Reference

### Deploy Everything

```bash
# 1. Prep cluster
kubectl create namespace flower-fl

# 2. Build & load images
docker build -f Dockerfile.server -t fl-weather-server:latest .
docker build -f Dockerfile.client -t fl-weather-client:latest .
kind load docker-image fl-weather-server:latest
kind load docker-image fl-weather-client:latest

# 3. Deploy server
kubectl apply -f fl-server.yaml

# 4. Deploy clients (auto-generate)
bash generate-fl-clients.sh | kubectl apply -f -

# 5. Wait for ready
kubectl wait --for=condition=ready pod -l app=flower -n flower-fl --timeout=300s

# 6. Check status
kubectl get pods -n flower-fl -l component=server -o wide
kubectl get pods -n flower-fl -l component=client -o wide

# 7. Monitor
kubectl logs -f deployment/fl-server -n flower-fl
```

### Monitor Training

```bash
# Terminal 1: Server logs
kubectl logs -f deployment/fl-server -n flower-fl

# Terminal 2: Client logs
kubectl logs -f pod/fl-client-0-berlin -n flower-fl

# Terminal 3: Resource usage
watch kubectl top pods -n flower-fl --sort-by=memory
```

### Cleanup

```bash
kubectl delete namespace flower-fl
```

---

**Status**: ✅ PRODUCTION READY  
**Version**: 1.0  
**Last Updated**: June 5, 2026  

Happy Federated Learning! 🚀
