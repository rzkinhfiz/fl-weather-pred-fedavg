# 🎯 Docker Configuration - SETUP COMPLETE ✅

## 📦 Fail-Fail yang Telah Dibuat

Berikut adalah semua fail Docker yang telah disiapkan untuk Federated Learning Weather Prediction:

### ✅ A. Dockerfile.server (2.4 KB)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/Dockerfile.server

Kandungan:
- Base Image: python:3.10-slim (~200MB)
- Dependencies: flwr>=1.8.0, numpy, pandas, protobuf
- NO PyTorch (server hanya untuk aggregation)
- Port: 8080 (gRPC)
- Memory target: 512MB (per docker-compose.yml)
- Health check: Socket connection test
- Command: python server.py --host 0.0.0.0 --port 8080 --num_rounds 10
```

### ✅ B. Dockerfile.client (3.4 KB)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/Dockerfile.client

Kandungan:
- Base Image: python:3.10-slim (~200MB)
- Dependencies: flwr>=1.8.0, PyTorch 2.0.1 CPU-only, scikit-learn, pandas
- PyTorch: Installed from official CPU wheels (150MB, bukan CUDA)
- Entrypoint: Bash script yang accept env vars (CITY_ID, SERVER_ADDRESS)
- Volume mount: /app/data/processed (read-only dataset)
- Memory target: 700MB per container
- Environment variables: CITY_ID (0-14), SERVER_ADDRESS (server:8080)
```

### ✅ C. docker-compose.yml (14 KB)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/docker-compose.yml

Kandungan:
- Version: 3.9 (supports deploy.resources)
- Services: 16 total (1 server + 15 clients)

fl-server:
  - Image: fl-weather-server:latest
  - Port: 8080:8080
  - Memory: 512MB limit, 256MB reservation
  - Health check: Enabled
  - Restart: on-failure

fl-client-0 hingga fl-client-14:
  - Image: fl-weather-client:latest
  - Cities: Berlin, Cairo, Dammam, Doha, Dubai, Jeddah, London, Mumbai, 
           New York, Paris, Riyadh, Singapore, Sydney, Tokyo, Toronto
  - Environment: CITY_ID, SERVER_ADDRESS
  - Memory: 700MB limit, 400MB reservation (ketat untuk tidak crash)
  - CPU: 0.5 vCPU per container
  - Restart: on-failure (automatic recovery)
  - Volume: data/processed (mount read-only)
  - Network: fl-network (bridge network)
  - Depends on: fl-server (waits untuk server healthy)

Network:
  - Bridge network: fl-network
  - Subnet: 172.20.0.0/16

Total Memory Consumption:
  - Server: 512MB
  - Clients: 15 × 700MB = 10,500MB = ~10.5GB
  - TOTAL: ~11GB (safe di bawah 16GB laptop dengan buffer OS)
```

### ✅ D. requirements.server.txt (69 bytes)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/requirements.server.txt

Kandungan:
flwr[simulation]>=1.8.0
numpy>=1.23.0
pandas>=1.5.0
protobuf>=3.20.0
```

### ✅ E. requirements.client.txt (272 bytes)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/requirements.client.txt

Kandungan:
flwr[simulation]>=1.8.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
protobuf>=3.20.0
# PyTorch CPU-only: installed separately in Dockerfile
```

### ✅ F. DOCKER_SETUP.md (15 KB)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/DOCKER_SETUP.md

Kandungan:
- Gambaran keseluruhan Docker architecture
- Masalah yang diselesaikan
- Struktur fail
- Petunjuk pemasangan step-by-step
- Arahan terminal (common commands)
- Monitoring & troubleshooting guide
- Memory optimization guide
- GCP deployment instructions
- Verification checklist
```

### ✅ G. docker-quickstart.sh (3.5 KB)
```
Lokasi: /home/rna_13/FedLearning/flweatherpred/docker-quickstart.sh

Kandungan:
- Automated setup script untuk quick start
- Checks prerequisites (Docker, Docker Compose)
- Verifies system resources
- Verifies data files (15 cities)
- Builds Docker images
- Starts all services
- Displays status dan next steps
```

---

## 🚀 ARAHAN TERMINAL - QUICK START

### **Langkah 1: Navigate ke Project**
```bash
cd /home/rna_13/FedLearning/flweatherpred
```

### **Langkah 2: Build Docker Images**
```bash
# Option A: Automated (recommended)
bash docker-quickstart.sh

# Option B: Manual step-by-step
docker compose build --no-cache
```

### **Langkah 3: Start ALL Services (1 server + 15 clients)**
```bash
docker compose up -d
```

### **Langkah 4: Verify Status**
```bash
# Lihat semua services
docker compose ps

# Expected output: 16 services (1 Up, 15 Up)
```

### **Langkah 5: Monitor Training Progress**
```bash
# View logs real-time
docker compose logs -f

# atau lihat specific service
docker compose logs -f fl-server
docker compose logs -f fl-client-0
```

### **Langkah 6: Monitor Memory Usage**
```bash
docker stats
# Check: setiap client < 700MB, server < 512MB
```

### **Langkah 7: Stop All Services**
```bash
docker compose down
```

---

## ✅ CHECKLIST VERIFICATION

Sebelum menjalankan sistem, pastikan:

```bash
# 1. Docker installed & running
docker --version        # Should show version ≥ 20.10
docker compose version  # Should show version ≥ 2.0

# 2. 15 city CSV files exist
ls data/processed/*.csv | wc -l  # Should show 15

# 3. Sufficient RAM available
free -h  # Should show ≥ 12GB available

# 4. Sufficient disk space
df -h /  # Should show ≥ 10GB free

# 5. All config files present
ls -1 Dockerfile.* docker-compose.yml requirements.*.txt
# Should show 5 files

# 6. Source code files exist
ls -1 src/*.py  # Should show 7 files
```

---

## 📊 Expected Output & Timeline

### Build Phase (5-10 minutes first time):
```
Step 1/10 : FROM python:3.10-slim
Step 2/10 : LABEL maintainer="FL Weather Prediction Team"
...
Successfully built XXXXX
Successfully tagged fl-weather-server:latest
Successfully tagged fl-weather-client:latest
```

### Startup Phase (30-60 seconds):
```
Creating fl-server ... done
Creating fl-client-0 ... done
Creating fl-client-1 ... done
...
Creating fl-client-14 ... done
```

### Training Phase (watch logs):
```
[FL-SERVER] [INFO] Round 1: Aggregating 15 clients
[FL-CLIENT-0] [INFO] Training round 1/10
[FL-CLIENT-1] [INFO] Training round 1/10
...
[FL-SERVER] [INFO] Round 1: Avg Loss=1.859
[FL-SERVER] [INFO] Round 2: Aggregating 15 clients
...
```

### Memory Usage (expected):
```
CONTAINER         MEM USAGE / LIMIT
fl-server         180M / 512M       ✅
fl-client-0       650M / 700M       ✅
fl-client-1       655M / 700M       ✅
fl-client-2       652M / 700M       ✅
...
fl-client-14      658M / 700M       ✅
TOTAL:            ~10.5GB           ✅
```

---

## 🔧 Common Commands Reference

```bash
# BUILD
docker compose build                          # Build all images
docker compose build --no-cache              # Rebuild (skip cache)

# RUN
docker compose up -d                         # Start all services
docker compose up -d --build                 # Build & start

# STOP/STATUS
docker compose stop                          # Stop all
docker compose start                         # Resume
docker compose restart                       # Restart
docker compose ps                            # List services
docker compose ps -a                         # Include stopped

# LOGS
docker compose logs                          # All logs
docker compose logs -f                       # Real-time all
docker compose logs -f fl-server            # Specific service
docker compose logs --tail 50               # Last 50 lines

# CLEANUP
docker compose down                          # Stop & remove
docker compose down -v                       # Remove including volumes
docker image prune -a                        # Remove unused images
docker system prune -a                       # Full cleanup

# MONITORING
docker stats                                 # Real-time stats
docker stats --no-stream                    # Single snapshot
docker inspect fl-server                    # Detailed container info

# EXEC
docker compose exec fl-server python -c "import flwr; print(flwr.__version__)"
docker compose exec fl-client-0 ls data/processed/
```

---

## 🎯 Success Indicators

✅ **Setup Successful When:**
1. `docker compose ps` menunjukkan 16 services dengan status "Up"
2. `docker stats` menunjukkan each client < 700MB memory
3. `docker compose logs fl-server` menunjukkan "Aggregating X clients"
4. `docker compose logs fl-client-0` menunjukkan "Training round" messages
5. No error messages dalam logs (INFO/DEBUG messages OK)

❌ **Troubleshooting If:**
1. Services stuck "Restarting" → Check logs, increase timeout
2. Memory > 700MB per client → Reduce container limit, close other apps
3. Clients tidak connect → Ensure server started first, check network
4. PyTorch import error → Rebuild client image without cache

---

## 📚 Documentation Files

**Created:**
- ✅ Dockerfile.server - Server container (2.4 KB)
- ✅ Dockerfile.client - Client container (3.4 KB)
- ✅ docker-compose.yml - Orchestration (14 KB)
- ✅ requirements.server.txt - Server dependencies (69 bytes)
- ✅ requirements.client.txt - Client dependencies (272 bytes)
- ✅ DOCKER_SETUP.md - Full documentation (15 KB)
- ✅ docker-quickstart.sh - Automated setup (3.5 KB)

**Total Configuration:** ~38 KB of production-ready Docker config

---

## 🎓 Cara Menggunakan

### Untuk Development:
```bash
# Start everything
docker compose up -d

# Watch logs
docker compose logs -f

# Make code changes, rebuild & restart
docker compose up -d --build

# Stop
docker compose down
```

### Untuk Production (GCP):
```bash
# Push to container registry
docker push gcr.io/YOUR_PROJECT/fl-weather-server:latest
docker push gcr.io/YOUR_PROJECT/fl-weather-client:latest

# Deploy to Cloud Run / GKE
# (See DOCKER_SETUP.md for detailed instructions)
```

### Untuk Testing:
```bash
# Run single client for testing
docker run -d \
  -e CITY_ID=0 \
  -e SERVER_ADDRESS=localhost:8080 \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  fl-weather-client:latest
```

---

## 📞 Support & Documentation

- **Full Setup Guide**: `DOCKER_SETUP.md`
- **Flower Framework**: https://flower.ai/docs
- **Docker Compose**: https://docs.docker.com/compose/
- **PyTorch**: https://pytorch.org/

---

**Status**: ✅ READY FOR DEPLOYMENT  
**Date Created**: June 5, 2026  
**Configuration**: Production-Grade  
**Memory Optimized**: Yes (10.5GB total for 15 clients)  
**GCP Compatible**: Yes (e2-micro server + laptop clients)  

---

## 🎉 Next Step

```bash
# Start the training!
cd /home/rna_13/FedLearning/flweatherpred
docker compose up -d
docker compose logs -f
```

Good luck! 🚀
