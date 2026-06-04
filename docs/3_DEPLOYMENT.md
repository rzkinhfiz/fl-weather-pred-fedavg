# Federated Learning Weather Prediction: Deployment & Setup Guide

**Document Version:** 1.0  
**Last Updated:** June 2026  
**Status:** Production-Ready  
**Classification:** Deployment & Operations Guide

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Installation Procedures](#installation-procedures)
4. [Environment Configuration](#environment-configuration)
5. [Running the Simulation](#running-the-simulation)
6. [Docker Deployment](#docker-deployment)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Logging](#monitoring--logging)
9. [Maintenance & Updates](#maintenance--updates)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### 1.1 Hardware Requirements

#### **Minimum Configuration** (For Testing)

```
┌─────────────────────────────────────┐
│     MINIMUM HARDWARE (Testing)      │
├─────────────────────────────────────┤
│ GPU:        NVIDIA GPU              │
│             (4 GB VRAM minimum)     │
│ CPU:        Intel/AMD 6+ cores      │
│ RAM:        8 GB DDR4/DDR5          │
│ Storage:    50 GB SSD               │
│ Network:    100 Mbps Ethernet       │
└─────────────────────────────────────┘

Time per round: ~15-20 seconds
Total 10 rounds: ~2-3 minutes
```

#### **Recommended Configuration** (Production)

```
┌─────────────────────────────────────┐
│   RECOMMENDED HARDWARE (Production) │
├─────────────────────────────────────┤
│ GPU:        NVIDIA RTX 40-series    │
│             (8+ GB VRAM)            │
│ CPU:        Intel Core i9 / Xeon    │
│             (16+ cores)             │
│ RAM:        32 GB DDR5              │
│ Storage:    500 GB NVMe SSD         │
│ Network:    1 Gbps Ethernet         │
└─────────────────────────────────────┘

Time per round: ~2-5 seconds
Total 10 rounds: ~20-50 seconds
Multi-GPU: Scales to 4-8 GPUs
```

#### **Cloud Deployment** (AWS/GCP/Azure)

```
Recommended Instance Types:

AWS:
  - Single GPU: p3.2xlarge (1× V100, 61 GB RAM)
  - Multi-GPU: p3.8xlarge (4× V100, 244 GB RAM)
  - Cost: $3.06-$12.24 per hour

GCP:
  - Single GPU: n1-standard-8 + 1× Tesla V100
  - Multi-GPU: n1-standard-32 + 4× Tesla V100
  - Cost: $1.30-$5.20 per GPU-hour

Azure:
  - Single GPU: Standard_NC6s_v3 (1× V100, 112 GB RAM)
  - Multi-GPU: Standard_NC24s_v3 (4× V100, 448 GB RAM)
  - Cost: $2.28-$9.12 per GPU-hour
```

### 1.2 Software Requirements

#### **Required Packages**

```
Python:             3.13.x
PyTorch:            >=2.3.0 with CUDA 13.3
Flower:             >=1.8.0
Ray:                >=2.10.0
scikit-learn:       >=1.3.0
pandas:             >=2.0.0
numpy:              >=1.24.0
```

#### **Optional Packages**

```
Jupyter:            For notebooks (EDA)
matplotlib:         For visualization
seaborn:            For advanced plots
tensorboard:        For metrics visualization
pytest:             For testing
black:              For code formatting
```

#### **System Dependencies** (Linux)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3.13-dev \
    python3.13-venv \
    git \
    curl \
    wget

# CUDA Toolkit (if not pre-installed)
# Download from: https://developer.nvidia.com/cuda-13-3-downloads
# Or use: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

# cuDNN (required for CUDA acceleration)
# Download from: https://developer.nvidia.com/cudnn
# Extract to: /usr/local/cuda
```

#### **System Dependencies** (macOS)

```bash
# Using Homebrew
brew install python@3.13
brew install git
brew install curl

# Note: Apple Silicon (M1/M2/M3) uses GPU via Metal
# CUDA not available; PyTorch uses MPS backend
```

#### **System Dependencies** (Windows)

```powershell
# Using Chocolatey (run as Administrator)
choco install python --version=3.13
choco install git
choco install visual-studio-build-tools

# CUDA Toolkit from NVIDIA website
# Visual Studio Build Tools required for CUDA
```

### 1.3 Network Requirements

```
Firewall Rules (if distributed across machines):
  - Ray Head Node: Port 6379 (Redis), 6380-6382
  - Ray Worker Nodes: Ephemeral ports (36000-36100)
  - Flower Server: Port 8080 (optional, for monitoring)
  
For in-process simulation:
  - No network ports required
  - Ray communicates via IPC
```

---

## Pre-Deployment Checklist

### 2.1 System Verification Checklist

```
HARDWARE VERIFICATION
□ GPU available and detected by CUDA
  Command: nvidia-smi
  Expected: NVIDIA GPU listed with compute capability >= 3.0
  
□ VRAM sufficient
  Command: nvidia-smi | grep "Default GPU / Memory"
  Expected: >= 4 GB (minimum), >= 8 GB (recommended)
  
□ CPU cores available
  Command: nproc
  Expected: >= 6 cores
  
□ RAM available
  Command: free -h
  Expected: >= 8 GB free (minimum)
  
□ Storage space
  Command: df -h /
  Expected: >= 50 GB free for logs, data, models

CUDA VERIFICATION
□ CUDA Toolkit installed
  Command: nvcc --version
  Expected: CUDA Compilation Tools, release 13.3
  
□ cuDNN installed
  Command: find /usr/local/cuda -name libcudnn*
  Expected: Multiple cuDNN library files found
  
□ CUDA environment variables
  Command: echo $CUDA_HOME
  Expected: /usr/local/cuda (or similar)

PYTHON VERIFICATION
□ Python 3.13 installed
  Command: python3 --version
  Expected: Python 3.13.x
  
□ pip installed
  Command: pip --version
  Expected: pip X.X.X from /path/to/python3.13
  
□ Virtual environment available
  Command: python3 -m venv --help
  Expected: Help text displayed (no error)
```

### 2.2 Pre-Deployment Security Checklist

```
SECURITY CHECKLIST
□ File permissions on data directory
  Command: ls -la data/
  Expected: Read/write access for current user
  
□ Data encryption at rest (optional)
  For production: Enable disk encryption for data/ directory
  
□ Model checkpoints secured
  Expected: Restricted read access (chmod 600)
  
□ Credentials not in code
  Check: grep -r "password\|api_key\|token" src/
  Expected: No matches (all in environment variables)
  
□ Requirements.txt pinned versions
  Expected: All packages have exact versions (==, not >=)
  
□ Git repository cleaned
  Command: git status
  Expected: No uncommitted changes in src/
  
□ Audit logging enabled
  Expected: logs/ directory exists and writable
```

### 2.3 Data Preparation Checklist

```
DATA CHECKLIST
□ All 15 city CSV files present
  Command: ls -1 data/processed/ | wc -l
  Expected: 15 files
  
□ Each CSV has exactly 365 rows
  Command: for f in data/processed/*.csv; do wc -l $f; done
  Expected: All show 365 (or 366 with header)
  
□ No missing values in numeric columns
  Expected: 0 NaN values in [Temperature_C, Humidity_%, Wind_Speed_km_h, Precipitation_mm]
  
□ Date coverage: Full year 2025
  Expected: January 1 - December 31, 2025
  
□ CSV formatting verified
  Command: head -1 data/processed/berlin.csv
  Expected: Comma-separated with headers
```

---

## Installation Procedures

### 3.1 Quick Start (5 Minutes)

#### **Step 1: Clone Repository**

```bash
# Clone the project
git clone https://github.com/your-org/flweatherpred.git
cd flweatherpred

# Verify structure
ls -la
# Expected: data/, src/, notebooks/, docs/, requirements.txt, README.md
```

#### **Step 2: Create Virtual Environment**

```bash
# Create isolated Python environment
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Verify activation (prompt should show (venv))
which python  # Should show /path/to/venv/bin/python
```

#### **Step 3: Install Dependencies**

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
# Expected: All packages installed successfully
# Time: ~2-5 minutes depending on network

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

#### **Step 4: Run Simulation**

```bash
# Quick test (no full training)
python -c "from src.config import CITIES; print(f'Loaded {len(CITIES)} cities')"
# Expected: Loaded 15 cities

# Run full FL simulation
python -m src.simulation
# Expected: 10 rounds complete in 2-3 minutes
# Output: logs/training_log.csv with round metrics
```

### 3.2 Detailed Installation (For Production)

#### **Step 1: Environment Preparation**

```bash
# Create project directory
mkdir -p /opt/fl-weather-pred
cd /opt/fl-weather-pred

# Clone repository
git clone <repo-url> .

# Create virtual environment
python3.13 -m venv /opt/fl-weather-pred/venv

# Activate
source /opt/fl-weather-pred/venv/bin/activate
```

#### **Step 2: System Dependencies**

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install build tools
sudo apt-get install -y \
    build-essential \
    python3.13-dev \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev

# Verify Python 3.13
python3.13 --version  # Should show 3.13.x

# Verify pip
python3.13 -m pip --version
```

#### **Step 3: CUDA Setup (Critical for GPU)**

```bash
# Check CUDA installation
nvidia-smi  # Should show GPU and CUDA version

# If CUDA not installed:
# Download from: https://developer.nvidia.com/cuda-13-3-downloads
# Installation example (Ubuntu):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-repo-ubuntu2204_12.3.1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204_12.3.1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-13-3

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-13.3
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA
nvcc --version  # Should show release 13.3
```

#### **Step 4: Python Dependencies**

```bash
# Activate virtual environment
source /opt/fl-weather-pred/venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.3
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu133
# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch CUDA available: True

# Install other requirements
pip install -r requirements.txt
# Expected: No errors, all packages installed

# Verify all imports
python -c "
from src.config import NUM_CLIENTS, NUM_ROUNDS
from src.model import WeatherLSTM
from src.utils import prepare_client_data
from src.client import WeatherClient
import flwr as fl
import ray
print('✓ All imports successful')
"
```

#### **Step 5: Data Verification**

```bash
# Check data files
ls -la data/processed/ | head -20
# Expected: 15 city CSV files listed

# Verify data integrity
python << 'EOF'
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")
cities = [f.stem for f in data_dir.glob("*.csv")]
print(f"Found {len(cities)} cities")

for city in sorted(cities)[:3]:  # Check first 3
    df = pd.read_csv(data_dir / f"{city}.csv")
    print(f"  {city}: {len(df)} rows, {df.isnull().sum().sum()} nulls")

print("✓ Data verification complete")
EOF
```

#### **Step 6: Configuration Setup**

```bash
# Review and adjust configuration
nano src/config.py  # Edit if needed

# Key settings to verify:
# - CITIES: Should list 15 cities
# - NUM_ROUNDS: Default 10 (can increase)
# - LOCAL_EPOCHS: Default 5
# - BATCH_SIZE: Default 32
# - DEVICE: Should auto-detect GPU

# Verify configuration
python -c "
from src.config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, DEVICE
print(f'Configuration:')
print(f'  Clients: {NUM_CLIENTS}')
print(f'  Rounds: {NUM_ROUNDS}')
print(f'  Local Epochs: {LOCAL_EPOCHS}')
print(f'  Batch Size: {BATCH_SIZE}')
print(f'  Device: {DEVICE}')
"
```

---

## Environment Configuration

### 4.1 Python Virtual Environment

#### **Creation & Activation**

```bash
# One-time setup
python3.13 -m venv /path/to/venv

# Every session - activate
source /path/to/venv/bin/activate

# Verify (prompt should show (venv))
which python

# To deactivate
deactivate
```

#### **Dependency Management**

```bash
# Generate requirements.txt from environment
pip freeze > requirements-frozen.txt

# Reinstall from requirements
pip install -r requirements.txt

# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade flower==1.9.0
```

### 4.2 Environment Variables

#### **Essential Variables**

```bash
# GPU/CUDA configuration
export CUDA_VISIBLE_DEVICES=0          # Use GPU 0 (if multi-GPU)
export CUDA_HOME=/usr/local/cuda-13.3
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# PyTorch configuration
export TORCH_HOME=/home/user/.cache/torch  # Cache directory
export OMP_NUM_THREADS=4                   # OpenMP threads

# Ray configuration
export RAY_memory=1000000000               # 1 GB per process
export RAY_object_store_memory=2000000000  # 2 GB object store

# FL configuration
export FL_NUM_ROUNDS=10
export FL_NUM_CLIENTS=15
export FL_LOG_LEVEL=DEBUG  # INFO, WARNING, ERROR

# Application configuration
export PYTHONUNBUFFERED=1  # Unbuffered output (for logging)
```

#### **Setting Variables (Linux/macOS)**

```bash
# Method 1: Set for current session
export CUDA_VISIBLE_DEVICES=0

# Method 2: Set for new sessions (~/.bashrc or ~/.zshrc)
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
source ~/.bashrc

# Method 3: Create .env file (load with python-dotenv)
cat > .env << 'EOF'
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=/home/user/.cache/torch
FL_NUM_ROUNDS=10
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

#### **Setting Variables (Windows PowerShell)**

```powershell
# Permanent (current user)
[Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "User")

# Temporary (current session)
$env:CUDA_VISIBLE_DEVICES = "0"

# Verify
$env:CUDA_VISIBLE_DEVICES
```

### 4.3 Logging Configuration

#### **Enable Debug Logging**

```bash
# Via environment variable
export FL_LOG_LEVEL=DEBUG

# Or in config.py
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)
```

#### **Log Rotation**

```python
# src/config.py or logging setup
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/training.log',
    maxBytes=10485760,  # 10 MB
    backupCount=5       # Keep 5 backup files
)
```

---

## Running the Simulation

### 5.1 Single-Run Execution

#### **Basic Execution**

```bash
# Activate virtual environment
source venv/bin/activate

# Run simulation
python -m src.simulation

# Expected output:
# ================================================================================
# FEDERATED LEARNING WEATHER PREDICTION SIMULATION
# ================================================================================
# System Configuration:
#   Clients (Cities): 15
#   FL Rounds: 10
# 
# Starting Flower Simulation...
# [Round 1/10] Global MSE: 1.859470
# [Round 2/10] Global MSE: 1.850180
# ...
# [Round 10/10] Global MSE: 1.849972
# 
# ✓ Training log saved to: logs/training_log.csv
# ================================================================================
# SIMULATION COMPLETE
# ================================================================================
```

#### **With Custom Parameters**

```bash
# Method 1: Modify config.py before running
python -m src.simulation

# Method 2: Command-line arguments (if implemented)
python -m src.simulation --num-rounds 20 --local-epochs 10

# Method 3: Environment variables
export FL_NUM_ROUNDS=20
export LOCAL_EPOCHS=10
python -m src.simulation
```

#### **Monitoring During Execution**

```bash
# In separate terminal, monitor GPU
watch -n 1 nvidia-smi

# Monitor memory usage
while true; do free -h; sleep 2; done

# Monitor CPU usage
top -b -n 1 | head -20
```

### 5.2 Batch Execution (Multiple Runs)

#### **Hyperparameter Search**

```bash
#!/bin/bash
# run_experiments.sh

# Test different learning rates
for LR in 0.0001 0.0005 0.001 0.005; do
    echo "Running with LEARNING_RATE=$LR"
    
    # Modify config
    sed -i "s/LEARNING_RATE = .*/LEARNING_RATE = $LR/" src/config.py
    
    # Run simulation
    python -m src.simulation
    
    # Save results
    cp logs/training_log.csv logs/training_log_lr_${LR}.csv
done

# Restore original config
git checkout src/config.py
```

#### **Multiple Rounds with Different Seeds**

```bash
#!/bin/bash
# run_multiple_seeds.sh

for SEED in 42 123 456 789 999; do
    echo "Running with SEED=$SEED"
    
    # Set seed
    export PYTHONHASHSEED=$SEED
    
    # Modify config
    sed -i "s/SEED = .*/SEED = $SEED/" src/config.py
    
    # Run
    python -m src.simulation
    
    # Archive
    mkdir -p results/seed_${SEED}
    cp logs/training_log.csv results/seed_${SEED}/
done
```

### 5.3 Output Artifacts

#### **Generated Files**

```
logs/training_log.csv
├─ Columns: round, global_mse
├─ Rows: 1 per FL round (10 total)
└─ Example:
   round,global_mse
   1,1.859470
   2,1.850180
   ...
   10,1.849972

models/global_model_final.pt (optional)
└─ Saved global model weights after final round

logs/debug.log (if DEBUG enabled)
└─ Detailed trace of execution

logs/ray_*.log (from Ray cluster)
└─ Ray scheduler and worker logs
```

#### **Analyzing Results**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('logs/training_log.csv')

# Print statistics
print(f"Final MSE: {df['global_mse'].iloc[-1]:.6f}")
print(f"Best MSE: {df['global_mse'].min():.6f} at round {df['global_mse'].idxmin() + 1}")
print(f"Improvement: {df['global_mse'].iloc[0] - df['global_mse'].iloc[-1]:.6f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(df['round'], df['global_mse'], marker='o', linewidth=2)
plt.xlabel('Round')
plt.ylabel('Global MSE')
plt.title('FL Convergence Curve')
plt.grid(True)
plt.savefig('convergence.png', dpi=150)
```

---

## Docker Deployment

### 6.1 Dockerfile

```dockerfile
# Dockerfile
FROM nvidia/cuda:13.3-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Default command
CMD ["python", "-m", "src.simulation"]
```

### 6.2 Docker Build & Run

#### **Build Image**

```bash
# Build Docker image
docker build -t fl-weather-pred:latest .

# Build with build args
docker build \
    --build-arg NUM_ROUNDS=20 \
    -t fl-weather-pred:v1.0 \
    .

# Verify image
docker images | grep fl-weather-pred
```

#### **Run Container**

```bash
# Run with GPU support
docker run --gpus all \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -e CUDA_VISIBLE_DEVICES=0 \
    fl-weather-pred:latest

# Run with resource limits
docker run --gpus all \
    --memory=8g \
    --cpus=4 \
    -v $(pwd)/logs:/app/logs \
    fl-weather-pred:latest

# Run interactive (for debugging)
docker run --gpus all -it \
    -v $(pwd):/app \
    fl-weather-pred:latest \
    bash
```

### 6.3 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  fl-weather:
    build: .
    image: fl-weather-pred:latest
    container_name: fl_weather_training
    
    # GPU support
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Volumes
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    
    # Environment
    environment:
      - FL_NUM_ROUNDS=10
      - PYTHONUNBUFFERED=1
```

#### **Docker Compose Commands**

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Clean up (remove volumes)
docker-compose down -v
```

---

## Production Deployment

### 7.1 Systemd Service

#### **Create Service File**

```bash
# Create service file
sudo nano /etc/systemd/system/fl-weather.service
```

```ini
[Unit]
Description=Federated Learning Weather Prediction
After=network.target nvidia-persistence-daemon.service
Wants=nvidia-persistence-daemon.service

[Service]
Type=simple
User=fl-user
Group=fl-user
WorkingDirectory=/opt/fl-weather-pred

# Activate virtual environment and run
ExecStart=/opt/fl-weather-pred/venv/bin/python -m src.simulation

# Restart policy
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fl-weather

# Environment variables
Environment="PATH=/opt/fl-weather-pred/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="CUDA_HOME=/usr/local/cuda"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

#### **Enable and Start Service**

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable (auto-start on boot)
sudo systemctl enable fl-weather.service

# Start service
sudo systemctl start fl-weather.service

# Check status
sudo systemctl status fl-weather.service

# View logs
sudo journalctl -u fl-weather.service -f

# Stop service
sudo systemctl stop fl-weather.service
```

### 7.2 Production Checklist

```
PRE-PRODUCTION VERIFICATION
□ All data files present and verified (15 cities, 365 days each)
□ Configuration optimized for production (NUM_ROUNDS, LOCAL_EPOCHS)
□ Logging enabled (logs/ directory writable)
□ Model checkpoints directory exists (models/)
□ CUDA and GPU verified working
□ Virtual environment created and all packages installed
□ Performance baseline established (measure time per round)
□ Backup of original config created

MONITORING SETUP
□ Log rotation configured (prevent disk full)
□ GPU monitoring enabled (nvidia-smi in cron job?)
□ Memory usage alerts set (threshold 90%)
□ Training progress logged (round metrics)
□ Failed round recovery procedure documented

SECURITY HARDENING
□ File permissions restricted (only fl-user can read/write)
□ No hardcoded credentials in code
□ Data encryption at rest (if sensitive)
□ Audit logging enabled
□ Access logs stored

OPERATIONAL READINESS
□ Runbooks created (how to start/stop/restart)
□ Rollback procedure documented
□ Backup strategy implemented (save logs, models periodically)
□ Disaster recovery plan drafted
□ On-call escalation procedure established
```

---

## Monitoring & Logging

### 8.1 Real-Time Monitoring

#### **GPU Monitoring**

```bash
# Real-time GPU stats (1-second refresh)
watch -n 1 nvidia-smi

# GPU memory profiling (per process)
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# GPU temperature and power
nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv
```

#### **System Monitoring**

```bash
# CPU and memory usage
top -b -n 1 | head -20

# Memory breakdown
free -h

# Storage usage
df -h /

# Process-specific stats
ps aux | grep python | grep simulation
```

#### **Application Monitoring**

```python
# Monitor training progress in Python
import subprocess
import time

while True:
    # Check if simulation running
    result = subprocess.run(['pgrep', '-f', 'src.simulation'], 
                          capture_output=True)
    if result.stdout:
        print("✓ Simulation running")
        
        # Check latest metrics
        try:
            df = pd.read_csv('logs/training_log.csv')
            latest_round = df['round'].max()
            latest_mse = df['global_mse'].iloc[-1]
            print(f"  Round: {latest_round}, MSE: {latest_mse:.6f}")
        except:
            pass
    else:
        print("✗ Simulation not running")
    
    time.sleep(60)  # Check every minute
```

### 8.2 Log Management

#### **Centralized Logging**

```python
# src/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(log_file='logs/training.log', level=logging.INFO):
    """Configure logging with file and console handlers."""
    
    # Create logger
    logger = logging.getLogger('fl-weather')
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage in src/simulation.py
from src.logging_config import setup_logging

logger = setup_logging()
logger.info(f"Starting FL round {server_round}")
logger.warning(f"Client {cid} convergence slow")
logger.error(f"Round {server_round} failed: {error}")
```

#### **Log Analysis**

```bash
# Count errors
grep "ERROR" logs/training.log | wc -l

# Find specific client issues
grep "Client 0" logs/training.log

# Extract metrics
grep "Global MSE" logs/training.log

# Timeline analysis
tail -20 logs/training.log
```

### 8.3 Metrics Export

#### **Prometheus Metrics**

```python
# Optional: Export to Prometheus
from prometheus_client import CollectorRegistry, Gauge, start_http_server

registry = CollectorRegistry()

# Create metrics
global_mse = Gauge('fl_global_mse', 'Global MSE', registry=registry)
round_duration = Gauge('fl_round_duration_seconds', 'Round duration', registry=registry)

# Update during training
global_mse.set(1.8594)
round_duration.set(2.35)

# Expose metrics (Prometheus scrapes at :8000/metrics)
start_http_server(8000, registry=registry)
```

#### **TensorBoard Visualization** (Optional)

```python
# Log metrics to TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/tensorboard')

# Log global MSE
writer.add_scalar('Training/GlobalMSE', global_mse, round_num)

# View in TensorBoard
# tensorboard --logdir logs/tensorboard
```

---

## Maintenance & Updates

### 9.1 Regular Maintenance

#### **Daily Checks**

```bash
#!/bin/bash
# daily_checks.sh - Run via cron daily

echo "=== Daily Maintenance Checks ==="

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "⚠️  WARNING: Disk usage at ${DISK_USAGE}%"
    # Archive old logs
    tar -czf logs/archive/logs_$(date +%Y%m%d).tar.gz logs/*.log
    rm logs/*.log
fi

# Check GPU health
nvidia-smi -q | grep -E "Temperature|Power"

# Verify data integrity
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/processed/berlin.csv')
assert len(df) == 365, 'Data corrupted!'
print('✓ Data integrity OK')
"

# Check logs for errors
ERROR_COUNT=$(grep -c "ERROR" logs/training.log)
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT errors in logs"
fi
```

#### **Schedule with Cron**

```bash
# Add to crontab (crontab -e)
0 2 * * * /opt/fl-weather-pred/daily_checks.sh >> /var/log/fl-weather-checks.log 2>&1

# Weekly backup
0 3 * * 0 tar -czf /backups/fl-weather-$(date +\%Y\%m\%d).tar.gz /opt/fl-weather-pred/
```

### 9.2 Package Updates

#### **Safe Update Procedure**

```bash
# Step 1: Check for updates
source venv/bin/activate
pip list --outdated

# Step 2: Review requirements.txt
cat requirements.txt | head -10

# Step 3: Update specific package (test first in dev)
pip install --upgrade flower==1.9.0

# Step 4: Verify compatibility
python -c "from src.simulation import main; print('✓ Compatibility OK')"

# Step 5: Run local test
python -m src.simulation --test-mode

# Step 6: Commit changes
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update flower to 1.9.0"
```

### 9.3 Backup & Recovery

#### **Backup Strategy**

```bash
# Backup script (backup.sh)
#!/bin/bash

BACKUP_DIR="/backups/fl-weather"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
mkdir -p $BACKUP_DIR/$TIMESTAMP

# Backup code
cp -r src/ $BACKUP_DIR/$TIMESTAMP/
cp -r docs/ $BACKUP_DIR/$TIMESTAMP/
cp requirements.txt $BACKUP_DIR/$TIMESTAMP/

# Backup data (optional - large files)
# cp -r data/ $BACKUP_DIR/$TIMESTAMP/

# Backup logs & models
cp -r logs/ $BACKUP_DIR/$TIMESTAMP/
cp -r models/ $BACKUP_DIR/$TIMESTAMP/

# Compress
tar -czf $BACKUP_DIR/fl-weather-$TIMESTAMP.tar.gz $BACKUP_DIR/$TIMESTAMP/

# Cleanup old backups (keep 10)
ls -t $BACKUP_DIR/*.tar.gz | tail -n +11 | xargs rm

echo "✓ Backup completed: $BACKUP_DIR/fl-weather-$TIMESTAMP.tar.gz"
```

#### **Recovery Procedure**

```bash
# List available backups
ls -lh /backups/fl-weather/*.tar.gz

# Extract backup
tar -xzf /backups/fl-weather/fl-weather-20260604_120000.tar.gz -C /opt/

# Restore to previous version
cp -r /backups/fl-weather/20260604_120000/src/* /opt/fl-weather-pred/src/
pip install -r /backups/fl-weather/20260604_120000/requirements.txt

# Verify
python -m src.simulation --dry-run
```

---

## Troubleshooting

### 10.1 Installation Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| **No module named 'torch'** | PyTorch not installed | `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu133` |
| **CUDA not available** | CUDA toolkit missing or path wrong | Check `echo $CUDA_HOME`, reinstall CUDA 13.3 |
| **torch.cuda.is_available() = False** | Correct CUDA not matched with PyTorch | Use `pip install torch --index-url https://download.pytorch.org/whl/cu133` |
| **Permission denied installing** | Virtual environment not activated | `source venv/bin/activate` then retry |

### 10.2 Runtime Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| **CUDA out of memory** | Batch too large or multiple clients | Reduce `BATCH_SIZE` from 32→16 or `MAX_WORKERS` from 2→1 |
| **FileNotFoundError: berlin.csv** | City file missing or wrong path | Verify `ls data/processed/` shows 15 files with lowercase names |
| **Connection refused (Ray)** | Ray head node not running | Ray starts automatically; if issue persists, check ports 6379-6382 |
| **shape mismatch in fit()** | Parameter count changed | Ensure config.py matches model.py (NUM_FEATURES, LSTM_HIDDEN_SIZE) |

### 10.3 Performance Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| **Slow training (>5 sec/round)** | Inefficient GPU usage | Check `nvidia-smi`, verify `flatten_parameters()` called, set `num_workers=0` |
| **High memory but low GPU util** | Transfer bottleneck | Reduce data transfer, increase `BATCH_SIZE`, check `pin_memory=True` |
| **Convergence not improving** | Non-IID or weak learning | Increase `LOCAL_EPOCHS` from 5→10, decrease `LEARNING_RATE` to 0.0005 |

---

## Document Metadata

**Author:** Federated Learning Weather System Team  
**Document ID:** FL-WP-DEPLOY-003  
**Review Status:** Final  
**Distribution:** DevOps, Operations Team

**Revision History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 2026 | Complete deployment guide |

---

**End of Deployment Document**

Previous: [2_IMPLEMENTATION.md](2_IMPLEMENTATION.md) - Code walkthrough and API reference

Next: [4_EVALUATION.md](4_EVALUATION.md) - Performance analysis and benchmarks
