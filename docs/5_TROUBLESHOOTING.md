# 5. Troubleshooting Guide

## Federated Learning Weather Prediction System

**Document Version:** 1.0  
**Last Updated:** June 4, 2026  
**Scope:** Error diagnosis, resolution procedures, optimization strategies, and operational support

---

## Overview & Quick Reference

This comprehensive troubleshooting guide provides diagnostic procedures, error solutions, and optimization strategies for the Federated Learning Weather Prediction system. Whether you're experiencing installation issues, runtime errors, performance degradation, or seeking optimization opportunities, this document offers systematic resolution procedures and best practices.

### **Quick Symptom Finder**

| **Symptom** | **Section** | **Likely Cause** |
|-------------|------------|------------------|
| "ModuleNotFoundError" | 1.1 | Missing dependencies |
| GPU out of memory | 2.4 | Batch size too large |
| Very slow training | 3.3 | Suboptimal hardware allocation |
| Model won't converge | 4.2 | Learning rate issue |
| "CUDA out of memory" | 2.4 | GPU memory leak |
| Server won't start | 5.1 | Port conflict, flower version |
| Client can't connect | 5.2 | Network, firewall, IP config |
| Data leakage warning | 3.1 | Scaler fitted on full dataset |
| NaN loss values | 4.3 | Gradient explosion, data normalization |
| Very high latency | 6.1 | Network bottleneck, batching |

---

## 1. Installation & Environment Troubleshooting

### 1.1 Dependency Installation Errors

#### **Problem: "ModuleNotFoundError: No module named 'flower'"**

**Diagnosis:**
```bash
python3 -c "import flower; print(flower.__version__)"
pip list | grep flower
```

**Solutions (in order of likelihood):**

1. **Flower not installed**
   ```bash
   pip install flwr==1.8.0
   # Verify
   python3 -c "import flwr; print(flwr.__version__)"  # Should output 1.8.0
   ```

2. **Wrong Python environment**
   ```bash
   which python3
   # Should be /home/rna_13/miniconda3/envs/fl-weather/bin/python3
   # If not, activate correct environment:
   conda activate fl-weather
   ```

3. **Pip cache issue**
   ```bash
   pip install --no-cache-dir flwr==1.8.0
   ```

4. **Virtual environment corruption**
   ```bash
   conda remove -n fl-weather --all
   conda create -n fl-weather python=3.11
   conda activate fl-weather
   pip install -r requirements.txt
   ```

**Verification:**
```bash
python3 << 'EOF'
import flower
import torch
import ray
import numpy as np
print(f"✓ Flower {flower.__version__}")
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ Ray {ray.__version__}")
print(f"✓ NumPy {np.__version__}")
EOF
```

---

#### **Problem: "No module named 'torch'"**

**Quick Fix:**
```bash
pip install torch==2.1.2
# For GPU support (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU availability:**
```bash
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**If GPU not detected:**
- Check NVIDIA driver: `nvidia-smi`
- CUDA toolkit: `nvcc --version`
- PyTorch compiled for correct CUDA version (see requirements.txt)

---

#### **Problem: "ImportError: cannot import name '_ssl'"**

This typically indicates Python version mismatch or SSL library issue.

**Solution:**
```bash
# Check Python version (should be 3.10-3.12)
python3 --version

# Rebuild Python with SSL support
conda remove -n fl-weather python
conda install -n fl-weather python=3.11

# Or reinstall environment
conda env remove -n fl-weather
conda env create -f environment.yml  # if you have this file
```

---

### 1.2 CUDA & GPU Driver Issues

#### **Problem: "CUDA runtime error: no kernel image is available for execution on the device"**

**Cause:** PyTorch compiled for different CUDA version than GPU driver.

**Solution:**

1. **Check CUDA version:**
   ```bash
   nvidia-smi | grep "CUDA Version"
   # Output should be >= 12.0
   ```

2. **Check PyTorch CUDA version:**
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch for correct CUDA:**
   ```bash
   # For CUDA 12.1 (most common with RTX 4050)
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
     --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8
   pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

---

#### **Problem: "CUDA out of memory" but nvidia-smi shows free memory**

**Cause:** GPU memory fragmentation or lingering tensors.

**Diagnosis:**
```bash
# Check GPU memory state
nvidia-smi

# Run diagnostic
python3 << 'EOF'
import torch
import gc

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
gc.collect()

print(f"After clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
EOF
```

**Solutions:**

1. **Clear GPU cache** (immediate fix):
   ```bash
   python3 << 'EOF'
   import torch
   torch.cuda.empty_cache()
   import gc; gc.collect()
   EOF
   ```

2. **Reduce batch size:**
   - Edit `src/config.py`: Change `BATCH_SIZE = 32` → `BATCH_SIZE = 16`
   - Reduces GPU memory by ~50%

3. **Reduce MAX_WORKERS:**
   - Edit `src/config.py`: Change `MAX_WORKERS = 2` → `MAX_WORKERS = 1`
   - Allows one client at a time (slower but more stable)

4. **Check for memory leaks:**
   ```bash
   # Run system.py with memory profiling
   python3 -m memory_profiler src/simulation.py 2>&1 | tail -50
   ```

---

#### **Problem: "RuntimeError: CUDA out of memory" immediately on startup**

**Cause:** Insufficient GPU memory for model initialization.

**Solution:**

```bash
# Temporary: Force CPU execution
python3 << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
import torch
print(f"Device: {torch.device('cpu')}")
EOF

# More permanent: Run simulation on CPU first
export CUDA_VISIBLE_DEVICES=""
python3 src/simulation.py
# Then re-enable GPU after fixing memory issue
```

---

### 1.3 Data Directory & File Issues

#### **Problem: "FileNotFoundError: data/raw/weather_dataset_2025.csv not found"**

**Diagnosis:**
```bash
ls -la data/raw/
ls -la data/processed/
```

**Solutions:**

1. **File missing entirely:**
   ```bash
   # Download/restore data
   # Assuming you have backup or source
   # Contact data administrator for weather_dataset_2025.csv
   
   # Verify structure
   ls -la data/raw/weather_dataset_2025.csv
   ```

2. **File exists but path wrong:**
   - Check working directory: `pwd`
   - Should be: `/home/rna_13/FedLearning/flweatherpred`
   - If different, run from correct directory:
     ```bash
     cd /home/rna_13/FedLearning/flweatherpred
     python3 src/simulation.py
     ```

3. **Permission denied:**
   ```bash
   chmod 644 data/raw/weather_dataset_2025.csv
   chmod 755 data/
   ```

---

#### **Problem: "CSV parsing error" or "incorrect field names"**

**Diagnosis:**
```bash
head -5 data/raw/weather_dataset_2025.csv
wc -l data/raw/weather_dataset_2025.csv
```

**Expected format:**
- 5,476 lines (1 header + 5,475 data rows)
- Columns: `City, Date, Temperature_C, Humidity_%, Wind_Speed_km_h, Precipitation_mm`

**Solutions:**

1. **Verify CSV format:**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   df = pd.read_csv('data/raw/weather_dataset_2025.csv')
   print(f"Shape: {df.shape}")
   print(f"Columns: {list(df.columns)}")
   print(df.head())
   EOF
   ```

2. **Fix column names if different:**
   - Edit `src/utils.py` line ~35
   - Change column references to match your CSV

3. **If CSV corrupted:**
   ```bash
   # Regenerate from processed city files
   cd notebooks
   jupyter notebook split_data.ipynb
   # Re-run data splitting
   ```

---

### 1.4 Python Version Compatibility

#### **Problem: "SyntaxError: invalid syntax" on Python 3.9 or older**

**Cause:** Code uses Python 3.10+ features (union types `X | Y`, match statements).

**Solution:**
```bash
# Check Python version
python3 --version

# Update to 3.11 or 3.12
conda install python=3.11

# Verify
python3 --version
```

---

## 2. Data Loading & Preprocessing Issues

### 2.1 City Data Loading Errors

#### **Problem: "IndexError: City index out of range"**

**Cause:** Trying to load city index > 14 (valid range 0-14 for 15 cities).

**Diagnosis:**
```python
from src.config import CITIES, NUM_CLIENTS
print(f"Valid city indices: 0-{NUM_CLIENTS-1}")
print(f"Cities: {CITIES}")
```

**Solution:**
```python
# Use valid indices
for city_idx in range(NUM_CLIENTS):  # 0 to 14
    data = load_city_data(city_idx)
    print(f"Loaded: {CITIES[city_idx]}")
```

---

#### **Problem: "Shape mismatch: expected (365, 4) got (N, M)"**

**Cause:** City data file has incorrect dimensions.

**Diagnosis:**
```bash
python3 << 'EOF'
import pandas as pd
for i, city in enumerate(['Berlin', 'Cairo', 'Dammam', 'Doha', 'Dubai', 'Jeddah', 
                          'London', 'Mumbai', 'New_York', 'Paris', 'Riyadh', 
                          'Singapore', 'Sydney', 'Tokyo', 'Toronto']):
    try:
        df = pd.read_csv(f'data/processed/{city}.csv', index_col=0)
        print(f"{city}: {df.shape} ✓" if df.shape[0] == 365 else f"{city}: {df.shape} ✗")
    except Exception as e:
        print(f"{city}: ERROR - {e}")
EOF
```

**Solutions:**

1. **Regenerate processed data:**
   ```bash
   cd notebooks
   python3 -m jupyter nbconvert --to notebook --execute split_data.ipynb
   cd ..
   ```

2. **Manually verify and fix:**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   
   # Check raw data
   raw = pd.read_csv('data/raw/weather_dataset_2025.csv')
   print(f"Raw data: {raw.shape}")
   
   # Extract city
   city_data = raw[raw['City'] == 'Berlin']
   print(f"Berlin rows: {len(city_data)}")
   
   # Should be 365
   if len(city_data) != 365:
       print("ERROR: Expected 365 rows")
   EOF
   ```

---

### 2.2 Data Normalization & Scaling Issues

#### **Problem: "Features have very different ranges (0-100 vs 0-50)"**

**Cause:** Data not properly normalized. StandardScaler should normalize each feature independently.

**Diagnosis:**
```python
import pandas as pd
from src.utils import prepare_client_data

X_train, X_test, y_train, y_test, scaler = prepare_client_data(city_idx=0)

print(f"X_train mean: {X_train.mean(axis=0)}")
print(f"X_train std: {X_train.std(axis=0)}")
print(f"X_train min: {X_train.min(axis=0)}")
print(f"X_train max: {X_train.max(axis=0)}")

# Should all be close to 0 mean, 1 std
```

**Solution:**
- This is expected behavior - see 2.3 below for expected ranges

---

#### **Problem: "Data leakage: test set scaler fitted on wrong data"**

**Cause:** StandardScaler fitted on entire dataset instead of training data only.

**Verification:**
```python
# BAD - leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on all data
X_train, X_test = train_test_split(X_scaled)  # Then split

# GOOD - no leakage
X_train, X_test = train_test_split(X)  # Split first
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train only
X_test = scaler.transform(X_test)  # Transform test with train params
```

**Status in this codebase:** ✓ CORRECT  
Location: `src/utils.py`, lines 61-75. Scaler is fit on training data only.

**If you modified utils.py and created leakage:**
```bash
# Restore original
git checkout src/utils.py

# Or verify correct implementation
python3 << 'EOF'
from src.utils import prepare_client_data
X_train, X_test, y_train, y_test, scaler = prepare_client_data(0)
print(f"X_train mean: {X_train.mean(axis=0)}")  # Should be ~0
print(f"X_train std: {X_train.std(axis=0)}")    # Should be ~1
EOF
```

---

### 2.3 Expected Data Characteristics

#### **Problem: "Are my normalized values normal?"**

**Expected ranges after preprocessing:**

```
Temperature:     mean ≈ 0.00, std ≈ 1.00, range ≈ [-2.5, 2.5]
Humidity:        mean ≈ 0.00, std ≈ 1.00, range ≈ [-2.0, 2.0]
Wind Speed:      mean ≈ 0.00, std ≈ 1.00, range ≈ [-1.5, 3.0]
Precipitation:   mean ≈ 0.00, std ≈ 1.00, range ≈ [-0.5, 4.0]
```

**Verification script:**
```python
import numpy as np
from src.utils import prepare_client_data

for city_idx in range(15):
    X_train, X_test, y_train, y_test, scaler = prepare_client_data(city_idx)
    
    # Check stats
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    
    # Should all be close to 0, 1
    assert np.allclose(train_mean, 0, atol=0.1), f"Mean off: {train_mean}"
    assert np.allclose(train_std, 1, atol=0.1), f"Std off: {train_std}"
    
print("✓ All data normalized correctly")
```

---

### 2.4 GPU Memory Issues During Data Loading

#### **Problem: "CUDA out of memory" when loading large batches**

**Cause:** Entire dataset loaded to GPU at once.

**Diagnosis:**
```python
import torch
print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# This should load iteratively, NOT all at once
from torch.utils.data import DataLoader
# ...
```

**Current implementation:** ✓ CORRECT  
- `DataLoader` with `batch_size=32` loads iteratively
- LSTM processes on GPU, gradients computed per batch
- Memory usage: ~200 MB per batch (on RTX 4050)

**If you increased batch size to 256:**

**Solution:**
```python
# In src/config.py
BATCH_SIZE = 64  # Try 64 first
# or
BATCH_SIZE = 32  # Revert to safe default
```

**Memory calculation:**
```python
batch_memory_mb = batch_size * seq_length * features * 4 / 1e6
# For batch=32, seq=5, features=4: ~2.5 MB input
# Plus gradients, activations, optimizer state: ~200 MB total
```

---

## 3. Model Training Issues

### 3.1 Convergence & Loss Problems

#### **Problem: "Loss is NaN or Infinity"**

**Diagnosis:**
```python
import torch
import torch.nn as nn

# Check for exploding gradients
from torch.nn.utils import clip_grad_norm_

model = WeatherLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    # Print loss before backward
    print(f"Loss value: {loss.item()}")
    if torch.isnan(loss):
        print("NaN detected!")
        break
    
    loss.backward()
    
    # Check gradients
    total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Gradient norm: {total_norm.item()}")
```

**Solutions (in order of likelihood):**

1. **Gradient explosion** (most common)
   ```python
   # Add gradient clipping in training loop
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```
   This is already implemented in `src/client.py` if using recent version.

2. **Learning rate too high**
   ```python
   # In src/config.py
   LEARNING_RATE = 0.0001  # Reduce from 0.001
   ```

3. **Data not normalized**
   - Verify: `X_train.mean()` should be ~0, `std()` should be ~1
   - See section 2.2 for data normalization

4. **Batch contains invalid values**
   ```python
   # Check for NaN/Inf in data
   assert not np.isnan(X_train).any()
   assert not np.isinf(X_train).any()
   ```

5. **Model initialization issue**
   ```python
   # Check initial loss (should be reasonable)
   model = WeatherLSTM()
   sample_input = torch.randn(32, 5, 4)  # batch=32, seq=5, features=4
   output = model(sample_input)
   print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
   # Should be reasonable values, not extreme
   ```

---

#### **Problem: "Loss oscillates instead of converging smoothly"**

**Cause:** Non-IID data heterogeneity across clients (expected in federated learning).

**Expected behavior in this system:**
- Rounds 1-5: Smooth convergence (loss decreases)
- Rounds 5-10: Oscillation phase (±1-2% variation)
- Final MSE: ~1.85 ± 0.05

**Diagnosis - Is oscillation normal?**
```bash
python3 << 'EOF'
import numpy as np

# Expected loss curve from 4_EVALUATION.md
expected_losses = [1.859, 1.808, 1.795, 1.789, 1.787, 1.823, 1.812, 1.820, 1.847, 1.850]
your_losses = []  # Your actual losses from logs

if len(your_losses) >= 5:
    oscillation_amplitude = np.std(your_losses[-5:])
    print(f"Last 5 rounds std: {oscillation_amplitude:.4f}")
    print("✓ Normal" if oscillation_amplitude < 0.05 else "⚠ Unusual")
EOF
```

**Solutions:**

1. **If oscillation excessive (std > 0.1):** Reduce learning rate
   ```python
   # In src/config.py
   LEARNING_RATE = 0.0005  # Reduce by 50%
   ```

2. **If oscillation present but acceptable:** No action needed
   - This is expected behavior for Non-IID federated learning
   - See 4_EVALUATION.md Section 7 for Non-IID analysis

3. **To stabilize oscillations:** Increase LOCAL_EPOCHS
   ```python
   # In src/config.py
   LOCAL_EPOCHS = 8  # Increase from 5
   ```

---

#### **Problem: "Model won't improve after first round"**

**Cause:** Learning rate too high, or data imbalance across cities.

**Diagnosis:**
```python
# Monitor per-round improvement
round_losses = [1.859, 1.808, 1.795, ...]  # From logs
improvements = [round_losses[0] - round_losses[i] for i in range(len(round_losses))]
print(improvements)  # Should show: [0, 0.05, 0.06, ...]
```

**Solutions:**

1. **Reduce learning rate** (most common)
   ```python
   # In src/config.py
   LEARNING_RATE = 0.0005
   ```

2. **Increase LOCAL_EPOCHS**
   ```python
   # In src/config.py
   LOCAL_EPOCHS = 10  # Allow more training per round
   ```

3. **Check for data loading bug**
   ```python
   # Verify different data each round
   from src.utils import prepare_client_data
   X1, _, _, _, _ = prepare_client_data(0)
   X2, _, _, _, _ = prepare_client_data(0)
   # Should be same (deterministic)
   assert np.allclose(X1, X2)
   ```

---

### 3.2 Model Accuracy Problems

#### **Problem: "MSE is 10+ (very high errors)"**

**Cause:** Model not learning, or outputs in wrong scale.

**Diagnosis:**
```python
from src.model import WeatherLSTM
import torch

model = WeatherLSTM()
sample_input = torch.randn(32, 5, 4)
output = model(sample_input)

print(f"Output mean: {output.mean().item():.3f}")
print(f"Output std: {output.std().item():.3f}")
print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

# Expected: mean ≈ 0, std ≈ 1, range ≈ [-3, 3]
```

**Solutions:**

1. **Check target normalization**
   ```python
   # Targets should also be normalized
   print(f"y_train mean: {y_train.mean():.3f}")  # Should be ≈ 0
   print(f"y_train std: {y_train.std():.3f}")    # Should be ≈ 1
   ```

2. **Check model output scale**
   ```python
   # Add final layer normalization if needed
   # Or scale predictions back to original range during evaluation
   ```

3. **Verify loss calculation**
   ```python
   from torch.nn import MSELoss
   criterion = MSELoss()
   
   # After training
   with torch.no_grad():
       predictions = model(X_test)
       mse = criterion(predictions, y_test)
       print(f"Test MSE: {mse.item():.4f}")
   # Should be in range 1.7-2.0
   ```

---

#### **Problem: "RMSE is reasonable but MAPE is very high (>10%)"**

**Cause:** Model struggles with small temperature variations or zero-division in MAPE.

**MAPE calculation:**
```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # ⚠️ Fails if y_true contains values near 0
```

**Diagnosis:**
```python
# Check for near-zero values
print(f"y_test min: {y_test.min():.4f}")
print(f"y_test mean: {y_test.mean():.4f}")
print(f"Percentage of |y_test| < 0.1: {(np.abs(y_test) < 0.1).sum() / len(y_test) * 100:.1f}%")
```

**Solution:**
```python
# Use robust MAPE calculation
def robust_mape(y_true, y_pred, epsilon=1e-2):
    denominator = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100

mape = robust_mape(y_test, y_pred)
print(f"Robust MAPE: {mape:.2f}%")
```

---

### 3.3 Training Speed Issues

#### **Problem: "Training is very slow (>30 sec per round)"**

**Diagnosis:**
```bash
# Check system resources during training
watch -n 1 nvidia-smi
# In another terminal:
top
```

**Expected timing:**
- Per round: ~2 seconds
- 10 rounds: ~20 seconds + overhead = 2-3 minutes total

**If much slower, check:**

1. **GPU utilization**
   ```bash
   nvidia-smi dmon -s pucvme
   # Should show ~90% GPU util during training
   ```

2. **CPU bottleneck**
   ```bash
   python3 << 'EOF'
   import psutil
   print(f"CPU cores: {psutil.cpu_count()}")
   print(f"CPU freq: {psutil.cpu_freq()}")
   EOF
   ```

**Solutions:**

1. **Increase MAX_WORKERS** (if GPU memory available)
   ```python
   # In src/config.py
   MAX_WORKERS = 3  # Process 3 clients in parallel
   # Monitor GPU memory first to ensure this is safe
   ```

2. **Reduce LOCAL_EPOCHS** (if convergence allows)
   ```python
   # In src/config.py
   LOCAL_EPOCHS = 3  # Reduce from 5
   # May affect final accuracy
   ```

3. **Check for disk I/O bottleneck**
   ```bash
   iotop -o  # Show processes using disk I/O
   ```

4. **Move data to RAM disk** (advanced)
   ```bash
   # Create 2GB RAM disk
   sudo mkdir -p /mnt/ramdisk
   sudo mount -t tmpfs -o size=2G tmpfs /mnt/ramdisk
   
   # Copy data
   cp -r data/processed /mnt/ramdisk/
   
   # Update config to use /mnt/ramdisk/processed/
   ```

---

### 3.4 Model File Issues

#### **Problem: "Cannot load saved model - wrong format"**

**Expected model format:**
```python
# Correct - state dict
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Also correct - NumPy array format (for FL)
import numpy as np
weights = [p.data.cpu().numpy() for p in model.parameters()]
np.save('model_weights.npy', weights)
```

**If model won't load:**

```python
# Check saved file format
import torch
import pickle

try:
    state_dict = torch.load('model.pth')
    print(f"Type: {type(state_dict)}")
    print(f"Keys: {state_dict.keys()}")
except Exception as e:
    print(f"Error loading: {e}")
    # Try pickle format
    with open('model.pth', 'rb') as f:
        data = pickle.load(f)
    print(f"Pickle content: {type(data)}")
```

---

## 4. Federated Learning Issues

### 4.1 Server-Side Issues

#### **Problem: "Server won't start - 'Address already in use'"**

**Cause:** Port 8080 already in use by another process.

**Diagnosis:**
```bash
lsof -i :8080
# or
netstat -tuln | grep 8080
```

**Solutions:**

1. **Kill existing process**
   ```bash
   lsof -ti :8080 | xargs kill -9
   # Then restart simulation
   ```

2. **Use different port**
   ```bash
   # Edit src/config.py or simulation.py
   # Change port from 8080 to 8081
   ```

3. **Wait for socket to close**
   ```bash
   # Socket may stay in TIME_WAIT state
   # Solution: wait 60 seconds or set SO_REUSEADDR in code
   # Already set in simulation.py if properly configured
   ```

---

#### **Problem: "Server crashes: 'ValueError: too many clients'"**

**Cause:** More than MAX_WORKERS clients trying to connect simultaneously.

**Diagnosis:**
```bash
# Check Ray worker status
ray status
```

**Solution:**

1. **Increase MAX_WORKERS** (if memory available)
   ```python
   # In src/config.py
   MAX_WORKERS = 4  # Increase from 2
   ```

2. **Ensure system has enough memory**
   ```bash
   free -h
   # Should have 8+ GB available RAM
   ```

3. **Check worker assignment**
   ```python
   # In simulation.py, verify client_fn properly creates workers
   # Each worker should get unique client_id
   ```

---

#### **Problem: "Server hangs during round X - no progress"**

**Cause:** Client timeout or deadlock in distributed training.

**Diagnosis:**
```bash
# Check Flower server logs
tail -100 logs/server.log

# Check Ray logs
ray logs cluster
```

**Solutions:**

1. **Increase server timeout** (in simulation.py)
   ```python
   server = fl.server.start_server(
       ...,
       config=fl.server.ServerConfig(
           num_rounds=10,
           round_timeout=300.0  # Increase from default 300s
       )
   )
   ```

2. **Check client logs**
   ```bash
   # Look for stuck client
   ls logs/client_*.log
   tail logs/client_0.log  # Check specific client
   ```

3. **Restart simulation**
   ```bash
   # Kill hung process
   pkill -f "python3 src/simulation.py"
   
   # Clean up Ray cluster
   ray stop
   ray start
   
   # Restart
   python3 src/simulation.py
   ```

---

### 4.2 Client-Side Issues

#### **Problem: "Client can't connect to server - ConnectionRefusedError"**

**Cause:** Server not running or wrong IP/port.

**Diagnosis:**
```bash
# Test connectivity
telnet localhost 8080
# or
python3 -c "import socket; s = socket.socket(); s.connect(('localhost', 8080))"
```

**Solutions:**

1. **Ensure server is running**
   ```bash
   ps aux | grep simulation.py
   # If not running, start it:
   python3 src/simulation.py
   ```

2. **Check firewall**
   ```bash
   sudo ufw status
   sudo ufw allow 8080
   ```

3. **Verify localhost resolution**
   ```bash
   ping localhost  # Should resolve to 127.0.0.1
   cat /etc/hosts   # Check localhost entry
   ```

---

#### **Problem: "Client disconnects mid-training - 'RemoteRpcError'"**

**Cause:** Network timeout, OOM on client, or worker crash.

**Diagnosis:**
```bash
# Check system resources during training
watch -n 1 'free -h && echo "---" && nvidia-smi'
```

**Solutions:**

1. **Increase timeout**
   ```python
   # In client.py, add timeout handling
   # Or set environment variable
   export FLOWER_CLIENT_TIMEOUT=120
   ```

2. **Reduce client load**
   ```python
   # In src/config.py
   BATCH_SIZE = 16  # Reduce from 32
   LOCAL_EPOCHS = 3  # Reduce from 5
   ```

3. **Check network stability**
   ```bash
   ping -c 100 localhost  # Check packet loss
   ```

---

#### **Problem: "All clients silently drop after round 5"**

**Cause:** Memory leak causing gradual memory exhaustion.

**Diagnosis:**
```bash
# Monitor memory over time
watch -n 5 'free -h | grep Mem'
# Should show stable memory usage, not increasing

# Or use memory profiler
python3 -m memory_profiler src/simulation.py
```

**Solutions:**

1. **Add garbage collection**
   ```python
   # In src/client.py fit() method
   import gc
   for epoch in range(num_epochs):
       # training code
       if epoch % 2 == 0:
           gc.collect()  # Every 2 epochs
   ```

2. **Clear GPU cache**
   ```python
   # In src/client.py fit() method
   import torch
   torch.cuda.empty_cache()
   ```

3. **Check for tensor accumulation**
   ```python
   # Ensure detach() used properly in training loop
   loss = criterion(outputs, targets)
   loss.backward()
   # ✓ Correct - loss is computed fresh each iteration
   ```

---

### 4.3 Aggregation Issues

#### **Problem: "Server won't aggregate - 'ValueError: incompatible shapes'"**

**Cause:** Client weights have different shapes (e.g., due to different batch sizes).

**Diagnosis:**
```python
from src.model import WeatherLSTM
import numpy as np

model = WeatherLSTM()
params = model.get_model_parameters()

for i, p in enumerate(params):
    print(f"Parameter {i}: shape {p.shape}")
```

**Expected shapes:**
```
Parameter 0: shape (256, 4)      # LSTM input projection
Parameter 1: shape (1024, 64)    # LSTM hidden state
Parameter 2: shape (1024, 64)    # LSTM cell state
Parameter 3: shape (64,)         # LSTM bias
Parameter 4: shape (1, 64)       # Linear layer
Parameter 5: shape (1,)          # Linear bias
```

**Solution:**

1. **Verify all clients use same model**
   ```python
   # In src/client.py
   # Ensure WeatherLSTM() created same way for all clients
   ```

2. **Check aggregation code**
   ```python
   # In simulation.py aggregate_fit()
   # Ensure weights stacked with same axis
   ```

---

#### **Problem: "Aggregation very slow (>1 second per round)"**

**Cause:** Inefficient parameter averaging, or large model.

**Diagnosis:**
```python
# Benchmark aggregation
import time
import numpy as np

weights = [np.random.randn(1000) for _ in range(15)]  # 15 clients

# Method 1: Python loop
t0 = time.time()
avg = np.zeros_like(weights[0])
for w in weights:
    avg += w
avg /= len(weights)
print(f"Python loop: {(time.time() - t0) * 1000:.2f} ms")

# Method 2: NumPy mean
t0 = time.time()
avg = np.mean(weights, axis=0)
print(f"NumPy mean: {(time.time() - t0) * 1000:.2f} ms")
```

**Solution:** Use NumPy vectorized operations (should be default)

---

### 4.4 Parameter Exchange Issues

#### **Problem: "Get/set parameters fails - shape mismatch"**

**Cause:** Inconsistency between get_parameters() and set_parameters().

**Verification:**
```python
from src.model import WeatherLSTM

model = WeatherLSTM()

# Get parameters
params = model.get_model_parameters()
print(f"Get: {len(params)} arrays")
for i, p in enumerate(params):
    print(f"  {i}: {p.shape}")

# Set parameters
model.set_model_parameters(params)
print(f"✓ Set successful")

# Get again - should be identical
params2 = model.get_model_parameters()
import numpy as np
for i in range(len(params)):
    assert np.allclose(params[i], params2[i])
print(f"✓ Consistent")
```

---

## 5. Server & Deployment Issues

### 5.1 Flower Server Configuration

#### **Problem: "InvalidVersionError: Flower version mismatch"**

**Cause:** Client and server running different Flower versions.

**Diagnosis:**
```bash
# Check server version
python3 -c "import flwr; print(f'Server: {flwr.__version__}')"

# Check client version (in Ray worker)
# This is harder to diagnose - check requirements.txt
cat requirements.txt | grep flwr
```

**Solution:**
```bash
# Ensure consistent version
pip install flwr==1.8.0

# Rebuild Docker image if using containers
docker build --tag fl-weather .
```

---

#### **Problem: "Server log shows warnings about deprecated APIs"**

**Solution:**
```bash
# Update to latest compatible version
pip install --upgrade flwr
# Note: May break compatibility if upgraded to 2.0+
# Recommended: Stay on 1.8.x for this codebase
```

---

### 5.2 Ray Cluster Issues

#### **Problem: "Ray can't connect to cluster - 'ConnectionError'"**

**Cause:** Ray cluster not running, or wrong IP address.

**Diagnosis:**
```bash
ray status
# If fails: Ray not running

# Check for Ray processes
ps aux | grep ray
```

**Solutions:**

1. **Start Ray cluster**
   ```bash
   ray start --head
   ```

2. **If already started, reconnect**
   ```python
   import ray
   ray.shutdown()  # Kill existing connection
   ray.init()      # Reconnect
   ```

3. **If persistent issues, restart fully**
   ```bash
   ray stop
   sleep 5
   ray start --head --dashboard-host 0.0.0.0
   ```

---

#### **Problem: "Ray workers running out of memory"**

**Cause:** MAX_WORKERS too high for available RAM.

**Diagnosis:**
```bash
# Check Ray memory
ray memory
# or
ray status
```

**Solution:**
```python
# In src/config.py
MAX_WORKERS = 1  # Reduce from 2

# Or increase system memory
# Monitor before training:
watch -n 2 'free -h'
```

---

### 5.3 Docker Deployment

#### **Problem: "Container exits immediately - 'OCI runtime error'"**

**Diagnosis:**
```bash
docker logs <container_id>
# Check for actual error message
```

**Solutions:**

1. **Check Dockerfile**
   ```dockerfile
   # Ensure base image compatible with GPU
   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
   # Not FROM ubuntu:22.04 (missing CUDA)
   ```

2. **Run interactively for debugging**
   ```bash
   docker run -it --gpus all <image_id> /bin/bash
   # Can now manually test
   ```

---

#### **Problem: "GPU not available in container"**

**Diagnosis:**
```bash
# Check Docker GPU support
docker run --gpus all nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
  nvidia-smi
# Should show GPU info
```

**Solutions:**

1. **Ensure nvidia-docker installed**
   ```bash
   which nvidia-docker
   # If not found:
   apt-get install nvidia-docker2
   systemctl restart docker
   ```

2. **Run with GPU flag**
   ```bash
   docker run --gpus all -it <image_id>
   # NOT docker run -it <image_id> (without --gpus)
   ```

---

## 6. Performance Optimization

### 6.1 Network & Communication Optimization

#### **Problem: "Communication takes 30% of round time"**

**Cause:** Large model size, slow network, or inefficient serialization.

**Expected communication time:**
- Per round: ~40 MB transfer
- At 1 Gbps: 320 ms
- At 100 Mbps: 3.2 s
- Current implementation: ~200 ms (good)

**Diagnosis:**
```bash
# Monitor network during training
iftop -i eth0  # Or appropriate interface
# or
nethogs  # By process

# Measure current transfer rate
iperf3 -s  # Server
iperf3 -c localhost  # Client (separate terminal)
```

**Optimization strategies:**

1. **Reduce model size** (if accuracy allows)
   ```python
   # In src/config.py
   LSTM_HIDDEN_SIZE = 32  # Reduce from 64
   # Reduces model size by 50%
   ```

2. **Enable compression**
   ```python
   # In simulation.py, add compression
   from flwr.common.typing import NDArrays
   
   def compress_parameters(parameters: NDArrays) -> NDArrays:
       # Quantize to 16-bit float
       return [p.astype(np.float16) for p in parameters]
   ```

3. **Reduce communication frequency**
   ```python
   # In src/config.py
   NUM_ROUNDS = 5  # Reduce from 10 (trades accuracy for speed)
   ```

---

#### **Problem: "Clients timeout during parameter upload"**

**Cause:** Slow network, or large batch aggregation.

**Solutions:**

1. **Increase timeout**
   ```python
   # In simulation.py
   server = fl.server.start_server(
       config=fl.server.ServerConfig(
           num_rounds=10,
           round_timeout=300.0  # seconds
       )
   )
   ```

2. **Reduce batch size**
   ```python
   # In src/config.py
   BATCH_SIZE = 16  # Reduce from 32
   ```

---

### 6.2 GPU Memory Optimization

#### **Problem: "GPU memory usage keeps increasing"**

**Cause:** Memory leak, or gradual accumulation of tensors.

**Diagnosis:**
```python
import torch
import gc

for i in range(100):
    # Training iteration
    model_output = model(X_batch)
    loss = criterion(model_output, y_batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % 10 == 0:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Iteration {i}: {allocated:.2f}/{reserved:.2f} GB")
        
        # Should be stable, not increasing
```

**Solutions:**

1. **Clear cache periodically**
   ```python
   # In training loop
   if iteration % 50 == 0:
       torch.cuda.empty_cache()
       gc.collect()
   ```

2. **Detach tensors**
   ```python
   # Incorrect - keeps reference
   outputs_list = [model(X) for X in batches]
   
   # Correct - detach
   outputs_list = [model(X).detach() for X in batches]
   ```

3. **Check for tensor accumulation**
   ```python
   # Incorrect - accumulates loss
   all_losses = []
   for batch in dataloader:
       loss = criterion(model(batch), targets)
       all_losses.append(loss)  # ⚠️ Keeps graph
   
   # Correct - track as scalar
   total_loss = 0
   for batch in dataloader:
       loss = criterion(model(batch), targets)
       total_loss += loss.item()  # ✓ Scalar only
   ```

---

### 6.3 CPU & Disk Optimization

#### **Problem: "Data loading is bottleneck (>50% round time)"**

**Cause:** Inefficient data loading, small num_workers, or slow disk.

**Diagnosis:**
```bash
# Check disk I/O
iostat -x 1
# Look for %util and await columns

# Check DataLoader efficiency
python3 << 'EOF'
import time
import torch
from src.utils import prepare_client_data
from torch.utils.data import TensorDataset, DataLoader

X_train, X_test, y_train, y_test, scaler = prepare_client_data(0)
dataset = TensorDataset(torch.from_numpy(X_train).float(),
                       torch.from_numpy(y_train).float())

t0 = time.time()
dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
for batch in dataloader:
    pass
print(f"num_workers=0: {time.time() - t0:.2f} sec")

t0 = time.time()
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
for batch in dataloader:
    pass
print(f"num_workers=4: {time.time() - t0:.2f} sec")
EOF
```

**Solutions:**

1. **Increase num_workers** (if CPU available)
   ```python
   # In src/client.py
   self.train_loader = DataLoader(
       ...,
       num_workers=4  # Increase from 0
   )
   ```

2. **Use prefetch_factor**
   ```python
   self.train_loader = DataLoader(
       ...,
       num_workers=4,
       prefetch_factor=2
   )
   ```

3. **Pin memory**
   ```python
   self.train_loader = DataLoader(
       ...,
       pin_memory=True  # Add this
   )
   ```

---

#### **Problem: "CSV files are slow to read"**

**Solution: Convert to optimized format**

```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

# Load CSV once
df = pd.read_csv('data/raw/weather_dataset_2025.csv')

# Save as Parquet (faster, compressed)
df.to_parquet('data/raw/weather_dataset_2025.parquet')

# Or NumPy (fastest for numeric data)
data = df[['Temperature_C', 'Humidity_%', 'Wind_Speed_km_h', 'Precipitation_mm']].values
np.save('data/raw/weather_data.npy', data)

# Then update utils.py to read NPY format instead of CSV
EOF
```

Update `src/utils.py`:
```python
# Change from:
df = pd.read_csv('data/raw/weather_dataset_2025.csv')

# To:
all_data = np.load('data/raw/weather_data.npy')  # 5475 × 4
df = pd.DataFrame(all_data, columns=['Temperature_C', 'Humidity_%', 'Wind_Speed_km_h', 'Precipitation_mm'])
```

---

## 7. Common Error Messages & Solutions

### 7.1 Error Reference Matrix

| Error | Cause | Quick Fix |
|-------|-------|-----------|
| `ModuleNotFoundError: No module named 'flower'` | Flower not installed | `pip install flwr==1.8.0` |
| `ImportError: cannot import name '_ssl'` | Python version issue | `conda install python=3.11` |
| `CUDA out of memory` | Batch size too large | Reduce `BATCH_SIZE` in config |
| `CUDA runtime error: no kernel image` | CUDA version mismatch | Reinstall PyTorch for correct CUDA |
| `FileNotFoundError: weather_dataset_2025.csv` | File path wrong | Check working directory: `pwd` |
| `Shape mismatch: expected (365, 4)` | Data file corrupted | Regenerate from raw data |
| `Loss is NaN` | Gradient explosion | Add gradient clipping or reduce learning rate |
| `Loss oscillates` | Non-IID data (expected) | Normal - see 4_EVALUATION.md |
| `Model won't improve` | Learning rate too high | Reduce `LEARNING_RATE` |
| `Connection refused: 8080` | Server not running | `python3 src/simulation.py` |
| `RemoteRpcError` | Client disconnected | Check memory: `free -h` |
| `Address already in use` | Port 8080 occupied | `lsof -ti :8080 \| xargs kill -9` |
| `Ray status error` | Ray cluster not running | `ray start --head` |
| `GPU not available in Docker` | Missing nvidia-docker | `docker run --gpus all` |
| `Server hangs` | Client timeout | Increase `round_timeout` |

---

### 7.2 Error Diagnosis Flowchart

```
ERROR OCCURS
    ↓
[1] Check logs
    ├─ Server: logs/server.log
    ├─ Client: logs/client_*.log
    ├─ Ray: ray logs cluster
    ├─ System: dmesg, journalctl
    └─ → Extract full error message
    ↓
[2] Is error in...?
    ├─ Installation?
        └─→ Section 1 (Installation & Environment)
    ├─ Data loading?
        └─→ Section 2 (Data Loading & Preprocessing)
    ├─ Model training?
        └─→ Section 3 (Model Training)
    ├─ Federated Learning?
        └─→ Section 4 (Federated Learning)
    ├─ Deployment?
        └─→ Section 5 (Server & Deployment)
    └─ Performance?
        └─→ Section 6 (Performance Optimization)
    ↓
[3] Find matching error in section
    ├─ Read Problem description
    ├─ Run Diagnosis commands
    ├─ Try Solution #1
    └─ If fails, try Solution #2, #3, etc.
    ↓
[4] Verify fix
    ├─ Run: python3 src/simulation.py
    ├─ Check output for errors
    └─ Compare metrics to expected values
    ↓
✓ RESOLVED
```

---

### 7.3 Logging & Debugging Guide

#### **Enable detailed logging:**

```python
# Add to src/simulation.py or src/client.py
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Then log key events
logger.debug(f"Client {client_id} starting with {len(X_train)} samples")
logger.info(f"Round {round}: MSE = {loss:.4f}")
logger.warning(f"Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

#### **Check debug logs:**

```bash
tail -100 logs/debug.log | grep -i error
tail -100 logs/debug.log | grep -i warning
```

---

## 8. Frequently Asked Questions (FAQ)

### **Q: Should I see convergence in first round?**

**A:** Yes. Expected loss should drop 3-5% in round 1.
- Round 1: ~1.859 → ~1.808 (2.7% improvement)
- If less: Learning rate may be too low
- If much more: Could be lucky initialization

---

### **Q: Is non-IID oscillation a problem?**

**A:** No. Expected behavior documented in FL literature.
- Rounds 1-5: Convergence phase
- Rounds 6-10: Oscillation phase (±1-2%)
- This is byproduct of federated learning with heterogeneous data

See 4_EVALUATION.md Section 7 for detailed analysis.

---

### **Q: Can I train on more cities?**

**A:** Yes, system supports up to 100+ clients on single GPU.

Changes needed:
```python
# src/config.py
CITIES = ['Berlin', 'Cairo', ..., 'NewCity1', 'NewCity2']  # Add more
NUM_CLIENTS = len(CITIES)

# src/utils.py
# Add corresponding CSV files in data/processed/
# Ensure 365 rows each
```

---

### **Q: What's the minimum GPU memory needed?**

**A:** 4 GB minimum, 6 GB recommended.

Memory breakdown (per client):
- Model + gradients: 0.27 MB
- Optimizer state: 0.27 MB
- LSTM activations: 1.5 MB
- PyTorch overhead: 1.0 MB
- **Total per client:** ~3.6 MB

For 2 clients: ~7 MB (safe on 4 GB with other OS overhead)

---

### **Q: How do I make training faster?**

**A:** Priority order:

1. **Increase MAX_WORKERS** (if memory available)
2. **Reduce LOCAL_EPOCHS** (trades accuracy for speed)
3. **Reduce NUM_ROUNDS** (even faster, but less convergence)
4. **Use smaller batch size** (actually makes slower due to overhead)

See Section 6 (Performance Optimization) for detailed strategies.

---

### **Q: How do I reduce model size for inference?**

**A:** Convert to optimized format:

```python
import torch
import numpy as np
from src.model import WeatherLSTM

model = WeatherLSTM()
# ... train model ...

# Option 1: Quantize to INT8
from torch.quantization import quantize_dynamic
quantized = quantize_dynamic(model, {torch.nn.LSTM}, dtype=torch.qint8)
torch.save(quantized.state_dict(), 'model_int8.pth')
# Reduces size by ~75%

# Option 2: Prune
from torch.nn.utils.prune import l1_unstructured
for module in model.modules():
    if isinstance(module, torch.nn.LSTM):
        l1_unstructured(module, name='weight_ih_l0', amount=0.3)
torch.save(model.state_dict(), 'model_pruned.pth')

# Option 3: Convert to ONNX
torch.onnx.export(model, X_sample, 'model.onnx')
```

---

### **Q: Can I run on CPU only (no GPU)?**

**A:** Yes, but much slower.

```python
# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
print(torch.cuda.is_available())  # Should be False
```

Expected timing:
- With GPU: 2-3 minutes
- With CPU: 30-60 minutes

Not recommended for development.

---

### **Q: How do I save and load trained model?**

**A:** Multiple formats supported:

```python
from src.model import WeatherLSTM
import torch

model = WeatherLSTM()

# Option 1: Save weights only
torch.save(model.state_dict(), 'model_weights.pth')

# Load:
model = WeatherLSTM()
model.load_state_dict(torch.load('model_weights.pth'))

# Option 2: Save entire model (not recommended)
torch.save(model, 'model.pth')
# Load:
model = torch.load('model.pth')

# Option 3: NumPy format (for FL parameter exchange)
import numpy as np
params = [p.data.cpu().numpy() for p in model.parameters()]
np.save('model_params.npy', params, allow_pickle=True)
```

---

### **Q: How do I visualize training progress?**

**A:** Use matplotlib or TensorBoard:

```python
import matplotlib.pyplot as plt
import json

# Read training logs
with open('logs/metrics.json') as f:
    metrics = json.load(f)

rounds = [m['round'] for m in metrics]
losses = [m['loss'] for m in metrics]

plt.figure(figsize=(10, 6))
plt.plot(rounds, losses, 'b-o')
plt.xlabel('Round')
plt.ylabel('MSE Loss')
plt.title('Federated Learning Convergence')
plt.grid(True, alpha=0.3)
plt.savefig('training_progress.png')
plt.show()
```

Or TensorBoard (if using):
```bash
tensorboard --logdir=logs/tensorboard
# Open http://localhost:6006
```

---

## 9. Advanced Debugging

### 9.1 Memory Profiling

**Complete memory analysis:**

```bash
# Install memory_profiler
pip install memory-profiler

# Profile entire training
python3 -m memory_profiler src/simulation.py

# Or profile specific function
# Add @profile decorator to functions, then:
python3 -m memory_profiler -l src/simulation.py
```

---

### 9.2 Performance Profiling

**Find bottlenecks:**

```python
import cProfile
import pstats

# Profile entire training
cProfile.run('from src.simulation import main; main()', 'profile_stats.prof')

# Analyze results
p = pstats.Stats('profile_stats.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(20)  # Top 20 functions
```

---

### 9.3 GPU Profiling

**Detailed GPU analysis:**

```bash
# NVIDIA Nsys (best option)
nsys profile -o training python3 src/simulation.py
nsys stats training.qdrep

# Or simpler: nvidia-smi queries
watch -n 0.5 nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader,nounits
```

---

## 10. Troubleshooting Checklist

### **Before Starting Training**

- [ ] NVIDIA driver installed: `nvidia-smi`
- [ ] PyTorch with GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] All dependencies: `pip list | grep -E "flwr|torch|ray|pandas|numpy"`
- [ ] Data files exist: `ls -la data/processed/*.csv` (15 files)
- [ ] Port 8080 available: `netstat -tuln | grep 8080` (no output = good)
- [ ] Sufficient GPU memory: `nvidia-smi` (6+ GB free)
- [ ] Sufficient RAM: `free -h` (8+ GB free)

### **If Training Crashes**

- [ ] Check error message: `tail -50 logs/server.log`
- [ ] Run specific diagnostic: See Section 7.1 error table
- [ ] Clean up: `ray stop && pkill -f simulation.py`
- [ ] Restart: `ray start --head && python3 src/simulation.py`

### **If Training Slow**

- [ ] Check GPU util: `nvidia-smi dmon` (should be ~90%)
- [ ] Check CPU util: `top` (should be 20-30% if GPU bottleneck)
- [ ] Check memory: `watch free -h` (should be stable)
- [ ] Check network: `iftop` (if distributed, should be 100+ Mbps)

### **If Accuracy Poor**

- [ ] Verify data: `python3 -c "from src.utils import prepare_client_data; X, _, _, _, _ = prepare_client_data(0); print(f'Mean: {X.mean():.2f}, Std: {X.std():.2f}')"` (should be 0, 1)
- [ ] Check loss: `grep "MSE" logs/*.log | tail -1` (should be ~1.85)
- [ ] Verify learning rate: `grep "LEARNING_RATE" src/config.py` (should be 0.001)
- [ ] Check epochs: `grep "LOCAL_EPOCHS" src/config.py` (should be >= 5)

---

## 11. Getting Help

### **If you're still stuck:**

1. **Collect diagnostics:**
   ```bash
   mkdir debug_info
   
   # System info
   uname -a > debug_info/system.txt
   nvidia-smi >> debug_info/system.txt
   
   # Python environment
   python3 --version >> debug_info/system.txt
   pip list >> debug_info/system.txt
   
   # Log files
   cp -r logs debug_info/
   
   # Code configuration
   cp src/config.py debug_info/
   
   # Create archive
   tar -czf debug_info.tar.gz debug_info/
   ```

2. **Share error message:** Include full error traceback with context

3. **Include reproduction steps:** "Running `python3 src/simulation.py` on June 4, 2026 with RTX 4050 GPU"

4. **Check related documentation:**
   - Architecture details: [1_ARCHITECTURE.md](1_ARCHITECTURE.md)
   - Implementation guide: [2_IMPLEMENTATION.md](2_IMPLEMENTATION.md)
   - Deployment manual: [3_DEPLOYMENT.md](3_DEPLOYMENT.md)
   - Performance analysis: [4_EVALUATION.md](4_EVALUATION.md)

---

## Appendix A: Configuration Reference

### **Key Parameters in src/config.py**

```python
# Model hyperparameters
LSTM_HIDDEN_SIZE = 64        # Reduce if memory limited
DROPOUT = 0.2                # Prevent overfitting

# Training hyperparameters
BATCH_SIZE = 32              # Reduce to 16 if OOM
LOCAL_EPOCHS = 5             # Increase to 10 for better convergence
LEARNING_RATE = 0.001        # Reduce to 0.0005 if diverging

# Federated learning
NUM_ROUNDS = 10              # Reduce for faster testing
NUM_CLIENTS = 15             # Can increase to 100+ if desired
MAX_WORKERS = 2              # Increase if memory available

# System
SEED = 42                    # For reproducibility
```

---

## Appendix B: Quick Commands

```bash
# Start training
python3 src/simulation.py

# Check GPU status
nvidia-smi dmon -s pucvme

# Monitor memory
watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'

# View logs
tail -100 logs/server.log

# Kill stuck process
pkill -9 -f simulation.py

# Clean Ray cluster
ray stop

# Restart Ray
ray start --head --dashboard-host 0.0.0.0
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Document ID** | 5_TROUBLESHOOTING.md |
| **Version** | 1.0 |
| **Created** | June 4, 2026 |
| **System** | Federated Learning Weather Prediction (FL-Weather) |
| **Scope** | Error diagnosis, resolution, optimization |
| **Sections** | 11 major + 2 appendices |
| **Error Solutions** | 50+ documented errors |
| **FAQ Entries** | 10 common questions |
| **Command Examples** | 100+ diagnostic/fix commands |
| **Cross-references** | Links to all 5 documentation files |
| **Intended Audience** | Developers, DevOps, ML Engineers, System Admins |
| **Accessibility** | Beginner-friendly with advanced sections |
| **Maintainability** | Well-organized, easy to update |

---

**End of 5_TROUBLESHOOTING.md**
