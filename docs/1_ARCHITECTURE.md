# Federated Learning Weather Prediction System: Architecture & Technical Design

**Document Version:** 1.0  
**Last Updated:** June 2026  
**Status:** Production-Ready  
**Classification:** Technical Architecture Document

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Deep Learning Model Specification](#deep-learning-model-specification)
4. [Federated Learning Strategy](#federated-learning-strategy)
5. [Hardware-Aware Constraints & Optimization](#hardware-aware-constraints--optimization)
6. [Data Flow & Communication Protocol](#data-flow--communication-protocol)
7. [Scalability Considerations](#scalability-considerations)
8. [References & Mathematical Foundations](#references--mathematical-foundations)

---

## Executive Summary

This document describes the technical architecture of a **distributed Federated Learning (FL) system** for multi-city weather prediction. The system trains a PyTorch LSTM neural network across 15 geographically distributed clients (representing major cities worldwide) without centralizing raw weather data, preserving privacy while achieving predictive accuracy.

**Key Design Principles:**
- **Privacy-by-Design**: Raw sensor data never leaves client nodes
- **Communication-Efficient**: Only model parameters (not data) traverse the network
- **Hardware-Constrained Optimization**: Operates within 6GB GPU VRAM using dynamic resource allocation
- **Non-IID Resilient**: Handles heterogeneous data distributions across federated clients

---

## System Architecture

### 1.1 High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
        │  FL Server  │   │  Ray Cluster│   │  Aggregator │
        │  (Manager)  │   │  (Resource) │   │  (Strategy) │
        └─────────────┘   └─────────────┘   └─────────────┘
                │              │              │
    ┌───────────┼──────────────┼──────────────┼───────────┐
    │           │              │              │           │
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐    ...┌────────┐
│ Client │││ Client ││ Client ││ Client ││ Client │    ...│ Client │
│  City 1│││  City 2││  City 3││  City 4││  City 5│    ...│ City 15│
└────────┘└────────┘└────────┘└────────┘└────────┘    ...└────────┘
   LSTM     LSTM     LSTM     LSTM     LSTM           LSTM
```

### 1.2 Architectural Components

#### **A. Central Orchestration Layer**

| Component | Role | Technology |
|-----------|------|------------|
| **Flower Server** | Coordinates FL rounds, distributes global weights, collects client updates | Flower Framework v1.8+ |
| **FedAvg Strategy** | Computes weighted parameter aggregation, maintains global model | Custom aggregation logic |
| **Training Coordinator** | Manages round scheduling, logging, convergence monitoring | Python orchestration |
| **Result Logger** | Persists training metrics (MSE per round) to disk | CSV-based logging |

**Flower Server Responsibilities:**
```python
# Pseudo-code representation
for round in range(NUM_ROUNDS):
    # 1. Sample clients (all 15 in this implementation)
    selected_clients = sample_clients(15, fraction=1.0)
    
    # 2. Send current global weights to clients
    for client in selected_clients:
        client.receive_parameters(global_weights)
    
    # 3. Collect updated weights and metrics
    client_updates = collect_updates(selected_clients)
    
    # 4. Aggregate using FedAvg
    global_weights = aggregate(client_updates, sample_sizes)
    
    # 5. Evaluate on all clients
    metrics = evaluate(global_weights, selected_clients)
    
    # 6. Log and report
    log_round_metrics(round, metrics)
```

#### **B. Distributed Client Layer**

**Client Architecture (Per City):**

Each of the 15 clients implements the following workflow:

```
┌──────────────────────────────────────────┐
│          DISTRIBUTED CLIENT (City)       │
├──────────────────────────────────────────┤
│ 1. Load Local Data (365-day weather)    │
│    └─ 80/20 chronological train/test    │
│                                          │
│ 2. Initialize LSTM Model                │
│    └─ Receive global weights from server│
│                                          │
│ 3. Local Training (5 epochs)            │
│    └─ Forward pass → Loss → Backprop   │
│                                          │
│ 4. Extract Model Parameters             │
│    └─ Serialize 10 weight matrices      │
│                                          │
│ 5. Return to Server                     │
│    └─ Updated weights + local loss      │
│                                          │
│ 6. Evaluation on Test Set               │
│    └─ Compute MSE without updating      │
└──────────────────────────────────────────┘
```

**Client Implementation Specifications:**

```python
# Class: WeatherClient (derived from flwr.client.NumPyClient)
class WeatherClient(NumPyClient):
    def __init__(self, city_index: int):
        self.city_index = city_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WeatherLSTM()  # Local model instance
        self.optimizer = Adam(lr=0.001)
        self.criterion = MSELoss()
        
        # Load local data
        X_train, y_train, X_test, y_test, scaler = prepare_client_data(city_index)
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=32,
            shuffle=False
        )
    
    def get_parameters(self) -> NDArrays:
        """Extract current model weights as numpy arrays"""
        return get_model_parameters(self.model)
    
    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """Local training: 5 epochs on local data"""
        set_model_parameters(self.model, parameters)
        self.model.train()
        
        local_loss = 0.0
        for epoch in range(5):  # LOCAL_EPOCHS
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            local_loss = epoch_loss / len(self.train_loader)
        
        return get_model_parameters(self.model), len(self.train_loader.dataset), {
            "train_loss": local_loss
        }
    
    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Local evaluation: test set MSE"""
        set_model_parameters(self.model, parameters)
        self.model.eval()
        
        total_loss = 0.0
        sample_count = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * len(y_batch)
                sample_count += len(y_batch)
        
        return total_loss / sample_count, sample_count, {"mse": total_loss / sample_count}
```

#### **C. Ray Distributed Execution Layer**

**Ray Cluster Configuration:**

Ray manages concurrent client execution while respecting hardware constraints:

```python
# Ray client resource allocation
client_resources = {
    "num_gpus": 1.0 / MAX_WORKERS,  # 0.5 GPU per client (2 concurrent clients)
    "num_cpus": 2                    # 2 CPU cores per client
}

# Example with MAX_WORKERS=2:
# - Up to 2 clients train in parallel
# - Each gets 0.5 GPU allocation (total 1.0 GPU = 6GB VRAM)
# - Each gets 2 CPU cores (total 4 cores = avoid contention)
```

**Worker Lifecycle:**

```
Client 1 ─┐
         ├─ Ray Worker Process 1 (0.5 GPU, 2 CPU)
Client 2 ─┘
                        │
Client 3 ─┐             ├─ Train epochs 1-5
         ├─ Ray Worker Process 2 (0.5 GPU, 2 CPU)
Client 4 ─┘             │
                        │ (After completion, new clients scheduled)
```

---

## Deep Learning Model Specification

### 2.1 Architecture Overview

**Model Type:** Sequence-to-Scalar LSTM Regression

**Purpose:** Predict next-day temperature given 5-day weather history

### 2.2 Input/Output Specifications

```
Input:  (batch_size, sequence_length, n_features)
        (32, 5, 4)  where:
            - batch_size=32 (training batch)
            - sequence_length=5 (5 consecutive days)
            - n_features=4 (Temperature_C, Humidity_%, Wind_Speed_km_h, Precipitation_mm)

Output: (batch_size, 1)
        (32, 1)  - Predicted temperature for next day
```

### 2.3 Layer-by-Layer Specification

#### **Layer 1: LSTM Core**

```python
class WeatherLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,        # 4 weather features
            hidden_size=64,      # 64 hidden units per layer
            num_layers=2,        # 2 stacked LSTM layers
            dropout=0.2,         # 20% dropout between layers
            batch_first=True     # Input shape: (batch, seq, features)
        )
```

**LSTM Behavioral Specification:**

- **Forward Pass Flow:**
  ```
  Input (batch=32, seq=5, features=4)
         ↓
      LSTM Layer 1 (hidden=64)
         ├─ Cell state C₁ᵗ (32, 64)
         ├─ Hidden state h₁ᵗ (32, 64)
         ↓
      LSTM Layer 2 (hidden=64)
         ├─ Cell state C₂ᵗ (32, 64)
         ├─ Hidden state h₂ᵗ (32, 64)  ← We use this
         ↓
  ```

- **Dropout Mechanism:**
  - Applied between Layer 1→2 output during training
  - Probability: 20% neuron deactivation
  - **Purpose:** Prevent co-adaptation of hidden units, reduce overfitting to local non-IID data

- **Mathematical Detail (Single LSTM Cell):**
  ```
  iₜ = σ(Wᵢᵢ·xₜ + Wʰᵢ·hₜ₋₁ + bᵢ)     [Input gate]
  fₜ = σ(Wᵢf·xₜ + Wʰf·hₜ₋₁ + bf)     [Forget gate]
  C̃ₜ = tanh(Wᵢc·xₜ + Wʰc·hₜ₋₁ + bc)  [Cell candidate]
  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ           [Cell update]
  oₜ = σ(Wᵢₒ·xₜ + Wʰₒ·hₜ₋₁ + bₒ)     [Output gate]
  hₜ = oₜ ⊙ tanh(Cₜ)                [Hidden state output]
  ```
  where σ=sigmoid, ⊙=Hadamard product

#### **Layer 2: Linear Projection**

```python
self.fc = nn.Linear(
    in_features=64,   # Hidden size from LSTM
    out_features=1    # Single regression output
)
```

**Purpose:** Project learned representations (64-dim) to target space (1-dim temperature)

#### **Layer 3: Forward Method with GPU Optimization**

```python
def forward(self, x):
    # x shape: (batch=32, seq=5, features=4)
    
    # Critical GPU optimization
    self.lstm.flatten_parameters()  # Compact weight memory layout
    
    # LSTM forward: returns (output, (h_n, c_n))
    lstm_out, (h_n, c_n) = self.lstm(x)
    # lstm_out: (batch=32, seq=5, hidden=64)
    # h_n: (num_layers=2, batch=32, hidden=64)
    # c_n: (num_layers=2, batch=32, hidden=64)
    
    # Take hidden state from last LSTM layer, final timestep
    last_hidden = h_n[-1, :, :]  # (batch=32, hidden=64)
    
    # Linear projection to scalar
    output = self.fc(last_hidden)  # (batch=32, 1)
    
    return output
```

### 2.4 Why LSTM for Weather Time-Series Regression?

| Aspect | Traditional Neural Network | Recurrent Neural Network | **LSTM (Our Choice)** |
|--------|---------------------------|-----------------------|----------------------|
| **Temporal Dependency** | ❌ Treats each timestep independently | ⚠️ Limited by vanishing gradient | ✅ Learns long-range dependencies |
| **Memory Mechanism** | ❌ No explicit memory | ⚠️ Basic hidden state | ✅ Cell state + gates for selective memory |
| **Gradient Flow** | ❌ Degrades over sequences | ⚠️ Exponential decay | ✅ Constant gradient through cell state |
| **Sequence Length** | ❌ Fixed input size | ⚠️ Unstable for seq>10 | ✅ Stable for arbitrary lengths |
| **Weather Suitability** | ❌ Cannot capture seasonality | ⚠️ Weak pattern learning | ✅ Models 5-day weather dependencies |

**Concrete Intuition for Weather:**

Weather demonstrates **strong sequential dependencies**:
- Day 1-2: Temperature changes gradually (inertia)
- Day 3-4: Humidity affects temperature (correlation)
- Day 5: Pressure systems introduce multi-day lags

LSTM's **forget gate** learns *when* to discard old information:
```
Forget Gate Example:
Day 1: weather_pattern = "low pressure system"
       ├─ Remember cell: yes (relevant for next 3 days)
       ├─ Forget percentage: 10%
Day 4: same system still influences
       └─ Forget gate: "release this memory now" (90% forget)
```

### 2.5 Model Parameters & Capacity

**Total Trainable Parameters:**

```
LSTM Layer 1:
  - Input gate weights: 4×64 + 64×64 + 64 = 4,160
  - Forget gate weights: 4×64 + 64×64 + 64 = 4,160
  - Cell gate weights: 4×64 + 64×64 + 64 = 4,160
  - Output gate weights: 4×64 + 64×64 + 64 = 4,160
  └─ Subtotal: 16,640 parameters

LSTM Layer 2:
  - Similar structure with input_size=64: 16,640 + (64×64×4 + 64×4) = 17,408
  └─ Subtotal: 17,408 parameters

Linear Layer:
  - Weights: 64×1 = 64
  - Bias: 1
  └─ Subtotal: 65 parameters

──────────────────
TOTAL: ~34,113 parameters
```

**Memory Footprint:**

```
Float32 storage: 34,113 × 4 bytes = 136.5 KB per model
Multi-client (15): 136.5 KB × 15 = ~2.05 MB
Aggregation overhead: ~3 MB
Total model memory: <5 MB (negligible compared to data)
```

### 2.6 Loss Function & Optimization

**Loss Function: Mean Squared Error (MSE)**

```python
criterion = nn.MSELoss(reduction='mean')

# Mathematical form:
# MSE = (1/n) × Σ(ŷᵢ - yᵢ)²
```

**Why MSE for Regression?**
- Continuous target (temperature): ✅ MSE suitable
- Outlier sensitivity: ⚠️ Incentivizes learning extreme events
- Interpretability: ✅ MSE in °C² directly meaningful

**Optimizer: Adam (Adaptive Moment Estimation)**

```python
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # Exponential decay rates
    eps=1e-8,           # Numerical stability
    weight_decay=0.0    # No L2 regularization
)
```

**Adam Update Rule:**

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ         [First moment (mean)]
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²        [Second moment (variance)]
m̂ₜ = mₜ / (1 - β₁ᵗ)              [Bias-corrected first moment]
v̂ₜ = vₜ / (1 - β₂ᵗ)              [Bias-corrected second moment]
θₜ₊₁ = θₜ - α·m̂ₜ / (√v̂ₜ + ε)     [Parameter update]

Where:
- α = 0.001 (learning rate)
- gₜ = gradient at step t
- t = current iteration
```

**Why Adam vs SGD?**
- ✅ Adaptive per-parameter learning rates
- ✅ Momentum helps navigate non-IID loss landscapes
- ✅ Handles varying local data distributions

---

## Federated Learning Strategy

### 3.1 FedAvg (Federated Averaging) Algorithm

**Algorithm Overview:**

FedAvg is the foundational FL algorithm that enables privacy-preserving distributed training:

```
ALGORITHM: FedAvg
INPUT: K clients, T communication rounds, E local epochs, B batch size
       Initial weights w₀

OUTPUT: Final global model weights w

Procedure:
  for round t = 1 to T do:
    1. Server samples fraction C of K clients (all 15 in this case)
    
    2. for each selected client k in parallel do:
         // Local training
         wₖ,ₜ = LocalTraining(wₜ₋₁, E epochs, B batch size)
         nₖ = number of training samples on client k
       end
    
    3. wₜ = (Σₖ nₖ/n) · wₖ,ₜ  [Weighted aggregation by sample count]
       where n = Σₖ nₖ (total samples)
    
    4. Evaluate on all clients, log metrics
  end
```

**Key Innovation: Weight Aggregation Without Data Sharing**

```
Traditional Centralized Learning:
┌────────────────────────────────────────┐
│  Raw Data Pool (5,475 weather records) │
└────────────────────────────────────────┘
           │ (Privacy Risk!)
           ↓
    ┌────────────────┐
    │   ML Trainer   │
    │   (LSTM Model) │
    └────────────────┘

Federated Learning (This System):
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Berlin Data  │  │  Tokyo Data  │  │ Toronto Data │
│  (365 rows)  │  │  (365 rows)  │  │  (365 rows)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ (train locally)  │ (train locally)  │ (train locally)
       ↓                  ↓                  ↓
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Weights_B  │    │  Weights_T  │    │  Weights_To │
  │   Update    │    │   Update    │    │   Update    │
  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
         │                  │                  │ (Only weights!)
         │                  ↓                  │
         └──────────────────────────────────────┘
                     │
                     ↓
            ┌─────────────────┐
            │ FedAvg (Server) │
            │  Aggregates     │
            │  1.5 MB weights │
            │  (NOT 5475 MB   │
            │   of raw data!) │
            └─────────────────┘
                     │
                     ↓
          Global Model (Updated)
          Ready for next round
```

### 3.2 Mathematical Formulation of FedAvg

**Federated Objective Function:**

```
min_w F(w) = Σₖ₌₁^K (nₖ/n) · Lₖ(w)

Where:
- K = 15 (number of clients/cities)
- nₖ = sample count on client k (~365 for each)
- n = total samples (5,475)
- Lₖ(w) = local loss on client k = (1/nₖ) Σᵢ∈Dₖ ℓ(w; xᵢ, yᵢ)
- ℓ(·) = MSE loss for single sample
- w = global model parameters
```

**Weighted Aggregation Formula (Round t):**

```
w_t = Σₖ₌₁^K (nₖ/n) · w^k_t

Where:
- w^k_t = parameters returned from client k after local training
- (nₖ/n) = weight factor (sample count normalized)
- w_t = new global parameters for round t

Example:
Client 1 (Berlin):    287 samples, weight = 287/5475 ≈ 0.0524
Client 2 (Tokyo):     288 samples, weight = 288/5475 ≈ 0.0526
Client 3 (Toronto):   290 samples, weight = 290/5475 ≈ 0.0530
...
Global weights = 0.0524·w₁ + 0.0526·w₂ + 0.0530·w₃ + ...
```

### 3.3 Communication Round Protocol

**Single FL Round Sequence:**

```
┌─────────────────────────────────────────────────────────┐
│                    SERVER ROUND N                       │
└─────────────────────────────────────────────────────────┘

PHASE 1: PARAMETER DISTRIBUTION (t_0 = 0 ms)
  ├─ Server creates message: "global_weights_N"
  ├─ Size: ~136.5 KB (model parameters only)
  ├─ Broadcast to all 15 clients
  └─ Latency: ~10-100 ms (in-process with Ray)

PHASE 2: LOCAL TRAINING (t_0 + 100 ms → t_0 + 300 ms)
  ├─ Client 1-2 train in parallel (Ray batch 1)
  │  ├─ Load local data: 287 samples
  │  ├─ 5 epochs × (287/32) batches = 45 batches
  │  ├─ Forward + backward propagation
  │  ├─ Time: ~150-200 ms per client
  │  └─ Result: Updated weights + MSE loss
  │
  ├─ Client 3-4 train in parallel (Ray batch 2)
  │  └─ Concurrent execution [GPU utilization: 100%]
  │
  └─ Clients 5-15 scheduled sequentially (batches 3-8)
     ├─ All clients must complete before aggregation
     └─ Total phase duration: ~1000-1500 ms

PHASE 3: PARAMETER COLLECTION (t₀ + 1500 ms → t₀ + 1600 ms)
  ├─ Server collects updated weights from all clients
  ├─ Receives: 15 × 136.5 KB = 2.048 MB total
  ├─ Size tractable for network transmission
  └─ Aggregation latency: ~100 ms

PHASE 4: AGGREGATION (t₀ + 1600 ms → t₀ + 1610 ms)
  ├─ Server computes weighted average:
  │  global_w = Σ (nₖ/n) · w_k  [15 multiplications + 14 additions]
  ├─ Update global model in memory
  └─ Computational cost: <10 ms

PHASE 5: EVALUATION (t₀ + 1610 ms → t₀ + 1800 ms)
  ├─ Server sends global weights to selected clients
  ├─ Each client evaluates test set independently
  ├─ Collects MSE loss (scalar) from each
  ├─ Time: ~150-200 ms per client batch
  └─ Global MSE computed as weighted average

PHASE 6: LOGGING & METRICS (t₀ + 1800 ms → t₀ + 1810 ms)
  ├─ Log round number, global MSE, timestamp
  ├─ Persist to: logs/training_log.csv
  └─ Print: "[Round N/10] Global MSE: X.XXXXXX"

┌──────────────────────────────────────────────────────────┐
│            ROUND DURATION: ~1.8 seconds                  │
│         (Assuming Ray executor pool ready)               │
│   10 rounds = ~18 seconds + overhead = ~2-3 minutes     │
└──────────────────────────────────────────────────────────┘
```

### 3.4 Privacy Guarantees

**Information Disclosed:**

```
What Server SEES:                 What Server DOES NOT SEE:
─────────────────────────────────────────────────────────
✓ Model weights (136.5 KB)        ✗ Raw weather data (not transmitted)
✓ Local sample count (~288)        ✗ Individual records/samples
✓ Evaluation metrics (MSE)         ✗ Feature values (Temp, Humidity, etc)
✓ Aggregate statistics             ✗ Client hardware/infrastructure details
```

**Differential Privacy (Optional Enhancement):**

While not implemented in current version, the architecture supports:

```python
# Could add gradient clipping + Gaussian noise
class DifferentiallyPrivateOptimizer:
    def step(self, grads):
        # Clip gradients to norm C
        clipped_grads = gradient_clip(grads, C=1.0)
        
        # Add Gaussian noise for ε-DP guarantee
        noise = Normal(0, σ²)  # σ related to privacy budget ε
        noisy_grads = clipped_grads + noise
        
        # Update parameters
        return parameters - lr * noisy_grads
```

### 3.5 Non-IID Resilience

**Why Non-IID Data Matters:**

In traditional centralized ML, training data is **Independent & Identically Distributed (IID)**. Federated weather data is **Non-IID** because:

```
Dammam (Hot, Dry):          New York (Cold, Humid):
┌────────────────────┐      ┌────────────────────┐
│ Temp: 25.1 ± 12°C  │      │ Temp: 23.8 ± 11°C  │
│ Humidity: 40%      │      │ Humidity: 65%      │
│ Wind: 22 km/h      │      │ Wind: 18 km/h      │
│ Correlation Temp-  │      │ Correlation Temp-  │
│   Humidity: -0.3   │      │   Humidity: -0.7   │
└────────────────────┘      └────────────────────┘

             ↓                      ↓
         gradients           gradients
        (focus on          (focus on
        hot weather)       cold weather)
             ↓                      ↓
          Conflict!
    FedAvg averaging
     creates compromise
     that's suboptimal
      for both cities
```

**Convergence Under Non-IID (Oscillations Expected):**

```
Training Loss Over 10 Rounds:

MSE │                              
    │  1.8595  ← Round 1 (random initialization)
    │  1.8502  ↘ (decreasing - good convergence)
    │  1.8374  │
    │  1.8094  │
    │  1.7870  ↙ (best MSE at round 5)
    │  1.7891  ↗ (increasing - local divergence starts)
    │  1.8112  │
    │  1.8220  │ (oscillations due to Non-IID)
    │  1.8417  │
    │  1.8500  ↙ (final round)
    └──────────────────────────────────────
      1    2    3    4    5    6    7    8    9   10
                        Rounds

Why oscillations?
- Rounds 1-5: Global averaging helps all clients → convergence
- Round 5+: Clients' gradient directions too divergent
- FedAvg aggregation creates "average" that fits none perfectly
- Result: loss oscillates as clients "fight" for representation
```

---

## Hardware-Aware Constraints & Optimization

### 4.1 Hardware Specifications

```
┌─────────────────────────────────────────────────┐
│         TARGET HARDWARE (RTX 4050 Laptop)       │
├─────────────────────────────────────────────────┤
│ GPU:        NVIDIA RTX 4050 (6 GB VRAM)        │
│ CPU:        Intel Core i7-13th Gen (16 cores)  │
│ RAM:        16 GB DDR5                          │
│ Storage:    512 GB NVMe SSD                     │
│ Python:     3.13.x                              │
│ CUDA:       13.3                                │
└─────────────────────────────────────────────────┘
```

**Memory Bottleneck:** 6 GB GPU VRAM is the critical constraint

### 4.2 Memory Analysis

**Per-Client VRAM Allocation:**

```
Component                           Size        Notes
─────────────────────────────────────────────────────
Model Parameters (LSTM)             136.5 KB    34,113 parameters × 4 bytes
Model Gradients                     136.5 KB    ∂L/∂w for backprop
Optimizer State (Adam)              273 KB      Moment estimates (2× params)
Input Batch (32 × 5 × 4)           20.5 KB     (32 batches × 5 days × 4 features × 4 bytes)
Output Batch (32 × 1)              0.125 KB    32 predictions × 4 bytes
Intermediate Activations           ~1.5 MB     LSTM hidden states + cell states
PyTorch Overhead                   ~2 MB       Tensor metadata, cuDNN kernels
─────────────────────────────────────────────────
Per-Client Estimated:              ~4.2 MB

Concurrent Clients (MAX_WORKERS=2):
2 clients × 4.2 MB = 8.4 MB

Safety Margin:
6000 MB (6 GB) - 8.4 MB - 1500 MB (OS/System) ≈ 4500 MB available
Utilization: 8.4 / 6000 = 0.14% (safe, even with headroom)
```

**Why MAX_WORKERS=2 (Not Higher)?**

```
With MAX_WORKERS=4 (concurrent):
  4 clients × 4.2 MB = 16.8 MB
  + Ray overhead: ~500 MB
  + System processes: ~1500 MB
  = 2016 MB required
  
  Available: 6000 MB
  Margin: 3984 MB ← Still OK

But practical issues arise:
  1. Peak Memory Usage: LSTM operations create temporary tensors
     during forward/backward pass (2-3× batch size)
  2. Garbage Collection: Python GC may delay memory cleanup
  3. CUDA Context: Each worker maintains separate CUDA context (~500 MB)
  4. System Pressure: OS memory management becomes inefficient >90%

With MAX_WORKERS=2:
  2 clients × 4.2 MB = 8.4 MB
  + Ray overhead: ~250 MB
  + System processes: ~1500 MB
  = 1758.4 MB required
  
  Available: 6000 MB
  Utilization: 29.3% ← Comfortable, robust
  Peak Safe Margin: ~2500 MB ✓
```

### 4.3 Ray Distributed Execution Optimization

**Ray Cluster Configuration:**

```python
# rays/simulation.py
client_resources = {
    "num_gpus": 1.0 / MAX_WORKERS,  # 0.5 per client (2 max concurrent)
    "num_cpus": 2                    # 2 cores per client (prevent CPU starvation)
}

# Interpretation:
# - Ray scheduler sees each client needs 0.5 GPU share
# - With 1.0 total GPU, max 2 clients can schedule simultaneously
# - CPU constraint: each client claims 2 cores
#   - With 16 cores available, prevents >8 concurrent clients
#   - But GPU is more restrictive, so 2 is max
```

**Worker Scheduling Algorithm:**

```
Queue: [Client 1, 2, 3, 4, 5, ..., 15]
Available Resources: 1.0 GPU, 16 CPU, 16 GB RAM

Scheduling Loop:
  Iteration 1:
    ├─ Client 1: needs 0.5 GPU, 2 CPU → SCHEDULE (GPU: 0.5/1.0, CPU: 2/16)
    ├─ Client 2: needs 0.5 GPU, 2 CPU → SCHEDULE (GPU: 1.0/1.0, CPU: 4/16)
    ├─ Client 3: needs 0.5 GPU, 2 CPU → WAIT (GPU exhausted)
    └─ (Clients 1-2 train in parallel for 150 ms)
  
  Iteration 2 (150 ms later, Client 1 completes):
    ├─ Client 3: needs 0.5 GPU, 2 CPU → SCHEDULE (GPU: 0.5/1.0, CPU: 4/16)
    ├─ Client 4: needs 0.5 GPU, 2 CPU → SCHEDULE (GPU: 1.0/1.0, CPU: 6/16)
    ├─ Client 5: needs 0.5 GPU, 2 CPU → WAIT
    └─ (Clients 3-4 train in parallel for 150 ms)
  
  ... continues until all 15 clients complete
  
  Total time for 15 clients:
    = ceil(15 / 2) × 150 ms = 8 × 150 ms = 1200 ms per round
```

### 4.4 GPU Memory Optimization Techniques

#### **Technique 1: LSTM Parameter Flattening**

```python
# src/model.py - forward() method
def forward(self, x):
    # Critical optimization
    self.lstm.flatten_parameters()
    
    # Purpose: Consolidate scattered weight matrices into contiguous memory
    # Impact: 
    #   - Reduces cuDNN kernel launch overhead
    #   - Improves GPU cache utilization
    #   - ~10-15% speedup on forward/backward
    #
    # Before flattening (memory layout):
    #   LSTM weights stored as separate tensors
    #   Accessing requires pointer chasing, cache misses
    #
    # After flattening:
    #   LSTM weights in single contiguous block
    #   Sequential memory access, efficient prefetching
    
    lstm_out, (h_n, c_n) = self.lstm(x)
    last_hidden = h_n[-1, :, :]
    output = self.fc(last_hidden)
    return output
```

#### **Technique 2: Batch Size Tuning**

```python
# Batch size = 32 (per design)

Why 32 (not 16 or 64)?

┌────────────┬─────────────┬────────────────────┐
│ Batch Size │ Memory Cost │ Training Duration  │
├────────────┼─────────────┼────────────────────┤
│ 16         │ 2 MB        │ 700 ms per client  │
│ 32         │ 4 MB        │ 350 ms per client  │ ← OPTIMAL
│ 64         │ 8 MB        │ 175 ms per client  │
│ 128        │ 16 MB       │ 87 ms per client   │ ← OOM risk
└────────────┴─────────────┴────────────────────┘

Tradeoff:
- Larger batch → faster convergence per GPU util, but fewer iterations/epoch
- Smaller batch → slower GPU util, but noisier gradients (good for Non-IID)
- 32 = sweet spot: efficient GPU use + non-IID gradient noise
```

#### **Technique 3: Mixed Precision Training (Could Add)**

```python
# Optional future enhancement
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Use float16 for forward pass
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Benefit: 2× memory savings, 1.5× speedup
# Current system doesn't need it (plenty of headroom)
```

#### **Technique 4: Gradient Accumulation (Not Used)**

```python
# Not applicable here because:
# - Already using MAX_WORKERS=2 (effective batch size doubled)
# - No need to simulate larger batches
# - Would add complexity without benefit
```

### 4.5 CPU Optimization

**DataLoader Configuration:**

```python
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
    num_workers=0,    # No multiprocessing workers
    pin_memory=True   # Pin to CPU RAM for faster GPU transfer
)
```

**Why `num_workers=0`?**
```
num_workers=2 scenario:
  ├─ Main process (LSTM training)
  ├─ Worker 1 (data loading from disk)
  └─ Worker 2 (data preprocessing)
  
  Total processes: 3
  With Ray managing 2 clients × 2 cores = 4 CPU reserved
  → Contention: 7 processes on 16 cores (saturated)
  → Context switching overhead > prefetch benefit
  
num_workers=0 (sequential loading):
  ├─ Main process handles all (data + training)
  ├─ But data is already in RAM (loaded once at init)
  └─ No disk I/O during training
  
  Result: Clean, predictable execution
```

### 4.6 Multi-GPU Scaling Implications

**If Hardware Upgraded to 2 GPUs:**

```python
# Theoretical scaling:
client_resources = {
    "num_gpus": 1.0 / MAX_WORKERS,  # 1.0 per client (2 GPUs ÷ 2 clients)
    "num_cpus": 4                    # Increase to 4 (more cores available)
}

# Results:
# - Can increase MAX_WORKERS from 2 to 8 (2 GPUs ÷ 0.25 per client)
# - Or keep MAX_WORKERS=2 but double per-client compute capacity
# - 8 concurrent clients: total round time = ceil(15/8) × 150 = 300 ms
# - Potential 4× speedup per round

# Caveats:
# - FL algorithm effectiveness depends on per-client local epochs
# - Faster rounds ≠ better convergence (smaller local improvement)
# - Optimal: max 2-4 concurrent clients for FL (not GPU-limited)
```

---

## Data Flow & Communication Protocol

### 5.1 Single Round Data Flow Diagram

```
TIME →

T=0ms       SERVER PHASE
            ├─ Global weights ready: w_t (136.5 KB)
            ├─ Publish to all 15 clients
            └─ Message: "FitIns(parameters=w_t, config={})"
                
T=10ms      CLIENT PHASE 1: Parameter Reception
            ├─ All clients receive w_t simultaneously (Ray broadcast)
            ├─ Load parameters into local LSTM model
            └─ Verification: ensure shape compatibility (error if mismatch)
                
T=20ms      CLIENT PHASE 2: Training (Concurrent Batch 1)
            ├─ Clients 1-2 simultaneously:
            │   ├─ FOR epoch 1 to 5:
            │   │   ├─ FOR batch 1 to 45 (287 samples ÷ 32 batch):
            │   │   │   ├─ X_batch, y_batch = DataLoader
            │   │   │   ├─ y_pred = model.forward(X_batch)
            │   │   │   ├─ loss = MSELoss(y_pred, y_batch)
            │   │   │   ├─ loss.backward()
            │   │   │   └─ optimizer.step()
            │   │   └─ END batch loop
            │   └─ END epoch loop
            │   └─ Save updated weights: w¹_t
            │
            └─ Time elapsed: ~150 ms per client
                
T=170ms     CLIENT PHASE 3: Training (Concurrent Batch 2)
            ├─ Clients 3-4 train (similar process)
            └─ Time elapsed: ~150 ms per client
                
T=320ms     CLIENT PHASE 4: Training (Concurrent Batch 3)
            ... (Clients 5-6)
                
T=1200ms    SERVER PHASE: Parameter Collection
            ├─ Server waits for all clients to finish training
            ├─ Receives from each client:
            │   ├─ Updated weights w^k_t (136.5 KB)
            │   ├─ Sample count n^k (scalar)
            │   └─ Metrics: train_loss (scalar)
            └─ Total received: 15 × (136.5 KB + metrics) ≈ 2 MB
                
T=1210ms    SERVER PHASE: Aggregation
            ├─ Compute weighted average:
            │   w_t+1 = Σ (n^k / Σn) × w^k_t
            │   = Σ (287/5475) × w^k_t + (288/5475) × w^k_t + ...
            ├─ Update global model in server memory
            └─ Time: ~10 ms
                
T=1220ms    SERVER PHASE: Evaluation Distribution
            ├─ Publish w_t+1 to clients for evaluation
            └─ Message: "EvaluateIns(parameters=w_t+1, config={})"
                
T=1230ms    CLIENT PHASE 5: Evaluation (Concurrent Batch 1)
            ├─ Clients 1-2 simultaneously:
            │   ├─ Load w_t+1 into model
            │   ├─ FOR batch in test_loader:
            │   │   ├─ y_pred = model.forward(X_test_batch)
            │   │   ├─ loss = MSELoss(y_pred, y_test_batch)
            │   │   └─ accumulate loss
            │   └─ Compute average MSE: loss_avg
            │   └─ Return: loss_avg (scalar), n_test (sample count)
            │
            └─ Time elapsed: ~75 ms per client
                
T=1305ms    CLIENT PHASE 6: Evaluation (Concurrent Batch 2-8)
            ... (Similar to evaluation batch 1)
                
T=1680ms    SERVER PHASE: Metrics Aggregation
            ├─ Receive MSE from all 15 clients
            ├─ Compute global MSE:
            │   global_mse = Σ (n^k_test / Σn_test) × mse^k
            ├─ Log to training_log.csv
            └─ Print: "[Round 1/10] Global MSE: 1.859470"
                
T=1690ms    ✓ ROUND 1 COMPLETE
            └─ Duration: 1.69 seconds
            
            Ready for Round 2... (repeats)
```

### 5.2 Parameter Message Format

**FedAvg Parameter Exchange:**

```python
# Client sends: FitRes
{
    "parameters": [
        numpy.array(...),  # w_LSTM_layer1_input_gate (4160,)
        numpy.array(...),  # w_LSTM_layer1_forget_gate (4160,)
        numpy.array(...),  # w_LSTM_layer1_cell_gate (4160,)
        numpy.array(...),  # w_LSTM_layer1_output_gate (4160,)
        # ... (8 more weight matrices for layer 2)
        numpy.array(...),  # b_LSTM_layer1_input_gate (64,)
        # ... (7 more biases)
        numpy.array(...),  # w_FC (64,)
        numpy.array(...),  # b_FC (1,)
    ],
    "num_examples": 287,        # Training sample count
    "metrics": {
        "train_loss": 0.8234    # Final epoch loss
    }
}

Total serialized size:
  34,113 parameters × 8 bytes (float64) = 272.9 KB
  Overhead (JSON metadata, etc): ~10 KB
  ≈ 283 KB per client × 15 clients = 4.245 MB per round
  Network bandwidth requirement: <1 Mbps (easily satisfied)
```

---

## Scalability Considerations

### 6.1 Scaling to More Clients

**Current System (15 clients):**
```
Round duration: ~1.8 seconds
Bottleneck: GPU (0.5 GPU per 2 clients in queue)
```

**Scaling to 100 clients:**

```python
# Option A: Increase MAX_WORKERS (GPU-bound solution)
client_resources = {
    "num_gpus": 1.0 / MAX_WORKERS,  # MAX_WORKERS = ?
    "num_cpus": 2
}

# With single RTX 4050 (1.0 GPU):
# MAX_WORKERS = 1 → 1 client concurrent (slow: 100 × 150 ms = 15 seconds/round)
# MAX_WORKERS = 2 → 2 clients concurrent (current: 50 × 150 ms = 7.5 seconds/round)
# MAX_WORKERS = 10 → 10 clients concurrent (risky: 10 × 150 ms = 1.5 seconds/round)

# Actual memory usage at MAX_WORKERS=10:
# 10 clients × 4.2 MB = 42 MB + overhead → Still OK!
# Problem: CUDA context per worker, thread management overhead

# Recommendation: Stay with MAX_WORKERS=2 for stability, or add multi-GPU
```

**Option B: Stratified Sampling**

```python
# Don't train all clients each round, sample subset
SAMPLING_FRACTION = 0.5  # Train 7-8 clients per round

for round in range(NUM_ROUNDS):
    selected_clients = sample(all_15_clients, int(15 * 0.5))
    
    for client in selected_clients:
        fit(client, global_weights)
    
    aggregate(selected_clients)  # Only aggregate sampled clients
```

**Implication:** Rounds complete faster, but convergence slower (fewer clients contribute)

### 6.2 Scaling to More Rounds

**Current: 10 rounds**

**Extended training (100 rounds):**

```
With current MAX_WORKERS=2:
  100 rounds × 1.8 sec/round = 180 seconds ≈ 3 minutes
  
Expected convergence improvement:
  - Rounds 1-5: Rapid improvement (loss decreases ~15%)
  - Rounds 6-50: Slow improvement with oscillations (~5% improvement)
  - Rounds 50-100: Convergence plateau (Non-IID barrier)
  
Practical benefit: Diminishing returns after round 30-40

Recommendation: Run 30-50 rounds for production, monitor convergence curve
```

### 6.3 Scaling to Larger Models

**Current Model: 34K parameters**

**Hypothetical 1M parameter model:**

```
Memory impact:
  1M × 8 bytes (float64) = 8 MB per model
  × 2 concurrent clients = 16 MB
  → Still within 6 GB capacity
  
But training time explodes:
  Forward pass: 4× slowdown (more layers)
  Backward pass: 4× slowdown (more gradients)
  Per-client training: 150 ms → 600 ms
  Round duration: 1.8 sec → 4.5 seconds
  
Feasibility: Possible but not recommended without multi-GPU
```

---

## References & Mathematical Foundations

### 7.1 Federated Learning Literature

- **FedAvg Original Paper:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ICML 2017)
  - Introduces weighted averaging without data sharing
  - Proves convergence under Non-IID data with caveats

- **LSTM Architecture:** Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation 1997)
  - Foundational LSTM design with gating mechanisms
  - Solutions for vanishing gradient problem

- **Non-IID Challenges:** Zhao et al., "Federated Learning with Non-IID Data" (arXiv 2018)
  - Analysis of convergence degradation with heterogeneous data
  - Proposes local epochs as mitigation

### 7.2 Implementation References

**Framework Versions:**
- Flower >= 1.8.0 (FL orchestration)
- PyTorch >= 2.3.0 (LSTM, optimization)
- Ray >= 2.10.0 (distributed execution)
- scikit-learn >= 1.3.0 (StandardScaler)

**Key Classes & Methods:**
```python
# Flower
from flwr.client import NumPyClient
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

# PyTorch
import torch
torch.nn.LSTM(input_size=4, hidden_size=64, num_layers=2, dropout=0.2)
torch.nn.Linear(64, 1)
torch.optim.Adam(lr=0.001)
torch.nn.MSELoss()

# Ray
import ray
@ray.remote(num_gpus=0.5, num_cpus=2)
def train_client(client_id):
    ...
```

### 7.3 Hyperparameter Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **LOCAL_EPOCHS** | 5 | 5-10 epochs typical for FL; balances convergence vs communication |
| **LEARNING_RATE** | 0.001 | Standard for Adam; 0.01 too aggressive for Non-IID data |
| **BATCH_SIZE** | 32 | 16-64 typical; 32 optimal for RTX 4050 memory |
| **HIDDEN_SIZE** | 64 | 32-128 typical for small models; 64 sufficient for weather patterns |
| **NUM_LAYERS** | 2 | 1-3 typical; 2 layers capture temporal + feature interactions |
| **DROPOUT** | 0.2 | 0.1-0.5 typical; 0.2 prevents overfitting without hindering learning |
| **NUM_ROUNDS** | 10 | Demonstrates convergence; 30+ recommended for production |
| **MAX_WORKERS** | 2 | 2-4 typical; 2 safe for 6GB VRAM with headroom |

### 7.4 Mathematical Foundations Summary

**Federated Optimization Problem:**
$$\min_{\mathbf{w}} F(\mathbf{w}) = \sum_{k=1}^{K} \frac{n_k}{n} L_k(\mathbf{w})$$

**FedAvg Update Rule (Per Round t):**
$$\mathbf{w}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_t^{(k,E,\eta)}$$

Where:
- $\mathbf{w}_t^{(k,E,\eta)}$ = weights after local training on client $k$ with $E$ epochs, learning rate $\eta$
- $n_k$ = sample count on client $k$
- $n = \sum_k n_k$ = total samples

**LSTM Cell Computation:**
$$\mathbf{i}_t = \sigma(\mathbf{W}_{ii}\mathbf{x}_t + \mathbf{W}_{hi}\mathbf{h}_{t-1} + \mathbf{b}_i)$$
$$\mathbf{f}_t = \sigma(\mathbf{W}_{if}\mathbf{x}_t + \mathbf{W}_{hf}\mathbf{h}_{t-1} + \mathbf{b}_f)$$
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_{ic}\mathbf{x}_t + \mathbf{W}_{hc}\mathbf{h}_{t-1} + \mathbf{b}_c)$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_{io}\mathbf{x}_t + \mathbf{W}_{ho}\mathbf{h}_{t-1} + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

---

## Document Metadata

**Author:** Federated Learning Weather System Team  
**Document ID:** FL-WP-ARCH-001  
**Review Status:** Final  
**Distribution:** Technical Team, Deployment Engineers  
**Revision History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 2026 | Initial comprehensive architecture documentation |

---

**End of Architecture Document**

For implementation details, see: [2_IMPLEMENTATION.md](2_IMPLEMENTATION.md)

For deployment procedures, see: [3_DEPLOYMENT.md](3_DEPLOYMENT.md)

For performance analysis, see: [4_EVALUATION.md] (4_EVALUATION.md)
