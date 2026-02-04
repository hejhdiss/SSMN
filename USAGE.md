# SSMN Usage Guide

Detailed examples and use cases for both Text-Native SSMN and Standard SSMN.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Text-Native SSMN Examples](#text-native-ssmn-examples)
3. [Standard SSMN Examples](#standard-ssmn-examples)
4. [Advanced Techniques](#advanced-techniques)
5. [Real-World Applications](#real-world-applications)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Prerequisites

```bash
# Required: GCC compiler
# Linux/Mac: usually pre-installed
# Windows: Install MinGW or use WSL

# Python dependencies
pip install numpy
```

### Compilation

```bash
# Navigate to the directory containing the .c files

# For Text-Native SSMN
gcc -shared -fPIC -o text_native_ssmn.so text_native_ssmn.c -lm -O3

# For Standard SSMN
gcc -shared -fPIC -o ssmn.so ssmn.c -lm -O3

# On Mac, replace .so with .dylib
# On Windows, replace .so with .dll and remove -fPIC
```

### Verify Installation

```python
# Test Text-Native SSMN
from text_native_ssmn import TextNativeSSMN
net = TextNativeSSMN(vocab_size=100, hidden_dim=64)
print("✓ Text-Native SSMN loaded successfully!")

# Test Standard SSMN
from ssmn import SparseStreamMemoryNetwork
net = SparseStreamMemoryNetwork(input_dim=32, hidden_dim=64, output_dim=16)
print("✓ Standard SSMN loaded successfully!")
```

---

## Text-Native SSMN Examples

### Example 1: Basic Token Processing

```python
from text_native_ssmn import TextNativeSSMN
import numpy as np

# Create network for a small vocabulary
net = TextNativeSSMN(
    vocab_size=1000,      # 1000 unique tokens
    embed_dim=64,         # 64-dim semantic embeddings
    hidden_dim=128,       # 128-dim hidden state
    window_size=32,       # Look back 32 tokens
    plasticity_eta=0.01,  # Learning rate for synaptic updates
    decay_lambda=0.001    # Decay rate
)

# Print network info
net.print_info()

# Process a sequence of tokens
tokens = [42, 17, 89, 3, 156, 91, 12]

for i, token in enumerate(tokens):
    probs = net.forward(token)
    
    # Get top-5 predicted next tokens
    top_5 = np.argsort(probs)[-5:][::-1]
    
    print(f"Token {token:3d} → Top-5 predictions: {top_5.tolist()}")
    print(f"  Synaptic Energy: {net.synaptic_energy:.6f}")
    print(f"  Importance: {net.avg_importance:.4f}")
```

### Example 2: Text Generation

```python
from text_native_ssmn import TextNativeSSMN
import numpy as np

# Create network
net = TextNativeSSMN(
    vocab_size=5000,
    embed_dim=128,
    hidden_dim=256,
    window_size=64
)

# Define prompt
prompt_tokens = [1, 42, 17, 89]  # Start tokens

# Generate continuation
generated = net.generate_sequence(
    prompt_tokens=prompt_tokens,
    max_length=50,      # Generate 50 total tokens
    temperature=0.8,    # Lower = more deterministic
    top_k=40           # Sample from top-40 tokens
)

print(f"Generated sequence ({len(generated)} tokens):")
print(generated)

# Analyze what the network learned
analysis = net.analyze_synaptic_state()
print("\nSynaptic State Analysis:")
for key, value in analysis.items():
    print(f"  {key}: {value}")
```

### Example 3: Semantic Compression Monitoring

```python
from text_native_ssmn import TextNativeSSMN
import numpy as np
import matplotlib.pyplot as plt

net = TextNativeSSMN(vocab_size=1000, hidden_dim=128, window_size=32)

# Track metrics over time
metrics = {
    'energy': [],
    'importance': [],
    'entropy': []
}

# Process a long sequence
sequence = np.random.randint(0, 1000, size=200)

for token in sequence:
    _ = net.forward(token)
    
    metrics['energy'].append(net.synaptic_energy)
    metrics['importance'].append(net.avg_importance)
    metrics['entropy'].append(net.attention_entropy)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(metrics['energy'])
axes[0].set_ylabel('Synaptic Energy')
axes[0].set_title('Memory Accumulation Over Time')

axes[1].plot(metrics['importance'])
axes[1].set_ylabel('Importance Gate')
axes[1].set_title('Semantic Anchor Detection')

axes[2].plot(metrics['entropy'])
axes[2].set_ylabel('Attention Entropy')
axes[2].set_xlabel('Token Position')
axes[2].set_title('Attention Distribution Spread')

plt.tight_layout()
plt.savefig('ssmn_metrics.png')
print("Saved metrics plot to ssmn_metrics.png")
```

### Example 4: Multi-Document Processing

```python
from text_native_ssmn import TextNativeSSMN

net = TextNativeSSMN(vocab_size=10000, hidden_dim=256, window_size=128)

# Simulate multiple documents
documents = [
    [10, 42, 73, 19, 85],     # Doc 1
    [42, 91, 12, 73, 34],     # Doc 2
    [19, 10, 85, 42, 99]      # Doc 3
]

results = []

for doc_id, doc_tokens in enumerate(documents):
    # Reset memory for each new document
    net.reset()
    
    # Process document
    probs = net.generate(doc_tokens, reset_memory=False)
    
    # Analyze what was stored
    analysis = net.analyze_synaptic_state()
    
    results.append({
        'doc_id': doc_id,
        'energy': analysis['energy'],
        'active_modes': analysis['num_active_modes']
    })
    
    print(f"Document {doc_id}:")
    print(f"  Energy: {analysis['energy']:.6f}")
    print(f"  Active Modes: {analysis['num_active_modes']}")
```

---

## Standard SSMN Examples

### Example 1: Time Series Prediction

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Generate synthetic time series (sine wave)
t = np.linspace(0, 10*np.pi, 1000)
signal = np.sin(t) + 0.1*np.random.randn(1000)

# Create sequences: input = current value, target = next value
X = signal[:-1].reshape(-1, 1).astype(np.float32)
y = signal[1:].reshape(-1, 1).astype(np.float32)

# Split train/test
split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create network
net = SparseStreamMemoryNetwork(
    input_dim=1,
    hidden_dim=64,
    output_dim=1,
    window_size=32,
    plasticity_eta=0.01,
    decay_lambda=0.001
)

# Train
print("Training...")
net.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate
predictions = net.predict(X_test)
mse = np.mean((predictions - y_test)**2)
print(f"\nTest MSE: {mse:.6f}")

# Plot predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(y_test[:100], label='True', alpha=0.7)
plt.plot(predictions[:100], label='Predicted', alpha=0.7)
plt.legend()
plt.title('Time Series Prediction')
plt.savefig('time_series_prediction.png')
```

### Example 2: Multi-Dimensional Regression

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Generate synthetic data: predict multiple outputs from multiple inputs
np.random.seed(42)
n_samples = 1000
input_dim = 20
output_dim = 10

X = np.random.randn(n_samples, input_dim).astype(np.float32)
# Target is a nonlinear function of input
y = np.tanh(X[:, :output_dim]) + 0.1*np.random.randn(n_samples, output_dim).astype(np.float32)

# Split
split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create network
net = SparseStreamMemoryNetwork(
    input_dim=input_dim,
    hidden_dim=128,
    output_dim=output_dim,
    window_size=64
)

net.print_info()

# Train
net.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# Evaluate
predictions = net.predict(X_test)
r2 = net.score(X_test, y_test, metric='r2')
mse = net.score(X_test, y_test, metric='mse')

print(f"\nTest R²: {r2:.4f}")
print(f"Test MSE: {mse:.6f}")

# Analyze memory
analysis = net.analyze_synaptic_state()
print("\nFinal Synaptic State:")
for key, value in analysis.items():
    print(f"  {key}: {value}")
```

### Example 3: Sequential Feature Extraction

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Create network for feature extraction
net = SparseStreamMemoryNetwork(
    input_dim=128,
    hidden_dim=256,
    output_dim=64,
    window_size=32
)

# Process a sequence and extract hidden states
sequence = np.random.randn(100, 128).astype(np.float32)

hidden_states = []
net.reset()

for i, x in enumerate(sequence):
    _ = net.forward(x)
    h = net.get_hidden_state()
    hidden_states.append(h)

hidden_states = np.array(hidden_states)  # Shape: (100, 256)

print(f"Extracted features shape: {hidden_states.shape}")
print(f"Feature statistics:")
print(f"  Mean: {hidden_states.mean():.4f}")
print(f"  Std: {hidden_states.std():.4f}")
print(f"  Min: {hidden_states.min():.4f}")
print(f"  Max: {hidden_states.max():.4f}")

# Use these features for downstream tasks
# e.g., clustering, classification, etc.
```

### Example 4: Adaptive Memory Control

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

net = SparseStreamMemoryNetwork(
    input_dim=64,
    hidden_dim=128,
    output_dim=32,
    window_size=32
)

# Scenario: Different data regimes require different memory behaviors

# Regime 1: Fast-changing data - need high plasticity
print("Regime 1: Fast-changing data")
net.plasticity_eta = 0.05   # High plasticity
net.decay_lambda = 0.005    # High decay
net.reset()

fast_data = np.random.randn(50, 64).astype(np.float32)
_ = net.predict(fast_data)

analysis1 = net.analyze_synaptic_state()
print(f"  Energy: {analysis1['energy']:.6f}")
print(f"  Update magnitude: {analysis1['update_magnitude']:.6f}")

# Regime 2: Stable data - need low plasticity
print("\nRegime 2: Stable data")
net.plasticity_eta = 0.005  # Low plasticity
net.decay_lambda = 0.001    # Low decay
net.reset()

stable_data = np.random.randn(50, 64).astype(np.float32) * 0.1
_ = net.predict(stable_data)

analysis2 = net.analyze_synaptic_state()
print(f"  Energy: {analysis2['energy']:.6f}")
print(f"  Update magnitude: {analysis2['update_magnitude']:.6f}")
```

---

## Advanced Techniques

### 1. Curriculum Learning with Dynamic Plasticity

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=64, output_dim=16, window_size=32
)

# Start with high plasticity for fast learning
initial_eta = 0.05
final_eta = 0.001

# Generate tasks of increasing difficulty
difficulties = [0.1, 0.3, 0.5, 0.8]

for epoch, difficulty in enumerate(difficulties):
    # Gradually decrease plasticity (curriculum learning)
    current_eta = initial_eta - (initial_eta - final_eta) * (epoch / len(difficulties))
    net.plasticity_eta = current_eta
    
    # Generate data with noise proportional to difficulty
    X = np.random.randn(100, 32).astype(np.float32)
    y = X[:, :16] + difficulty * np.random.randn(100, 16).astype(np.float32)
    
    net.fit(X, y, epochs=10, verbose=0)
    
    print(f"Difficulty {difficulty:.1f}, η={current_eta:.4f}: "
          f"Energy={net.synaptic_energy:.6f}")
```

### 2. Multi-Task Learning

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Shared network for multiple tasks
net = SparseStreamMemoryNetwork(
    input_dim=64, hidden_dim=128, output_dim=32, window_size=32
)

tasks = {
    'task_A': (np.random.randn(100, 64).astype(np.float32), 
               np.random.randn(100, 32).astype(np.float32)),
    'task_B': (np.random.randn(100, 64).astype(np.float32), 
               np.random.randn(100, 32).astype(np.float32))
}

# Train on interleaved tasks
for epoch in range(20):
    for task_name, (X, y) in tasks.items():
        # Reset synaptic memory between tasks (optional)
        # net.reset_synaptic_memory()
        
        net.fit(X, y, epochs=1, verbose=0)
        
        if epoch % 5 == 0:
            score = net.score(X, y, metric='mse')
            print(f"Epoch {epoch}, {task_name}: MSE={score:.6f}")
```

### 3. Memory Visualization

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np
import matplotlib.pyplot as plt

net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=64, output_dim=16, window_size=16
)

# Process some data
X = np.random.randn(100, 32).astype(np.float32)
_ = net.predict(X)

# Get synaptic weights
W_f = net.get_synaptic_weights()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Heatmap
im = axes[0].imshow(W_f, cmap='RdBu', vmin=-1, vmax=1)
axes[0].set_title('Synaptic Weight Matrix')
plt.colorbar(im, ax=axes[0])

# Eigenvalue spectrum
eigenvalues = np.linalg.eigvalsh(W_f)
axes[1].bar(range(len(eigenvalues)), sorted(np.abs(eigenvalues), reverse=True))
axes[1].set_title('Eigenvalue Magnitude')
axes[1].set_xlabel('Mode')
axes[1].set_ylabel('|Eigenvalue|')

# Weight distribution
axes[2].hist(W_f.flatten(), bins=50)
axes[2].set_title('Weight Distribution')
axes[2].set_xlabel('Weight Value')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('synaptic_analysis.png')
print("Saved visualization to synaptic_analysis.png")
```

---

## Real-World Applications

### 1. Sentiment Analysis (Text-Native SSMN)

```python
from text_native_ssmn import TextNativeSSMN
import numpy as np

# Assume we have a tokenizer
# tokenizer = YourTokenizer()

# Create network
net = TextNativeSSMN(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    window_size=64
)

# Example: Process movie reviews
reviews = [
    "this movie was amazing and thrilling",
    "terrible waste of time and money",
    "pretty good but could be better"
]

# In practice, you'd:
# 1. Tokenize reviews
# 2. Process with SSMN
# 3. Use hidden state for classification

for review in reviews:
    # tokens = tokenizer.encode(review)
    tokens = [np.random.randint(0, 10000) for _ in range(10)]  # Mock
    
    _ = net.generate(tokens)
    
    # Extract representation
    hidden = net.get_hidden_state()
    analysis = net.analyze_synaptic_state()
    
    print(f"Review: '{review[:30]}...'")
    print(f"  Synaptic Energy: {analysis['energy']:.6f}")
    print(f"  Active Modes: {analysis['num_active_modes']}")
```

### 2. Anomaly Detection (Standard SSMN)

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Train on normal data
normal_data = np.random.randn(1000, 32).astype(np.float32)

net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=64, output_dim=32, window_size=32
)

# Train to reconstruct normal patterns
net.fit(normal_data, normal_data, epochs=20, verbose=1)

# Test on normal and anomalous data
test_normal = np.random.randn(100, 32).astype(np.float32)
test_anomaly = 5 * np.random.randn(100, 32).astype(np.float32)  # Different distribution

normal_error = net.score(test_normal, test_normal, metric='mse')
anomaly_error = net.score(test_anomaly, test_anomaly, metric='mse')

print(f"\nReconstruction Error:")
print(f"  Normal data: {normal_error:.6f}")
print(f"  Anomalous data: {anomaly_error:.6f}")
print(f"  Ratio: {anomaly_error/normal_error:.2f}x")
```

### 3. Control Systems (Standard SSMN)

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Simulate a simple control task
# State: [position, velocity], Action: [force]

def simulate_physics(state, action, dt=0.1):
    """Simple 1D physics simulation"""
    pos, vel = state
    force = action[0]
    
    # F = ma, m = 1
    acc = force
    vel_new = vel + acc * dt
    pos_new = pos + vel_new * dt
    
    return np.array([pos_new, vel_new])

# Create controller network
net = SparseStreamMemoryNetwork(
    input_dim=2,   # [position, velocity]
    hidden_dim=32,
    output_dim=1,  # [force]
    window_size=16
)

# Training data: collect trajectories
states = []
actions = []

state = np.array([0.0, 0.0])
target_pos = 1.0

for step in range(100):
    # Random exploration
    action = np.random.randn(1) * 0.1
    
    states.append(state)
    actions.append(action)
    
    state = simulate_physics(state, action)

states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

# Train controller
net.fit(states, actions, epochs=50, verbose=1)

# Test controller
print("\nTesting controller...")
state = np.array([0.0, 0.0], dtype=np.float32)
for step in range(20):
    action = net.forward(state)
    state = simulate_physics(state, action)
    print(f"Step {step:2d}: pos={state[0]:.3f}, vel={state[1]:.3f}, force={action[0]:.3f}")
```

---

## Performance Optimization

### 1. Batch Processing

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np
import time

net = SparseStreamMemoryNetwork(
    input_dim=64, hidden_dim=128, output_dim=32, window_size=32
)

# Process data in batches for efficiency
data = np.random.randn(10000, 64).astype(np.float32)

# Method 1: One-by-one (slow)
start = time.time()
net.reset()
for x in data[:1000]:
    _ = net.forward(x)
time_sequential = time.time() - start

# Method 2: Using predict (faster - fewer Python calls)
start = time.time()
_ = net.predict(data[:1000])
time_batch = time.time() - start

print(f"Sequential: {time_sequential:.3f}s")
print(f"Batch: {time_batch:.3f}s")
print(f"Speedup: {time_sequential/time_batch:.2f}x")
```

### 2. Memory Management

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=64, output_dim=16, window_size=32
)

# For very long sequences, periodically reset window
# This prevents the window buffer from affecting cache performance

long_sequence = np.random.randn(100000, 32).astype(np.float32)
chunk_size = 1000

for i in range(0, len(long_sequence), chunk_size):
    chunk = long_sequence[i:i+chunk_size]
    
    # Process chunk
    _ = net.predict(chunk, reset_memory=False)
    
    # Optionally: check if synaptic memory is becoming too large
    if net.synaptic_energy > 10.0:
        print(f"Warning: High synaptic energy at step {i}")
        # Could reduce plasticity or increase decay
        net.plasticity_eta *= 0.9
```

### 3. Hyperparameter Tuning

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Generate validation data
X_val = np.random.randn(200, 32).astype(np.float32)
y_val = np.random.randn(200, 16).astype(np.float32)

# Grid search over plasticity parameters
best_score = float('inf')
best_params = None

for eta in [0.001, 0.005, 0.01, 0.05]:
    for lam in [0.0001, 0.001, 0.01]:
        net = SparseStreamMemoryNetwork(
            input_dim=32, hidden_dim=64, output_dim=16, window_size=32,
            plasticity_eta=eta, decay_lambda=lam
        )
        
        # Quick training
        X_train = np.random.randn(500, 32).astype(np.float32)
        y_train = np.random.randn(500, 16).astype(np.float32)
        net.fit(X_train, y_train, epochs=10, verbose=0)
        
        # Evaluate
        score = net.score(X_val, y_val, metric='mse')
        
        if score < best_score:
            best_score = score
            best_params = (eta, lam)
        
        print(f"η={eta:.4f}, λ={lam:.4f}: MSE={score:.6f}")

print(f"\nBest params: η={best_params[0]:.4f}, λ={best_params[1]:.4f}")
print(f"Best MSE: {best_score:.6f}")
```

---

## Troubleshooting

### Issue 1: Synaptic Energy Exploding

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=64, output_dim=16, window_size=32,
    plasticity_eta=0.1,  # Too high!
    decay_lambda=0.0001  # Too low!
)

X = np.random.randn(100, 32).astype(np.float32)

for i in range(10):
    _ = net.forward(X[i])
    
    if net.synaptic_energy > 100:
        print(f"Warning: Energy exploding at step {i}!")
        print(f"  Current energy: {net.synaptic_energy:.2f}")
        print(f"  Solution: Reduce plasticity_eta or increase decay_lambda")
        
        # Fix: Reduce plasticity
        net.plasticity_eta = 0.01
        net.decay_lambda = 0.001
        print(f"  Adjusted: η={net.plasticity_eta}, λ={net.decay_lambda}")
        break
```

### Issue 2: Poor Predictions

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Check various potential issues

# 1. Input normalization
X = np.random.randn(100, 32).astype(np.float32)
print(f"Input statistics:")
print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")
print(f"  Min: {X.min():.4f}, Max: {X.max():.4f}")

# If not normalized, normalize:
X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# 2. Check network capacity
net = SparseStreamMemoryNetwork(
    input_dim=32, hidden_dim=128,  # Try increasing hidden_dim
    output_dim=16, window_size=64   # Try increasing window_size
)

# 3. Monitor training
y = np.random.randn(100, 16).astype(np.float32)
net.fit(X_normalized, y, epochs=50, verbose=1)

# 4. Check for NaN/Inf
predictions = net.predict(X_normalized[:10])
if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
    print("ERROR: NaN or Inf in predictions!")
    print("  Check plasticity parameters")
else:
    print("✓ Predictions are valid")
```

---

## Summary

This guide covered:
- ✅ Installation and setup
- ✅ Basic usage for both architectures
- ✅ Advanced techniques (curriculum learning, multi-task, visualization)
- ✅ Real-world applications (sentiment analysis, anomaly detection, control)
- ✅ Performance optimization
- ✅ Common troubleshooting scenarios

For more details, see the README.md or run the demo scripts:
```bash
python text_native_ssmn.py  # Text-Native SSMN demos
python ssmn.py              # Standard SSMN demos
```
