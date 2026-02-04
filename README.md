# Sparse-Stream Memory Networks (SSMN)

> Member of [MNNN](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK) Family 


**Revolutionary neural architectures that replace expensive global attention with "continuous ink" of synaptic weights.**

This repository contains two implementations:
1. **Text-Native SSMN**: Language and Memory are unified - stores geometric relationships between concepts
2. **Standard SSMN**: Sliding window attention + neural synaptic memory

Both architectures achieve **O(n·w) complexity** instead of O(n²) for transformers, with **no global KV cache required**.

---

## 🏗️ Architecture Overview

### Core Innovation: The MN (Memory-Native) Layer

Instead of searching through past tokens with attention, information flows out of the sliding window and gets **compressed into synaptic weights** that update during the forward pass:

```
ΔW_f = η(h_t ⊗ h_{t-1}) - λW_f
```

- **η (Plasticity)**: Absorbs current context into weights
- **λ (Decay)**: Prunes old/irrelevant information, prevents bloat

### Brain-Inspired Design

```
📊 Layer Distribution:
├─ 80% Static Layers ──────► Grammar, basic logic (cortex)
└─ 20% Plastic Layers ─────► Memory hubs (hippocampus)
```

---

## 📦 What's Included

```
.
├── text_native_ssmn.c      # Text-Native SSMN C library
├── text_native_ssmn.py     # Text-Native SSMN Python wrapper
├── ssmn.c                  # Standard SSMN C library
├── ssmn.py                 # Standard SSMN Python wrapper
├── README.md               # This file
└── USAGE.md                # Detailed usage examples
```

---

## 🚀 Quick Start

### 1. Compile C Libraries

**Linux/Mac:**
```bash
# Text-Native SSMN
gcc -shared -fPIC -o text_native_ssmn.so text_native_ssmn.c -lm -O3

# Standard SSMN
gcc -shared -fPIC -o ssmn.so ssmn.c -lm -O3

# Custom SSMN
gcc -shared -fPIC -o ssmn_custom.so ssmn_custom.c -lm -O3
```

**Windows:**
```bash
# Text-Native SSMN
gcc -shared -o text_native_ssmn.dll text_native_ssmn.c -lm -O3

# Standard SSMN
gcc -shared -o ssmn.dll ssmn.c -lm -O3

# Custom SSMN
gcc -shared -o ssmn_custom.dll ssmn_custom.c -lm -O3
```

**Mac (with Homebrew GCC):**
```bash
# Text-Native SSMN
gcc -shared -fPIC -o text_native_ssmn.dylib text_native_ssmn.c -lm -O3

# Standard SSMN
gcc -shared -fPIC -o ssmn.dylib ssmn.c -lm -O3

# Custom SSMN
gcc -shared -fPIC -o ssmn_custom.so ssmn_custom.c -lm -O3
```

### 2. Run Python Demos

**Text-Native SSMN:**
```bash
python text_native_ssmn.py
```

**Standard SSMN:**
```bash
python ssmn.py
```

---

## 🎯 Text-Native SSMN

### Key Features

✨ **Neural Semantic Encoder**: Converts tokens → "thought embeddings" capturing intent  
🪟 **Sliding Window Attention**: O(n·w) local context  
🧠 **Semantic Anchors**: Only updates important synaptic connections  
💬 **Internal Recurrent Chat**: Model re-reads its own synaptic state  
🔗 **Unified Memory**: Language IS memory - geometric concept relationships  

### Architecture Flow

```
Token ID
   ↓
[Neural Semantic Encoder] ──► Thought Embedding
   ↓
[Sliding Window Attention] ──► Local Context
   ↓
[Static Layers 80%] ────────► Grammar & Logic
   ↓
[Plastic Layers 20%] ───────► Memory Hubs
   │
   ├─► Update: ΔW_f = Gate(Importance)·[η(h⊗h_prev) - λW_f]
   │
   └─► Internal Chat: Re-read synaptic state
   ↓
Output Probabilities
```

### Python Example

```python
from text_native_ssmn import TextNativeSSMN

# Create network
net = TextNativeSSMN(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    window_size=512,
    plasticity_eta=0.01,
    decay_lambda=0.001
)

# Process sequence
tokens = [42, 123, 7, 891, 34]
probs = net.generate(tokens)

# Generate text
generated = net.generate_sequence(
    prompt_tokens=[1, 2, 3],
    max_length=100,
    temperature=0.8
)

# Analyze synaptic state
analysis = net.analyze_synaptic_state()
print(f"Synaptic Energy: {analysis['energy']:.6f}")
print(f"Active Modes: {analysis['num_active_modes']}")
```

---

## 🎯 Standard SSMN

### Key Features

👁️ **Sliding Window Attention**: O(n·w) local attention (The Eyes)  
🧠 **Neural Synaptic Memory**: Fast-weight matrix W_f (The Brain)  
🔄 **Decaying Latent Blocks**: Hybrid attention + synaptic cells  
⚡ **Linear Complexity**: Process infinite sequences  
🎚️ **Tunable Plasticity**: Adjust η and λ for different memory behaviors  

### Architecture Flow

```
Input Vector
   ↓
[Input Projection]
   ↓
[Sliding Window Attention] ──► O(n·w) local context
   ↓
[Static Layers 80%] ────────► Grammar & Logic
   ↓
[Plastic Layers 20%] ───────► Memory Hubs
   │
   ├─► Synaptic Update: ΔW_f = η(h⊗h_prev) - λW_f
   │
   └─► Apply W_f: output += W_f·h
   ↓
[Output Projection]
   ↓
Output Vector
```

### Python Example

```python
from ssmn import SparseStreamMemoryNetwork
import numpy as np

# Create network
net = SparseStreamMemoryNetwork(
    input_dim=128,
    hidden_dim=256,
    output_dim=64,
    window_size=512,
    plasticity_eta=0.01,
    decay_lambda=0.001
)

# Train on sequential data
X_train = np.random.randn(1000, 128).astype(np.float32)
y_train = np.random.randn(1000, 64).astype(np.float32)

net.fit(X_train, y_train, epochs=20, verbose=1)

# Make predictions
X_test = np.random.randn(100, 128).astype(np.float32)
predictions = net.predict(X_test)

# Analyze memory
analysis = net.analyze_synaptic_state()
print(f"Synaptic Energy: {analysis['energy']:.6f}")
print(f"Spectral Radius: {analysis['spectral_radius']:.4f}")
```

---

## 🔬 Key Concepts

### 1. Sliding Window Attention

Instead of every token attending to every other token (O(n²)), each token only attends to its **local neighborhood** (O(n·w)):

```python
# Global Attention (Transformer)
complexity = O(n²)  # Quadratic!

# Sliding Window Attention (SSMN)
complexity = O(n·w)  # Linear! (w is constant)
```

### 2. Synaptic Memory Layer

Information that "falls off" the sliding window doesn't disappear - it gets **compressed into synaptic weights**:

```python
# At each step:
ΔW_f = η·(h_current ⊗ h_previous) - λ·W_f

# η controls how fast new info is absorbed
# λ controls how fast old info decays
```

### 3. The 80/20 Split

Like the brain (cortex vs hippocampus):

- **80% Static Layers**: Handle grammar, basic logic - don't change during inference
- **20% Plastic Layers**: Memory hubs that adapt via synaptic updates

### 4. Text-Native Design (Text-Native SSMN only)

Traditional LLMs: `Token → Embedding → Processing`

Text-Native SSMN: `Token → Thought Embedding (captures intent) → Semantic Processing`

The model doesn't store words - it stores **geometric relationships between concepts**.

---

## 📊 Performance Characteristics

### Memory Usage

| Architecture | Per-Token Memory | Total Memory |
|--------------|------------------|--------------|
| Transformer  | O(n)             | O(n²)        |
| SSMN         | O(1)             | O(n)         |

### Computational Complexity

| Operation          | Transformer | SSMN    |
|--------------------|-------------|---------|
| Attention          | O(n²)       | O(n·w)  |
| Synaptic Update    | -           | O(d²)   |
| Total per token    | O(n²)       | O(n·w)  |

Where:
- `n` = sequence length
- `w` = window size (constant, e.g., 512)
- `d` = hidden dimension

---

## 🎛️ Hyperparameters

### Text-Native SSMN

```python
vocab_size       # Vocabulary size
embed_dim        # Semantic embedding dimension (default: 128)
hidden_dim       # Hidden state dimension (default: 256)
window_size      # Sliding window size (default: 512)
plasticity_eta   # η - plasticity rate (default: 0.01)
decay_lambda     # λ - decay rate (default: 0.001)
```

**Tuning Tips:**
- ↑ `embed_dim`: Better semantic representation, more memory
- ↑ `window_size`: More local context, slower
- ↑ `plasticity_eta`: Faster learning, more volatile memory
- ↑ `decay_lambda`: Faster forgetting, more stable

### Standard SSMN

```python
input_dim        # Input vector dimension
hidden_dim       # Hidden state dimension (default: 256)
output_dim       # Output vector dimension
window_size      # Sliding window size (default: 512)
plasticity_eta   # η - plasticity rate (default: 0.01)
decay_lambda     # λ - decay rate (default: 0.001)
```

**Tuning Tips:**
- ↑ `hidden_dim`: More capacity, slower
- ↑ `window_size`: Better local context, more memory
- Balance `η` and `λ`: High η + low λ = long memory, Low η + high λ = short memory

---

## 🔍 Monitoring & Debugging

### Key Statistics

Both implementations provide real-time statistics:

```python
# Synaptic memory health
net.synaptic_energy          # How much info is stored
net.synaptic_update_magnitude # How fast memory is changing

# Attention patterns
net.attention_entropy        # Attention distribution spread

# Window state
net.window_fill              # How full is the window

# Full analysis
analysis = net.analyze_synaptic_state()
# Returns: energy, active_modes, spectral_radius, sparsity, etc.
```

### Common Issues

**Problem**: Synaptic energy exploding  
**Solution**: Decrease `plasticity_eta` or increase `decay_lambda`

**Problem**: Network "forgetting" too fast  
**Solution**: Decrease `decay_lambda` or increase `plasticity_eta`

**Problem**: Low attention entropy (peaked distribution)  
**Solution**: Check input normalization, adjust window size

**Problem**: Spectral radius > 1  
**Solution**: Decrease `plasticity_eta` (network may be unstable)

---

## 🆚 Comparison: Text-Native vs Standard SSMN

| Feature                    | Text-Native SSMN | Standard SSMN |
|----------------------------|------------------|---------------|
| **Input Type**             | Token IDs        | Continuous vectors |
| **Semantic Encoding**      | ✅ Yes           | ❌ No          |
| **Importance Gating**      | ✅ Yes           | ❌ No          |
| **Internal Chat**          | ✅ Yes           | ❌ No          |
| **Best For**               | NLP, chatbots    | Time series, RL |
| **Memory Selectivity**     | High (gated)     | Medium         |
| **Complexity**             | Higher           | Lower          |

**When to use Text-Native SSMN:**
- Building language models
- Chat/dialogue systems
- Semantic reasoning tasks
- When you need the model to "understand" intent

**When to use Standard SSMN:**
- Time series prediction
- Reinforcement learning
- Control systems
- Continuous data streams

---

## 📚 References

### Theoretical Foundations

1. **Sliding Window Attention**: Reduces complexity from O(n²) to O(n·w)
2. **Fast Weights / Hebbian Learning**: Outer product updates `h_t ⊗ h_{t-1}`
3. **Hippocampus-Cortex Model**: 80/20 static/plastic layer split
4. **Eigenvalue Stability**: Spectral radius monitoring prevents explosion

---

## 🛠️ Troubleshooting

### Compilation Issues

**Error**: `library not found`  
**Fix**: Ensure compiled library is in same directory as .py file

**Error**: `undefined symbol`  
**Fix**: Check that `-lm` flag is included (links math library)

### Runtime Issues

**Error**: `ValueError: Expected input_dim=X`  
**Fix**: Check input shape matches network configuration

**Error**: Numerical instability  
**Fix**: Reduce `plasticity_eta` or add gradient clipping

**Warning**: Slow performance  
**Fix**: Compile with `-O3` optimization flag

---

## 📝 License

GPL V3

---

**Built with ❤️ for efficient, brain-inspired AI**
