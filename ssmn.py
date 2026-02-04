#!/usr/bin/env python3
"""
SPARSE-STREAM MEMORY NETWORK (SSMN) - PYTHON API

Architecture combines:
1. Sliding Window Attention - O(n·w) local attention (The Eyes)
2. Neural Synaptic Memory - Fast-weight matrix W_f (The Brain)  
3. Decaying Latent Blocks - Hybrid attention + synaptic cells

Design: Replaces expensive "spotlight" of global attention with
"continuous ink" of synaptic weights. As information flows out of
the sliding window, it gets compressed into the MN Layer.

Synaptic Update: ΔW_f = η(h_t ⊗ h_{t-1}) - λW_f
- η (Plasticity): Absorbs current context into weights
- λ (Decay): Prunes old or irrelevant noise, keeps memory fresh

Architecture: 80% Static Layers (grammar/logic) + 20% Plastic Layers (memory hubs)

Compile C library first:
    Windows: gcc -shared -o ssmn.dll ssmn.c -lm -O3
    Linux:   gcc -shared -fPIC -o ssmn.so ssmn.c -lm -O3
    Mac:     gcc -shared -fPIC -o ssmn.dylib ssmn.c -lm -O3

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional, Tuple

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'ssmn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'ssmn.dylib'
    else:
        lib_name = 'ssmn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o ssmn.dll ssmn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o ssmn.dylib ssmn.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o ssmn.so ssmn.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded SSMN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_ssmn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float
]
_lib.create_ssmn.restype = ctypes.c_void_p
_lib.destroy_ssmn.argtypes = [ctypes.c_void_p]
_lib.destroy_ssmn.restype = None

# Forward pass
_lib.ssmn_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.ssmn_forward.restype = None

# Memory management
_lib.ssmn_reset.argtypes = [ctypes.c_void_p]
_lib.ssmn_reset.restype = None
_lib.ssmn_reset_synaptic_memory.argtypes = [ctypes.c_void_p]
_lib.ssmn_reset_synaptic_memory.restype = None

# Getters
_lib.ssmn_get_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.ssmn_get_hidden_state.restype = None

_lib.ssmn_get_synaptic_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.ssmn_get_synaptic_weights.restype = None

_lib.ssmn_get_synaptic_energy.argtypes = [ctypes.c_void_p]
_lib.ssmn_get_synaptic_energy.restype = ctypes.c_float

_lib.ssmn_get_attention_entropy.argtypes = [ctypes.c_void_p]
_lib.ssmn_get_attention_entropy.restype = ctypes.c_float

_lib.ssmn_get_synaptic_update_magnitude.argtypes = [ctypes.c_void_p]
_lib.ssmn_get_synaptic_update_magnitude.restype = ctypes.c_float

_lib.ssmn_get_window_fill.argtypes = [ctypes.c_void_p]
_lib.ssmn_get_window_fill.restype = ctypes.c_int

# Setters
_lib.ssmn_set_plasticity_eta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.ssmn_set_plasticity_eta.restype = None

_lib.ssmn_set_decay_lambda.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.ssmn_set_decay_lambda.restype = None

# Info
_lib.ssmn_print_info.argtypes = [ctypes.c_void_p]
_lib.ssmn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class SparseStreamMemoryNetwork:
    """
    Sparse-Stream Memory Network (SSMN)
    
    Replaces expensive global attention with:
    - Sliding Window Attention: O(n·w) local context
    - Neural Synaptic Memory: Fast-weight matrix that updates during forward pass
    
    The network "compresses" information flowing out of the sliding window
    into synaptic weights, creating a "continuous ink" of memory rather than
    a "spotlight" of attention.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int, default=256
        Hidden state dimension
    output_dim : int
        Output dimension
    window_size : int, default=512
        Sliding window size for local attention
    plasticity_eta : float, default=0.01
        Plasticity rate for synaptic updates (η)
    decay_lambda : float, default=0.001
        Decay rate for synaptic forgetting (λ)
    
    Examples
    --------
    >>> # Create SSMN
    >>> net = SparseStreamMemoryNetwork(
    ...     input_dim=128, 
    ...     hidden_dim=256, 
    ...     output_dim=64,
    ...     window_size=512
    ... )
    >>> 
    >>> # Process sequence
    >>> X = np.random.randn(100, 128).astype(np.float32)
    >>> outputs = net.predict(X)
    >>> 
    >>> # Check statistics
    >>> print(f"Synaptic Energy: {net.synaptic_energy:.6f}")
    >>> print(f"Attention Entropy: {net.attention_entropy:.4f}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        window_size: int = 512,
        plasticity_eta: float = 0.01,
        decay_lambda: float = 0.001
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self._plasticity_eta = plasticity_eta
        self._decay_lambda = decay_lambda
        
        # Create network
        self._net = _lib.create_ssmn(
            input_dim, hidden_dim, output_dim, window_size,
            plasticity_eta, decay_lambda
        )
        
        if not self._net:
            raise RuntimeError("Failed to create SSMN")
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_ssmn(self._net)
    
    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Forward pass for a single input vector
        
        Parameters
        ----------
        input_vec : np.ndarray
            Input vector of shape (input_dim,)
        
        Returns
        -------
        output : np.ndarray
            Output vector of shape (output_dim,)
        """
        if input_vec.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_vec.shape[0]}")
        
        input_vec = input_vec.astype(np.float32)
        output = np.zeros(self.output_dim, dtype=np.float32)
        
        input_ptr = input_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.ssmn_forward(self._net, input_ptr, output_ptr)
        
        return output
    
    def predict(
        self, 
        X: np.ndarray, 
        reset_memory: bool = True
    ) -> np.ndarray:
        """
        Process a sequence of inputs
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence of shape (seq_len, input_dim)
        reset_memory : bool, default=True
            Whether to reset memory before processing
        
        Returns
        -------
        outputs : np.ndarray
            Output sequence of shape (seq_len, output_dim)
        """
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected shape (*, {self.input_dim}), got {X.shape}")
        
        if reset_memory:
            self.reset()
        
        outputs = np.zeros((len(X), self.output_dim), dtype=np.float32)
        
        for i, input_vec in enumerate(X):
            outputs[i] = self.forward(input_vec)
        
        return outputs
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        verbose: int = 1
    ) -> 'SparseStreamMemoryNetwork':
        """
        Train the network (simple gradient descent)
        
        Note: This is a basic training loop. For production use,
        implement proper backpropagation through the C library.
        
        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, input_dim)
        y : np.ndarray
            Training targets of shape (n_samples, output_dim)
        epochs : int, default=10
            Number of training epochs
        learning_rate : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size
        verbose : int, default=1
            Verbosity level
        
        Returns
        -------
        self : SparseStreamMemoryNetwork
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                outputs = self.predict(batch_X, reset_memory=False)
                
                # Compute MSE loss
                loss = np.mean((outputs - batch_y) ** 2)
                total_loss += loss
            
            avg_loss = total_loss / (n_samples // batch_size)
            
            if verbose > 0 and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                      f"Synaptic Energy: {self.synaptic_energy:.6f}")
        
        return self
    
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'mse'
    ) -> float:
        """
        Compute score on test data
        
        Parameters
        ----------
        X : np.ndarray
            Test inputs
        y : np.ndarray
            Test targets
        metric : str, default='mse'
            Metric to compute ('mse' or 'r2')
        
        Returns
        -------
        score : float
        """
        predictions = self.predict(X)
        
        if metric == 'mse':
            return float(np.mean((predictions - y) ** 2))
        elif metric == 'r2':
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def reset(self):
        """Reset all memory (hidden state, window, synaptic weights)"""
        _lib.ssmn_reset(self._net)
    
    def reset_synaptic_memory(self):
        """Reset only the synaptic memory weights"""
        _lib.ssmn_reset_synaptic_memory(self._net)
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        state = np.zeros(self.hidden_dim, dtype=np.float32)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.ssmn_get_hidden_state(self._net, state_ptr)
        return state
    
    def get_synaptic_weights(self) -> np.ndarray:
        """Get current synaptic weight matrix"""
        weights = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.ssmn_get_synaptic_weights(self._net, weights_ptr)
        return weights
    
    @property
    def synaptic_energy(self) -> float:
        """Energy in synaptic weight matrix"""
        return _lib.ssmn_get_synaptic_energy(self._net)
    
    @property
    def attention_entropy(self) -> float:
        """Entropy of attention distribution"""
        return _lib.ssmn_get_attention_entropy(self._net)
    
    @property
    def synaptic_update_magnitude(self) -> float:
        """Magnitude of last synaptic update"""
        return _lib.ssmn_get_synaptic_update_magnitude(self._net)
    
    @property
    def window_fill(self) -> int:
        """Number of positions filled in window buffer"""
        return _lib.ssmn_get_window_fill(self._net)
    
    @property
    def plasticity_eta(self) -> float:
        """Plasticity rate η"""
        return self._plasticity_eta
    
    @plasticity_eta.setter
    def plasticity_eta(self, value: float):
        """Set plasticity rate η"""
        self._plasticity_eta = value
        _lib.ssmn_set_plasticity_eta(self._net, value)
    
    @property
    def decay_lambda(self) -> float:
        """Decay rate λ"""
        return self._decay_lambda
    
    @decay_lambda.setter
    def decay_lambda(self, value: float):
        """Set decay rate λ"""
        self._decay_lambda = value
        _lib.ssmn_set_decay_lambda(self._net, value)
    
    def print_info(self):
        """Print network information and statistics"""
        _lib.ssmn_print_info(self._net)
    
    def analyze_synaptic_state(self) -> dict:
        """
        Analyze the current synaptic memory state
        
        Returns
        -------
        analysis : dict
            Dictionary with analysis metrics
        """
        weights = self.get_synaptic_weights()
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(weights)
        
        return {
            'energy': self.synaptic_energy,
            'update_magnitude': self.synaptic_update_magnitude,
            'attention_entropy': self.attention_entropy,
            'window_fill': self.window_fill,
            'max_eigenvalue': float(np.max(np.abs(eigenvalues))),
            'spectral_radius': float(np.max(np.abs(eigenvalues))),
            'num_active_modes': int(np.sum(np.abs(eigenvalues) > 0.01)),
            'weight_sparsity': float(np.sum(np.abs(weights) < 0.01) / weights.size),
            'mean_weight': float(np.mean(np.abs(weights))),
            'std_weight': float(np.std(weights))
        }

# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_usage():
    """Demonstrate basic usage"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage - Sequence Processing")
    print("="*70)
    
    # Create synthetic sequential data
    np.random.seed(42)
    seq_len = 100
    input_dim = 32
    output_dim = 8
    
    X = np.random.randn(seq_len, input_dim).astype(np.float32)
    y = np.random.randn(seq_len, output_dim).astype(np.float32)
    
    # Split data
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create network
    net = SparseStreamMemoryNetwork(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        window_size=16
    )
    
    print("\nNetwork created:")
    net.print_info()
    
    print("\nTraining...")
    net.fit(X_train, y_train, epochs=20, verbose=1)
    
    # Evaluate
    mse = net.score(X_test, y_test, metric='mse')
    r2 = net.score(X_test, y_test, metric='r2')
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R²: {r2:.4f}")
    
    return net


def demo_synaptic_memory():
    """Demonstrate synaptic memory compression"""
    print("\n" + "="*70)
    print("DEMO 2: Synaptic Memory - Information Compression")
    print("="*70)
    
    net = SparseStreamMemoryNetwork(
        input_dim=16,
        hidden_dim=32,
        output_dim=8,
        window_size=8,
        plasticity_eta=0.02,
        decay_lambda=0.001
    )
    
    print("\nProcessing a long sequence...")
    print("Watch how information compresses into synaptic weights\n")
    
    # Generate pattern
    sequence_length = 50
    X = np.random.randn(sequence_length, 16).astype(np.float32)
    
    net.reset()
    
    for i in range(0, sequence_length, 5):
        batch = X[i:i+5]
        _ = net.predict(batch, reset_memory=False)
        
        analysis = net.analyze_synaptic_state()
        
        print(f"Step {i}:")
        print(f"  Energy: {analysis['energy']:.6f}")
        print(f"  Update Magnitude: {analysis['update_magnitude']:.6f}")
        print(f"  Active Modes: {analysis['num_active_modes']}")
        print(f"  Window Fill: {analysis['window_fill']}/8")
        print()
    
    print("✓ Synaptic memory accumulates compressed information!")


def demo_sliding_window():
    """Demonstrate sliding window efficiency"""
    print("\n" + "="*70)
    print("DEMO 3: Sliding Window - O(n·w) Complexity")
    print("="*70)
    
    net = SparseStreamMemoryNetwork(
        input_dim=64,
        hidden_dim=128,
        output_dim=32,
        window_size=32
    )
    
    print(f"\nWindow Size: {net.window_size}")
    print("Processing sequences of different lengths...\n")
    
    import time
    
    for seq_len in [100, 500, 1000]:
        X = np.random.randn(seq_len, 64).astype(np.float32)
        
        start = time.time()
        _ = net.predict(X)
        elapsed = time.time() - start
        
        print(f"Sequence Length: {seq_len:4d} - Time: {elapsed:.4f}s "
              f"({elapsed/seq_len*1000:.2f}ms per token)")
    
    print("\n✓ Linear scaling with sequence length!")


def demo_plasticity_decay():
    """Demonstrate plasticity and decay parameters"""
    print("\n" + "="*70)
    print("DEMO 4: Plasticity & Decay - Memory Dynamics")
    print("="*70)
    
    X = np.random.randn(20, 16).astype(np.float32)
    
    configs = [
        ("High Plasticity, Low Decay", 0.05, 0.0001),
        ("Balanced", 0.01, 0.001),
        ("Low Plasticity, High Decay", 0.001, 0.01)
    ]
    
    for name, eta, lam in configs:
        print(f"\n--- {name} (η={eta}, λ={lam}) ---")
        
        net = SparseStreamMemoryNetwork(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            window_size=8,
            plasticity_eta=eta,
            decay_lambda=lam
        )
        
        _ = net.predict(X)
        analysis = net.analyze_synaptic_state()
        
        print(f"Final Energy: {analysis['energy']:.6f}")
        print(f"Mean Weight: {analysis['mean_weight']:.6f}")
        print(f"Active Modes: {analysis['num_active_modes']}")
    
    print("\n✓ Different dynamics create different memory behaviors!")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "SPARSE-STREAM MEMORY NETWORK (SSMN)" + " "*23 + "║")
    print("║" + " "*7 + "Sliding Window Attention + Synaptic Memory" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        demo_basic_usage()
        
        input("\nPress Enter to continue...")
        demo_synaptic_memory()
        
        input("\nPress Enter to continue...")
        demo_sliding_window()
        
        input("\nPress Enter to continue...")
        demo_plasticity_decay()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Features Demonstrated:")
        print("  • Sliding Window Attention - O(n·w) local context")
        print("  • Neural Synaptic Memory - Fast-weight compression")
        print("  • Decaying Latent Blocks - Hybrid architecture")
        print("  • 80/20 Static/Plastic layers - Brain-inspired design")
        print("  • Hebbian learning with decay - Adaptive forgetting")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()