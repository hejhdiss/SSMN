#!/usr/bin/env python3
"""
SPARSE-STREAM MEMORY NETWORK (SSMN) - PYTHON API - CUSTOM SPLITTABLE VERSION

NEW FEATURE: Adjustable static/plastic layer split from Python!

Now you can control the ratio of plastic layers:
- plastic_ratio=0.2 → 20% plastic (default, brain-like)
- plastic_ratio=0.5 → 50% plastic (balanced)
- plastic_ratio=0.9 → 90% plastic (highly adaptive)

Compile C library first:
    Windows: gcc -shared -o ssmn_custom.dll ssmn_custom.c -lm -O3
    Linux:   gcc -shared -fPIC -o ssmn_custom.so ssmn_custom.c -lm -O3
    Mac:     gcc -shared -fPIC -o ssmn_custom.dylib ssmn_custom.c -lm -O3

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
        lib_name = 'ssmn_custom.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'ssmn_custom.dylib'
    else:
        lib_name = 'ssmn_custom.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o ssmn_custom.dll ssmn_custom.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o ssmn_custom.dylib ssmn_custom.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o ssmn_custom.so ssmn_custom.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded SSMN Custom C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_ssmn_custom.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float  # NEW: plastic_ratio
]
_lib.create_ssmn_custom.restype = ctypes.c_void_p
_lib.destroy_ssmn_custom.argtypes = [ctypes.c_void_p]
_lib.destroy_ssmn_custom.restype = None

# Forward pass
_lib.ssmn_custom_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.ssmn_custom_forward.restype = None

# Memory management
_lib.ssmn_custom_reset.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_reset.restype = None
_lib.ssmn_custom_reset_synaptic_memory.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_reset_synaptic_memory.restype = None

# Getters
_lib.ssmn_custom_get_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.ssmn_custom_get_hidden_state.restype = None

_lib.ssmn_custom_get_synaptic_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.ssmn_custom_get_synaptic_weights.restype = None

_lib.ssmn_custom_get_synaptic_energy.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_synaptic_energy.restype = ctypes.c_float

_lib.ssmn_custom_get_attention_entropy.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_attention_entropy.restype = ctypes.c_float

_lib.ssmn_custom_get_synaptic_update_magnitude.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_synaptic_update_magnitude.restype = ctypes.c_float

_lib.ssmn_custom_get_window_fill.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_window_fill.restype = ctypes.c_int

_lib.ssmn_custom_get_num_static_layers.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_num_static_layers.restype = ctypes.c_int

_lib.ssmn_custom_get_num_plastic_layers.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_num_plastic_layers.restype = ctypes.c_int

_lib.ssmn_custom_get_plastic_ratio.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_get_plastic_ratio.restype = ctypes.c_float

# Setters
_lib.ssmn_custom_set_plasticity_eta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.ssmn_custom_set_plasticity_eta.restype = None

_lib.ssmn_custom_set_decay_lambda.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.ssmn_custom_set_decay_lambda.restype = None

# Info
_lib.ssmn_custom_print_info.argtypes = [ctypes.c_void_p]
_lib.ssmn_custom_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class CustomSplitSSMN:
    """
    Sparse-Stream Memory Network with Custom Layer Split
    
    NEW: You can now control the plastic/static layer ratio!
    
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
    plastic_ratio : float, default=0.2
        **NEW**: Ratio of plastic layers (0.0 to 1.0)
        - 0.2 = 20% plastic (default, brain-like)
        - 0.5 = 50% plastic (balanced)
        - 0.9 = 90% plastic (highly adaptive)
    
    Examples
    --------
    >>> # Brain-like split (20% plastic)
    >>> net = CustomSplitSSMN(
    ...     input_dim=128, hidden_dim=256, output_dim=64,
    ...     plastic_ratio=0.2
    ... )
    >>> 
    >>> # Balanced split (50% plastic)
    >>> net = CustomSplitSSMN(
    ...     input_dim=128, hidden_dim=256, output_dim=64,
    ...     plastic_ratio=0.5
    ... )
    >>> 
    >>> # Highly adaptive (90% plastic)
    >>> net = CustomSplitSSMN(
    ...     input_dim=128, hidden_dim=256, output_dim=64,
    ...     plastic_ratio=0.9
    ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        window_size: int = 512,
        plasticity_eta: float = 0.01,
        decay_lambda: float = 0.001,
        plastic_ratio: float = 0.2  # NEW PARAMETER!
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self._plasticity_eta = plasticity_eta
        self._decay_lambda = decay_lambda
        self._plastic_ratio = plastic_ratio
        
        # Validate plastic_ratio
        if plastic_ratio < 0.0 or plastic_ratio > 1.0:
            raise ValueError(f"plastic_ratio must be between 0.0 and 1.0, got {plastic_ratio}")
        
        # Create network with custom split
        self._net = _lib.create_ssmn_custom(
            input_dim, hidden_dim, output_dim, window_size,
            plasticity_eta, decay_lambda, plastic_ratio
        )
        
        if not self._net:
            raise RuntimeError("Failed to create Custom SSMN")
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_ssmn_custom(self._net)
    
    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """Forward pass for a single input vector"""
        if input_vec.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_vec.shape[0]}")
        
        input_vec = input_vec.astype(np.float32)
        output = np.zeros(self.output_dim, dtype=np.float32)
        
        input_ptr = input_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.ssmn_custom_forward(self._net, input_ptr, output_ptr)
        
        return output
    
    def predict(
        self, 
        X: np.ndarray, 
        reset_memory: bool = True
    ) -> np.ndarray:
        """Process a sequence of inputs"""
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
    ) -> 'CustomSplitSSMN':
        """Train the network"""
        n_samples = len(X)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                outputs = self.predict(batch_X, reset_memory=False)
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
        """Compute score on test data"""
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
        """Reset all memory"""
        _lib.ssmn_custom_reset(self._net)
    
    def reset_synaptic_memory(self):
        """Reset only synaptic memory"""
        _lib.ssmn_custom_reset_synaptic_memory(self._net)
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        state = np.zeros(self.hidden_dim, dtype=np.float32)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.ssmn_custom_get_hidden_state(self._net, state_ptr)
        return state
    
    def get_synaptic_weights(self) -> np.ndarray:
        """Get synaptic weight matrix"""
        weights = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.ssmn_custom_get_synaptic_weights(self._net, weights_ptr)
        return weights
    
    @property
    def synaptic_energy(self) -> float:
        """Energy in synaptic weights"""
        return _lib.ssmn_custom_get_synaptic_energy(self._net)
    
    @property
    def attention_entropy(self) -> float:
        """Attention distribution entropy"""
        return _lib.ssmn_custom_get_attention_entropy(self._net)
    
    @property
    def synaptic_update_magnitude(self) -> float:
        """Magnitude of last synaptic update"""
        return _lib.ssmn_custom_get_synaptic_update_magnitude(self._net)
    
    @property
    def window_fill(self) -> int:
        """Number of filled positions in window"""
        return _lib.ssmn_custom_get_window_fill(self._net)
    
    @property
    def num_static_layers(self) -> int:
        """Number of static layers"""
        return _lib.ssmn_custom_get_num_static_layers(self._net)
    
    @property
    def num_plastic_layers(self) -> int:
        """Number of plastic layers"""
        return _lib.ssmn_custom_get_num_plastic_layers(self._net)
    
    @property
    def plastic_ratio(self) -> float:
        """Plastic layer ratio"""
        return _lib.ssmn_custom_get_plastic_ratio(self._net)
    
    @property
    def plasticity_eta(self) -> float:
        """Plasticity rate η"""
        return self._plasticity_eta
    
    @plasticity_eta.setter
    def plasticity_eta(self, value: float):
        """Set plasticity rate η"""
        self._plasticity_eta = value
        _lib.ssmn_custom_set_plasticity_eta(self._net, value)
    
    @property
    def decay_lambda(self) -> float:
        """Decay rate λ"""
        return self._decay_lambda
    
    @decay_lambda.setter
    def decay_lambda(self, value: float):
        """Set decay rate λ"""
        self._decay_lambda = value
        _lib.ssmn_custom_set_decay_lambda(self._net, value)
    
    def print_info(self):
        """Print network information"""
        _lib.ssmn_custom_print_info(self._net)
    
    def analyze_synaptic_state(self) -> dict:
        """Analyze synaptic memory state"""
        weights = self.get_synaptic_weights()
        eigenvalues = np.linalg.eigvalsh(weights)
        
        return {
            'energy': self.synaptic_energy,
            'update_magnitude': self.synaptic_update_magnitude,
            'attention_entropy': self.attention_entropy,
            'window_fill': self.window_fill,
            'num_static_layers': self.num_static_layers,
            'num_plastic_layers': self.num_plastic_layers,
            'plastic_ratio': self.plastic_ratio,
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

def demo_custom_splits():
    """Demonstrate different plastic/static splits"""
    print("\n" + "="*70)
    print("DEMO 1: Custom Plastic/Static Splits")
    print("="*70)
    
    splits = [
        ("Brain-like (20% plastic)", 0.2),
        ("Balanced (50% plastic)", 0.5),
        ("Highly Adaptive (80% plastic)", 0.8)
    ]
    
    X = np.random.randn(100, 32).astype(np.float32)
    y = np.random.randn(100, 16).astype(np.float32)
    
    for name, ratio in splits:
        print(f"\n--- {name} ---")
        
        net = CustomSplitSSMN(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            window_size=16,
            plastic_ratio=ratio
        )
        
        net.print_info()
        
        # Train briefly
        net.fit(X, y, epochs=10, verbose=0)
        
        analysis = net.analyze_synaptic_state()
        print(f"\nAfter training:")
        print(f"  Synaptic Energy: {analysis['energy']:.6f}")
        print(f"  Active Modes: {analysis['num_active_modes']}")
        print(f"  Static Layers: {analysis['num_static_layers']}")
        print(f"  Plastic Layers: {analysis['num_plastic_layers']}")


def demo_split_comparison():
    """Compare performance across different splits"""
    print("\n" + "="*70)
    print("DEMO 2: Split Comparison on Same Task")
    print("="*70)
    
    # Generate data
    X_train = np.random.randn(200, 32).astype(np.float32)
    y_train = np.random.randn(200, 16).astype(np.float32)
    X_test = np.random.randn(50, 32).astype(np.float32)
    y_test = np.random.randn(50, 16).astype(np.float32)
    
    ratios = [0.1, 0.2, 0.5, 0.8]
    results = []
    
    for ratio in ratios:
        net = CustomSplitSSMN(
            input_dim=32, hidden_dim=64, output_dim=16,
            window_size=16, plastic_ratio=ratio
        )
        
        net.fit(X_train, y_train, epochs=20, verbose=0)
        
        mse = net.score(X_test, y_test, metric='mse')
        results.append((ratio, mse, net.synaptic_energy))
        
        print(f"Plastic Ratio: {ratio:.1f} ({ratio*100:.0f}%) → "
              f"Test MSE: {mse:.6f}, Energy: {net.synaptic_energy:.6f}")
    
    print("\n✓ Different splits achieve different trade-offs!")


def demo_dynamic_adaptation():
    """Show how plastic ratio affects adaptation"""
    print("\n" + "="*70)
    print("DEMO 3: Adaptation Speed vs Stability")
    print("="*70)
    
    # Low plastic - more stable
    stable_net = CustomSplitSSMN(
        input_dim=16, hidden_dim=32, output_dim=8,
        window_size=8, plastic_ratio=0.1
    )
    
    # High plastic - more adaptive
    adaptive_net = CustomSplitSSMN(
        input_dim=16, hidden_dim=32, output_dim=8,
        window_size=8, plastic_ratio=0.9
    )
    
    # Process sequence
    X = np.random.randn(50, 16).astype(np.float32)
    
    print("\nStable Network (10% plastic):")
    for i in range(0, 50, 10):
        _ = stable_net.forward(X[i])
        print(f"  Step {i}: Energy={stable_net.synaptic_energy:.6f}, "
              f"Update={stable_net.synaptic_update_magnitude:.6f}")
    
    print("\nAdaptive Network (90% plastic):")
    for i in range(0, 50, 10):
        _ = adaptive_net.forward(X[i])
        print(f"  Step {i}: Energy={adaptive_net.synaptic_energy:.6f}, "
              f"Update={adaptive_net.synaptic_update_magnitude:.6f}")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*8 + "CUSTOM SPLIT SPARSE-STREAM MEMORY NETWORK" + " "*19 + "║")
    print("║" + " "*12 + "Now with adjustable layer ratios!" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        demo_custom_splits()
        
        input("\nPress Enter to continue...")
        demo_split_comparison()
        
        input("\nPress Enter to continue...")
        demo_dynamic_adaptation()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Features Demonstrated:")
        print("  • Adjustable plastic/static split from Python")
        print("  • Brain-like (20%), Balanced (50%), Adaptive (80%) configs")
        print("  • Performance comparison across splits")
        print("  • Adaptation speed vs stability trade-offs")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()