#!/usr/bin/env python3
"""
TEXT-NATIVE SPARSE-STREAM MEMORY NETWORK (Text-Native SSMN) - PYTHON API

Architecture unifies Language and Memory:
1. Neural Semantic Encoder - "Thought embeddings" capture intent, not just words
2. Sliding Window Attention - O(n·w) local attention for immediate context
3. Synaptic Memory Layer - Fast-weight matrix W_f stores semantic relationships
4. Internal Recurrent Chat - Model "re-reads" its own synaptic state

Key Innovation: The model stores geometric relationships between concepts 
in active synapses. Language IS memory.

Compile C library first:
    Windows: gcc -shared -o text_native_ssmn.dll text_native_ssmn.c -lm -O3
    Linux:   gcc -shared -fPIC -o text_native_ssmn.so text_native_ssmn.c -lm -O3
    Mac:     gcc -shared -fPIC -o text_native_ssmn.dylib text_native_ssmn.c -lm -O3

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional, List, Union

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'text_native_ssmn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'text_native_ssmn.dylib'
    else:
        lib_name = 'text_native_ssmn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o text_native_ssmn.dll text_native_ssmn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o text_native_ssmn.dylib text_native_ssmn.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o text_native_ssmn.so text_native_ssmn.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded Text-Native SSMN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_text_native_ssmn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_float, ctypes.c_float
]
_lib.create_text_native_ssmn.restype = ctypes.c_void_p
_lib.destroy_text_native_ssmn.argtypes = [ctypes.c_void_p]
_lib.destroy_text_native_ssmn.restype = None

# Forward pass
_lib.text_native_ssmn_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
]
_lib.text_native_ssmn_forward.restype = None

# Memory management
_lib.text_native_ssmn_reset.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_reset.restype = None
_lib.text_native_ssmn_reset_synaptic_memory.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_reset_synaptic_memory.restype = None

# Getters
_lib.text_native_ssmn_get_hidden_state.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
]
_lib.text_native_ssmn_get_hidden_state.restype = None

_lib.text_native_ssmn_get_avg_importance.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_get_avg_importance.restype = ctypes.c_float

_lib.text_native_ssmn_get_avg_synaptic_energy.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_get_avg_synaptic_energy.restype = ctypes.c_float

_lib.text_native_ssmn_get_avg_attention_entropy.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_get_avg_attention_entropy.restype = ctypes.c_float

_lib.text_native_ssmn_get_synaptic_weights.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
]
_lib.text_native_ssmn_get_synaptic_weights.restype = None

# Info
_lib.text_native_ssmn_print_info.argtypes = [ctypes.c_void_p]
_lib.text_native_ssmn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class TextNativeSSMN:
    """
    Text-Native Sparse-Stream Memory Network
    
    A neural architecture where Language and Memory are unified. The model
    doesn't store words; it stores the geometric relationships between 
    concepts in its active synapses.
    
    Architecture:
    - Neural Semantic Encoder: Converts tokens to "thought embeddings"
    - Sliding Window Attention: O(n·w) local attention
    - Synaptic Memory: Fast-weight matrix W_f updated during forward pass
    - Internal Recurrent Chat: Model re-reads its own synaptic state
    
    Parameters
    ----------
    vocab_size : int
        Size of vocabulary
    embed_dim : int, default=128
        Dimension of semantic embeddings
    hidden_dim : int, default=256
        Hidden state dimension
    window_size : int, default=512
        Sliding window size for local attention
    plasticity_eta : float, default=0.01
        Learning rate for synaptic updates (η)
    decay_lambda : float, default=0.001
        Decay rate for synaptic forgetting (λ)
    
    Examples
    --------
    >>> # Create text-native SSMN
    >>> net = TextNativeSSMN(vocab_size=10000, hidden_dim=256, window_size=512)
    >>> 
    >>> # Process a sequence of tokens
    >>> tokens = [42, 123, 7, 891, 34]
    >>> probs = net.generate(tokens)
    >>> 
    >>> # Get statistics
    >>> print(f"Synaptic Energy: {net.synaptic_energy:.6f}")
    >>> print(f"Attention Entropy: {net.attention_entropy:.4f}")
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        window_size: int = 512,
        plasticity_eta: float = 0.01,
        decay_lambda: float = 0.001
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.plasticity_eta = plasticity_eta
        self.decay_lambda = decay_lambda
        
        # Create network
        self._net = _lib.create_text_native_ssmn(
            vocab_size, embed_dim, hidden_dim, window_size,
            plasticity_eta, decay_lambda
        )
        
        if not self._net:
            raise RuntimeError("Failed to create Text-Native SSMN")
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_text_native_ssmn(self._net)
    
    def forward(self, token_id: int) -> np.ndarray:
        """
        Forward pass for a single token
        
        Parameters
        ----------
        token_id : int
            Input token ID
        
        Returns
        -------
        probs : np.ndarray
            Output probability distribution over vocabulary
        """
        probs = np.zeros(self.vocab_size, dtype=np.float32)
        probs_ptr = probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.text_native_ssmn_forward(self._net, token_id, probs_ptr)
        
        return probs
    
    def generate(
        self, 
        tokens: List[int], 
        reset_memory: bool = True
    ) -> np.ndarray:
        """
        Process a sequence of tokens
        
        Parameters
        ----------
        tokens : List[int]
            Sequence of token IDs
        reset_memory : bool, default=True
            Whether to reset memory before processing
        
        Returns
        -------
        probs : np.ndarray
            Output probabilities for last token
        """
        if reset_memory:
            self.reset()
        
        probs = None
        for token_id in tokens:
            probs = self.forward(token_id)
        
        return probs
    
    def generate_sequence(
        self,
        prompt_tokens: List[int],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[int]:
        """
        Generate a sequence of tokens
        
        Parameters
        ----------
        prompt_tokens : List[int]
            Initial prompt tokens
        max_length : int, default=100
            Maximum sequence length to generate
        temperature : float, default=1.0
            Sampling temperature (higher = more random)
        top_k : int, default=50
            Sample from top-k tokens only
        
        Returns
        -------
        generated : List[int]
            Generated token sequence
        """
        self.reset()
        generated = list(prompt_tokens)
        
        # Process prompt
        for token_id in prompt_tokens:
            _ = self.forward(token_id)
        
        # Generate
        for _ in range(max_length - len(prompt_tokens)):
            probs = self.forward(generated[-1])
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
            
            # Top-k sampling
            if top_k > 0:
                top_indices = np.argsort(probs)[-top_k:]
                top_probs = probs[top_indices]
                top_probs = top_probs / np.sum(top_probs)
                next_token = np.random.choice(top_indices, p=top_probs)
            else:
                next_token = np.random.choice(self.vocab_size, p=probs)
            
            generated.append(int(next_token))
        
        return generated
    
    def reset(self):
        """Reset all memory (hidden state, window, synaptic weights)"""
        _lib.text_native_ssmn_reset(self._net)
    
    def reset_synaptic_memory(self):
        """Reset only the synaptic memory weights"""
        _lib.text_native_ssmn_reset_synaptic_memory(self._net)
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        state = np.zeros(self.hidden_dim, dtype=np.float32)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.text_native_ssmn_get_hidden_state(self._net, state_ptr)
        return state
    
    def get_synaptic_weights(self) -> np.ndarray:
        """Get current synaptic weight matrix"""
        weights = np.zeros(
            (self.hidden_dim, self.hidden_dim), 
            dtype=np.float32
        )
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.text_native_ssmn_get_synaptic_weights(self._net, weights_ptr)
        return weights
    
    @property
    def avg_importance(self) -> float:
        """Average importance gate activation"""
        return _lib.text_native_ssmn_get_avg_importance(self._net)
    
    @property
    def synaptic_energy(self) -> float:
        """Energy in synaptic weight matrix"""
        return _lib.text_native_ssmn_get_avg_synaptic_energy(self._net)
    
    @property
    def attention_entropy(self) -> float:
        """Entropy of attention distribution"""
        return _lib.text_native_ssmn_get_avg_attention_entropy(self._net)
    
    def print_info(self):
        """Print network information and statistics"""
        _lib.text_native_ssmn_print_info(self._net)
    
    def analyze_synaptic_state(self):
        """
        Analyze the current synaptic memory state
        
        Returns
        -------
        analysis : dict
            Dictionary with analysis metrics
        """
        weights = self.get_synaptic_weights()
        
        # Compute eigenvalues to understand memory structure
        eigenvalues = np.linalg.eigvalsh(weights)
        
        return {
            'energy': self.synaptic_energy,
            'importance': self.avg_importance,
            'attention_entropy': self.attention_entropy,
            'max_eigenvalue': float(np.max(np.abs(eigenvalues))),
            'spectral_radius': float(np.max(np.abs(eigenvalues))),
            'num_active_modes': int(np.sum(np.abs(eigenvalues) > 0.01)),
            'weight_sparsity': float(np.sum(np.abs(weights) < 0.01) / weights.size)
        }

# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_generation():
    """Demonstrate basic token generation"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Token Generation")
    print("="*70)
    
    # Create small vocabulary network
    net = TextNativeSSMN(
        vocab_size=100,
        embed_dim=32,
        hidden_dim=64,
        window_size=16
    )
    
    print("\nNetwork created:")
    net.print_info()
    
    # Process a sequence
    print("\nProcessing sequence: [5, 12, 7, 23, 45]")
    tokens = [5, 12, 7, 23, 45]
    
    for token in tokens:
        probs = net.forward(token)
        top_5 = np.argsort(probs)[-5:][::-1]
        
        print(f"\nToken {token}:")
        print(f"  Synaptic Energy: {net.synaptic_energy:.6f}")
        print(f"  Importance: {net.avg_importance:.4f}")
        print(f"  Top-5 predictions: {top_5.tolist()}")
    
    print("\n✓ Network processes tokens and updates synaptic memory!")


def demo_semantic_compression():
    """Demonstrate synaptic text compression"""
    print("\n" + "="*70)
    print("DEMO 2: Synaptic Text Compression")
    print("="*70)
    
    net = TextNativeSSMN(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        window_size=32,
        plasticity_eta=0.02,
        decay_lambda=0.001
    )
    
    print("\nSimulating a 'conversation' - processing related tokens")
    
    # Simulate topic: "cat", "dog", "animal", "pet"
    # In real use, these would be actual token IDs
    topic_tokens = [42, 43, 44, 45, 42, 45, 43]  # Related concepts
    
    print("\nProcessing related tokens (simulating semantic field)...")
    for i, token in enumerate(topic_tokens):
        _ = net.forward(token)
        
        if i % 2 == 0:
            analysis = net.analyze_synaptic_state()
            print(f"\nStep {i}:")
            print(f"  Energy: {analysis['energy']:.6f}")
            print(f"  Active modes: {analysis['num_active_modes']}")
            print(f"  Spectral radius: {analysis['spectral_radius']:.4f}")
    
    print("\n--- Synaptic State Analysis ---")
    final_analysis = net.analyze_synaptic_state()
    for key, value in final_analysis.items():
        print(f"{key}: {value}")
    
    print("\n✓ Network compresses semantic relationships into synaptic weights!")


def demo_internal_chat():
    """Demonstrate internal recurrent chatting"""
    print("\n" + "="*70)
    print("DEMO 3: Internal Recurrent Chat")
    print("="*70)
    
    net = TextNativeSSMN(
        vocab_size=500,
        embed_dim=64,
        hidden_dim=128,
        window_size=64
    )
    
    print("\nThe model 're-reads' its own synaptic state")
    print("This creates an internal dialogue for reasoning\n")
    
    # Process a sequence and observe internal state evolution
    sequence = [10, 20, 30, 40, 50, 60, 70, 80]
    
    states = []
    for token in sequence:
        _ = net.forward(token)
        state = net.get_hidden_state()
        states.append(state)
    
    # Compute state changes (shows internal dynamics)
    print("Hidden State Dynamics:")
    for i in range(1, len(states)):
        delta = np.linalg.norm(states[i] - states[i-1])
        print(f"  Step {i}: Δh = {delta:.4f}")
    
    print("\n✓ Internal chat creates dynamic state evolution!")


def demo_sliding_window():
    """Demonstrate sliding window attention"""
    print("\n" + "="*70)
    print("DEMO 4: Sliding Window Attention")
    print("="*70)
    
    net = TextNativeSSMN(
        vocab_size=200,
        embed_dim=32,
        hidden_dim=64,
        window_size=8  # Small window for demo
    )
    
    print(f"\nWindow Size: {net.window_size}")
    print("Processing long sequence to see windowing effect...\n")
    
    # Long sequence
    long_sequence = list(range(0, 50, 3))  # [0, 3, 6, 9, ...]
    
    for i, token in enumerate(long_sequence):
        _ = net.forward(token)
        
        if i % 5 == 0:
            print(f"Token {i}: Attention Entropy = {net.attention_entropy:.4f}")
    
    print("\n✓ Sliding window provides O(n·w) complexity!")
    print(f"  Processed {len(long_sequence)} tokens")
    print(f"  Each token only attends to {net.window_size} neighbors")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*8 + "TEXT-NATIVE SPARSE-STREAM MEMORY NETWORK" + " "*20 + "║")
    print("║" + " "*12 + "Language IS Memory: Unified Architecture" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        demo_basic_generation()
        
        input("\nPress Enter to continue...")
        demo_semantic_compression()
        
        input("\nPress Enter to continue...")
        demo_internal_chat()
        
        input("\nPress Enter to continue...")
        demo_sliding_window()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Features Demonstrated:")
        print("  • Neural Semantic Encoder - Thought embeddings, not word vectors")
        print("  • Sliding Window Attention - O(n·w) local context")
        print("  • Synaptic Memory - Geometric concept relationships")
        print("  • Internal Recurrent Chat - Self-reading mechanism")
        print("  • Text-Native Design - Language and memory are unified")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()