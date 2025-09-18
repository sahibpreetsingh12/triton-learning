"""
Blog Post 1: Basic Attention in Triton - Understanding the Fundamentals
=====================================================================

This is the starting point of our Triton attention series. Here we implement
a simple, easy-to-understand attention kernel that mirrors PyTorch's behavior.

Key Learning Goals:
- Understand how attention works at the kernel level
- Learn Triton basics: program_id, load/store, vectorization
- Establish correctness before optimizing for performance

Performance Note: This implementation is intentionally simple and NOT optimized.
We'll improve it step by step in subsequent blog posts.
"""

import torch
import triton
import triton.language as tl
import math
import time
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AttentionConfig:
    """Configuration for our attention kernel"""
    BLOCK_SIZE_SEQ: int = 128    # Max sequence length we can handle in one block
    BLOCK_SIZE_DIM: int = 128    # Max dimension size we can handle in one block

    def __post_init__(self):
        """Ensure our block sizes are powers of 2 (Triton works best this way)"""
        assert (self.BLOCK_SIZE_SEQ & (self.BLOCK_SIZE_SEQ - 1)) == 0, "BLOCK_SIZE_SEQ must be power of 2"
        assert (self.BLOCK_SIZE_DIM & (self.BLOCK_SIZE_DIM - 1)) == 0, "BLOCK_SIZE_DIM must be power of 2"


# ============================================================================
# THE MAIN KERNEL - Simple and Educational
# ============================================================================

@triton.jit
def basic_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr
):
    """
    Basic attention kernel - each GPU thread handles one query position.

    This is the "naive" approach that prioritizes clarity over performance.
    We'll optimize this in later blog posts!

    Algorithm:
    1. Each thread loads one query vector
    2. Loop through all keys, compute attention scores
    3. Apply softmax to get attention weights
    4. Loop through all values, compute weighted sum
    5. Store the result

    Args:
        Q_ptr, K_ptr, V_ptr: Pointers to query, key, value tensors
        Output_ptr: Pointer to output tensor
        seq_len: Sequence length (number of tokens)
        d_model: Model dimension (size of each vector)
        scale: Attention scaling factor (1/sqrt(d_model))
        BLOCK_SIZE_*: Compile-time constants for memory management
    """

    # Step 1: Figure out which query position this thread handles
    query_idx = tl.program_id(0)  # Each thread gets a unique ID

    if query_idx >= seq_len:
        return  # Skip if we're beyond the sequence

    # Step 2: Set up memory access patterns
    # We need to load vectors of size d_model, but our block might be larger
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)  # [0, 1, 2, ..., BLOCK_SIZE_DIM-1]
    dim_mask = dim_offsets < d_model            # [True, True, ..., False] masks out padding

    # Step 3: Load our query vector
    # Memory layout: Q[query_idx] starts at Q_ptr + query_idx * d_model
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # Step 4: Compute attention scores against ALL keys
    # Initialize scores with -infinity (will become 0 after softmax)
    scores = tl.full([BLOCK_SIZE_SEQ], value=-float('inf'), dtype=tl.float32)

    # The inefficient part: loop through each key individually
    # (We'll vectorize this in the next blog post!)
    for k_idx in range(seq_len):
        if k_idx < BLOCK_SIZE_SEQ:
            # Load key vector k_idx
            k_ptrs = K_ptr + k_idx * d_model + dim_offsets
            key = tl.load(k_ptrs, mask=dim_mask, other=0.0)

            # Compute attention score: dot product + scaling
            score = tl.sum(query * key) * scale

            # Store score at position k_idx in our scores array
            # (This is a bit clunky but works for learning purposes)
            score_mask = tl.arange(0, BLOCK_SIZE_SEQ) == k_idx
            scores = tl.where(score_mask, score, scores)

    # Step 5: Apply softmax to convert scores to probabilities
    # Mask out positions beyond our sequence length
    seq_mask = tl.arange(0, BLOCK_SIZE_SEQ) < seq_len
    scores = tl.where(seq_mask, scores, -float('inf'))

    # Numerically stable softmax: subtract max to prevent overflow
    max_score = tl.max(scores, axis=0)
    scores_shifted = scores - max_score
    exp_scores = tl.exp(scores_shifted)
    exp_scores = tl.where(seq_mask, exp_scores, 0.0)  # Zero out invalid positions

    sum_exp = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / sum_exp

    # Step 6: Compute output as weighted sum of values
    output = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    # Another inefficient loop - we'll fix this too!
    for v_idx in range(seq_len):
        if v_idx < BLOCK_SIZE_SEQ:
            # Load value vector v_idx
            v_ptrs = V_ptr + v_idx * d_model + dim_offsets
            value = tl.load(v_ptrs, mask=dim_mask, other=0.0)

            # Get the attention weight for this value
            weight_mask = tl.arange(0, BLOCK_SIZE_SEQ) == v_idx
            weight = tl.sum(tl.where(weight_mask, attn_weights, 0.0))

            # Add this value's contribution to our output
            output += weight * value

    # Step 7: Store the final result
    o_ptrs = Output_ptr + query_idx * d_model + dim_offsets
    tl.store(o_ptrs, output, mask=dim_mask)


# ============================================================================
# WRAPPER FUNCTION - Clean Interface
# ============================================================================

def basic_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Clean, PyTorch-like interface for our basic attention kernel.

    Args:
        Q, K, V: Input tensors of shape [seq_len, d_model]

    Returns:
        output: Attention output of shape [seq_len, d_model]
    """
    seq_len, d_model = Q.shape
    assert K.shape == V.shape == (seq_len, d_model), "Q, K, V must have same shape"

    # Create output tensor
    output = torch.empty_like(Q)

    # Standard attention scaling
    scale = 1.0 / math.sqrt(d_model)

    # Choose block sizes (powers of 2 that fit our data)
    config = AttentionConfig()

    # Make sure our blocks are large enough for the actual data
    BLOCK_SIZE_SEQ = config.BLOCK_SIZE_SEQ
    while BLOCK_SIZE_SEQ < seq_len:
        BLOCK_SIZE_SEQ *= 2

    BLOCK_SIZE_DIM = config.BLOCK_SIZE_DIM
    while BLOCK_SIZE_DIM < d_model:
        BLOCK_SIZE_DIM *= 2

    # Launch one thread per query position
    grid = (seq_len,)

    basic_attention_kernel[grid](
        Q, K, V, output,
        seq_len, d_model, scale,
        BLOCK_SIZE_SEQ, BLOCK_SIZE_DIM
    )

    return output


# ============================================================================
# EDUCATIONAL FUNCTIONS
# ============================================================================

def explain_attention_algorithm():
    """Print step-by-step explanation of what our kernel does"""
    print("🎓 Basic Attention Algorithm Explanation")
    print("=" * 50)
    print("""
    Our kernel implements the standard attention mechanism:

    Attention(Q,K,V) = softmax(QK^T/√d)V

    Step by step:
    1. For each query position i:
       a. Load query vector Q[i]
       b. Compute scores = Q[i] · K[j] for all j (dot products)
       c. Apply scaling: scores = scores / √d_model
       d. Apply softmax: weights = softmax(scores)
       e. Compute output: O[i] = Σ(weights[j] * V[j])
       f. Store O[i]

    Why this works:
    - Dot products measure similarity between query and keys
    - Softmax converts similarities to probabilities (sum to 1)
    - Weighted sum combines values based on attention weights

    Performance issues with this version:
    - Each key/value is loaded multiple times (once per query)
    - No vectorization - we compute one dot product at a time
    - Memory bandwidth not fully utilized

    Next blog post: We'll fix these issues with vectorization!
    """)


def compare_with_pytorch():
    """Demonstrate correctness by comparing with PyTorch"""
    print("\n🔬 Correctness Test vs PyTorch")
    print("=" * 40)

    # Small test case
    seq_len, d_model = 16, 64
    Q = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
    K = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
    V = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)

    # Our implementation
    output_triton = basic_attention(Q, K, V)

    # PyTorch reference
    scale = 1.0 / math.sqrt(d_model)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output_pytorch = torch.matmul(attn_weights, V)

    # Check correctness
    max_diff = torch.max(torch.abs(output_triton - output_pytorch)).item()
    is_correct = torch.allclose(output_triton, output_pytorch, rtol=1e-4, atol=1e-4)

    print(f"Input shape: {Q.shape}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Correct: {is_correct} {'✅' if is_correct else '❌'}")

    if is_correct:
        print("🎉 Perfect! Our basic implementation matches PyTorch!")
    else:
        print("❌ Something's wrong - let's debug...")
        print(f"Triton sample: {output_triton[0, :5]}")
        print(f"PyTorch sample: {output_pytorch[0, :5]}")


def basic_performance_test():
    """Quick performance test (we expect this to be slow!)"""
    print("\n⏱️  Basic Performance Test")
    print("=" * 35)
    print("Note: This is intentionally unoptimized - we're prioritizing clarity!")

    seq_len, d_model = 512, 128
    Q = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
    K = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
    V = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = basic_attention(Q, K, V)

    # Time our version
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        output_triton = basic_attention(Q, K, V)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 50

    # Time PyTorch
    scale = 1.0 / math.sqrt(d_model)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_pytorch = torch.matmul(attn_weights, V)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 50

    slowdown = triton_time / pytorch_time

    print(f"Basic Triton: {triton_time*1000:.2f} ms")
    print(f"PyTorch:      {pytorch_time*1000:.2f} ms")
    print(f"Slowdown:     {slowdown:.1f}x slower {'🐌' if slowdown > 1 else '🚀'}")
    print("\n💡 That's expected! PyTorch is highly optimized.")
    print("   Our goal is education first, performance second.")
    print("   Next blog post: We'll make this much faster!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This tutorial needs a GPU.")
        exit(1)

    print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    print("\n" + "="*60)
    print("BLOG POST 1: BASIC ATTENTION IN TRITON")
    print("="*60)

    # Educational walkthrough
    explain_attention_algorithm()

    # Correctness verification
    compare_with_pytorch()

    # Performance baseline
    basic_performance_test()

    print("\n🎯 Key Takeaways:")
    print("- We successfully implemented attention from scratch in Triton")
    print("- Our kernel is correct but unoptimized")
    print("- Next: We'll vectorize operations for better performance")
    print("\n📖 Ready for Blog Post 2: Vectorized Attention!")