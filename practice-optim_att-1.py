import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Tuple

# ========================================
# 1. YOUR ORIGINAL TRITON KERNEL
# ========================================

@triton.jit
def attention_kernel_original(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr
):
    """
    Your original kernel - each thread handles one query position.
    This is the "naive" approach but helps us understand the basics.
    """
    # Which query position does this thread handle?
    query_idx = tl.program_id(0)

    if query_idx >= seq_len:
        return

    # Create indices for loading vectors - think of this as preparing addresses
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)  # [0, 1, 2, ..., 127] 
    dim_mask = dim_offsets < d_model            # [True, True, ..., False] - only use first d_model elements

    # Load the query vector for this thread's assigned position
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # Initialize scores - tl.full creates array filled with same value (like np.full)
    scores = tl.full([BLOCK_SIZE_SEQ], value=-float('inf'), dtype=tl.float32)

    # The inefficient part: load each key one by one in a loop
    for k_idx in range(seq_len):
        if k_idx < BLOCK_SIZE_SEQ:
            # Load one key vector at a time (this is wasteful!)
            k_ptrs = K_ptr + k_idx * d_model + dim_offsets
            key = tl.load(k_ptrs, mask=dim_mask, other=0.0)

            # Compute attention score: dot product of query and key
            score = tl.sum(query * key) * scale

            # Store the score at the right position in our scores array
            scores = tl.where(tl.arange(0, BLOCK_SIZE_SEQ) == k_idx, score, scores)

    # Apply softmax to convert scores to probabilities
    seq_mask = tl.arange(0, BLOCK_SIZE_SEQ) < seq_len
    scores = tl.where(seq_mask, scores, -float('inf'))

    max_score = tl.max(scores, axis=0)
    scores_shifted = scores - max_score
    exp_scores = tl.exp(scores_shifted)
    exp_scores = tl.where(seq_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / sum_exp

    # Another inefficient loop: load each value one by one
    output = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    for v_idx in range(seq_len):
        if v_idx < BLOCK_SIZE_SEQ:
            # Load one value vector at a time (wasteful again!)
            v_ptrs = V_ptr + v_idx * d_model + dim_offsets
            value = tl.load(v_ptrs, mask=dim_mask, other=0.0)

            # Get the attention weight for this value
            weight_mask = tl.arange(0, BLOCK_SIZE_SEQ) == v_idx
            weight = tl.sum(tl.where(weight_mask, attn_weights, 0.0))

            # Add this value's contribution to the output
            output += weight * value

    # Store the final result
    o_ptrs = Output_ptr + query_idx * d_model + dim_offsets
    tl.store(o_ptrs, output, mask=dim_mask)


# ========================================
# 2. VECTORIZED TRITON KERNEL
# ========================================

@triton.jit
def attention_kernel_vectorized(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr
):
    """
    Vectorized version - still one thread per query, but loads all keys/values at once.
    This eliminates the wasteful loops from the original version.
    """
    query_idx = tl.program_id(0)
    if query_idx >= seq_len:
        return

    # Same setup as before
    seq_offsets = tl.arange(0, BLOCK_SIZE_SEQ)  # [0, 1, 2, 3, ..., BLOCK_SIZE_SEQ-1]
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)  # [0, 1, 2, 3, ..., BLOCK_SIZE_DIM-1]
    seq_mask = seq_offsets < seq_len             # Which sequence positions are valid?
    dim_mask = dim_offsets < d_model             # Which dimensions are valid?

    # Load query vector (same as original)
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # THE MAGIC: Load ALL keys at once instead of one by one
    # seq_offsets[:, None] creates a column vector [0, 1, 2, 3].T 
    # dim_offsets[None, :] creates a row vector [0, 1, 2, ..., 127]
    # Broadcasting gives us a 2D grid of memory addresses!
    k_ptrs = K_ptr + seq_offsets[:, None] * d_model + dim_offsets[None, :]
    all_keys = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)

    # Vectorized attention scores: compute ALL dot products at once
    # query[None, :] reshapes query from [128] to [1, 128] for broadcasting
    # all_keys is [seq_len, 128], so we get [seq_len] scores in one operation!
    scores = tl.sum(query[None, :] * all_keys, axis=1) * scale
    scores = tl.where(seq_mask, scores, -float('inf'))

    # Softmax (same as original)
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    exp_scores = tl.where(seq_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / sum_exp

    # THE MAGIC AGAIN: Load ALL values at once and compute weighted sum vectorized
    v_ptrs = V_ptr + seq_offsets[:, None] * d_model + dim_offsets[None, :]
    all_values = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
    
    # attn_weights[:, None] reshapes from [seq_len] to [seq_len, 1] for broadcasting
    # This gives us the weighted sum in one vectorized operation!
    output = tl.sum(attn_weights[:, None] * all_values, axis=0)

    # Store result
    o_ptrs = Output_ptr + query_idx * d_model + dim_offsets
    tl.store(o_ptrs, output, mask=dim_mask)


# ========================================
# 3. WRAPPER FUNCTIONS
# ========================================

def fused_attention_original(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Wrapper for the original kernel - keeps it simple and readable"""
    seq_len, d_model = Q.shape
    out = torch.empty_like(Q)
    scale = 1.0 / (d_model ** 0.5)  # Standard attention scaling
    
    # Calculate block size: find next power of 2 that fits d_model
    BLOCK_SIZE_DIM = 64
    while BLOCK_SIZE_DIM < d_model:
        BLOCK_SIZE_DIM *= 2
    
    # Each thread handles one query position
    grid = (seq_len,)
    attention_kernel_original[grid](
        Q, K, V, out, seq_len, d_model, scale,
        BLOCK_SIZE_SEQ=128, BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )
    return out

def fused_attention_vectorized(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Wrapper for the vectorized kernel"""
    seq_len, d_model = Q.shape
    out = torch.empty_like(Q)
    scale = 1.0 / (d_model ** 0.5)
    
    BLOCK_SIZE_DIM = 64
    while BLOCK_SIZE_DIM < d_model:
        BLOCK_SIZE_DIM *= 2
    
    grid = (seq_len,)
    attention_kernel_vectorized[grid](
        Q, K, V, out, seq_len, d_model, scale,
        BLOCK_SIZE_SEQ=128, BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )
    return out

def pytorch_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """PyTorch's optimized attention for comparison"""
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


# ========================================
# 4. BENCHMARKING FUNCTIONS
# ========================================

def warmup_gpu():
    """
    Warm up the GPU to get consistent timing results.
    First few GPU operations are slower due to initialization overhead.
    """
    print("🔥 Warming up GPU... (this prevents timing inconsistencies)")
    dummy = torch.randn(1000, 1000, device='cuda')
    for _ in range(10):
        torch.matmul(dummy, dummy)
    torch.cuda.synchronize()
    print("✅ GPU warmed up!")

def benchmark_function(func, Q, K, V, num_runs=100):
    """
    Accurately benchmark a function by running it multiple times.
    We use torch.cuda.synchronize() to ensure GPU operations complete.
    """
    # Warm up this specific function
    for _ in range(10):
        _ = func(Q, K, V)
    torch.cuda.synchronize()
    
    # Actual timing
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = func(Q, K, V)
    torch.cuda.synchronize()  # Wait for all GPU work to finish
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms, result

def run_benchmarks(sequence_lengths: List[int], d_model: int = 128) -> Tuple[List[float], List[float], List[float]]:
    """
    Run comprehensive benchmarks across different sequence lengths.
    Returns timing results for all three implementations.
    """
    pytorch_times = []
    original_times = []
    vectorized_times = []
    
    print(f"🚀 Starting benchmarks with d_model={d_model}")
    print("=" * 60)
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Create test data - using half precision like real models
        Q = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float16)
        K = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float16)
        V = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float16)
        
        # Convert to float32 for our kernels (Triton works better with float32 for now)
        Q_f32 = Q.float()
        K_f32 = K.float()
        V_f32 = V.float()
        
        try:
            # Benchmark PyTorch (the gold standard)
            print("  🐍 PyTorch...", end=" ")
            pytorch_time, pytorch_result = benchmark_function(pytorch_attention, Q, K, V)
            pytorch_times.append(pytorch_time)
            print(f"{pytorch_time:.2f}ms")
            
            # Benchmark original Triton kernel
            print("  🔧 Original Triton...", end=" ")
            original_time, original_result = benchmark_function(fused_attention_original, Q_f32, K_f32, V_f32)
            original_times.append(original_time)
            print(f"{original_time:.2f}ms")
            
            # Benchmark vectorized Triton kernel
            print("  ⚡ Vectorized Triton...", end=" ")
            vectorized_time, vectorized_result = benchmark_function(fused_attention_vectorized, Q_f32, K_f32, V_f32)
            vectorized_times.append(vectorized_time)
            print(f"{vectorized_time:.2f}ms")
            
            # Quick correctness check - results should be similar
            original_error = torch.max(torch.abs(original_result.half() - pytorch_result)).item()
            vectorized_error = torch.max(torch.abs(vectorized_result.half() - pytorch_result)).item()
            print(f"  ✅ Max error vs PyTorch: Original={original_error:.6f}, Vectorized={vectorized_error:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error at seq_len {seq_len}: {e}")
            # Use previous value or NaN for failed runs
            pytorch_times.append(float('nan'))
            original_times.append(float('nan'))
            vectorized_times.append(float('nan'))
    
    return pytorch_times, original_times, vectorized_times

def create_visualizations(sequence_lengths: List[int], pytorch_times: List[float], 
                         original_times: List[float], vectorized_times: List[float], d_model: int):
    """
    Create beautiful and informative plots showing the performance comparison.
    Now includes a bar chart for easier comparison!
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Line graph - Raw timing comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(sequence_lengths, pytorch_times, 'o-', label='PyTorch (Optimized)', 
             linewidth=3, markersize=8, color='#1f77b4')
    ax1.plot(sequence_lengths, original_times, 's-', label='Your Original Triton', 
             linewidth=3, markersize=8, color='#ff7f0e')
    ax1.plot(sequence_lengths, vectorized_times, '^-', label='Vectorized Triton', 
             linewidth=3, markersize=8, color='#2ca02c')
    
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance vs Sequence Length (d_model={d_model})\n(Lower is Better)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Bar chart for direct comparison
    ax2 = plt.subplot(2, 2, 2)
    x_pos = np.arange(len(sequence_lengths))
    width = 0.25
    
    bars1 = ax2.bar(x_pos - width, pytorch_times, width, label='PyTorch', 
                    color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x_pos, original_times, width, label='Original Triton', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x_pos + width, vectorized_times, width, label='Vectorized Triton', 
                    color='#2ca02c', alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars (makes it super easy to read)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.annotate(f'{height:.1f}ms',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2) 
    add_value_labels(bars3)
    
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Direct Timing Comparison (d_model={d_model})', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sequence_lengths)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Speedup factors
    ax3 = plt.subplot(2, 2, 3)
    original_speedup = [o/p for o, p in zip(original_times, pytorch_times)]
    vectorized_speedup = [v/p for v, p in zip(vectorized_times, pytorch_times)]
    original_vs_vectorized = [o/v for o, v in zip(original_times, vectorized_times)]
    
    ax3.plot(sequence_lengths, original_speedup, 's-', label='Original vs PyTorch', 
             linewidth=3, markersize=8, color='#ff7f0e')
    ax3.plot(sequence_lengths, vectorized_speedup, '^-', label='Vectorized vs PyTorch', 
             linewidth=3, markersize=8, color='#2ca02c')
    ax3.plot(sequence_lengths, original_vs_vectorized, 'd-', label='Original vs Vectorized', 
             linewidth=3, markersize=8, color='#d62728')
    
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (1.0x)')
    ax3.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Performance\n(Values > 1.0 = slower than reference)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Speedup bar chart (easier to see the wins!)
    ax4 = plt.subplot(2, 2, 4)
    
    # Focus on the vectorization improvement (most interesting)
    improvement_bars = ax4.bar(x_pos, original_vs_vectorized, width*2, 
                              color='#2ca02c', alpha=0.8, edgecolor='black',
                              label='Speedup from Vectorization')
    
    # Add speedup labels
    for i, bar in enumerate(improvement_bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax4.annotate(f'{height:.2f}x faster',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No improvement')
    ax4.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax4.set_title('Vectorization Improvement\n(How much faster vectorized is)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sequence_lengths)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(bottom=0.8)  # Start y-axis just below 1.0 for better visibility
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    
    for i, seq_len in enumerate(sequence_lengths):
        if not any(np.isnan([pytorch_times[i], original_times[i], vectorized_times[i]])):
            print(f"\nSequence Length {seq_len}:")
            print(f"  PyTorch:     {pytorch_times[i]:8.2f}ms")
            print(f"  Original:    {original_times[i]:8.2f}ms ({original_times[i]/pytorch_times[i]:.2f}x vs PyTorch)")
            print(f"  Vectorized:  {vectorized_times[i]:8.2f}ms ({vectorized_times[i]/pytorch_times[i]:.2f}x vs PyTorch)")
            print(f"  Improvement: {original_times[i]/vectorized_times[i]:.2f}x speedup from vectorization!")


# ========================================
# 5. MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # Make sure we're using CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This benchmark needs a GPU.")
        exit(1)
    
    print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    
    # Warm up the GPU first
    warmup_gpu()
    
    # Test on various sequence lengths to see scaling behavior
    sequence_lengths = [64, 128, 256, 512, 1024, 2048]
    d_model = 128  # Common model dimension
    
    # Run the benchmarks
    pytorch_times, original_times, vectorized_times = run_benchmarks(sequence_lengths, d_model)
    
    # Create visualizations
    create_visualizations(sequence_lengths, pytorch_times, original_times, vectorized_times, d_model)
    
    print("\n🎉 Benchmarking complete! You should see clear improvements from vectorization.")
    print("💡 Key takeaway: Vectorization eliminates redundant memory operations and uses GPU parallelism better!")