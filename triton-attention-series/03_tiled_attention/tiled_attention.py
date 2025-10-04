"""
Blog Post 3: Scalable Attention with Tiling (Flash Attention)
=================================================================

This post combines vectorization and tiling to build a memory-efficient,
scalable attention kernel that mirrors the principles of Flash Attention.

Key Concepts:
- Tiling for memory efficiency
- Online softmax for numerical stability
- Fusing operations to avoid storing large intermediate matrices
- Benchmarking and performance analysis
"""

import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sys
import os

# Add parent directories to path to import from other blog posts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_basic_attention'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_vectorized_attention'))

from basic_attention import basic_attention
from vectorized_attention import vectorized_attention

# ============================================================================
# UTILITY FUNCTIONS (simplified versions)
# ============================================================================

def pytorch_attention_reference(Q, K, V):
    """Reference PyTorch attention implementation"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)

def benchmark_function(func, Q, K, V, num_runs=100, warmup_runs=10):
    """Simple benchmark function"""
    # Warmup
    for _ in range(warmup_runs):
        _ = func(Q, K, V)
    torch.cuda.synchronize()
    
    # Actual timing
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = func(Q, K, V)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms, result

# ============================================================================
# TILED (FLASH) ATTENTION KERNEL
# ============================================================================

@triton.jit
def tiled_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    stride_qm, stride_qk, stride_km, stride_kk, stride_vm, stride_vk, stride_om, stride_ok,
    seq_len, d_model, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Tiled attention kernel - this is the Flash Attention style implementation
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm) + (offs_d[None, :] * stride_qk)
    q_mask = offs_m[:, None] < seq_len
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    output_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    running_max_score = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    running_sum_exp = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start_idx in range(0, seq_len, BLOCK_N):
        offs_n = k_start_idx + tl.arange(0, BLOCK_N)
        
        k_ptrs = K_ptr + (offs_n[:, None] * stride_km) + (offs_d[None, :] * stride_kk)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vm) + (offs_d[None, :] * stride_vk)
        kv_mask = offs_n[:, None] < seq_len

        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        score_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
        score_tile = tl.where(offs_n[None,:] < seq_len, score_tile, -float('inf'))

        current_max_score = tl.max(score_tile, axis=1)
        new_running_max = tl.maximum(running_max_score, current_max_score)

        exp_scale = tl.exp(running_max_score - new_running_max)
        running_sum_exp *= exp_scale
        output_acc *= exp_scale[:, None]

        score_tile_shifted = tl.exp(score_tile - new_running_max[:, None])
        running_sum_exp += tl.sum(score_tile_shifted, axis=1)
        output_acc += tl.dot(score_tile_shifted.to(v_tile.dtype), v_tile)

        running_max_score = new_running_max

    output_acc = output_acc / running_sum_exp[:, None]
    
    o_ptrs = Output_ptr + (offs_m[:, None] * stride_om) + (offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, output_acc.to(Output_ptr.dtype.element_ty), mask=q_mask)

# ============================================================================
# WRAPPER FUNCTION
# ============================================================================

def tiled_attention(Q, K, V):
    seq_len, d_model = Q.shape
    output = torch.empty_like(Q)
    scale = 1.0 / (d_model ** 0.5)

    BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, d_model
    grid = (triton.cdiv(seq_len, BLOCK_M),)

    tiled_attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), output.stride(0), output.stride(1),
        seq_len, d_model, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
    return output

# ============================================================================
# MAIN EXECUTION: BENCHMARK & VISUALIZE
# ============================================================================

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across different sequence lengths"""
    
    # Define implementations to test
    implementations = {
        "Basic (Loop)": basic_attention,
        "Vectorized": vectorized_attention, 
        "Tiled (Flash)": tiled_attention,
        "PyTorch (Baseline)": pytorch_attention_reference
    }
    
    # Test configurations
    sequence_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    d_model = 128
    
    print(f"🚀 Running comprehensive benchmarks (d_model={d_model})")
    print("=" * 60)
    
    # Store results
    results = {name: [] for name in implementations.keys()}
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Create test tensors
        Q = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
        V = torch.randn(seq_len, d_model, device='cuda', dtype=torch.float32)
        
        for name, func in implementations.items():
            try:
                time_ms, output = benchmark_function(func, Q, K, V, num_runs=50)
                results[name].append(time_ms)
                print(f"  {name:20}: {time_ms:6.2f} ms")
                
                # Quick correctness check vs PyTorch
                if name != "PyTorch (Baseline)":
                    pytorch_output = pytorch_attention_reference(Q, K, V)
                    max_diff = torch.max(torch.abs(output - pytorch_output)).item()
                    is_correct = torch.allclose(output, pytorch_output, rtol=1e-3, atol=1e-3)
                    status = "✅" if is_correct else "❌"
                    print(f"  {' '*20}   Correctness: {status} (max_diff: {max_diff:.6f})")
                
            except Exception as e:
                print(f"  {name:20}: ❌ Error: {e}")
                results[name].append(float('nan'))
    
    return results, sequence_lengths

def create_performance_plots(results, sequence_lengths):
    """Create performance comparison plots"""
    
    # Filter out NaN values for plotting
    clean_results = {}
    for name, times in results.items():
        clean_times = [t if not np.isnan(t) else None for t in times]
        clean_results[name] = clean_times
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Raw performance comparison
    colors = {'Basic (Loop)': '#ff7f0e', 'Vectorized': '#d62728', 'Tiled (Flash)': '#2ca02c', 'PyTorch (Baseline)': '#1f77b4'}
    
    for name, times in clean_results.items():
        valid_seq_lens = [seq_len for seq_len, time_val in zip(sequence_lengths, times) if time_val is not None]
        valid_times = [time_val for time_val in times if time_val is not None]
        
        if valid_times:
            ax1.plot(valid_seq_lens, valid_times, 'o-', label=name, linewidth=2, markersize=6, color=colors.get(name, 'gray'))
    
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Attention Performance vs Sequence Length\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup vs PyTorch
    pytorch_times = clean_results.get('PyTorch (Baseline)', [])
    
    for name, times in clean_results.items():
        if name == 'PyTorch (Baseline)':
            continue
            
        speedups = []
        valid_seq_lens = []
        
        for i, (pytorch_time, triton_time) in enumerate(zip(pytorch_times, times)):
            if pytorch_time is not None and triton_time is not None and triton_time > 0:
                speedups.append(pytorch_time / triton_time)
                valid_seq_lens.append(sequence_lengths[i])
        
        if speedups:
            ax2.plot(valid_seq_lens, speedups, 'o-', label=f'{name} vs PyTorch', linewidth=2, markersize=6, color=colors.get(name, 'gray'))
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='PyTorch Baseline (1.0x)')
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor vs PyTorch', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Performance to PyTorch\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📊 SUMMARY:")
    print("=" * 30)
    
    for seq_len in [512, 2048, 4096]:  # Show key sequence lengths
        if seq_len in sequence_lengths:
            idx = sequence_lengths.index(seq_len)
            print(f"\nSequence Length {seq_len}:")
            for name, times in clean_results.items():
                if idx < len(times) and times[idx] is not None:
                    time_ms = times[idx]
                    if name != 'PyTorch (Baseline)' and clean_results['PyTorch (Baseline)'][idx] is not None:
                        speedup = clean_results['PyTorch (Baseline)'][idx] / time_ms
                        print(f"  {name:20}: {time_ms:6.2f} ms ({speedup:4.2f}x vs PyTorch)")
                    else:
                        print(f"  {name:20}: {time_ms:6.2f} ms")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)

    print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    print("\n" + "="*70)
    print("BLOG POST 3: FROM VECTORIZATION TO TILING (FLASH ATTENTION)")
    print("="*70)
    
    # Run benchmarks
    results, sequence_lengths = run_comprehensive_benchmark()
    
    # Create visualizations
    create_performance_plots(results, sequence_lengths)
    
    print("\n🎉 Benchmark complete! Tiled attention shows superior scaling.")
    print("💡 This demonstrates how tiling avoids the memory bottleneck of simple vectorization.")
