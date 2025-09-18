"""
Shared Testing Utilities for Triton Attention Blog Series
========================================================

This module provides common testing functions used across all blog posts
to ensure consistency and avoid code duplication.
"""

import torch
import math
import time
from typing import Callable, Tuple, List, Dict, Any
import numpy as np


def create_test_tensors(seq_len: int, d_model: int, device: str = 'cuda', dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create standard test tensors for attention benchmarking.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        Tuple of (Q, K, V) tensors
    """
    Q = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    K = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    V = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    return Q, K, V


def pytorch_attention_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Reference PyTorch attention implementation.

    This is our "ground truth" for correctness testing.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def test_correctness(
    triton_func: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test correctness of a Triton attention implementation.

    Args:
        triton_func: Function to test
        Q, K, V: Input tensors
        rtol, atol: Tolerance for torch.allclose
        verbose: Whether to print results

    Returns:
        Dictionary with test results
    """
    # Get outputs
    output_triton = triton_func(Q, K, V)
    output_pytorch = pytorch_attention_reference(Q, K, V)

    # Compute differences
    max_diff = torch.max(torch.abs(output_triton - output_pytorch)).item()
    mean_diff = torch.mean(torch.abs(output_triton - output_pytorch)).item()
    is_correct = torch.allclose(output_triton, output_pytorch, rtol=rtol, atol=atol)

    results = {
        'is_correct': is_correct,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'output_triton': output_triton,
        'output_pytorch': output_pytorch
    }

    if verbose:
        print(f"✅ Correctness Test Results:")
        print(f"   Input shape: {Q.shape}")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Within tolerance: {is_correct} {'✅' if is_correct else '❌'}")

        if not is_correct:
            print(f"   Triton sample: {output_triton[0, :5]}")
            print(f"   PyTorch sample: {output_pytorch[0, :5]}")

    return results


def warmup_gpu(iterations: int = 10):
    """Warm up GPU to get consistent timing results"""
    dummy = torch.randn(1000, 1000, device='cuda')
    for _ in range(iterations):
        torch.matmul(dummy, dummy)
    torch.cuda.synchronize()


def benchmark_function(
    func: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Tuple[float, torch.Tensor]:
    """
    Accurately benchmark a function.

    Args:
        func: Function to benchmark
        Q, K, V: Input tensors
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs

    Returns:
        Tuple of (average_time_ms, output)
    """
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


def comprehensive_test_suite(
    triton_func: Callable,
    func_name: str = "Triton Function",
    test_configs: List[Tuple[int, int]] = None
) -> List[Dict[str, Any]]:
    """
    Run comprehensive test suite across multiple problem sizes.

    Args:
        triton_func: Triton function to test
        func_name: Name for reporting
        test_configs: List of (seq_len, d_model) tuples to test

    Returns:
        List of test results
    """
    if test_configs is None:
        test_configs = [
            (64, 64),    # Small
            (128, 128),  # Medium
            (256, 128),  # Large
            (512, 256),  # Very large
        ]

    print(f"🧪 Comprehensive Test Suite: {func_name}")
    print("=" * 50)

    results = []
    warmup_gpu()

    for seq_len, d_model in test_configs:
        print(f"\nTesting {seq_len}×{d_model}...")

        try:
            # Create test data
            Q, K, V = create_test_tensors(seq_len, d_model)

            # Test correctness
            correctness_results = test_correctness(triton_func, Q, K, V, verbose=False)

            # Benchmark performance
            triton_time, _ = benchmark_function(triton_func, Q, K, V)
            pytorch_time, _ = benchmark_function(pytorch_attention_reference, Q, K, V)

            speedup = pytorch_time / triton_time

            result = {
                'config': (seq_len, d_model),
                'config_str': f"{seq_len}×{d_model}",
                'is_correct': correctness_results['is_correct'],
                'max_diff': correctness_results['max_diff'],
                'triton_time_ms': triton_time,
                'pytorch_time_ms': pytorch_time,
                'speedup_vs_pytorch': speedup,
                'success': True
            }

            print(f"   Correctness: {'✅' if result['is_correct'] else '❌'}")
            print(f"   {func_name}: {triton_time:.2f} ms")
            print(f"   PyTorch: {pytorch_time:.2f} ms")
            print(f"   Speedup: {speedup:.2f}x")

            results.append(result)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            result = {
                'config': (seq_len, d_model),
                'config_str': f"{seq_len}×{d_model}",
                'error': str(e),
                'success': False
            }
            results.append(result)

    # Summary
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        avg_speedup = np.mean([r['speedup_vs_pytorch'] for r in successful_results])
        all_correct = all(r['is_correct'] for r in successful_results)

        print(f"\n📊 Summary:")
        print(f"   Tests passed: {len(successful_results)}/{len(results)}")
        print(f"   All correct: {'✅' if all_correct else '❌'}")
        print(f"   Average speedup: {avg_speedup:.2f}x vs PyTorch")

    return results


def compare_implementations(
    implementations: Dict[str, Callable],
    test_config: Tuple[int, int] = (512, 128)
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple implementations head-to-head.

    Args:
        implementations: Dict mapping names to functions
        test_config: (seq_len, d_model) for testing

    Returns:
        Dict mapping implementation names to results
    """
    seq_len, d_model = test_config
    Q, K, V = create_test_tensors(seq_len, d_model)

    print(f"🏁 Head-to-Head Comparison ({seq_len}×{d_model})")
    print("=" * 50)

    warmup_gpu()
    results = {}

    # Get PyTorch baseline
    pytorch_time, pytorch_output = benchmark_function(pytorch_attention_reference, Q, K, V)

    for name, func in implementations.items():
        try:
            # Test correctness
            output = func(Q, K, V)
            max_diff = torch.max(torch.abs(output - pytorch_output)).item()
            is_correct = torch.allclose(output, pytorch_output, rtol=1e-3, atol=1e-3)

            # Test performance
            time_ms, _ = benchmark_function(func, Q, K, V)
            speedup = pytorch_time / time_ms

            results[name] = {
                'time_ms': time_ms,
                'speedup_vs_pytorch': speedup,
                'is_correct': is_correct,
                'max_diff': max_diff
            }

            print(f"{name:20}: {time_ms:6.2f} ms ({speedup:4.2f}x) {'✅' if is_correct else '❌'}")

        except Exception as e:
            print(f"{name:20}: ❌ Error - {e}")
            results[name] = {'error': str(e)}

    print(f"{'PyTorch (baseline)':20}: {pytorch_time:6.2f} ms (1.00x) ✅")

    return results