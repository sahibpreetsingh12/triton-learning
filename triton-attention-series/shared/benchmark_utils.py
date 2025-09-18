"""
Shared Benchmarking Utilities for Triton Attention Blog Series
==============================================================

This module provides common benchmarking and visualization functions
for performance analysis across all blog posts.
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Dict, Any, Tuple, Optional
from .test_utils import create_test_tensors, pytorch_attention_reference


class PerformanceProfiler:
    """
    Comprehensive performance profiler for attention implementations.
    """

    def __init__(self, warmup_runs: int = 10, timing_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
        self.results = {}

    def profile_function(
        self,
        func: Callable,
        name: str,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> Dict[str, float]:
        """Profile a single function"""

        # Warmup
        for _ in range(self.warmup_runs):
            _ = func(Q, K, V)
        torch.cuda.synchronize()

        # Timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(self.timing_runs):
            output = func(Q, K, V)

        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / self.timing_runs
        return {
            'name': name,
            'time_seconds': avg_time,
            'time_ms': avg_time * 1000,
            'output_shape': output.shape
        }

    def profile_scaling(
        self,
        implementations: Dict[str, Callable],
        seq_lengths: List[int],
        d_model: int = 128
    ) -> Dict[str, List[float]]:
        """Profile how implementations scale with sequence length"""

        print(f"📈 Scaling Analysis (d_model={d_model})")
        print("=" * 50)

        results = {name: [] for name in implementations.keys()}
        results['seq_lengths'] = seq_lengths

        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            Q, K, V = create_test_tensors(seq_len, d_model)

            for name, func in implementations.items():
                try:
                    profile_result = self.profile_function(func, name, Q, K, V)
                    time_ms = profile_result['time_ms']
                    results[name].append(time_ms)
                    print(f"  {name:20}: {time_ms:6.2f} ms")
                except Exception as e:
                    print(f"  {name:20}: ❌ {e}")
                    results[name].append(float('nan'))

        return results

    def profile_dimensions(
        self,
        implementations: Dict[str, Callable],
        d_models: List[int],
        seq_len: int = 512
    ) -> Dict[str, List[float]]:
        """Profile how implementations scale with model dimension"""

        print(f"📐 Dimension Scaling Analysis (seq_len={seq_len})")
        print("=" * 50)

        results = {name: [] for name in implementations.keys()}
        results['d_models'] = d_models

        for d_model in d_models:
            print(f"\nModel dimension: {d_model}")
            Q, K, V = create_test_tensors(seq_len, d_model)

            for name, func in implementations.items():
                try:
                    profile_result = self.profile_function(func, name, Q, K, V)
                    time_ms = profile_result['time_ms']
                    results[name].append(time_ms)
                    print(f"  {name:20}: {time_ms:6.2f} ms")
                except Exception as e:
                    print(f"  {name:20}: ❌ {e}")
                    results[name].append(float('nan'))

        return results


class VisualizationHelper:
    """
    Helper class for creating consistent visualizations across blog posts.
    """

    @staticmethod
    def plot_performance_comparison(
        implementations: Dict[str, List[float]],
        x_values: List[int],
        x_label: str,
        title: str,
        save_path: Optional[str] = None
    ):
        """Create a performance comparison plot"""

        plt.figure(figsize=(12, 8))

        # Define colors for consistency
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_map = {}

        i = 0
        for name, times in implementations.items():
            if name in ['seq_lengths', 'd_models']:  # Skip metadata
                continue

            color = colors[i % len(colors)]
            color_map[name] = color

            plt.plot(x_values, times, 'o-', label=name, linewidth=2, markersize=6, color=color)
            i += 1

        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_speedup_comparison(
        implementations: Dict[str, List[float]],
        baseline_name: str,
        x_values: List[int],
        x_label: str,
        title: str,
        save_path: Optional[str] = None
    ):
        """Create a speedup comparison plot"""

        if baseline_name not in implementations:
            print(f"Baseline '{baseline_name}' not found in results")
            return

        plt.figure(figsize=(12, 8))

        baseline_times = implementations[baseline_name]
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        i = 0
        for name, times in implementations.items():
            if name in ['seq_lengths', 'd_models', baseline_name]:
                continue

            speedups = [b/t if not np.isnan(t) and t > 0 else np.nan
                       for b, t in zip(baseline_times, times)]

            color = colors[i % len(colors)]
            plt.plot(x_values, speedups, 'o-', label=f'{name} vs {baseline_name}',
                    linewidth=2, markersize=6, color=color)
            i += 1

        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline (1.0x)')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_memory_analysis(seq_lengths: List[int], d_model: int):
        """Visualize theoretical memory usage patterns"""

        plt.figure(figsize=(12, 6))

        # Calculate theoretical memory usage
        basic_memory = []
        vectorized_memory = []

        for seq_len in seq_lengths:
            # Basic: load each key/value individually
            basic_ops = seq_len * seq_len * 2  # Load each K,V for each Q
            basic_memory.append(basic_ops * d_model * 4 / (1024**2))  # MB

            # Vectorized: load all keys/values once per query
            vectorized_ops = seq_len * 2  # Load all K,V once per Q
            vectorized_memory.append(vectorized_ops * d_model * 4 / (1024**2))  # MB

        plt.subplot(1, 2, 1)
        plt.plot(seq_lengths, basic_memory, 'o-', label='Basic (redundant loads)', color='#ff7f0e', linewidth=2)
        plt.plot(seq_lengths, vectorized_memory, 'o-', label='Vectorized (efficient)', color='#2ca02c', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Traffic (MB)')
        plt.title('Theoretical Memory Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')

        plt.subplot(1, 2, 2)
        efficiency_ratio = [b/v for b, v in zip(basic_memory, vectorized_memory)]
        plt.plot(seq_lengths, efficiency_ratio, 'o-', color='#d62728', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Efficiency Ratio')
        plt.title('Basic vs Vectorized Memory Efficiency')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')

        plt.tight_layout()
        plt.show()


def quick_benchmark(
    implementations: Dict[str, Callable],
    seq_len: int = 512,
    d_model: int = 128,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Quick benchmark for development/debugging"""

    Q, K, V = create_test_tensors(seq_len, d_model)
    profiler = PerformanceProfiler(warmup_runs=5, timing_runs=50)

    results = {}
    if verbose:
        print(f"🚀 Quick Benchmark ({seq_len}×{d_model})")
        print("-" * 40)

    for name, func in implementations.items():
        try:
            result = profiler.profile_function(func, name, Q, K, V)
            results[name] = result

            if verbose:
                print(f"{name:20}: {result['time_ms']:6.2f} ms")

        except Exception as e:
            if verbose:
                print(f"{name:20}: ❌ {e}")
            results[name] = {'error': str(e)}

    return results


def memory_efficiency_analysis(seq_lengths: List[int], d_model: int = 128):
    """Analyze memory efficiency patterns"""

    print("💾 Memory Efficiency Analysis")
    print("=" * 40)

    for seq_len in seq_lengths:
        # Theoretical calculations
        element_size = 4  # float32
        total_elements = seq_len * d_model

        # Memory per attention computation
        qkv_memory = 3 * total_elements * element_size  # Q, K, V storage
        output_memory = total_elements * element_size   # Output storage

        # Different access patterns
        basic_loads = seq_len * seq_len * 2 * d_model * element_size  # Redundant K,V loads
        vectorized_loads = seq_len * 2 * d_model * element_size       # Efficient K,V loads

        efficiency_improvement = basic_loads / vectorized_loads

        print(f"\nSequence Length: {seq_len}")
        print(f"  Total data: {(qkv_memory + output_memory) / (1024**2):.1f} MB")
        print(f"  Basic memory traffic: {basic_loads / (1024**2):.1f} MB")
        print(f"  Vectorized memory traffic: {vectorized_loads / (1024**2):.1f} MB")
        print(f"  Efficiency improvement: {efficiency_improvement:.1f}x")


def create_blog_summary_table(results: List[Dict[str, Any]]) -> str:
    """Create a markdown table summarizing results for blog posts"""

    if not results:
        return "No results to display"

    table = "| Configuration | Triton Time (ms) | PyTorch Time (ms) | Speedup | Correct |\n"
    table += "|---------------|------------------|-------------------|---------|----------|\n"

    for result in results:
        if result.get('success', False):
            config = result['config_str']
            triton_time = result['triton_time_ms']
            pytorch_time = result['pytorch_time_ms']
            speedup = result['speedup_vs_pytorch']
            correct = "✅" if result['is_correct'] else "❌"

            table += f"| {config} | {triton_time:.2f} | {pytorch_time:.2f} | {speedup:.2f}x | {correct} |\n"
        else:
            config = result['config_str']
            table += f"| {config} | Error | Error | - | ❌ |\n"

    return table