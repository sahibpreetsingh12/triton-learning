"""
Shared utilities for Triton Attention Blog Series
================================================

This package provides common testing, benchmarking, and visualization
utilities used across all blog posts in the series.
"""

from .test_utils import (
    create_test_tensors,
    pytorch_attention_reference,
    test_correctness,
    warmup_gpu,
    benchmark_function,
    comprehensive_test_suite,
    compare_implementations
)

from .benchmark_utils import (
    PerformanceProfiler,
    VisualizationHelper,
    quick_benchmark,
    memory_efficiency_analysis,
    create_blog_summary_table
)

__all__ = [
    # Test utilities
    'create_test_tensors',
    'pytorch_attention_reference',
    'test_correctness',
    'warmup_gpu',
    'benchmark_function',
    'comprehensive_test_suite',
    'compare_implementations',

    # Benchmark utilities
    'PerformanceProfiler',
    'VisualizationHelper',
    'quick_benchmark',
    'memory_efficiency_analysis',
    'create_blog_summary_table'
]