# -*- coding: utf-8 -*-
"""practice-3-4.ipynb
Original file is located at
    https://colab.research.google.com/drive/12YQ6QjVZsvVO4buGSduuyqWkqOlskz0Z
"""

!pip install triton
!nvidia-smi

import triton
import triton.language as tl
import torch

"""## Row Softmax

"""

import torch
import triton
import triton.language as tl

@triton.jit
def row_softmax(X_ptr, Y_ptr, M, N, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_ptr = X_ptr + col_offsets * stride_n + pid * stride_m

    row_value = tl.load(row_ptr, mask=mask, other=0.0)

    # Find max value for numerical stability
    max_val = tl.max(row_value)

    # Subtract max (numerical stability trick)
    new_row_value = row_value - max_val

    # Compute exponentials
    exp_value = tl.exp(new_row_value)

    # Sum all exponentials
    row_sum = tl.sum(exp_value)

    # Normalize to get probabilities
    row_divide = exp_value / row_sum

    # Store result
    output_ptr = Y_ptr + col_offsets * stride_n + pid * stride_m
    tl.store(output_ptr, row_divide, mask=mask)


def triton_softmax(X: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax to each row of the input matrix
    """
    assert X.dim() == 2, "Input must be 2D"
    assert X.is_cuda, "Input must be on CUDA"
    assert X.is_contiguous(), "Input must be contiguous"

    M, N = X.shape

    # Create output tensor
    Y = torch.empty_like(X)

    # For power-of-2 constraint, find next power of 2 >= N
    BLOCK_SIZE = 1
    while BLOCK_SIZE < N:
        BLOCK_SIZE *= 2

    # Grid: one program per row
    grid = (M,)

    # Launch kernel
    row_softmax[grid](
        X, Y, M, N,
        X.stride(0), X.stride(1),  # stride_m, stride_n
        BLOCK_SIZE
    )

    return Y


# Test the function
def test_softmax():
    print("Testing Triton Softmax Implementation")
    print("=" * 40)

    # Test case 1: Your 3x4 example
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]], device='cuda')

    print("Input matrix (3x4):")
    print(X)

    # Our Triton softmax
    Y_triton = triton_softmax(X)

    # PyTorch reference
    Y_torch = torch.softmax(X, dim=1)  # dim=1 means row-wise

    print("\nTriton softmax result:")
    print(Y_triton)

    print("\nPyTorch reference:")
    print(Y_torch)

    print(f"\nResults match: {torch.allclose(Y_triton, Y_torch, rtol=1e-5)}")

    # Verify each row sums to 1
    print(f"\nRow sums (should be ~1.0):")
    print(f"Triton: {Y_triton.sum(dim=1)}")
    print(f"PyTorch: {Y_torch.sum(dim=1)}")

    # Test case 2: Larger matrix
    print("\n" + "=" * 40)
    print("Test case 2: Larger matrix")

    X2 = torch.randn(5, 8, device='cuda')
    Y2_triton = triton_softmax(X2)
    Y2_torch = torch.softmax(X2, dim=1)

    print(f"5x8 matrix - Results match: {torch.allclose(Y2_triton, Y2_torch, rtol=1e-5)}")
    print(f"Row sums close to 1: {torch.allclose(Y2_triton.sum(dim=1), torch.ones(5, device='cuda'))}")


if __name__ == "__main__":
    test_softmax()

