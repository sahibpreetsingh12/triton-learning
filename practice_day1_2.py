# -*- coding: utf-8 -*-
"""practice_day1-2.ipynb

Original file is located at
    https://colab.research.google.com/drive/1ft9_CPQeIgs76opymduGZ5dn_iq-N6uo
"""

!pip install triton
!nvidia-smi

import triton
import triton.language as tl
import torch

"""# Scale each element by a constant
# Input: X = [1, 2, 3, 4], scale = 2.5
# Output: [2.5, 5.0, 7.5, 10.0]

## For scaling a vector
"""

@triton.jit
def scale_vector(input_ptr, scale_factor, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the input block
    input_block = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Scale it (no need for intermediate variable)
    scaled_block = input_block * scale_factor

    # Store the result
    tl.store(output_ptr + offsets, scaled_block, mask=mask)

"""# Sum each row of a 2D matrix
# Input: X =
[[1, 2, 3],

[4, 5, 6]]
# Output: [6, 15]
"""

@triton.jit
def sum_rows(input_ptr, output_ptr, M, N, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = pid_m * BLOCK_SIZE +tl.arange(0, BLOCK_SIZE)
    mask_m = offsets_m < N


    offsets_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offsets_n < M

    input_row = input_ptr + (offsets_m[:, None] * stride_m + offsets_n[None, :]*stride_n)
    input_block = tl.load(input_row, mask=mask_m[:, None], other=0.0)

    output_vector = sum(input_block, axis=0)
    output_row = output_ptr + pid_m
    tl.store(output_row, output_vector, mask=mask_n)

"""# Sum each row of a 2D matrix
# Input: X =

[[1, 2, 3],

[4, 5, 6]]     


Output: [6, 15]
"""

@triton.jit
def sum_rows(input_ptr, output_ptr, M, N, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # Which rows does this program handle?
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < M  # ← This should be M, not N!

    # For each of my rows, I need ALL columns [0, 1, 2, ..., N-1]
    col_offsets = tl.arange(0, N)

    # Create 2D pointer grid: [BLOCK_SIZE, N]
    ptrs = input_ptr + row_offsets[:, None] * stride_m + col_offsets[None, :] * stride_n

    # Load my block of rows (complete rows)
    values = tl.load(ptrs, mask=row_mask[:, None], other=0.0)

    # Sum each row separately (sum along axis 1 = columns)
    row_sums = tl.sum(values, axis=1)

    # Store results for my rows
    tl.store(output_ptr + row_offsets, row_sums, mask=row_mask)

"""## Transpose"""

import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, input_stride_m, input_stride_n, output_stride_m, output_stride_n, BLOCK_SIZE: tl.constexpr):
    """
    Matrix transpose kernel using block-based approach.

    Strategy: Each program handles a block of input rows and transposes them to output columns.
    For a 2x3 to 3x2 transpose:
    - Program reads input rows [0,1]
    - Transposes to output columns [0,1]
    """

    # Step 1: Figure out which chunk of work I'm responsible for
    pid = tl.program_id(axis=0)  # Which program am I in the grid?

    # Calculate which input rows this program will process
    # If pid=0, BLOCK_SIZE=2: I handle rows [0,1]
    input_row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_row_mask = input_row_offsets < M  # Don't go past the actual number of rows

    # Step 2: Handle Triton's "power of 2" constraint for column access
    # Our matrix has 3 columns, but tl.arange needs power of 2, so use 4
    input_col_offsets = tl.arange(0, 4)  # [0,1,2,3] - next power of 2 after 3
    input_col_mask = input_col_offsets < 3  # Only use first 3: [T,T,T,F]

    # Step 3: Calculate memory addresses for loading input data
    # Broadcasting magic: create 2D grid of memory addresses
    # input_row_offsets[:, None] = [[0],[1]] (column vector)
    # input_col_offsets[None, :] = [[0,1,2,3]] (row vector)
    # Result: 2x4 grid of pointers for input[rows, cols]
    input_ptrs = input_ptr + input_row_offsets[:, None] * input_stride_m + input_col_offsets[None, :] * input_stride_n

    # Step 4: Load input data with safety masks
    # Combine row mask (valid rows) with column mask (valid columns)
    # This prevents reading garbage data or going out of bounds
    combined_mask = input_row_mask[:, None] & input_col_mask[None, :]
    input_values = tl.load(input_ptrs, mask=combined_mask, other=0.0)

    # Step 5: The core operation - transpose the loaded block
    # input_values shape: [BLOCK_SIZE, 4] → [4, BLOCK_SIZE]
    # For our 2x3 case: [2,4] → [4,2] (with padding from power-of-2 constraint)
    transposed_values = tl.trans(input_values)

    # Step 6: Calculate where to store the transposed data in output
    # My input rows become output columns
    # If I processed input rows [0,1], I write to output columns [0,1]
    output_col_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_col_mask = output_col_offsets < M  # Don't write past valid columns

    # For output rows: again need power of 2 for the 3 rows → use 4
    output_row_offsets = tl.arange(0, 4)  # [0,1,2,3]
    output_row_mask = output_row_offsets < 3  # Only use [0,1,2]: [T,T,T,F]

    # Step 7: Calculate memory addresses for storing output data
    # Note the dimension swap: output is [N, M] while input was [M, N]
    # output_row_offsets[:, None] = [[0],[1],[2],[3]] (column vector)
    # output_col_offsets[None, :] = [[0,1]] (row vector)
    # Result: 4x2 grid of pointers for output[rows, cols]
    output_ptrs = output_ptr + output_row_offsets[:, None] * output_stride_m + output_col_offsets[None, :] * output_stride_n

    # Step 8: Store the transposed data with safety masks
    # Combine output row mask (valid rows) with output column mask (valid columns)
    combined_output_mask = output_row_mask[:, None] & output_col_mask[None, :]
    tl.store(output_ptrs, transposed_values, mask=combined_output_mask)


def run_transpose_example():
    """
    Example: Transpose a 2x3 matrix to 3x2

    Input:  [[1, 2, 3],     to    Output: [[1, 4],
             [4, 5, 6]]                     [2, 5],
                                           [3, 6]]

    Process:
    1. One program handles both input rows (BLOCK_SIZE=2)
    2. Loads input as 2x4 block (padded for power-of-2)
    3. Transposes to 4x2 block
    4. Stores as 3x2 output (masked to remove padding)
    """

    # Create test input: 2 rows, 3 columns
    X = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], device='cuda')

    M, N = X.shape  # M=2 rows, N=3 columns

    # Create output tensor: 3 rows, 2 columns (dimensions swapped)
    Y = torch.empty((N, M), device='cuda', dtype=X.dtype)

    # Kernel configuration
    BLOCK_SIZE = 2  # Each program handles 2 rows (must be power of 2)
    grid = (1,)     # Only need 1 program since BLOCK_SIZE=2 covers all M=2 rows

    # Launch the kernel
    transpose_kernel[grid](
        X, Y, M,                    # Input tensor, output tensor, number of rows
        X.stride(0), X.stride(1),   # Input memory layout: row stride, column stride
        Y.stride(0), Y.stride(1),   # Output memory layout: row stride, column stride
        BLOCK_SIZE                  # How many rows each program processes
    )

    # Show results
    print("Input matrix (2x3):")
    print(X)
    print("\nTransposed output (3x2):")
    print(Y)



# Run the example
if __name__ == "__main__":
    run_transpose_example()



