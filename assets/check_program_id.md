# Triton Grid Design - Quick Reference Table

## **Program Grid Decision Matrix**

| **Operation Type** | **Independent Unit** | **Typical Grid** | **Program ID Meaning** | **Example Code** |
|---|---|---|---|---|
| **Vector Operations** | Each element | `(N,)` | `id(0)` = element index | `y[i] = x[i] * scale` |
| **Row-wise Matrix** | Each row | `(M,)` | `id(0)` = row index | Layer norm, row softmax |
| **Column-wise Matrix** | Each column | `(N,)` | `id(0)` = column index | Column normalization |
| **Element-wise Matrix** | Each element | `(M,N)` or `(M*N,)` | `id(0,1)` = row,col | `C[i,j] = A[i,j] + B[i,j]` |
| **Block-wise Matrix** | Each block | `(blocks_M, blocks_N)` | `id(0,1)` = block row,col | Tiled matrix multiply |
| **Attention** | Each query | `(seq_len,)` | `id(0)` = query index | Fused attention |
| **Convolution** | Each output pixel | `(batch, out_h*out_w)` | `id(0,1)` = batch, pixel | 2D convolution |
| **Matrix Transpose** | Each element | `(M,N)` or `(N,M)` | `id(0,1)` = input or output coords | `B[j,i] = A[i,j]` |
| **Reduction Operations** | Each output element | `(output_size,)` | `id(0)` = output index | Sum along axis |
| **Broadcast Operations** | Each output element | `(output_shape)` | `id(...)` = output coordinates | Element-wise with broadcasting |

---

## **Grid Dimension Guidelines**

### **1D Grid: `(N,)`**
- **Use for**: Sequential processing, row/column operations, most reductions
- **Advantages**: Simple indexing, better occupancy, cache-friendly
- **Program ID**: `tl.program_id(0)` gives position in sequence
- **Best for**: Vector ops, matrix rows, attention queries

### **2D Grid: `(M, N)`**
- **Use for**: Element-wise matrix operations, image processing, convolutions
- **Advantages**: Natural coordinate mapping, intuitive for 2D problems
- **Program IDs**: 
  - `tl.program_id(0)` = first dimension (usually rows/height)
  - `tl.program_id(1)` = second dimension (usually cols/width)
- **Best for**: Matrix addition, transpose, convolution kernels

### **3D Grid: `(X, Y, Z)`**
- **Use for**: 3D tensor operations, batched 2D operations, volume processing
- **Advantages**: Direct mapping to 3D problems
- **Program IDs**:
  - `tl.program_id(0)` = first dimension (usually batch/depth)
  - `tl.program_id(1)` = second dimension (usually height)
  - `tl.program_id(2)` = third dimension (usually width)
- **Best for**: 3D convolutions, batched operations, volume rendering

---

## **Common Patterns**

### **Pattern 1: One-to-One Mapping**
```python
# Each program computes exactly one output element
grid = output_shape
program_coords = (tl.program_id(0), tl.program_id(1), ...)
output[program_coords] = compute_function(input[program_coords])
```

### **Pattern 2: One-to-Many Mapping**  
```python
# Each program computes multiple related outputs
grid = (num_groups,)
group_id = tl.program_id(0)
for i in range(elements_per_group):
    output[group_id * elements_per_group + i] = compute_function(...)
```

### **Pattern 3: Many-to-One Mapping**
```python
# Each program processes multiple inputs to compute one output
grid = (num_outputs,)
output_id = tl.program_id(0)
result = 0
for i in range(inputs_per_output):
    result += input[output_id * inputs_per_output + i]
output[output_id] = result
```

---

## **Decision Tree**

```
Start: What am I computing?
├─ Single vector/array
│  └─ Grid: (N,) where N = vector length
├─ Matrix with row-wise operations
│  └─ Grid: (M,) where M = number of rows
├─ Matrix with element-wise operations
│  ├─ Simple case: Grid: (M*N,) (flatten to 1D)
│  └─ Complex case: Grid: (M, N)
├─ 3D tensor operations
│  ├─ Simple case: Grid: (total_elements,)
│  └─ Complex case: Grid: (X, Y, Z)
└─ Custom operations
   └─ Grid: (number_of_independent_work_units,)
```

---

## **Performance Tips**

### **Prefer 1D When Possible**
- **Reason**: Better GPU occupancy and simpler addressing
- **Exception**: When 2D/3D makes memory access much more natural

### **Grid Size Considerations**
- **Small grids** (< 1K programs): Any dimension works
- **Medium grids** (1K-1M programs): 1D often better
- **Large grids** (> 1M programs): May need 2D due to GPU limits

### **Memory Access Patterns**
- **Row-major data**: Use row-wise grids when possible
- **Column-major data**: Use column-wise grids when possible
- **Blocked data**: Use block-wise grids

---

## **Example Grid Calculations**

### **Matrix Operations**
```python
# Input: Matrix A[1024, 512]
# Operation: Row normalization

# Decision: Each row is independent
grid = (1024,)  # One program per row
row_id = tl.program_id(0)  # Which row am I? (0 to 1023)
```

### **Attention**
```python
# Input: Q, K, V [seq_len=256, d_model=768]
# Operation: Self-attention

# Decision: Each query position is independent  
grid = (256,)  # One program per query
query_id = tl.program_id(0)  # Which query am I? (0 to 255)
```

### **Convolution**
```python
# Input: Image [batch=32, channels=3, height=224, width=224]
# Output: Features [batch=32, out_channels=64, out_h=222, out_w=222]
# Operation: 2D convolution

# Decision: Each output pixel is independent
grid = (32 * 64, 222 * 222)  # (batch*channels, spatial)
batch_channel = tl.program_id(0)  # Which (batch,channel) combo?
spatial_pos = tl.program_id(1)    # Which spatial position?
```

This reference table should be your go-to guide for designing any Triton kernel grid!
